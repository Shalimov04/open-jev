"""PLAN-4 §3: the pair scorer's inference contract and the fold exclusion."""
import json
from pathlib import Path

import torch
import yaml

from openjev import base
from openjev.spec import load_task


def _task(tmp_path, body):
    p = tmp_path / "t.yaml"
    p.write_text(body + "data: {source: {csv: f}}\n")
    return load_task(p)


class _FakeModel:
    """Scores a pair by the length of segment A: deterministic, no download."""

    def __init__(self):
        self.seen = []

    def parameters(self):
        yield torch.zeros(1)

    def __call__(self, **enc):
        n = enc["input_ids"].shape[0]
        self.seen.append(n)
        return type("O", (), {"logits": torch.arange(n, dtype=torch.float).view(n, 1)})


def test_predict_logits_is_renormalised_log_probs(tmp_path, monkeypatch):
    t = _task(tmp_path, "name: t\ntype: choice\nquestion: Q?\noptions: [a, b, c, d]\n")
    monkeypatch.setattr(base, "_encode", lambda tok, b, dev: {"input_ids": torch.zeros(len(b), 4, dtype=torch.long)})
    out = base.predict_logits(None, _FakeModel(), t, ["x", "y"], batch_size=8)
    assert out.shape == (2, 4)
    assert torch.allclose(out.exp().sum(-1), torch.ones(2), atol=1e-5)   # a distribution, per row
    # batch 8 = row0's four pairs then row1's: sigmoid(0..3) then sigmoid(4..7), renormalised
    p = torch.sigmoid(torch.arange(8.0)).view(2, 4)
    assert torch.allclose(out.exp(), p / p.sum(-1, keepdim=True), atol=1e-5)


def test_predict_logits_noul_scores_one_pair(tmp_path, monkeypatch):
    """noul renders the same pair for both labels: score it once, return [log p, log(1-p)]."""
    t = _task(tmp_path, "name: t\ntype: noul\nstatement: S.\n")
    m = _FakeModel()
    monkeypatch.setattr(base, "_encode", lambda tok, b, dev: {"input_ids": torch.zeros(len(b), 4, dtype=torch.long)})
    out = base.predict_logits(None, m, t, ["x", "y", "z"], batch_size=8)
    assert m.seen == [3] and out.shape == (3, 2)       # 3 texts -> 3 forward passes, not 6
    assert torch.allclose(out.exp()[:, 0], torch.sigmoid(torch.arange(3.0)), atol=1e-5)
    assert torch.allclose(out.exp().sum(-1), torch.ones(3), atol=1e-5)


def test_excluded_is_the_whole_text_source_group(tmp_path):
    """Placebo: folding by task name would train on civil_comments while 'holding out' toxic."""
    assert base.excluded("none") == set() and base.excluded(None) == set()
    f1 = base.excluded("F1")
    assert {"toxic", "civil-insult", "kinopoisk", "swde-field", "swde-vertical"} <= f1
    assert "georeview" not in f1
    assert base.stem("kinopoisk-jev") == "kinopoisk" and base.stem("toxic") == "toxic"
    spec = yaml.safe_load(base.FOLDS.read_text())
    named = {t for k, v in spec.items() if k != "eval" for g in v.values() for t in g}
    for fold, tasks in spec["eval"].items():          # every evaluated task is in its own fold
        assert set(tasks) <= set(base.excluded(fold)) <= named


def test_calib_variants_differ_and_only_gold500_uses_gold(tmp_path):
    t = _task(tmp_path, "name: t\ntype: choice\nquestion: Q?\noptions: [a, b, c]\n")
    g = torch.randn(200, 3)
    logits = (g.softmax(-1)).log()
    rows = [{"gold": int(l.argmax()), "probs": p.tolist(), "id": str(i)}
            for i, (l, p) in enumerate(zip(g, g.softmax(-1)))]
    out = {v: base._calib_json(t, logits, rows, v) for v in base.VARIANTS}
    assert out["raw"] == {"temperature": 1.0, "method": "none", "bias": None, "target": "raw"}
    assert out["prior"]["bias"] is not None and out["prior"]["temperature"] == 1.0
    assert out["prior"]["target"] == "prior-uniform"          # no `prior:` in the YAML
    assert out["teacher500"]["bias"] is None and out["teacher500"]["temperature"] > 0
    assert out["gold500"]["target"] == "gold"
    assert json.dumps(out)                                     # every variant is JSON-serialisable


class _TinyModel(torch.nn.Module):
    """One parameter, so AdamW and .backward() work; logits ignore the input."""

    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(1))

    def forward(self, **enc):
        n = enc["input_ids"].shape[0]
        return type("O", (), {"logits": self.w.expand(n).unsqueeze(-1)})

    def save_pretrained(self, d):
        Path(d).mkdir(parents=True, exist_ok=True)
        (Path(d) / "saved").write_text("1")


def _fake_train(monkeypatch, bces, ntasks=3):
    """base.train with the mixture, the model and the calib BCE all faked; returns the run info."""
    from openjev import pairs as P

    idx = {f"t{i}": [(0, 2)] * 40 for i in range(ntasks)}
    monkeypatch.setattr(base, "device", lambda: "cpu")
    monkeypatch.setattr(P, "index", lambda split="train", **kw: dict(idx))
    monkeypatch.setattr(P, "tokenizer", lambda: type("T", (), {"save_pretrained": lambda self, d: None})())
    monkeypatch.setattr(P, "sample", lambda rows, cap, rng, **kw: iter(
        [{"seg_a": "a", "seg_b": "b", "target": 0.5} for _ in range(base.pairs.epoch_pairs(rows, cap))]))
    monkeypatch.setattr(base, "_encode", lambda tok, b, dev: {"input_ids": torch.zeros(len(b), 4, dtype=torch.long)})
    monkeypatch.setattr(base.AutoModelForSequenceClassification, "from_pretrained",
                        staticmethod(lambda *a, **kw: _TinyModel()))
    seq = iter(bces)
    monkeypatch.setattr(base, "_bce", lambda tok, model, rows, **kw: next(seq, bces[-1]))
    return idx


def test_max_epochs_sets_the_step_ceiling_and_not_converged(tmp_path, monkeypatch):
    """Placebo: fall back to --minutes and the run is a clock again, not 6 epochs."""
    _fake_train(monkeypatch, [0.5 - 0.01 * i for i in range(20)])  # always improving: never early-stops
    info = base.train("none", tmp_path / "r", cap=8, batch=4, eval_every=2, calib_cap=1,
                      max_epochs=3, min_delta=0.001, patience=3)
    assert info["pairs_per_epoch"] == 24               # 3 tasks x cap 8
    assert info["steps_planned"] == 3 * 24 // 4 == info["steps_run"]
    assert info["epochs_seen"] == 3
    assert info["converged"] is False                  # hit the ceiling: STOP rule §5


def test_patience_counts_min_delta_improvements_but_keeps_the_best_checkpoint(tmp_path, monkeypatch):
    """Placebo: with `improved = cb < best` a 0.0001 drift resets patience and the run never stops."""
    _fake_train(monkeypatch, [0.50, 0.4990, 0.4985, 0.4980])   # drifts by < min_delta three times
    info = base.train("none", tmp_path / "r", cap=8, batch=4, eval_every=2, calib_cap=1,
                      max_epochs=9, min_delta=0.001, patience=3)
    assert info["converged"] is True and info["steps_run"] == 8      # stopped at the 4th evaluation
    assert info["best_calib_bce"] == 0.4980                         # best, not the last non-improver
    assert (tmp_path / "r/student/saved").exists()


def test_only_restricts_the_mixture_to_the_named_run_dirs(tmp_path, monkeypatch):
    """Placebo: ignore `only` and the ablation arm trains on the 27 new tasks it is meant to drop."""
    _fake_train(monkeypatch, [0.5, 0.4], ntasks=4)
    info = base.train("none", tmp_path / "r", cap=8, batch=4, eval_every=2, calib_cap=1,
                      max_epochs=1, only=["t0", "t2"])
    assert info["train_tasks"] == ["t0", "t2"] and info["pairs_per_epoch"] == 16


def test_cli_base_train_passes_the_prereg_flags(monkeypatch):
    """Placebo: drop --min-delta from the CLI and the preregistered stop rule is not what ran."""
    from openjev import cli
    got = {}
    monkeypatch.setattr(base, "train", lambda *a, **kw: got.update(args=a, kw=kw) or {})
    cli.main(["base", "train", "--fold", "F1", "--tag", "v2-orig", "--seed", "2", "--max-epochs", "6",
              "--patience", "3", "--min-delta", "0.001", "--only", "agnews", "banking77"])
    assert got["args"][0] == "F1" and str(got["args"][1]).endswith("runs/base-F1-v2-orig")
    assert got["kw"]["max_epochs"] == 6 and got["kw"]["min_delta"] == 0.001
    assert got["kw"]["patience"] == 3 and got["kw"]["only"] == ["agnews", "banking77"]
    assert got["kw"]["seed"] == 2
