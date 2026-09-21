"""PLAN-4 §3: the pair scorer's inference contract and the fold exclusion."""
import json

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
