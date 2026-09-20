import json
import random
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from openjev.calibrate import ece, fit_temperature
from openjev.views import view


def test_views():
    c = view("choice", ["a", "b", "c"], [0.1, 0.7, 0.2])
    assert c["choice"] == "b" and c["confidence"] == pytest.approx(0.7) and set(c["probabilities"]) == {"a", "b", "c"}
    s = view("score", ["1", "2", "3"], [0.2, 0.3, 0.5], [1.0, 2.0, 3.0])
    assert s["score"] == pytest.approx(2.3) and 1 <= s["score"] <= 3
    n = view("noul", ["true", "false"], [0.3, 0.7])
    assert n == {"probability": pytest.approx(0.3), "confidence": pytest.approx(0.7)}


def test_temperature_scaling_fixes_overconfidence():
    torch.manual_seed(0)
    y = torch.randint(0, 4, (2000,))
    logits = torch.randn(2000, 4) + 1.5 * torch.nn.functional.one_hot(y, 4)
    logits *= 4  # overconfident
    t = fit_temperature(logits, y)
    assert t > 1.5
    nll = lambda z: torch.nn.functional.cross_entropy(z, y).item()
    assert nll(logits / t) < nll(logits)
    assert ece((logits / t).softmax(-1), y) < ece(logits.softmax(-1), y)


def test_temperature_no_gold_fits_the_teachers_softness():
    """B1: with no gold the target is the teacher's soft probs, so T flattens the student toward
    the teacher instead of sharpening it onto the teacher's argmax."""
    torch.manual_seed(0)
    logits = 2.0 * torch.randn(500, 4)
    teacher = (logits / 2.5).softmax(-1)  # the teacher is softer than the raw student
    t_soft = fit_temperature(logits, teacher)
    assert t_soft == pytest.approx(2.5, abs=0.1)
    assert fit_temperature(logits, teacher.argmax(-1)) < 1.0 < t_soft  # the old, sharpening target


def test_run_dir_suffixes(tmp_path):
    """The run dir is the experiment's identity: seed 0 is the baseline dir, everything else adds a
    suffix, and variants symlink the base run's (frozen) teacher labels."""
    import argparse

    from openjev.cli import setup

    (tmp_path / "t.yaml").write_text(
        "name: toy\ntype: choice\nquestion: q\noptions: [a, b]\ndata: {source: {jsonl: x.jsonl}}\n")

    def dir_for(**kw):
        a = argparse.Namespace(**{"task": tmp_path / "t.yaml", "runs": str(tmp_path / "runs"),
                                  "student": None, "gold_weight": None, "batch_size": None,
                                  "epochs": None, "seed": 0, "tag": None, "gold_n": None, **kw})
        return setup(a)[1].name

    assert dir_for() == "toy"
    assert dir_for(seed=2) == "toy-s2"
    assert dir_for(tag="e12") == "toy-e12"
    assert dir_for(tag="nosynth", seed=1) == "toy-nosynth-s1"
    assert dir_for(gold_weight=1.0, gold_n=500) == "toy-gold-n500"
    assert (tmp_path / "runs/toy-s2/teacher.jsonl").is_symlink()


def test_report_groups_seeds(tmp_path):
    from openjev.evaluate import report

    base = {"type": "choice", "k": 2, "lang": "en", "n_train": 10, "gold_weight": 0,
            "calib_target": "gold", "temperature": 1.0, "eval_n": 4, "student_model": "m",
            "teacher": {"acc": 0.5, "macro_f1": 0.5}, "agreement": {"argmax": 0.5}}
    res = tmp_path / "results"
    res.mkdir()
    for name, acc in (("toy", 0.80), ("toy-s1", 0.90), ("toy-s2", 0.70)):
        (res / f"{name}.json").write_text(json.dumps(
            dict(base, student={"acc": acc, "macro_f1": acc, "ece_raw": 0.1, "ece_cal": 0.05, "brier": 0.2})))
    rows = [l for l in report(res, tmp_path / "R.md").splitlines() if l.startswith("| toy")]
    assert len(rows) == 1, rows                      # the three seeds are one row
    assert "toy (n=3)" in rows[0] and "0.800 ± 0.100" in rows[0]
    assert "0.500 /" in rows[0]                      # frozen teacher: sd 0, no "±"
    assert rows[0].rstrip().endswith("| – | – |")    # no latency measured -> dash, not 0.0


def test_end_to_end_tiny(tmp_path):
    from fastapi.testclient import TestClient

    from openjev import calibrate, evaluate, serve, train
    from openjev.spec import load_task

    (tmp_path / "t.yaml").write_text(
        "name: toy\ntype: choice\nquestion: sport or not?\noptions: [sport, other]\n"
        "data: {source: {jsonl: x.jsonl}}\nstudent: {batch_size: 8, epochs: 1, max_len: 32}\n")
    task = load_task(tmp_path / "t.yaml")
    run_dir = tmp_path / "toy"
    run_dir.mkdir()
    rng = random.Random(0)
    with open(run_dir / "teacher.jsonl", "w") as f:
        for split, n in (("train", 32), ("calib", 16), ("eval", 16)):
            for j in range(n):
                g = rng.randint(0, 1)
                f.write(json.dumps({"id": f"{split}{j}", "split": split, "gold": g,
                                    "text": ["the team won the match", "stocks fell today"][g],
                                    "probs": [[0.9, 0.1], [0.1, 0.9]][g], "raw": {}, "source": "data"}) + "\n")
    info = train.run(task, run_dir, max_steps=3)
    assert (run_dir / "student" / "config.json").exists() and info["history"]
    cal = calibrate.run(task, run_dir)
    assert cal["temperature"] > 0 and cal["target"] == "gold"
    assert (run_dir / "calib_logits.pt").exists()
    res = evaluate.run(task, run_dir, results_dir=tmp_path / "results", latency=False)
    assert 0 <= res["student"]["acc"] <= 1 and (run_dir / "openjev.json").exists()
    er = [json.loads(l) for l in (run_dir / "eval_rows.jsonl").read_text().splitlines()]
    assert len(er) == 16 and set(er[0]) == {"id", "gold", "student_probs", "student_logits", "teacher_probs"}
    assert sum(er[0]["student_probs"]) == pytest.approx(1.0)
    # A8: what produced the number. Placebo: drop `_env` from the res.update and this fails.
    saved = json.loads((tmp_path / "results" / "toy.json").read_text())
    assert saved["env"]["torch"] == torch.__version__ and saved["env"]["student_model"]
    evaluate.report(tmp_path / "results", tmp_path / "README.md")
    assert "| toy | choice | 2 |" in (tmp_path / "README.md").read_text()
    client = TestClient(serve.app_for([run_dir], "cpu"))
    r = client.post("/v1/systemone", json={"model": "toy", "input": "a goal"})
    assert r.status_code == 200 and r.json()["choice"] in ("sport", "other")

    # batch in, list out; one item per input, in order
    r = client.post("/v1/systemone", json={"model": "toy", "input": ["a goal", "stocks fell"],
                                           "escalate_below": 0.99, "coverage": 0.9})
    body = r.json()
    assert r.status_code == 200 and isinstance(body, list) and len(body) == 2
    assert all(b["escalate"] == (b["confidence"] < 0.99) for b in body)
    assert all(b["set"] and set(b["set"]) <= {"sport", "other"} for b in body)
    assert body[0]["set_target"] == "gold" and (run_dir / "conformal.json").exists()
    # coverage 1.0 cannot be certified from 16 calib rows -> every label, never an empty set
    r = client.post("/v1/systemone", json={"model": "toy", "input": "a goal", "coverage": 1.0})
    assert sorted(r.json()["set"]) == ["other", "sport"]
    # with no escalate_below in the request, the default comes from openjev.json's parity tau
    meta = json.loads((run_dir / "openjev.json").read_text())
    b = client.post("/v1/systemone", json={"model": "toy", "input": "a goal"}).json()
    assert ("escalate" in b) == (meta["escalate_below"] is not None)
    if meta["escalate_below"] is not None:
        assert b["escalate"] == (b["confidence"] < meta["escalate_below"])

    from openjev import hub
    card = hub.card(run_dir, "u/toy")
    assert "No gold labels were used in the loss" in card and "sport" in card
    assert hub.push(run_dir, "u/toy", dry_run=True)["files"] == [
        "README.md", "conformal.json", "openjev.json", "student/config.json",
        "student/model.safetensors", "student/tokenizer.json", "student/tokenizer_config.json"]
    assert hub.resolve("runs/x") == "runs/x"  # only hf: is special-cased


def test_jev_adapter_shape(monkeypatch):
    """The `jev` adapter builds the documented request and reads the documented answer. No key and
    no network: a fake client records the body. Shape from https://docs.typesafe.ai/api.md."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "compare_api", Path(__file__).resolve().parent.parent / "scripts" / "compare_api.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setenv("JEV_API_KEY", "k")
    seen = {}

    class Fake:
        def post(self, url, json=None, headers=None):
            seen.update(url=url, body=json, headers=headers)
            return SimpleNamespace(
                raise_for_status=lambda: None,
                json=lambda: {"answers": {"q": {"type": "choice", "choice": "b",
                                                "probabilities": {"a": 0.3, "b": 0.7},
                                                "confidence": 0.6}}})

    meta = {"labels": ["a", "b"], "task": {"type": "choice", "question": "which?"}}
    assert mod.jev_adapter(Fake(), "https://api.typesafe.ai", "jev-latest", meta, "hi") == [0.3, 0.7]
    assert seen["url"].endswith("/v1/systemone")
    assert seen["body"]["state"] == "hi" and seen["body"]["model"] == "jev-latest"
    assert seen["body"]["questions"]["q"]["type"] == "choice"
    assert sorted(seen["body"]["questions"]["q"]["criteria"]) == ["a", "b"]
    assert seen["headers"]["Authorization"] == "Bearer k"


def test_oracle_next_arm_is_exactly_wrong(tmp_path):
    """Oracle placebo: point the teacher at gold instead of (gold+1)%K and both asserts flip.

    A teacher that always answers the *next* option travels the real train/calibrate/eval path and
    must come out at accuracy 0; for K=2 the student then agrees with it on exactly the rows it
    gets wrong. Catches a gold/teacher mix-up anywhere in eval's scoring.
    """
    from openjev import calibrate, evaluate, train
    from openjev.spec import load_task

    (tmp_path / "t.yaml").write_text(
        "name: oracle\ntype: choice\nquestion: sport or not?\noptions: [sport, other]\n"
        "data: {source: {jsonl: x.jsonl}}\nstudent: {batch_size: 8, epochs: 1, max_len: 32}\n")
    task = load_task(tmp_path / "t.yaml")
    run_dir = tmp_path / "oracle"
    run_dir.mkdir()
    rng = random.Random(0)
    with open(run_dir / "teacher.jsonl", "w") as f:
        for split, n in (("train", 32), ("calib", 16), ("eval", 16)):
            for j in range(n):
                g = rng.randint(0, 1)
                nxt = (g + 1) % task.k                       # the oracle-next arm
                f.write(json.dumps({"id": f"{split}{j}", "split": split, "gold": g,
                                    "text": ["the team won the match", "stocks fell today"][g],
                                    "probs": [[0.9, 0.1], [0.1, 0.9]][nxt], "raw": {},
                                    "source": "data"}) + "\n")
    train.run(task, run_dir, max_steps=3)
    calibrate.run(task, run_dir)
    res = evaluate.run(task, run_dir, results_dir=tmp_path / "results", latency=False)
    assert res["teacher"]["acc"] == 0.0
    assert res["agreement"]["argmax"] == pytest.approx(1 - res["student"]["acc"])
