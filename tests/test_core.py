import json
import random

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
    res = evaluate.run(task, run_dir, results_dir=tmp_path / "results", latency=False)
    assert 0 <= res["student"]["acc"] <= 1 and (run_dir / "openjev.json").exists()
    evaluate.report(tmp_path / "results", tmp_path / "README.md")
    assert "| toy | choice | 2 |" in (tmp_path / "README.md").read_text()
    r = TestClient(serve.app_for([run_dir])).post("/v1/systemone", json={"model": "toy", "input": "a goal"})
    assert r.status_code == 200 and r.json()["choice"] in ("sport", "other")
