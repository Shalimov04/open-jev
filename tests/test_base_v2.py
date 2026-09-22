"""docs/prereg/base-v2.md §4: the two verdict rules, on synthetic run dirs.

Placebo for the whole file: drop the MAE sign flip and a *worse* arm reads as an improvement.
"""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
spec = importlib.util.spec_from_file_location("base_v2", ROOT / "scripts/base_v2.py")
bv2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bv2)


def _dir(tmp, name, probs, gold, teacher=None, type_="choice", labels=("a", "b"), values=None):
    d = tmp / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "openjev.json").write_text(json.dumps(
        {"task": {"type": type_}, "labels": list(labels), "values": values}))
    with open(d / "eval_rows.jsonl", "w") as f:
        for i, (p, g) in enumerate(zip(probs, gold)):
            f.write(json.dumps({"id": str(i), "gold": g, "student_probs": p,
                                "teacher_probs": (teacher or probs)[i]}) + "\n")
    return d


def _split(n, acc):
    """n rows, gold alternating 0/1 (so a uniform matrix scores 0.5), right on `acc` of them."""
    gold = [i % 2 for i in range(n)]
    probs = [[0.9, 0.1] if (g == 0) == (i < int(n * acc)) else [0.1, 0.9]
             for i, g in enumerate(gold)]
    return probs, gold


@pytest.fixture
def patched(tmp_path, monkeypatch):
    monkeypatch.setattr(bv2, "RUNS", tmp_path)
    monkeypatch.setattr(bv2, "MIN_ROWS", {**bv2.MIN_ROWS, "t": 10})
    return tmp_path


def test_ablation_calls_a_real_gap_an_improvement(patched):
    for s in ("", "-s1", "-s2"):
        _dir(patched, f"A{s}-zs-t-gold500", *_split(400, 0.80))
        _dir(patched, f"B{s}-zs-t-gold500", *_split(400, 0.60))
    r, = bv2.ablation("gold500", a="A", b="B", tasks=["t"], n_boot=200)
    assert r["delta"] == pytest.approx(0.20) and r["excludes_0"] and r["same_sign"]
    assert r["improvement"] and r["non_inferior"] and not r["inconclusive"]
    assert r["per_seed"] == pytest.approx([0.20] * 3)


def test_ablation_below_threshold_is_not_an_improvement(patched):
    """1 pt with a tight CI: the CI excludes 0, the preregistered 2-pt threshold does not."""
    for s in ("", "-s1", "-s2"):
        _dir(patched, f"A{s}-zs-t-gold500", *_split(4000, 0.805))
        _dir(patched, f"B{s}-zs-t-gold500", *_split(4000, 0.795))
    r, = bv2.ablation("gold500", a="A", b="B", tasks=["t"], n_boot=200)
    assert r["excludes_0"] and r["delta"] == pytest.approx(0.01) and not r["improvement"]


def test_ablation_mae_sign_is_flipped(patched):
    """score tasks: a *lower* MAE is the better arm, so Δ > 0 must mean ABL-all is better."""
    kw = dict(type_="score", labels=("1", "2"), values=[1.0, 2.0])
    for s in ("", "-s1", "-s2"):
        _dir(patched, f"A{s}-zs-t-gold500", [[1.0, 0.0]] * 200, [0] * 200, **kw)   # MAE 0.0
        _dir(patched, f"B{s}-zs-t-gold500", [[0.5, 0.5]] * 200, [0] * 200, **kw)   # MAE 0.5
    r, = bv2.ablation("gold500", a="A", b="B", tasks=["t"], n_boot=200)
    assert r["metric"] == "mae" and r["a"] == pytest.approx(0.0) and r["b"] == pytest.approx(0.5)
    assert r["delta"] == pytest.approx(0.5) and r["improvement"] and r["per_seed"][0] > 0


def test_unseen_reports_advantage_over_chance_and_the_teacher_ratio(patched, monkeypatch):
    monkeypatch.setattr(bv2, "UNSEEN", ["t"])
    probs, gold = _split(400, 0.70)
    teacher, _ = _split(400, 0.90)
    _dir(patched, "M-zs-t-gold500", probs, gold, teacher=teacher)
    r, = bv2.unseen("gold500", arm="M", n_boot=200)
    assert (r["chance"], r["base"], r["teacher"]) == pytest.approx((0.5, 0.70, 0.90))
    assert r["adv"] == pytest.approx(0.20) and r["beats_chance"] and r["rows_ok"]
    assert r["normalised"] == pytest.approx(0.5)      # half of what the teacher recovers
