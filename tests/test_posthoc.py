"""M1 post-hoc suite: vector scaling + its 2-fold guard, the declared-prior bias, the cascade curve
and the paired-bootstrap `compare`. All CPU, no run dir needed except two the tests fabricate."""
import json

import torch

from openjev import calibrate, evaluate


def _logits(n=600, k=3, shift=None, seed=0):
    g = torch.Generator().manual_seed(seed)
    y = torch.randint(k, (n,), generator=g)
    z = torch.randn(n, k, generator=g) + 2.0 * torch.nn.functional.one_hot(y, k)
    if shift is not None:
        z = z + torch.tensor(shift)
    return z, y


def test_apply_is_the_one_calibration_path():
    z = torch.tensor([[1.0, 2.0, 3.0]])
    assert torch.allclose(calibrate.apply(z, {"temperature": 2.0}), z / 2)
    assert torch.allclose(calibrate.apply(z, {"temperature": 2.0, "bias": [1.0, 0.0, -1.0]}),
                          z / 2 + torch.tensor([1.0, 0.0, -1.0]))
    assert torch.allclose(calibrate.apply(z, {}), z)  # a calib.json without a bias still works


def test_fit_vector_beats_temperature_when_a_class_is_shifted():
    # a student whose logits carry a constant offset on class 0: exactly what a K-dim bias fixes
    z, y = _logits(shift=[3.0, 0.0, 0.0])
    t = calibrate.fit_temperature(z, y)
    tv, b = calibrate.fit_vector(z, y)
    assert calibrate._nll(z / tv + b, y) < calibrate._nll(z / t, y)
    assert calibrate.select_method(z, y) == "vector"
    assert b.argmin().item() == 0  # the bias pushes the over-predicted class back down


def test_vector_guard_rejects_a_bias_that_only_fits_noise():
    # K=40 with 200 rows: the bias has more parameters than it has evidence, and must be refused
    z, y = _logits(n=200, k=40, seed=1)
    assert calibrate.select_method(z, y) == "temperature"


def test_prior_bias_moves_the_mean_probability_onto_the_declared_prior():
    z, _ = _logits(shift=[3.0, 0.0, 0.0])
    prior = [1 / 3, 1 / 3, 1 / 3]
    assert (z.softmax(-1).mean(0)[0] > 0.5)  # skewed before
    b = calibrate.fit_prior_bias(z, prior)
    assert torch.allclose((z + b).softmax(-1).mean(0), torch.tensor(prior), atol=1e-3)


def test_prior_vector_forms():
    task = evaluate.SimpleNamespace(type="choice", labels=["a", "b"], values=None, k=2, prior=None)
    assert calibrate.prior_vector(task) is None
    assert calibrate.prior_vector(task, "uniform") == [0.5, 0.5]
    assert calibrate.prior_vector(task, {"a": 3, "b": 1}) == [0.75, 0.25]


def _task(type="choice", k=3):
    return evaluate.SimpleNamespace(type=type, labels=[str(i) for i in range(k)],
                                    values=[float(i) for i in range(k)], k=k)


def test_cascade_curve_and_parity():
    n = 400
    y = torch.arange(n) % 3
    teacher = torch.nn.functional.one_hot(y, 3).float() * 0.8 + 0.0667      # teacher is always right
    student = teacher.clone()
    student[:100] = torch.tensor([0.4, 0.35, 0.25])   # a quarter of the rows: unsure, mostly wrong
    curve, parity = evaluate.cascade(_task(), student, teacher, y)
    assert [c["tau"] for c in curve][:3] == [0.3, 0.35, 0.4]
    assert curve[0]["escalation"] == 0.0 and curve[-1]["escalation"] == 1.0
    assert [c["escalation"] for c in curve] == sorted(c["escalation"] for c in curve)
    assert curve[-1]["acc"] == parity["teacher"] == 1.0          # escalate everything = the teacher
    # the unconfident quarter is exactly the wrong quarter, so parity costs 25 % escalation
    assert parity["parity"]["escalation"] == 0.25


def _fake_run(d, probs, gold, ids=None):
    d.mkdir(parents=True, exist_ok=True)
    (d / "openjev.json").write_text(json.dumps(
        {"task": {"type": "choice"}, "labels": ["0", "1"], "values": None}))
    with open(d / "eval_rows.jsonl", "w") as f:
        for i, (p, g) in enumerate(zip(probs, gold)):
            f.write(json.dumps({"id": (ids or range(len(gold)))[i], "gold": int(g),
                                "student_probs": list(p), "teacher_probs": [0.5, 0.5]}) + "\n")


def test_compare_calls_a_tie_a_tie_and_a_real_gap_a_gap(tmp_path, capsys):
    n = 400
    gold = [i % 2 for i in range(n)]
    right = [[1.0, 0.0] if g == 0 else [0.0, 1.0] for g in gold]
    wrong = [list(reversed(p)) for p in right]
    _fake_run(tmp_path / "a", right, gold)
    _fake_run(tmp_path / "b", right, gold)
    _fake_run(tmp_path / "c", wrong[:n // 4] + right[n // 4:], gold)
    tie = evaluate.compare(tmp_path / "a", tmp_path / "b", n_boot=200)
    assert tie["delta"] == 0 and tie["ci"] == [0, 0]
    assert tie["verdict"].startswith("no measurable difference (Δ = +0.0 ± 0.0 pts")
    gap = evaluate.compare(tmp_path / "a", tmp_path / "c", n_boot=200)
    assert gap["ci"][0] > 0 and gap["verdict"].startswith("A is better")
    assert "| seed | A | B |" in capsys.readouterr().out


def test_compare_pairs_seeds_and_shared_rows_only(tmp_path):
    gold = [0, 1] * 50
    right = [[1.0, 0.0] if g == 0 else [0.0, 1.0] for g in gold]
    for name in ("a", "a-s1", "b", "b-s1", "b-s2"):
        _fake_run(tmp_path / name, right, gold)
    r = evaluate.compare(tmp_path / "a", tmp_path / "b", n_boot=50)
    assert r["seeds"] == [0, 1]           # b-s2 has no partner in arm A
    assert r["n_rows"] == len(gold)
