"""Temperature scaling (LBFGS on NLL) + ECE/Brier helpers."""
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from openjev.data import read_rows
from openjev.train import load_student, predict_logits


def ece(probs, targets, bins=15):
    conf, pred = probs.max(-1)
    correct = (pred == targets).float()
    edges = torch.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.float().mean() * (conf[m].mean() - correct[m].mean()).abs()
    return float(total)


def brier(probs, targets):
    return float(((probs - F.one_hot(targets, probs.shape[-1]).float()) ** 2).sum(-1).mean())


def fit_temperature(logits, targets):
    """`targets`: class indices, or a [N, K] probability matrix (soft-target cross-entropy)."""
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(logits / log_t.exp(), targets)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.detach().exp())


def targets_for(rows):
    """Gold when every row has it, else teacher argmax. Returns (targets, 'gold'|'teacher')."""
    if all(r.get("gold") is not None for r in rows):
        return torch.tensor([r["gold"] for r in rows]), "gold"
    return torch.tensor([max(range(len(r["probs"])), key=r["probs"].__getitem__) for r in rows]), "teacher"


def run(task, run_dir):
    run_dir = Path(run_dir)
    rows = read_rows(run_dir, "calib")
    tok, model = load_student(run_dir / "student")
    logits = predict_logits(tok, model, [r["text"] for r in rows], task.student.max_len)
    y, target = targets_for(rows)
    if target == "gold":
        t = fit_temperature(logits, y)
    else:  # no gold: fit to the teacher's *soft* probs, the thing the student was trained on.
        p = torch.tensor([r["probs"] for r in rows]).clamp_min(1e-6)
        t = fit_temperature(logits, p / p.sum(-1, keepdim=True))
        target = "teacher-soft"
    out = {"temperature": t, "target": target,
           "ece_before": ece(logits.softmax(-1), y), "ece_after": ece((logits / t).softmax(-1), y)}
    (run_dir / "calib.json").write_text(json.dumps(out, indent=2))
    return out
