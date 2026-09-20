"""Temperature / vector scaling (LBFGS on NLL) + ECE/Brier helpers.

`apply(logits, calib)` is the one place calibration happens: `logits / T + bias`. calibrate, evaluate
and serve all go through it, so a run dir's calib.json fully describes what the served probs are."""
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


def fit_vector(logits, targets):
    """Vector scaling: `logits / T + b`, b in R^K. Same CE loss as fit_temperature. -> (T, b)."""
    log_t = torch.zeros(1, requires_grad=True)
    b = torch.zeros(logits.shape[-1], requires_grad=True)
    opt = torch.optim.LBFGS([log_t, b], lr=0.1, max_iter=200)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(logits / log_t.exp() + b, targets)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.detach().exp()), b.detach()


def fit_prior_bias(logits, prior, iters=50):
    """No gold at all: b such that the *mean* calibrated probability over these rows equals the
    declared prior. Multiplicative fixed point, ~50 steps (it converges geometrically)."""
    p = torch.as_tensor(prior, dtype=torch.float)
    b = torch.zeros(len(p))
    for _ in range(iters):
        b = b + (p / (logits + b).softmax(-1).mean(0).clamp_min(1e-9)).log()
    return b - b.mean()  # softmax is shift-invariant; centre it so the numbers are readable


def prior_vector(task, prior=None):
    """`prior: uniform | {label: p}` -> a normalised K-vector, or None when nothing is declared."""
    prior = prior if prior is not None else getattr(task, "prior", None)
    if prior is None:
        return None
    if prior == "uniform":
        return [1.0 / task.k] * task.k
    if isinstance(prior, dict):
        missing = [l for l in task.labels if l not in prior]
        if missing:
            raise ValueError(f"prior: missing labels {missing} (needs a weight for every label)")
        v = [float(prior[l]) for l in task.labels]
        return [x / sum(v) for x in v]
    raise ValueError(f"prior must be 'uniform' or a {{label: weight}} mapping, got {prior!r}")


def apply(logits, calib):
    """The single calibration path: logits / T (+ bias). Returns *logits*, not probs."""
    z = logits / calib.get("temperature", 1.0)
    b = calib.get("bias")
    return z + torch.as_tensor(b, dtype=z.dtype) if b else z


def _nll(logits, targets):
    return float(F.cross_entropy(logits, targets))


def select_method(logits, targets, folds=2):
    """A K-dim bias fitted on 500 rows overfits (K=77). Split the calib rows, fit each half, score
    NLL on the other: `vector` only if it wins on *every* held-out half."""
    idx = torch.arange(len(logits))
    for f in range(folds):
        te = idx % folds == f
        tr = ~te
        t = fit_temperature(logits[tr], targets[tr])
        tv, bv = fit_vector(logits[tr], targets[tr])
        if _nll(logits[te] / tv + bv, targets[te]) >= _nll(logits[te] / t, targets[te]):
            return "temperature"
    return "vector"


def targets_for(rows, ignore_gold=False):
    """Gold when every row has it, else teacher argmax. Returns (targets, 'gold'|'teacher')."""
    if not ignore_gold and all(r.get("gold") is not None for r in rows):
        return torch.tensor([r["gold"] for r in rows]), "gold"
    return torch.tensor([max(range(len(r["probs"])), key=r["probs"].__getitem__) for r in rows]), "teacher"


def run(task, run_dir, ignore_gold=False, prior=None):
    """ignore_gold: pretend the calib split has no gold labels (this is what makes a
    `prior-declared` number a genuine no-gold number)."""
    run_dir = Path(run_dir)
    rows = read_rows(run_dir, "calib")
    tok, model = load_student(run_dir / "student")
    logits = predict_logits(tok, model, [r["text"] for r in rows], task.student.max_len)
    torch.save({"ids": [r["id"] for r in rows], "logits": logits}, run_dir / "calib_logits.pt")
    y, target = targets_for(rows, ignore_gold)
    prior = prior_vector(task, prior)
    method, bias = "temperature", None
    if target == "gold":
        method = select_method(logits, y)
        t, bias = fit_vector(logits, y) if method == "vector" else (fit_temperature(logits, y), None)
    else:  # no gold: fit T to the teacher's *soft* probs, the thing the student was trained on.
        p = torch.tensor([r["probs"] for r in rows]).clamp_min(1e-6)
        t = fit_temperature(logits, p / p.sum(-1, keepdim=True))
        target = "teacher-soft"
        if prior is not None:  # ... then move the marginal onto the declared prior. Still no gold.
            bias, method, target = fit_prior_bias(logits / t, prior), "vector", "prior-declared"
    out = {"temperature": t, "method": method, "bias": bias if bias is None else bias.tolist(),
           "target": target}
    before, after = ece(logits.softmax(-1), y), ece(apply(logits, out).softmax(-1), y)
    if after > before and method == "temperature":
        # the original guard, unchanged: on an already-calibrated model, a T fitted on ~500 rows
        # just adds noise, so keep 1.0. It deliberately does *not* apply to a `vector` fit: that one
        # has already passed a stricter, held-out test (NLL on both 2-fold halves). A bias moves the
        # argmax, a temperature never does, so the two need different guards: on headlines seed 1
        # this guard vetoed a +7 pt accuracy gain because calib-split ECE moved 0.038 -> 0.051.
        # ECE is reported, never the headline (PLAN-2 §5).
        out.update(temperature=1.0, method="none", target=target + "-kept-1.0")
        after = before
    out.update(ece_before=before, ece_after=after)
    (run_dir / "calib.json").write_text(json.dumps(out, indent=2))
    return out
