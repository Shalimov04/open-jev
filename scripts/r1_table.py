#!/usr/bin/env python
"""R1/R2 tables for docs/experiments.md, entirely post-hoc and entirely on the CPU.

    python scripts/r1_table.py [run_dir ...]        # default: every run dir with eval_rows.jsonl

For each run it refits all four calibrations on the *calib* split (never on eval) and scores each one
on the stored eval logits:

  raw              no calibration
  T-only           temperature scaling on the 500 gold calib rows
  vector           logits / T + b on the same rows (b in R^K), with the 2-fold selection verdict
  prior-declared   T fitted to the teacher's soft labels, b fitted so the mean calibrated probability
                   over the calib rows matches a declared uniform prior -- no gold anywhere
  oracle           the same fit with the *eval* gold marginal as the target instead of a declared
                   prior. It says how much of the gap is marginal at all -- not a bound on accuracy
                   (matching the marginal is not the accuracy-optimal bias, so prior-declared can
                   land above it). Fitted on eval: a diagnostic, never a claim, never in the README.

R2 reads the cascade curve `eval` already stored in results/<run>.json.
"""
import json
import sys
from pathlib import Path

import torch

from openjev import calibrate
from openjev.data import read_rows
from openjev.evaluate import _task_view, metric_fn, paired_ci

ROOT = Path(__file__).resolve().parent.parent


def _eval(run_dir):
    rows = [json.loads(l) for l in open(run_dir / "eval_rows.jsonl") if l.strip()]
    logits = torch.tensor([r["student_logits"] for r in rows])
    teacher = torch.tensor([r["teacher_probs"] for r in rows])
    gold = [r["gold"] for r in rows]
    y = (torch.tensor(gold) if all(g is not None for g in gold) else teacher.argmax(-1))
    return logits, teacher, y, all(g is not None for g in gold)


def score(task, logits, y, teacher=None):
    p = logits.softmax(-1)
    name, higher, f = metric_fn(task, p, y)
    return {name: f(), "f": f, "ece": calibrate.ece(p, y), "marginal": p.mean(0).tolist(),
            "agree": None if teacher is None else float((p.argmax(-1) == teacher.argmax(-1)).float().mean())}


def row(run_dir, prior="uniform"):
    run_dir = Path(run_dir)
    task = _task_view(run_dir)
    task.k = len(task.labels)
    logits, teacher, y, has_gold = _eval(run_dir)
    cal = torch.load(run_dir / "calib_logits.pt", weights_only=False)["logits"]
    crows = read_rows(run_dir, "calib")
    cp = torch.tensor([r["probs"] for r in crows]).clamp_min(1e-6)
    cy, ctarget = calibrate.targets_for(crows)
    out = {"run": run_dir.name, "metric": metric_fn(task, logits.softmax(-1), y)[0],
           "has_gold": has_gold and ctarget == "gold", "raw": score(task, logits, y, teacher)}
    if ctarget == "gold":
        t = calibrate.fit_temperature(cal, cy)
        out["T"] = t
        out["temperature"] = score(task, logits / t, y, teacher)
        tv, b = calibrate.fit_vector(cal, cy)
        out["method"] = calibrate.select_method(cal, cy)
        out["vector"] = score(task, logits / tv + b, y, teacher)
    # no gold at all: T on the teacher's soft labels, then b onto the declared prior
    ts = calibrate.fit_temperature(cal, cp / cp.sum(-1, keepdim=True))
    pv = calibrate.prior_vector(task, prior)
    out["prior"] = pv
    out["prior-declared"] = score(task, logits / ts + calibrate.fit_prior_bias(cal / ts, pv), y, teacher)
    if has_gold:  # DIAGNOSTIC: the bias that matches the eval gold marginal, fitted on eval itself
        gm = torch.bincount(y, minlength=len(task.labels)).float()
        out["oracle"] = score(task, logits / ts + calibrate.fit_prior_bias(logits / ts, gm / gm.sum()),
                              y, teacher)
    out["teacher"] = score(task, (teacher.clamp_min(1e-9)).log(), y)
    base = "temperature" if "temperature" in out else "raw"   # what the run shipped before M1
    for k in ("vector", "prior-declared"):
        if k in out:
            d, lo, hi = paired_ci(out[k]["f"], out[base]["f"], len(y))
            out[f"ci_{k}"] = {"vs": base, "delta": d, "lo": lo, "hi": hi}
    return out


def pts(ci, metric):
    s = 1 if metric == "mae" else 100
    d, lo, hi = ci["delta"] * s, ci["lo"] * s, ci["hi"] * s
    return f"{d:+.2f} ({lo:+.2f}..{hi:+.2f})" + ("" if lo <= 0 <= hi else " *")


def at(curve, e, m):
    """The cheapest point on the curve that escalates at least a fraction `e` of the rows."""
    c = [p for p in curve if p["escalation"] >= e]
    return f"{min(c, key=lambda p: p['escalation'])[m]:.3f}" if c else "–"


def r2(dirs, results=None):
    """R2: the cascade curve `eval` already stored in results/<run>.json. No new teacher call."""
    results = Path(results or ROOT / "results")
    print("| run | metric | student alone | 10 % escalated | 25 % | 50 % | teacher (100 %) "
          "| parity: escalation | tau |")
    print("|" + "---|" * 9)
    for d in dirs:
        p = results / f"{Path(d).name}.json"
        if not p.exists():
            p = results / "variants" / f"{Path(d).name}.json"   # the -prior arms live here
        if not p.exists():
            continue
        res = json.loads(p.read_text())
        c, par = res.get("cascade"), res.get("cascade_parity")
        if not c:
            continue
        m = par["metric"]
        pt = par["parity"]
        esc = f"{pt['escalation']:.1%}" if pt else "never"
        print(f"| {Path(d).name} | {m} | {c[0][m]:.3f} | {at(c, 0.10, m)} | {at(c, 0.25, m)} "
              f"| {at(c, 0.50, m)} | {par['teacher']:.3f} | {esc} | {pt['tau'] if pt else '–'} |")


def main(dirs=None):
    dirs = [Path(d) for d in dirs] if dirs else sorted(
        p.parent for p in (ROOT / "runs").glob("*/eval_rows.jsonl"))
    rows = []
    for d in dirs:
        try:
            rows.append(row(d))
        except (FileNotFoundError, KeyError) as e:
            print(f"skip {d.name}: {type(e).__name__} {e}", file=sys.stderr)
    print("| run | metric | teacher | raw | T-only | vector | prior-declared (no gold) | oracle "
          "(fitted on eval) | 2-fold verdict | Δ vector (95 % CI) | Δ prior-declared (95 % CI) |")
    print("|" + "---|" * 11)
    for r in rows:
        m = r["metric"]
        g = lambda k: f"{r[k][m]:.3f}" if k in r else "–"  # noqa: E731
        c = lambda k: pts(r[f"ci_{k}"], m) if f"ci_{k}" in r else "–"  # noqa: E731
        print(f"| {r['run']} | {m} | {r['teacher'][m]:.3f} | {g('raw')} | {g('temperature')} "
              f"| {g('vector')} | {g('prior-declared')} | {g('oracle')} | {r.get('method', '– (no gold)')} "
              f"| {c('vector')} | {c('prior-declared')} |")
    print("\n`*` = the 95 % CI excludes 0. Δ is against the calibration the run shipped before M1 "
          "(T-only, or raw when the calib split has no gold), paired bootstrap over eval rows, "
          "2000 resamples, one seed per run.\n")
    r2([r["run"] for r in rows])
    return rows


if __name__ == "__main__":
    main(sys.argv[1:])
