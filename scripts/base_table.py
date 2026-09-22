"""PLAN-4 M3: the leave-one-source-out and calibration-under-transfer tables, from results/ only.

    python3 scripts/base_table.py            # LOTO: chance / teacher / per-task student / base zero-shot
    python3 scripts/base_table.py --calib    # ECE and NLL per calibration variant
    python3 scripts/base_table.py --warm     # M4 warm-start curve

No hand-typed cell: every number is read from results/*.json or recomputed from the run dirs'
eval_rows.jsonl with the same `evaluate` helpers the per-task pipeline uses.
"""
import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from openjev import evaluate  # noqa: E402
from openjev.base import VARIANTS  # noqa: E402

RESULTS = ROOT / "results"
BASE = ROOT / "results/base"
RUNS = ROOT / "runs"


def folds():
    return yaml.safe_load((ROOT / "tasks/base/folds.yaml").read_text())


def arms(base_dir, student_dir):
    """(ids, y, task, {name: scorer}) over the eval rows the two arms share. The student arm is
    averaged over its seeds exactly as `openjev compare` does."""
    srcs = evaluate._seed_dirs(student_dir)
    rows = {("student", s): evaluate._eval_rows(d) for s, d in srcs.items()}
    rows[("base", 0)] = evaluate._eval_rows(base_dir)
    ids = sorted(set.intersection(*(set(r) for r in rows.values())))
    task = evaluate._task_view(next(iter(srcs.values())))
    gold = [rows[("base", 0)][i]["gold"] for i in ids]
    if any(g is None for g in gold):
        y = torch.tensor([max(range(len(rows[("base", 0)][i]["teacher_probs"])),
                              key=rows[("base", 0)][i]["teacher_probs"].__getitem__) for i in ids])
    else:
        y = torch.tensor(gold)
    k = len(rows[("base", 0)][ids[0]]["teacher_probs"])

    def scorer(probs):
        return evaluate.metric_fn(task, probs, y)

    fns = {}
    per_seed = [scorer(torch.tensor([rows[("student", s)][i]["student_probs"] for i in ids]))[2]
                for s in sorted(srcs)]
    fns["student"] = lambda i=None: sum(f(i) for f in per_seed) / len(per_seed)
    name, higher, fns["base"] = scorer(torch.tensor([rows[("base", 0)][i]["student_probs"] for i in ids]))
    fns["teacher"] = scorer(torch.tensor([rows[("base", 0)][i]["teacher_probs"] for i in ids]))[2]
    fns["chance"] = scorer(torch.full((len(ids), k), 1.0 / k))[2]
    return ids, name, higher, fns, len(srcs)


def loto(variant="gold500", n_boot=2000, tag=""):
    out = []
    ev = folds()["eval"]
    for fold, tasks in ev.items():
        for t in tasks:
            b = RUNS / f"base-{fold}{tag and '-' + tag}-zs-{t}-{variant}"
            s = RUNS / t
            if not (b / "eval_rows.jsonl").exists():
                continue
            ids, name, higher, fns, nseeds = arms(b, s)
            n = len(ids)
            row = {"fold": fold, "tag": tag, "task": t, "metric": name, "n": n, "seeds": nseeds,
                   "variant": variant, "chance": fns["chance"](), "teacher": fns["teacher"](),
                   "student": fns["student"](), "base": fns["base"]()}
            for other in ("student", "teacher"):
                d, lo, hi = evaluate.paired_ci(fns["base"], fns[other], n, n_boot)
                row[f"d_{other}"] = [d, lo, hi]
                row[f"verdict_{other}"] = ("no measurable difference" if lo <= 0 <= hi else
                                           ("base better" if (d > 0) == higher else "base worse"))
            out.append(row)
    return out


def fmt(rows):
    w = max(len(r["task"]) for r in rows)
    lines = [f"| {'task':{w}} | fold | metric | n | chance | teacher | student | base-zs | "
             f"Δ vs student (95 % CI) | Δ vs teacher (95 % CI) |",
             f"|{'-' * (w + 2)}|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        d = lambda k: (f"{r[k][0]:+.3f} ({r[k][1]:+.3f}..{r[k][2]:+.3f})")  # noqa: E731
        lines.append(f"| {r['task']:{w}} | {r['fold']} | {r['metric']} | {r['n']} | "
                     f"{r['chance']:.3f} | {r['teacher']:.3f} | {r['student']:.3f} | "
                     f"{r['base']:.3f} | {d('d_student')} | {d('d_teacher')} |")
    return "\n".join(lines)


def calib_table(tag=""):
    """ECE_raw / ECE_cal / NLL of base zero-shot per variant, against the per-task student's."""
    rows, arm = [], lambda fold: f"base-{fold}{tag and '-' + tag}"
    ev = folds()["eval"]
    for fold, tasks in ev.items():
        for t in tasks:
            st = RESULTS / f"{t}.json"
            have = [v for v in VARIANTS if (BASE / f"{arm(fold)}-zs-{t}-{v}.json").exists()]
            if not st.exists() or not have:      # no base model for this fold yet
                continue
            s = json.loads(st.read_text())["student"]
            rows.append({"task": t, "arm": "per-task student", "acc": s["acc"],
                         "ece_raw": s["ece_raw"], "ece_cal": s["ece_cal"], "nll": s["nll"]})
            for v in have:
                b = json.loads((BASE / f"{arm(fold)}-zs-{t}-{v}.json").read_text())["student"]
                rows.append({"task": t, "arm": f"{arm(fold)} {v}", "acc": b["acc"],
                             "ece_raw": b["ece_raw"], "ece_cal": b["ece_cal"], "nll": b["nll"]})
    w = max(len(r["arm"]) for r in rows)
    lines = [f"| task | {'arm':{w}} | acc | ECE raw | ECE cal | NLL |", f"|---|{'-' * (w + 2)}|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['task']} | {r['arm']:{w}} | {r['acc']:.3f} | {r['ece_raw']:.4f} | "
                     f"{r['ece_cal']:.4f} | {r['nll']:.3f} |")
    return "\n".join(lines)


def warm_table(task="kinopoisk", n_boot=2000):
    """M4: init in {mmBERT-small, fold model} x N, 3 seeds. Every arm is an ordinary run dir."""
    lines = ["| task | N | init | seeds | metric | mean | Δ vs scratch (95 % CI) |", "|---|---|---|---|---|---|---|"]
    for n in (100, 250, 500, 1000):
        a, b = RUNS / f"{task}-warm-n{n}", RUNS / f"{task}-scratch-n{n}"
        if not (a / "eval_rows.jsonl").exists() or not (b / "eval_rows.jsonl").exists():
            continue
        r = evaluate.compare(a, b, n_boot=n_boot, out=lambda *x: None)
        mean = lambda d: sum(s[d] for s in r["per_seed"]) / len(r["per_seed"])  # noqa: E731
        lines.append(f"| {task} | {n} | base fold model | {len(r['seeds'])} | {r['metric']} | "
                     f"{mean('a'):.4f} | {r['delta']:+.4f} ({r['ci'][0]:+.4f}..{r['ci'][1]:+.4f}) |")
        lines.append(f"| {task} | {n} | mmBERT-small | {len(r['seeds'])} | {r['metric']} | "
                     f"{mean('b'):.4f} | — |")
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="gold500", choices=list(VARIANTS))
    ap.add_argument("--calib", action="store_true")
    ap.add_argument("--warm", action="store_true")
    ap.add_argument("--task", default="kinopoisk")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--tag", default="", help="run-dir tag of the base models (e.g. v2)")
    a = ap.parse_args()
    if a.calib:
        print(calib_table(a.tag))
    elif a.warm:
        print(warm_table(a.task))
    else:
        rows = loto(a.variant, tag=a.tag)
        (BASE / f"base-loto{a.tag and '-' + a.tag}-{a.variant}.json").write_text(json.dumps(rows, indent=1))
        if a.json:
            print(json.dumps(rows, indent=1))
            raise SystemExit
        print(fmt(rows))
        for r in rows:
            print(f"{r['task']}: vs student {r['verdict_student']}, vs teacher {r['verdict_teacher']} "
                  f"(95 % CI {r['d_student'][1]:+.3f}..{r['d_student'][2]:+.3f} / "
                  f"{r['d_teacher'][1]:+.3f}..{r['d_teacher'][2]:+.3f})")
