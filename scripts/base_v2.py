"""The two `docs/prereg/base-v2.md` readouts that are not the LOTO table.

    python3 scripts/base_v2.py --unseen     # §3 readout 1: base-none-v2 on the three unseen sets
    python3 scripts/base_v2.py --ablation   # §3 readout 2: ABL-all - ABL-orig on fold F1

Every threshold below is quoted from the preregistration and none of it is tunable from the command
line: improvement = CI excludes 0 AND |Δ| >= 2.0 pts (0.04 MAE) AND all per-seed Δ share a sign.
"""
import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from openjev import evaluate  # noqa: E402

RUNS, BASE = ROOT / "runs", ROOT / "results/base"
UNSEEN = ["yahoo-topics", "sst5", "ru-inappropriate"]
ABL_TASKS = ["kinopoisk", "swde-field", "toxic"]          # prereg §3 readout 2
MIN_ROWS = {"yahoo-topics": 450, "sst5": 450, "ru-inappropriate": 450,   # prereg §5
            "kinopoisk": 1400, "swde-field": 1900, "toxic": 2800}
THR = {"mae": 0.04, "acc": 0.02, "auroc": 0.02}           # prereg §4 improvement / NI margin


def _load(dirs):
    """(ids, y, task, k) over the eval rows every dir in `dirs` shares."""
    rows = [evaluate._eval_rows(d) for d in dirs]
    ids = sorted(set.intersection(*(set(r) for r in rows)))
    task = evaluate._task_view(dirs[0])
    gold = [rows[0][i]["gold"] for i in ids]
    y = torch.tensor(gold if all(g is not None for g in gold) else
                     [max(range(len(rows[0][i]["teacher_probs"])),
                          key=rows[0][i]["teacher_probs"].__getitem__) for i in ids])
    return ids, rows, y, task, len(rows[0][ids[0]]["teacher_probs"])


def _fn(task, rows, ids, y, key="student_probs"):
    return evaluate.metric_fn(task, torch.tensor([rows[i][key] for i in ids]), y)


def unseen(variant="gold500", arm="base-none-v2", n_boot=2000):
    """Readout 1: metric, chance, teacher, CI on the advantage over chance, teacher-normalised."""
    out = []
    for t in UNSEEN:
        d = RUNS / f"{arm}-zs-{t}-{variant}"
        if not (d / "eval_rows.jsonl").exists():
            continue
        ids, (r,), y, task, k = _load([d])
        name, higher, model = _fn(task, r, ids, y)
        teacher = _fn(task, r, ids, y, "teacher_probs")[2]
        chance = evaluate.metric_fn(task, torch.full((len(ids), k), 1.0 / k), y)[2]
        # "advantage over chance", always positive-is-better (MAE flips, prereg §4)
        a, b = (model, chance) if higher else (chance, model)
        adv, lo, hi = evaluate.paired_ci(a, b, len(ids), n_boot)
        t_adv = (teacher() - chance()) if higher else (chance() - teacher())
        out.append({"task": t, "variant": variant, "metric": name, "n": len(ids), "k": k,
                    "chance": chance(), "teacher": teacher(), "base": model(),
                    "adv": adv, "ci": [lo, hi], "teacher_adv": t_adv,
                    "normalised": adv / t_adv if t_adv else float("nan"),
                    "beats_chance": not (lo <= 0 <= hi) and adv > 0,
                    "rows_ok": len(ids) >= MIN_ROWS.get(t, 0)})
    return out


def _seeded(prefix, task, variant):
    """The three seed dirs of one arm: `<prefix>`, `<prefix>-s1`, `<prefix>-s2` (prereg §2 names)."""
    return [RUNS / f"{prefix}{s}-zs-{task}-{variant}" for s in ("", "-s1", "-s2")]


def ablation(variant="gold500", a="base-F1-v2", b="base-F1-v2-orig", tasks=ABL_TASKS, n_boot=2000):
    """Readout 2: Δ = ABL-all − ABL-orig, pooled over seeds exactly as `openjev compare` pools."""
    out = []
    for t in tasks:
        A = [d for d in _seeded(a, t, variant) if (d / "eval_rows.jsonl").exists()]
        B = [d for d in _seeded(b, t, variant) if (d / "eval_rows.jsonl").exists()]
        if not A or not B:
            continue
        ids, rows, y, task, k = _load(A + B)
        name, higher, _ = _fn(task, rows[0], ids, y)
        fns = [_fn(task, r, ids, y)[2] for r in rows]
        fa, fb = fns[:len(A)], fns[len(A):]
        sign = 1.0 if higher else -1.0            # Δ is always in "better is more" units
        per_seed = [sign * (fa[i]() - fb[i]()) for i in range(min(len(A), len(B)))]
        mean = lambda f: (lambda i=None: sum(g(i) for g in f) / len(f))  # noqa: E731
        better, worse = (mean(fa), mean(fb)) if higher else (mean(fb), mean(fa))
        d, lo, hi = evaluate.paired_ci(better, worse, len(ids), n_boot)  # + = ABL-all is better
        half = (hi - lo) / 2
        same_sign = len(per_seed) == 3 and len({v > 0 for v in per_seed}) == 1
        excludes_0 = not (lo <= 0 <= hi)
        out.append({"task": t, "variant": variant, "metric": name, "n": len(ids),
                    "seeds_a": len(A), "seeds_b": len(B), "a": mean(fa)(), "b": mean(fb)(),
                    "delta": d, "ci": [lo, hi], "half_width": half, "per_seed": per_seed,
                    "same_sign": same_sign, "excludes_0": excludes_0,
                    "improvement": excludes_0 and abs(d) >= THR[name] and same_sign and d > 0,
                    "non_inferior": lo > -THR[name],
                    "inconclusive": (not excludes_0) and half > THR[name],
                    "rows_ok": len(ids) >= MIN_ROWS.get(t, 0)})
    return out


def fmt_unseen(rows):
    lines = ["| task | metric | n | K | chance | teacher | base-none-v2 | advantage over chance "
             "(95 % CI) | teacher-normalised |", "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['task']} | {r['metric']} | {r['n']} | {r['k']} | {r['chance']:.3f} | "
                     f"{r['teacher']:.3f} | {r['base']:.3f} | {r['adv']:+.3f} "
                     f"({r['ci'][0]:+.3f}..{r['ci'][1]:+.3f}) | {r['normalised']:.2f} |")
    beats = sum(r["beats_chance"] for r in rows)
    half = sum(r["normalised"] >= 0.5 for r in rows)
    lines.append(f"\nprereg §4: beats chance on {beats}/{len(rows)} (need all), "
                 f"teacher-normalised ≥ 0.5 on {half}/{len(rows)} (need ≥ 2) -> "
                 f"**{'zero-shot claim holds' if beats == len(rows) == 3 and half >= 2 else 'NO zero-shot claim'}**")
    return "\n".join(lines)


def fmt_ablation(rows):
    lines = ["| task | metric | n | seeds | ABL-all | ABL-orig | Δ (95 % CI) | per-seed Δ | "
             "signs agree | verdict |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        v = ("improvement" if r["improvement"] else
             "inconclusive" if r["inconclusive"] else
             "no measurable difference" if not r["excludes_0"] else
             "ABL-orig better" if r["delta"] < 0 else "below the 2-pt threshold")
        lines.append(f"| {r['task']} | {r['metric']} | {r['n']} | {r['seeds_a']}v{r['seeds_b']} | "
                     f"{r['a']:.3f} | {r['b']:.3f} | {r['delta']:+.4f} "
                     f"({r['ci'][0]:+.4f}..{r['ci'][1]:+.4f}) | "
                     f"{', '.join(f'{p:+.3f}' for p in r['per_seed'])} | "
                     f"{'yes' if r['same_sign'] else 'no'} | {v} |")
    wins = sum(r["improvement"] for r in rows)
    ni = sum(r["non_inferior"] for r in rows)
    lines.append(f"\nprereg §4: improvement on {wins}/{len(rows)} tasks (need ≥ 2) -> "
                 f"**{'diversity claim holds' if wins >= 2 else 'NO diversity claim'}**; "
                 f"non-inferiority (CI lower bound > −2.0 pts) on {ni}/{len(rows)} (need all)")
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="gold500")
    ap.add_argument("--unseen", action="store_true")
    ap.add_argument("--ablation", action="store_true")
    ap.add_argument("--arm", default="base-none-v2", help="--unseen: which model to score")
    ap.add_argument("--tasks", nargs="+", default=ABL_TASKS, help="--ablation: tasks to read out")
    a = ap.parse_args()
    BASE.mkdir(parents=True, exist_ok=True)
    if a.unseen:
        rows = unseen(a.variant, a.arm)
        (BASE / f"unseen-{a.arm}-{a.variant}.json").write_text(json.dumps(rows, indent=1))
        print(fmt_unseen(rows))
    if a.ablation:
        rows = ablation(a.variant, tasks=a.tasks)
        name = "ablation" + ("-unseen" if a.tasks != ABL_TASKS else "")
        (BASE / f"{name}-{a.variant}.json").write_text(json.dumps(rows, indent=1))
        print(fmt_ablation(rows))
