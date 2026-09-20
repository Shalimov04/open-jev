#!/usr/bin/env python
"""Option-order permutation diagnostic: does the teacher's class marginal follow the *option order*?

Relabels the calib+eval rows of `runs/<task>/` with the option list reversed (gold remapped
g -> K-1-g) into its own run dir `runs/<task>-rev/` -- the original teacher.jsonl is frozen and is
never touched. Then prints, for the original order, the reversed order and their average:
teacher accuracy, the predicted marginal, and mean max-p.

    python scripts/perm_check.py kinopoisk [--limit N]

A *positional* bias shows up as the marginal following the letters; a *semantic* bias stays put.
Reads `runs/<task>/teacher.jsonl`, so it spends teacher time only on rows that are already labeled
(~2000 rows, ~6 min at concurrency 32). Resumable: rows already in runs/<task>-rev are skipped.
"""
import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

for _var in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
    os.environ.pop(_var, None)  # the teacher is localhost; the box's SOCKS proxy must not see it

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openjev import teacher  # noqa: E402
from openjev.data import read_jsonl  # noqa: E402
from openjev.spec import load_task  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def reversed_task(task):
    """Same task with the options listed back to front. Only `options`/`labels` change, so the
    prompt is identical apart from the order (and the letter each option gets)."""
    if task.type != "choice":
        raise SystemExit(f"perm_check is for choice tasks; {task.name} is {task.type}")
    rev = load_task(task.path)
    rev.options = list(reversed(task.options))
    rev.labels = list(reversed(task.labels))
    return rev


def stats(probs, gold, labels):
    k = len(labels)
    pred = [max(range(k), key=p.__getitem__) for p in probs]
    acc = sum(p == g for p, g in zip(pred, gold)) / len(gold)
    marg = [pred.count(i) / len(pred) for i in range(k)]
    maxp = sum(max(p) for p in probs) / len(probs)
    return acc, marg, maxp


def show(name, probs, gold, labels):
    acc, marg, maxp = stats(probs, gold, labels)
    print(f"{name:<10} acc {acc:.3f}   mean max-p {maxp:.3f}   "
          + "  ".join(f"{l} {m:.3f}" for l, m in zip(labels, marg)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task")
    ap.add_argument("--limit", type=int, help="label at most N rows (smoke test)")
    ap.add_argument("--runs", default=str(ROOT / "runs"))
    a = ap.parse_args()

    task = load_task(ROOT / f"tasks/{a.task}.yaml")
    src, dst = Path(a.runs) / task.name, Path(a.runs) / f"{task.name}-rev"
    k = task.k
    rows = [r for r in read_jsonl(src / "teacher.jsonl") if r["split"] in ("calib", "eval")]
    rows = sorted(rows, key=lambda r: r["id"])[: a.limit]
    if not rows:
        raise SystemExit(f"no calib/eval rows in {src / 'teacher.jsonl'}")
    if any(r.get("gold") is None for r in rows):
        raise SystemExit(f"{task.name} has no gold; the diagnostic needs it")

    rev = reversed_task(task)
    dst.mkdir(parents=True, exist_ok=True)
    have = {r["id"] for r in read_jsonl(dst / "teacher.jsonl")}
    todo = [{"id": r["id"], "split": r["split"], "text": r["text"], "gold": k - 1 - r["gold"]}
            for r in rows if r["id"] not in have]
    print(f"{task.name}: {len(rows)} rows, {len(rows) - len(todo)} cached, {len(todo)} to relabel "
          f"into {dst}", flush=True)
    if todo:
        asyncio.run(teacher._run(rev, todo, dst / "teacher.jsonl"))

    by_id = {r["id"]: r for r in read_jsonl(dst / "teacher.jsonl")}
    missing = [r["id"] for r in rows if r["id"] not in by_id]
    if missing:
        print(f"warning: {len(missing)} rows failed to relabel, dropped", flush=True)
    rows = [r for r in rows if r["id"] in by_id]
    gold = [r["gold"] for r in rows]
    orig = [r["probs"] for r in rows]
    # reversed run's probs are in reversed-label order: flip back onto the original labels
    revp = [list(reversed(by_id[r["id"]]["probs"])) for r in rows]
    avg = [[(a_ + b_) / 2 for a_, b_ in zip(p, q)] for p, q in zip(orig, revp)]

    print(f"\n{task.name}: n={len(rows)} (calib+eval), options {task.labels}")
    gm = [gold.count(i) / len(gold) for i in range(k)]
    print(f"{'gold':<10} {'':>32}" + "  ".join(f"{l} {m:.3f}" for l, m in zip(task.labels, gm)))
    for name, p in (("original", orig), ("reversed", revp), ("averaged", avg)):
        show(name, p, gold, task.labels)
    (dst / "perm_check.json").write_text(json.dumps(
        {"task": task.name, "n": len(rows), "labels": task.labels,
         "gold_marginal": gm,
         **{name: dict(zip(("acc", "marginal", "mean_max_p"), stats(p, gold, task.labels)))
            for name, p in (("original", orig), ("reversed", revp), ("averaged", avg))}}, indent=2,
        ensure_ascii=False))


if __name__ == "__main__":
    main()
