"""PLAN-4's cheap quality gate over the S1 tasks, plus what each one cost.

    python3 scripts/base_gate.py            # one row per tasks/base/*.yaml labelled in runs/<name>-jev

Gate (PLAN-4 §2, "a task with weak gold"): keep a task only if the argmax marginal's top class
is <= 80 % and mean max-p >= 0.55. Accuracy vs gold is printed where the task has gold; it is
information, not part of the gate.
"""
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from openjev.data import read_jsonl  # noqa: E402
from openjev.spec import load_task  # noqa: E402


def row(path):
    task = load_task(path)
    run = ROOT / "runs" / f"{task.name}-jev"
    rows = read_jsonl(run / "teacher.jsonl")
    if not rows:
        return {"task": task.name, "n": 0}
    pred = [max(range(task.k), key=lambda i: r["probs"][i]) for r in rows]
    top = Counter(pred).most_common(1)[0]
    maxp = sum(max(r["probs"]) for r in rows) / len(rows)
    gold = [(p, r["gold"]) for p, r in zip(pred, rows) if r.get("gold") is not None]
    label = json.loads((run / "label.json").read_text()) if (run / "label.json").exists() else {}
    return {"task": task.name, "type": task.type, "k": task.k, "n": len(rows),
            "top_class": task.labels[top[0]], "top_share": top[1] / len(rows), "mean_maxp": maxp,
            "acc": (sum(p == g for p, g in gold) / len(gold)) if gold else None,
            "cost": label.get("cost"), "calls": label.get("calls"),
            "pass": top[1] / len(rows) <= 0.80 and maxp >= 0.55}


def main():
    rows = [row(p) for p in sorted((ROOT / "tasks/base").glob("*.yaml"))]
    rows = [r for r in rows if r["n"]]
    w = max(len(r["task"]) for r in rows)
    print(f"{'task':{w}} {'prim':6} {'K':>2} {'n':>5} {'top class':>22} {'top%':>6} {'maxp':>6} "
          f"{'acc':>6} {'$':>8}  gate")
    for r in rows:
        acc = f"{r['acc']:.3f}" if r["acc"] is not None else "    –"
        print(f"{r['task']:{w}} {r['type']:6} {r['k']:2} {r['n']:5} {r['top_class'][:22]:>22} "
              f"{100 * r['top_share']:6.1f} {r['mean_maxp']:6.3f} {acc:>6} "
              f"{r['cost'] or 0:8.4f}  {'pass' if r['pass'] else 'FAIL'}")
    print(f"{len(rows)} tasks, {sum(r['n'] for r in rows)} rows, "
          f"{sum(r['calls'] or 0 for r in rows)} calls, ${sum(r['cost'] or 0 for r in rows):.4f}, "
          f"{sum(not r['pass'] for r in rows)} failed the gate")


if __name__ == "__main__":
    main()
