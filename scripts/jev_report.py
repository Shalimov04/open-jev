#!/usr/bin/env python
"""Head-to-head table: Jev (runs/<task>-jev) vs our vLLM teacher and our student (results/<task>.json).

    python scripts/jev_report.py agnews kinopoisk georeview arb-success

Same rows, same gold, same metric code as `openjev eval` (evaluate._cls / metric_fn, calibrate.ece).
"""
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from openjev.data import read_rows  # noqa: E402
from openjev.evaluate import _cls, metric_fn  # noqa: E402
from openjev.spec import load_task  # noqa: E402


def probs_of(rows):
    p = torch.tensor([r["probs"] for r in rows]).clamp_min(1e-6)
    return p / p.sum(-1, keepdim=True)


def main(tasks):
    out = []
    for name in tasks:
        task = load_task(ROOT / "tasks" / f"{name}.yaml")
        jev_rows = read_rows(ROOT / "runs" / f"{name}-jev", "eval")
        base_rows = {r["id"]: r for r in read_rows(ROOT / "runs" / name, "eval")}
        jev_rows = [r for r in jev_rows if r["id"] in base_rows]
        assert jev_rows and all(r["gold"] is not None for r in jev_rows), name
        y = torch.tensor([r["gold"] for r in jev_rows])
        jev, teacher = probs_of(jev_rows), probs_of([base_rows[r["id"]] for r in jev_rows])
        res = json.loads((ROOT / "results" / f"{name}.json").read_text())
        row = {"task": name, "type": task.type, "n": len(jev_rows),
               "jev": _cls(jev, y), "teacher_local": _cls(teacher, y), "student": res["student"],
               "jev_mean_maxprob": float(jev.max(-1).values.mean()),
               "jev_zero_prob_rows": float((jev.max(-1).values > 0.999).float().mean())}
        conf = [r["raw"].get("confidence") for r in jev_rows if "confidence" in r["raw"]]
        row["jev_mean_reported_confidence"] = sum(conf) / len(conf) if conf else None
        mname, _, f_jev = metric_fn(task, jev, y)
        _, _, f_loc = metric_fn(task, teacher, y)
        row["metric"] = mname
        row["jev_metric"], row["teacher_local_metric"] = f_jev(), f_loc()
        row["student_metric"] = (res.get("score", {}) or {}).get("mae") or \
            (res.get("noul", {}) or {}).get("auroc") or res["student"]["acc"]
        out.append(row)
        print(json.dumps(row, indent=1))
    hdr = ("| task | type | n | Jev acc | teacher acc | student acc | Jev macro-F1 | teacher macro-F1 "
           "| headline (Jev / teacher / student) | Jev ECE | teacher ECE | Jev mean conf |")
    print("\n" + hdr + "\n|" + "---|" * 12)
    for r in out:
        f = (lambda v: f"{v:.3f}")
        print(f"| {r['task']} | {r['type']} | {r['n']} | {f(r['jev']['acc'])} | {f(r['teacher_local']['acc'])} "
              f"| {f(r['student']['acc'])} | {f(r['jev']['macro_f1'])} | {f(r['teacher_local']['macro_f1'])} "
              f"| {r['metric']} {f(r['jev_metric'])} / {f(r['teacher_local_metric'])} / {f(r['student_metric'])} "
              f"| {f(r['jev']['ece'])} | {f(r['teacher_local']['ece'])} | {f(r['jev_mean_maxprob'])} |")
    (ROOT / "results" / "api").mkdir(parents=True, exist_ok=True)
    (ROOT / "results" / "api" / "jev-teacher.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:])
