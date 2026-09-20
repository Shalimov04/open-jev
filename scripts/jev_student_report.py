#!/usr/bin/env python
"""Four-way table: Qwen teacher / Qwen student / Jev teacher / Jev student, same eval rows, same gold.

    python scripts/jev_student_report.py georeview kinopoisk

Students are the mean over the seeds that have been eval'd (sd in parentheses); teachers are the
`teacher.jsonl` eval rows of the respective arm. Writes results/api/jev-student.json.
"""
import json
import statistics
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from openjev.data import read_rows  # noqa: E402
from openjev.evaluate import _cls, _eval_rows, _seed_dirs, metric_fn  # noqa: E402
from openjev.spec import load_task  # noqa: E402


def norm(p):
    p = torch.tensor(p).clamp_min(1e-6)
    return p / p.sum(-1, keepdim=True)


def main(tasks):
    out = []
    for name in tasks:
        task = load_task(ROOT / "tasks" / f"{name}.yaml")
        arms = {a: _seed_dirs(ROOT / "runs" / a) for a in (name, f"{name}-jev")}
        teach = {a: {r["id"]: r for r in read_rows(ROOT / "runs" / a, "eval")} for a in arms}
        sets = [set(d) for d in teach.values()] + [set(_eval_rows(d)) for s in arms.values() for d in s.values()]
        ids = sorted(set.intersection(*sets))
        y = torch.tensor([teach[name][i]["gold"] for i in ids])
        row = {"task": name, "type": task.type, "n": len(ids),
               "seeds": {a: sorted(s) for a, s in arms.items()}, "arms": {}}
        for a, dirs in arms.items():
            tag = "qwen" if a == name else "jev"
            p = norm([teach[a][i]["probs"] for i in ids])
            mname, higher, f = metric_fn(task, p, y)
            row["metric"], row["higher_is_better"] = mname, higher
            row["arms"][f"{tag}_teacher"] = {**_cls(p, y), mname: f()}
            per = []
            for s in sorted(dirs):
                ev = _eval_rows(dirs[s])
                sp = norm([ev[i]["student_probs"] for i in ids])
                per.append({"seed": s, **_cls(sp, y), mname: metric_fn(task, sp, y)[2]()})
            agg = {k: statistics.mean(d[k] for d in per) for k in per[0] if k != "seed"}
            agg["sd_" + mname] = statistics.stdev([d[mname] for d in per]) if len(per) > 1 else 0.0
            row["arms"][f"{tag}_student"] = {**agg, "per_seed": per}
            row["arms"][f"{tag}_student"]["calib"] = {
                s: {k: v for k, v in json.loads((dirs[s] / "calib.json").read_text()).items()
                    if k in ("method", "temperature", "ece_before", "ece_after")} for s in sorted(dirs)}
        out.append(row)

    fmt = lambda v: f"{v:.3f}"  # noqa: E731
    print("\n| task | metric | n | Qwen teacher | Qwen student | Jev teacher | Jev student |")
    print("|" + "---|" * 7)
    for r in out:
        m = r["metric"]
        c = [f"{fmt(r['arms'][k][m])}" + (f" ± {r['arms'][k]['sd_' + m]:.3f}" if "student" in k else "")
             for k in ("qwen_teacher", "qwen_student", "jev_teacher", "jev_student")]
        print(f"| {r['task']} | {m} | {r['n']} | " + " | ".join(c) + " |")
    print("\n| task | arm | acc | macro-F1 | ECE |\n|" + "---|" * 5)
    for r in out:
        for k, v in r["arms"].items():
            print(f"| {r['task']} | {k} | {fmt(v['acc'])} | {fmt(v['macro_f1'])} | {fmt(v['ece'])} |")
    (ROOT / "results" / "api").mkdir(parents=True, exist_ok=True)
    (ROOT / "results" / "api" / "jev-student.json").write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    main(sys.argv[1:])
