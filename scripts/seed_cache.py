"""Seed a new task's teacher.jsonl from another run's, for the ids the two tasks share.

    python scripts/seed_cache.py tasks/headlines-8k.yaml runs/headlines

Teacher probabilities are frozen (PLAN-2 §2): a row that both tasks contain keeps the exact
probabilities it was labelled with, so only genuinely new rows cost teacher time. The cached line's
`split` is rewritten to the *new* task's role for that id -- headlines-8k promotes 59 of headlines'
calib rows to train, and read_rows() filters on that field.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from openjev.data import examples, read_jsonl  # noqa: E402
from openjev.spec import load_task  # noqa: E402

task, src = load_task(sys.argv[1]), Path(sys.argv[2])
dst = Path(sys.argv[3] if len(sys.argv) > 3 else f"runs/{task.name}")
role = {e["id"]: e["split"] for e in examples(task)}
assert not (dst / "teacher.jsonl").exists(), f"{dst}/teacher.jsonl already exists"
dst.mkdir(parents=True, exist_ok=True)
kept = 0
with open(dst / "teacher.jsonl", "w") as f:
    for r in read_jsonl(src / "teacher.jsonl"):
        if r["id"] in role and r.get("source", "data") == "data":
            f.write(json.dumps({**r, "split": role[r["id"]]}, ensure_ascii=False) + "\n")
            kept += 1
print(f"{dst}: {kept} cached rows reused of {len(role)}; {len(role) - kept} still to label")
