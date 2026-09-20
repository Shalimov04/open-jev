#!/usr/bin/env python
"""Offline-only: what is the zero-gold agnews student worth against the *real* ag_news labels?

The agnews-nogold task has no `data.gold`, so the pipeline never sees gold: the temperature is
fitted to the teacher's soft labels and eval is reported against the teacher. This script joins the
eval rows back to the dataset by row id ("test:<idx>") and scores the calibrated student and the
teacher against the true labels. Writes `gold_acc_offline` (and `teacher_gold_acc_offline`) into
results/variants/agnews-nogold-goldacc.json -- a file `openjev eval` never writes, so the number
survives a re-eval. Not part of the pipeline; run it by hand:

    python scripts/nogold_gold_acc.py
"""
import json
from pathlib import Path

import torch
from datasets import load_dataset

from openjev.data import read_rows
from openjev.train import load_student, predict_logits

ROOT = Path(__file__).resolve().parent.parent
RUN = ROOT / "runs/agnews-nogold"

rows = read_rows(RUN, "eval")
ds = load_dataset("fancyzhx/ag_news", split="test")
idx = [int(r["id"].split(":")[1]) for r in rows]
assert all(ds[i]["text"][:2000] == r["text"] for i, r in zip(idx, rows)), "id join is off"
gold = torch.tensor([ds[i]["label"] for i in idx])

T = json.loads((RUN / "calib.json").read_text())["temperature"]
tok, model = load_student(RUN / "student")
probs = (predict_logits(tok, model, [r["text"] for r in rows], 256) / T).softmax(-1)
teacher = torch.tensor([r["probs"] for r in rows])

acc = float((probs.argmax(-1) == gold).float().mean())
t_acc = float((teacher.argmax(-1) == gold).float().mean())
print(f"n={len(rows)} student vs gold {acc:.4f}  teacher vs gold {t_acc:.4f}")

res_path = ROOT / "results/variants/agnews-nogold-goldacc.json"
res_path.write_text(json.dumps(
    {"run": "agnews-nogold", "n": len(rows), "gold_acc_offline": acc,
     "teacher_gold_acc_offline": t_acc}, indent=2) + "\n")
print(f"wrote gold_acc_offline to {res_path}")
