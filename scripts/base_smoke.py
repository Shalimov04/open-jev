"""PLAN-4 M0 smoke: train the pair scorer (num_labels=1, BCE vs teacher p) on the mixture for a few
hundred steps and measure pairs/s, then print the per-task cap that keeps one epoch <= 60 min.

  scripts/gpu_queue.sh runs/queue-smoke.jobs     # the job line: python scripts/base_smoke.py
"""
import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from transformers import AutoModelForSequenceClassification, get_linear_schedule_with_warmup  # noqa: E402

from openjev import pairs  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--steps", type=int, default=300)
ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--warm", type=int, default=20, help="steps left out of the timing")
ap.add_argument("--cap", type=int, default=12000, help="per-task pairs for the smoke's own epoch")
ap.add_argument("--n-tasks", type=int, nargs="*", default=[35, 65], help="also print the cap for these task counts")
a = ap.parse_args()

torch.manual_seed(0)
rng = random.Random(0)
dev = "cuda" if torch.cuda.is_available() else "cpu"
tok = pairs.tokenizer()
model = AutoModelForSequenceClassification.from_pretrained(
    "jhu-clsp/mmBERT-small", num_labels=1, attn_implementation="sdpa").to(dev).train()
opt = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
sched = get_linear_schedule_with_warmup(opt, int(0.06 * a.steps), a.steps)

rows = pairs.index()
print(f"{len(rows)} tasks, {sum(len(v) for v in rows.values())} train rows, "
      f"{pairs.epoch_pairs(rows, a.cap)} pairs/epoch at cap {a.cap}", flush=True)
stream = pairs.sample(rows, a.cap, rng)
if dev == "cuda":
    torch.cuda.reset_peak_memory_stats()
losses, lens, t0, t_warm = [], [], time.time(), None
for step in range(a.steps):
    batch = [r for _, r in zip(range(a.batch), stream)]
    if not batch:
        break
    enc = pairs.encode(tok, [r["seg_a"] for r in batch], [r["seg_b"] for r in batch],
                       padding=True, return_tensors="pt").to(dev)
    y = torch.tensor([r["target"] for r in batch], dtype=torch.float, device=dev)
    with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
        logit = model(**enc).logits.float().squeeze(-1)
    loss = F.binary_cross_entropy_with_logits(logit, y)
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    sched.step()
    losses.append(loss.item())  # .item() syncs, so wall time below includes the GPU
    lens.append(enc["input_ids"].shape[1])
    if step + 1 == a.warm:
        t_warm = time.time()
    if (step + 1) % 50 == 0:
        print(f"step {step + 1} loss {sum(losses[-50:]) / 50:.4f} pad_len {sum(lens[-50:]) / 50:.0f}", flush=True)

timed = len(losses) - a.warm
rate = timed * a.batch / (time.time() - t_warm)
budget = int(3600 * rate)  # pairs per epoch that fit in 60 min
lo, hi = 0, max(len(v) for v in rows.values()) * pairs.KEEP
while lo < hi:  # largest cap whose epoch (over the tasks in pairs.jsonl now) fits in 3600 s
    mid = (lo + hi + 1) // 2
    if pairs.epoch_pairs(rows, mid) <= budget:
        lo = mid
    else:
        hi = mid - 1
uncapped = pairs.epoch_pairs(rows, hi + 1) <= budget  # the whole pool already fits
out = {"pairs_per_s": round(rate, 1), "steps_timed": timed, "batch": a.batch,
       "mean_pad_len": round(sum(lens) / len(lens)), "first_loss": round(losses[0], 4),
       "last_loss": round(sum(losses[-50:]) / 50, 4),
       "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2) if dev == "cuda" else 0.0,
       "pairs_per_hour": budget,
       "cap_now": {"tasks": len(rows), "cap": None if uncapped else lo,
                   "pairs_per_epoch": pairs.epoch_pairs(rows, lo),
                   "minutes": round(pairs.epoch_pairs(rows, lo) / rate / 60, 1)},
       "cap_by_n_tasks": {n: math.floor(budget / n) for n in a.n_tasks},
       "left_for_new_tasks_at_cap": {c: budget - pairs.epoch_pairs(rows, c) for c in (8000, 12000)}}
print(json.dumps(out, indent=1))
print(f"pairs/s {rate:.1f} = {budget} pairs/hour. Tasks on disk: "
      + (f"the whole pool ({pairs.epoch_pairs(rows, lo)} pairs/epoch after negative sampling) fits, no cap needed"
         if uncapped else f"cap {lo} per task")
      + f"; N tasks all at cap: {out['cap_by_n_tasks']}; pairs left for new tasks with the originals "
      f"at cap 8k/12k: {out['left_for_new_tasks_at_cap']}")
