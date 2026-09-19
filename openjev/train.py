"""Distill teacher probs into a small encoder: KL(teacher || student) (+ gold_weight * CE), early stop on calib KL."""
import json
import math
import os
import random
import time
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup


def read_rows(run_dir, split):
    with open(Path(run_dir) / "teacher.jsonl") as f:
        return [r for r in map(json.loads, f) if r["split"] == split]


def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_student(path, dev=None):
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path, attn_implementation="sdpa")
    return tok, model.to(dev or device()).eval()


def _batch(tok, texts, max_len, dev):
    return tok(texts, truncation=True, max_length=max_len, padding=True, return_tensors="pt").to(dev)


@torch.no_grad()
def predict_logits(tok, model, texts, max_len=256, batch_size=64):
    dev = next(model.parameters()).device
    out = []
    for i in range(0, len(texts), batch_size):
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            out.append(model(**_batch(tok, texts[i:i + batch_size], max_len, dev)).logits.float().cpu())
    return torch.cat(out)


def _targets(rows):
    p = torch.tensor([r["probs"] for r in rows], dtype=torch.float).clamp_min(1e-6)
    return p / p.sum(-1, keepdim=True)


def run(task, run_dir, student=None, epochs=None, batch_size=None, max_steps=None, patience=2, seed=0):
    run_dir = Path(run_dir)
    s = task.student
    name = student or s.model
    epochs, bs = epochs or s.epochs, batch_size or s.batch_size
    torch.manual_seed(seed)
    random.seed(seed)
    dev = device()
    train, calib = read_rows(run_dir, "train"), read_rows(run_dir, "calib")
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSequenceClassification.from_pretrained(
        name, num_labels=task.k, id2label=dict(enumerate(task.labels)),
        label2id={l: i for i, l in enumerate(task.labels)}, attn_implementation="sdpa").to(dev)
    steps = epochs * math.ceil(len(train) / bs)
    if max_steps:
        steps = min(steps, max_steps)
    opt = torch.optim.AdamW(model.parameters(), lr=s.lr, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
    calib_texts, calib_p = [r["text"] for r in calib], _targets(calib)
    if dev == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0, step, best, best_epoch, bad, history = time.time(), 0, float("inf"), -1, 0, []
    for epoch in range(epochs):
        model.train()
        random.shuffle(train)
        losses = []
        for i in range(0, len(train), bs):
            rows = train[i:i + bs]
            enc = _batch(tok, [r["text"] for r in rows], s.max_len, dev)
            with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
                logits = model(**enc).logits.float()
            logp = F.log_softmax(logits, -1)
            loss = F.kl_div(logp, _targets(rows).to(dev), reduction="batchmean")
            gold = [(j, r["gold"]) for j, r in enumerate(rows) if r.get("gold") is not None]
            if s.gold_weight and gold:
                idx, y = zip(*gold)
                loss = loss + s.gold_weight * F.nll_loss(logp[list(idx)], torch.tensor(y, device=dev))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            losses.append(loss.item())
            step += 1
            if step >= steps:
                break
        model.eval()
        q = F.log_softmax(predict_logits(tok, model, calib_texts, s.max_len), -1)
        kl = F.kl_div(q, calib_p, reduction="batchmean").item()
        history.append({"epoch": epoch, "train_loss": sum(losses) / len(losses), "calib_kl": kl})
        print(f"epoch {epoch} train_loss {history[-1]['train_loss']:.4f} calib_kl {kl:.4f}", flush=True)
        if kl < best:
            best, best_epoch, bad = kl, epoch, 0
            model.save_pretrained(run_dir / "student")
            tok.save_pretrained(run_dir / "student")
        else:
            bad += 1
            if bad >= patience:
                break
        if step >= steps:
            break
    info = {"student": name, "n_train": len(train), "n_synth": sum(r.get("source") == "synth" for r in train),
            "gold_weight": s.gold_weight, "best_epoch": best_epoch, "best_calib_kl": best, "history": history,
            "train_minutes": (time.time() - t0) / 60,
            "peak_gpu_gb": torch.cuda.max_memory_allocated() / 2**30 if dev == "cuda" else 0.0}
    (run_dir / "train.json").write_text(json.dumps(info, indent=2))
    return info
