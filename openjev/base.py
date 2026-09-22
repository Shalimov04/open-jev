"""`open-jev-base`: one pair scorer for a typed question (PLAN-4 §3).

`AutoModelForSequenceClassification(num_labels=1)` trained pointwise with BCE against the teacher's
probability of that option. Inference renormalises the per-option sigmoids over the K options offered
and returns *log-probabilities*, so the `[N, K]` matrix drops straight into `calibrate.apply` /
`evaluate.run` and every downstream table (cascade, selective, compare) is unchanged.

  openjev base train --fold F1                      -> runs/base-F1/
  openjev base eval --fold F1 --task tasks/kinopoisk.yaml   -> runs/base-F1-zs-kinopoisk-<variant>/
"""
import json
import os
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from openjev import calibrate, pairs
from openjev.data import read_rows

ROOT = Path(__file__).resolve().parent.parent
FOLDS = ROOT / "tasks/base/folds.yaml"
VARIANTS = ("raw", "prior", "teacher500", "gold500")


def stem(name):
    """Run-dir name -> task stem (`kinopoisk-jev` is the same texts as `kinopoisk`)."""
    return name[: -len("-jev")] if name.endswith("-jev") else name


def folds(path=FOLDS):
    return yaml.safe_load(Path(path).read_text())


def excluded(fold, path=FOLDS):
    """Every run-dir stem held out with `fold` -- the whole text-source group, not one task name."""
    if fold in (None, "none"):
        return set()
    spec = folds(path)
    if fold not in spec or fold == "eval":
        raise ValueError(f"unknown fold {fold!r}; have {[k for k in spec if k != 'eval']} or none")
    return {t for group in spec[fold].values() for t in group}


def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def load(path, dev=None):
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path, attn_implementation="sdpa")
    return tok, model.to(dev or device()).eval()


def _encode(tok, batch, dev):
    enc = pairs.encode(tok, [r["seg_a"] for r in batch], [r["seg_b"] for r in batch],
                       padding=True, return_tensors="pt")
    return {k: v.to(dev) for k, v in enc.items()}


@torch.no_grad()
def _bce(tok, model, rows, batch_size=64):
    dev = next(model.parameters()).device
    tot, n = 0.0, 0
    for i in range(0, len(rows), batch_size):
        b = rows[i:i + batch_size]
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            z = model(**_encode(tok, b, dev)).logits.float().squeeze(-1)
        y = torch.tensor([r["target"] for r in b], dtype=torch.float, device=dev)
        tot += float(F.binary_cross_entropy_with_logits(z, y, reduction="sum"))
        n += len(b)
    return tot / n


@torch.no_grad()
def predict_logits(tok, model, task, texts, batch_size=64, max_len=pairs.MAX_LEN):
    """[N, K] renormalised log-probabilities. K forward passes per decision; `noul` is one pair
    scored once and returned as [log p, log(1-p)] so `views.view` and `evaluate` need no change."""
    dev = next(model.parameters()).device
    k = 1 if task.type == "noul" else task.k
    flat = [pairs.render(task, i, t) for t in texts for i in range(k)]
    out = []
    for i in range(0, len(flat), batch_size):
        b = [{"seg_a": a, "seg_b": s} for a, s in flat[i:i + batch_size]]
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            out.append(model(**_encode(tok, b, dev)).logits.float().squeeze(-1).cpu())
    p = torch.cat(out).sigmoid().view(len(texts), k).clamp(1e-6, 1 - 1e-6)
    if task.type == "noul":  # labels are ["true", "false"]
        p = torch.cat([p, 1 - p], -1)
    return (p / p.sum(-1, keepdim=True)).log()


def train(fold, run_dir, cap=12000, steps=None, minutes=55, batch=32, lr=5e-5, seed=0,
          eval_every=1500, calib_cap=120, patience=3, min_delta=0.0, max_epochs=None,
          max_len=pairs.MAX_LEN, init="jhu-clsp/mmBERT-small", only=None, limit=None):
    """One pair model. `steps` defaults to `max_epochs` epochs over the capped mixture, else to what
    fits in `minutes` at the measured rate; rows (and the K > 8 negatives) are redrawn every epoch.
    Model selection: mean BCE on the held-in tasks' *calib* pairs, evaluated every `eval_every`
    steps; a checkpoint improves only if it beats the best by `min_delta`, `patience` non-improving
    evaluations stop the run. Reaching the step ceiling without that is `converged: false`."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    drop = excluded(fold)
    torch.manual_seed(seed)
    rng = random.Random(seed)
    dev = device()
    tok = pairs.tokenizer()

    def pool(split):
        d = {n: v for n, v in pairs.index(split=split).items() if stem(n) not in drop}
        if only:  # warm start: exact run-dir names (runs/kinopoisk, not also kinopoisk-jev)
            d = {n: v for n, v in d.items() if n in set(only)}
        if limit and split == "train":
            d = {n: random.Random(7).sample(v, min(limit, len(v))) for n, v in d.items()}
        return d

    rows, crows = pool("train"), pool("calib")
    per_epoch = pairs.epoch_pairs(rows, cap)
    calib = list(pairs.sample(crows, calib_cap, random.Random(12345)))

    model = AutoModelForSequenceClassification.from_pretrained(
        init, num_labels=1, attn_implementation="sdpa").to(dev).train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    if steps is None and max_epochs:  # the prereg's ceiling: N epochs over the capped mixture
        steps = max(1, -(-max_epochs * per_epoch // batch))
    if steps is None:  # 74.7 pairs/s measured at len 512 (M0); re-measured below and recorded
        steps = max(1, int(minutes * 60 * 74.7 / batch))
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
    if dev == "cuda":
        torch.cuda.reset_peak_memory_stats()

    print(f"fold {fold}: {len(rows)} train tasks (dropped {sorted(drop) or 'nothing'}), "
          f"{per_epoch} pairs/epoch at cap {cap}, {len(calib)} calib pairs, {steps} steps",
          flush=True)
    t0, step, epoch, best, bad, history, losses = time.time(), 0, 0, float("inf"), 0, [], []
    stopped = False
    stream = iter(())
    while step < steps:
        batch_rows = [r for _, r in zip(range(batch), stream)]
        if len(batch_rows) < batch:
            epoch += 1
            stream = pairs.sample(rows, cap, rng)
            batch_rows += [r for _, r in zip(range(batch - len(batch_rows)), stream)]
        with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
            z = model(**_encode(tok, batch_rows, dev)).logits.float().squeeze(-1)
        y = torch.tensor([r["target"] for r in batch_rows], dtype=torch.float, device=dev)
        loss = F.binary_cross_entropy_with_logits(z, y)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        losses.append(loss.item())
        step += 1
        if step % eval_every == 0 or step == steps:
            model.eval()
            cb = _bce(tok, model, calib)
            model.train()
            rate = step * batch / (time.time() - t0)
            history.append({"step": step, "epoch": epoch, "train_bce": sum(losses[-eval_every:])
                            / len(losses[-eval_every:]), "calib_bce": cb, "pairs_per_s": rate})
            print(f"step {step} epoch {epoch} train_bce {history[-1]['train_bce']:.4f} "
                  f"calib_bce {cb:.4f} {rate:.1f} pairs/s", flush=True)
            improved = cb < best - min_delta      # patience counts min_delta improvements...
            if cb < best:                         # ...but `student/` is always the best checkpoint
                best = cb
                model.save_pretrained(run_dir / "student")
                tok.save_pretrained(run_dir / "student")
            bad = 0 if improved else bad + 1
            if bad >= patience:
                stopped = True
                print(f"early stop at step {step} (best calib_bce {best:.4f})", flush=True)
                break
    info = {"fold": fold, "student": init, "pair_model": True, "seed": seed,
            "only": only, "limit": limit,
            "excluded": sorted(drop), "train_tasks": sorted(rows), "cap": cap, "batch": batch,
            "lr": lr, "max_len": max_len, "steps_planned": steps, "steps_run": step,
            "epochs_seen": epoch, "pairs_per_epoch": per_epoch, "n_calib_pairs": len(calib),
            "n_train": sum(len(v) for v in rows.values()), "n_synth": 0, "augment_round": 0,
            "gold_weight": 0.0, "best_calib_bce": best, "history": history,
            "patience": patience, "min_delta": min_delta, "max_epochs": max_epochs,
            "converged": stopped,
            "train_minutes": (time.time() - t0) / 60,
            "peak_gpu_gb": torch.cuda.max_memory_allocated() / 2**30 if dev == "cuda" else 0.0}
    (run_dir / "train.json").write_text(json.dumps(info, indent=2))
    return info


def _calib_json(task, logits, rows, variant):
    """The four calibration variants of PLAN-4 §M2, all fitted on the task's existing 500 calib
    rows: nothing here ever touches an eval row."""
    if variant == "raw":
        return {"temperature": 1.0, "method": "none", "bias": None, "target": "raw"}
    if variant == "prior":  # no labels at all: move the mean probability onto the declared prior
        prior = calibrate.prior_vector(task) or [1.0 / task.k] * task.k
        bias = calibrate.fit_prior_bias(logits, prior)
        return {"temperature": 1.0, "method": "vector", "bias": bias.tolist(),
                "target": "prior-declared" if getattr(task, "prior", None) else "prior-uniform"}
    if variant == "teacher500":  # "one minute of teacher": T fitted to the teacher's soft probs
        p = torch.tensor([r["probs"] for r in rows]).clamp_min(1e-6)
        t = calibrate.fit_temperature(logits, p / p.sum(-1, keepdim=True))
        return {"temperature": t, "method": "temperature", "bias": None, "target": "teacher-soft"}
    y, target = calibrate.targets_for(rows)  # gold500: exactly what the per-task student got
    method = calibrate.select_method(logits, y)
    t, bias = (calibrate.fit_vector(logits, y) if method == "vector"
               else (calibrate.fit_temperature(logits, y), None))
    return {"temperature": t, "method": method, "bias": None if bias is None else bias.tolist(),
            "target": target}


def zs_eval(base_dir, task, src_run, runs=None, variants=VARIANTS, batch_size=64, name=None):
    """Run a base model zero-shot on `src_run`'s existing calib/eval rows and write one ordinary
    run dir per calibration variant, so `openjev compare` / `eval` work on them unchanged."""
    from openjev import evaluate

    base_dir, src_run = Path(base_dir), Path(src_run)
    runs = Path(runs or ROOT / "runs")
    tok, model = load(base_dir / "student")
    train_info = json.loads((base_dir / "train.json").read_text())
    held_out = stem(src_run.name) in train_info["excluded"]
    if not held_out and train_info["fold"] not in (None, "none"):
        raise ValueError(f"{src_run.name} was in {base_dir.name}'s training mixture — that is a leak, "
                         f"not a zero-shot number (excluded: {train_info['excluded']})")

    if name and len(variants) != 1:
        raise ValueError("an explicit run-dir name takes exactly one calibration variant")
    cache = {}

    def logits_fn(texts):
        key = (len(texts), texts[0], texts[-1])
        if key not in cache:
            cache[key] = predict_logits(tok, model, task, texts, batch_size)
        return cache[key]

    crows = read_rows(src_run, "calib")
    clogits = logits_fn([r["text"] for r in crows])
    out = {}
    for v in variants:
        d = runs / (name if name else f"{base_dir.name}-zs-{task.name}-{v}")
        d.mkdir(parents=True, exist_ok=True)
        for f in ("teacher.jsonl", "label.json"):
            if (src_run / f).exists() and not (d / f).is_symlink():
                (d / f).symlink_to(os.path.relpath(src_run / f, d))
        cal = _calib_json(task, clogits, crows, v)
        y, _ = calibrate.targets_for(crows)
        cal["ece_before"] = calibrate.ece(clogits.softmax(-1), y)
        cal["ece_after"] = calibrate.ece(calibrate.apply(clogits, cal).softmax(-1), y)
        cal["variant"] = v
        (d / "calib.json").write_text(json.dumps(cal, indent=2))
        (d / "train.json").write_text(json.dumps({**train_info, "zero_shot_on": task.name,
                                                  "calib_variant": v}, indent=2))
        torch.save({"ids": [r["id"] for r in crows], "logits": clogits}, d / "calib_logits.pt")
        # results/base/: `openjev report` globs results/*.json for the README table, and a base
        # zero-shot row is a different experiment (PLAN-4 M7).
        res = evaluate.run(task, d, results_dir=ROOT / "results/base", latency=False,
                           logits_fn=logits_fn)
        out[v] = res
        print(f"{d.name}: {res['student']} ", flush=True)
    return out
