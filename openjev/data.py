"""Load a task's rows, apply the text template, map gold, and draw deterministic disjoint splits.

Example dict: {"id": "<source_split>:<row_idx>", "split": "train|calib|eval", "text": str,
               "gold": int|None, "gold_prob": float (only if data.gold_prob is set)}
"""
import json
import random
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset

ROLES = ("train", "calib", "eval")


def _load(src: dict, split: str):
    if "hf" in src:
        return load_dataset(src["hf"], src.get("config"), revision=src.get("revision"), split=split)
    kind = "csv" if "csv" in src else "json"
    return load_dataset(kind, data_files=src.get("csv") or src["jsonl"], split=split)


def gold_index(task, raw):
    d = task.data
    if raw is None:
        return None
    if d.gold_map is not None:
        raw = d.gold_map[raw]
    if task.type == "noul" and isinstance(raw, (bool, int, float)):
        return 0 if raw >= d.gold_threshold else 1  # labels = ["true", "false"]; True/1 -> "true"
    if isinstance(raw, str):
        return task.labels.index(raw)
    return int(raw)


def _pick(pool, golds, n, balance, k, rng):
    rng.shuffle(pool)
    if not balance:
        return pool[:n]
    by_cls = defaultdict(list)
    for i in pool:
        by_cls[golds[i]].append(i)
    per = n // k
    chosen = [i for c in by_cls for i in by_cls[c][:per]]
    taken = set(chosen)
    chosen += [i for i in pool if i not in taken][: n - len(chosen)]  # fill if a class is short
    rng.shuffle(chosen)
    return chosen


def examples(task):
    """All examples, in role order. Train and calib drawn from the same source split never
    overlap (roles are drawn in order from the not-yet-used rows)."""
    d = task.data
    used = defaultdict(set)
    out = []
    cache = {}
    # A balanced draw takes rows *by gold* and depletes the pool, so the natural-rate roles are
    # drawn first: otherwise calib/eval no longer carry the source prior. sorted() is stable, so
    # with no balanced split this is train -> calib -> eval as before.
    for role in sorted(ROLES, key=lambda r: getattr(d, r).balance):
        sp = getattr(d, role)
        if sp.split not in cache:
            cache[sp.split] = _load(d.source, sp.split)
        ds = cache[sp.split]
        golds = [gold_index(task, g) for g in ds[d.gold]] if d.gold else [None] * len(ds)
        pool = [i for i in range(len(ds)) if i not in used[sp.split]]
        idx = _pick(pool, golds, sp.n, sp.balance, task.k, random.Random(f"{d.seed}:{role}"))
        if len(idx) < sp.n:
            raise ValueError(
                f"{role}: asked for {sp.n} rows from split {sp.split!r} but only {len(idx)} are left "
                f"(the split has {len(ds)} rows and the other roles already took {len(used[sp.split])}). "
                f"Lower data.{role}.n, or point the roles at different splits.")
        used[sp.split].update(idx)
        rows = ds.select(idx)
        for i, row in zip(idx, rows):
            ex = {"id": f"{sp.split}:{i}", "split": role,
                  "text": d.text.format(**row)[: d.max_chars], "gold": golds[i]}
            if d.gold_prob:
                ex["gold_prob"] = float(row[d.gold_prob])
            out.append(ex)
    return sorted(out, key=lambda e: ROLES.index(e["split"]))  # drawn in pool order, returned in role order


def read_jsonl(path):
    try:
        with open(path) as f:
            lines = [l for l in f if l.strip()]
        out = []
        for l in lines:
            try:
                out.append(json.loads(l))
            except json.JSONDecodeError:  # truncated last line after a crash
                pass
        return out
    except FileNotFoundError:
        return []


def read_rows(run_dir, split):
    """Teacher rows for one split, sorted by id. Tolerates a line left half-written by a crash
    mid-flush. B6: the file is appended in teacher-completion order, so sorting here is what makes
    the training batch order depend on the seed only (seeded, not bit-exact — CUDA still drifts)."""
    rows = [r for r in read_jsonl(Path(run_dir) / "teacher.jsonl") if r["split"] == split]
    return sorted(rows, key=lambda r: r["id"])


def append_jsonl(f, rows):
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
    f.flush()
