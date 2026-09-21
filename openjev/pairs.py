"""Pair rendering and the per-epoch mixture for `open-jev-base` (PLAN-4 §3).

render(task, option_idx, text) -> (seg_a, seg_b): segment A is the decision (primitive tag, question,
the option line, the contrast set), never truncated; segment B is the text and takes the rest of the
window: `tok(seg_a, seg_b, truncation="only_second", max_length=512)`.

Mixture over data/pairs.jsonl (scripts/build_pairs.py; K consecutive lines per teacher row):
per epoch, per task, at most `cap` pairs, rows redrawn each epoch. K > 8 keeps the teacher-argmax
pair + 4 negatives drawn in proportion to teacher p + 3 uniform; K <= 8 keeps every pair.
"""
import json
import re
import string
from pathlib import Path

from openjev.spec import load_task
from openjev.teacher import option_lines

ROOT = Path(__file__).resolve().parent.parent
MAX_LEN = 512
OPTIONS_TOKENS = 96   # the contrast set is cut here, "…(+N more)"
KEEP = 8              # pairs kept per row when K > KEEP: argmax + HARD + UNIFORM
HARD, UNIFORM = 4, 3
NOUL_HEAD = "Is the following statement about the text true?"
_ROW_START = re.compile(r'\{"task": "([^"]*)", "source_id": "[^"]*", "pair_id": "[^"]*\|0", ')

_tok = None
_options_cache = {}


def tokenizer():
    global _tok
    if _tok is None:
        from transformers import AutoTokenizer
        _tok = AutoTokenizer.from_pretrained("jhu-clsp/mmBERT-small")
    return _tok


def n_tokens(s):
    return len(tokenizer()(s, add_special_tokens=False)["input_ids"])


def options_text(task):
    """All option lines joined by " | ", cut at OPTIONS_TOKENS tokens with "…(+N more)". Cached
    per task: the same string goes into every pair of the task."""
    key = (task.name, tuple(task.labels))
    if key not in _options_cache:
        lines = option_lines(task)
        kept, used = [], 0
        for line in lines:
            used += n_tokens(line) + 1  # +1 for the " |" separator
            if used > OPTIONS_TOKENS:
                break
            kept.append(line)
        s = " | ".join(kept)
        if len(kept) < len(lines):
            s += f"…(+{len(lines) - len(kept)} more)"
        _options_cache[key] = s
    return _options_cache[key]


def render(task, option_idx, text):
    """(seg_a, seg_b). Segment A reads like the teacher prompt (same option lines)."""
    if task.type == "noul":
        head = task.question or NOUL_HEAD
        return f"noul | Q: {head} | statement: {task.statement}", text
    if task.name == "m2w-element":
        # ponytail: the one task-name special case. The 16 options are "candidate A..P" and mean
        # nothing outside the row: the option is the candidate's own "G) <tag> …" line from the text,
        # and the contrast set is omitted because the text already holds all 16.
        letter = string.ascii_uppercase[option_idx]
        line = next((l for l in text.split("\n") if l.startswith(f"{letter}) ")), f"{letter}) ")
        return f"{task.type} | Q: {task.question} | option: {line}", text
    line = option_lines(task)[option_idx]
    return f"{task.type} | Q: {task.question} | option: {line} | options: {options_text(task)}", text


def encode(tok, seg_a, seg_b, max_len=MAX_LEN, **kw):
    return tok(seg_a, seg_b, truncation="only_second", max_length=max_len, **kw)


def task_for(name, tasks=ROOT / "tasks"):
    """Run-dir name -> Task (`<name>-jev` is the same task labelled by Jev; S1 tasks live in tasks/base/)."""
    stem = name[: -len("-jev")] if name.endswith("-jev") else name
    for p in (tasks / f"{stem}.yaml", tasks / "base" / f"{stem}.yaml"):
        if p.exists():
            return load_task(p)
    raise FileNotFoundError(f"no task YAML for {name}")


def keep_pairs(probs, rng):
    """Indices of the pairs kept from one teacher row. K <= KEEP: all. K > KEEP: argmax + HARD
    drawn ∝ p without replacement (p == 0 only fills when fewer than HARD have p > 0) + UNIFORM."""
    k = len(probs)
    if k <= KEEP:
        return list(range(k))
    top = max(range(k), key=probs.__getitem__)
    rest = [i for i in range(k) if i != top]
    # weighted sampling without replacement (Efraimidis-Spirakis): key = u^(1/w); w == 0 sorts last
    key = {i: rng.random() ** (1 / probs[i]) if probs[i] > 0 else -rng.random() for i in rest}
    hard = sorted(rest, key=key.__getitem__, reverse=True)[:HARD]
    left = [i for i in rest if i not in set(hard)]
    return [top, *hard, *rng.sample(left, UNIFORM)]


def index(path=ROOT / "data/pairs.jsonl", split="train"):
    """{task: [(offset, k), ...]} for every teacher row of `split`: the byte offset of its first pair.
    One pass over the file; only the first pair of each row is parsed."""
    rows = {}
    with open(path, "rb") as f:
        off = 0
        for line in f:
            if line.startswith(b'{"task": "'):
                head = line[:200].decode("utf-8", "ignore")
                if _ROW_START.match(head):
                    r = json.loads(line)
                    if r["split"] == split:
                        rows.setdefault(r["task"], []).append((off, r["k"]))
            off += len(line)
    return rows


def per_row(k):
    return min(k, KEEP)


def epoch_pairs(rows, cap):
    """Pairs one epoch yields under a per-task cap (what the smoke run rescales)."""
    return sum(min(len(v), cap // per_row(v[0][1])) * per_row(v[0][1]) for v in rows.values())


def sample(rows, cap, rng, path=ROOT / "data/pairs.jsonl", tasks=None):
    """One epoch of the mixture: yields {seg_a, seg_b, target, row_id, task, k}. Rows are drawn
    fresh per call (≤ cap pairs per task), negatives redrawn; reads the file by seeking, so the
    memory is one row at a time."""
    tasks = tasks if tasks is not None else {}
    order = []
    for name, v in rows.items():
        k = v[0][1]
        order += [(name, off) for off, _ in rng.sample(v, min(len(v), cap // per_row(k)))]
    rng.shuffle(order)
    with open(path, "rb") as f:
        for name, off in order:
            if name not in tasks:
                tasks[name] = task_for(name)
            task = tasks[name]
            f.seek(off)
            pairs = [json.loads(f.readline()) for _ in range(task.k)]
            for i in keep_pairs([p["p"] for p in pairs], rng):
                p = pairs[i]
                a, b = render(task, p["option_index"], p["text"])
                yield {"seg_a": a, "seg_b": b, "target": p["p"], "row_id": p["source_id"],
                       "task": name, "k": task.k}
