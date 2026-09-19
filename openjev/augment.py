"""PGKD-lite active loop: the teacher writes new training texts aimed at the student's weak classes.

Reads runs/<name>/eval.json (per-class recall + confusion on the **calib** split, never eval), asks the
teacher for new texts for the weakest classes and the top confusion pairs, and appends them to the
append-only runs/<name>/teacher.jsonl as `{"source": "synth", "split": "train"}` rows with ids
"synth:<round>:<i>" — ids the dataset can never produce ("<source_split>:<row_idx>"), so `data.examples`
still enumerates only the dataset and resume keeps working unchanged. Labeling goes through the normal
teacher path (`teacher.run(..., extra=...)`); the teacher may disagree with the intended class and that
label is what is kept.
"""
import json
import re

import httpx

from openjev import teacher
from openjev.data import read_rows

SYSTEM = ("You write realistic training examples for a text classifier. "
          "Output ONLY a JSON array of strings and nothing else.")
MIN_CHARS = 10  # shorter than this is JSON punctuation or a stray fence, not an example


def targets(ev, cap=6, n_weak=3, n_pairs=3):
    """[(class_idx, confused_with_idx|None)]: weakest classes by recall, then top confusion pairs."""
    cs = ev["calib_split"]
    rec, cm = list(cs["recall"].values()), cs["confusion"]
    weak = sorted(range(len(rec)), key=lambda i: rec[i])[:n_weak]
    pairs = sorted(((cm[i][j], i, j) for i in range(len(cm)) for j in range(len(cm)) if i != j),
                   reverse=True)[:n_pairs]
    out = [(i, None) for i in weak] + [(i, j) for c, i, j in pairs if c]
    return list(dict.fromkeys(out))[:cap]


def parse_texts(raw, max_chars, seen):
    """Lenient: a JSON array if one can be found and parsed, else one text per line (bullets,
    numbering and JSON quoting stripped). Drops empties, texts over max_chars and duplicates —
    `seen` holds whitespace/case-normalised texts already in the run and is updated in place."""
    items = None
    m = re.search(r"\[.*]", raw, re.S)
    if m:
        try:
            items = [x for x in json.loads(m.group(0)) if isinstance(x, str)]
        except (json.JSONDecodeError, TypeError):
            items = None
    if items is None:
        items = [re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", l).strip().strip(",").strip('"')
                 for l in raw.splitlines()]
    out = []
    for t in items:
        t = t.strip()
        key = " ".join(t.lower().split())
        if len(t) < MIN_CHARS or len(t) > max_chars or key in seen:
            continue
        seen.add(key)
        out.append(t)
    return out


def generate(client, task, model, cls, other, n, shots):
    lines = teacher.option_lines(task)
    ask = [f"Task: {task.question or task.statement}",
           f'Write {n} new realistic texts in language "{task.lang}", each one a correct example of '
           f'the label "{lines[cls]}".']
    if other is not None:
        ask.append(f'Make about half of them superficially look like "{lines[other]}" but really be '
                   f'"{lines[cls]}" — those are the hard cases the classifier gets wrong.')
    ask += ["Match the style, length and register of these real examples:",
            *(f"- {s[:400]}" for s in shots),
            f"Do not copy them. Output a JSON array of exactly {n} strings."]
    r = client.post(f"{teacher.URL}/chat/completions", json={
        "model": model, "temperature": 0.9, "max_tokens": min(8000, 200 * n + 500),
        "messages": [{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": "\n".join(ask)}],
        "chat_template_kwargs": {"enable_thinking": False}})
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def round_once(task, run_dir, per_class=100, cap=6, label=True):
    """One round: generate, dedup, append as synth rows, label. Returns (round, n_synth)."""
    ev = json.loads((run_dir / "eval.json").read_text())
    train = read_rows(run_dir, "train")
    rnd = max((int(r["id"].split(":")[1]) for r in train if r.get("source") == "synth"), default=0) + 1
    seen = {" ".join(r["text"].lower().split()) for r in train}
    by_cls = {}
    for r in train:  # few-shot pool: gold when the task has it, else the teacher's argmax
        i = r["gold"] if r.get("gold") is not None else max(range(task.k), key=lambda j: r["probs"][j])
        by_cls.setdefault(i, []).append(r["text"])
    new = []
    with httpx.Client(trust_env=False, timeout=600) as c:
        model = c.get(f"{teacher.URL}/models").json()["data"][0]["id"]
        for cls, other in targets(ev, cap):
            texts = parse_texts(generate(c, task, model, cls, other, per_class, by_cls.get(cls, [])[:3]),
                                task.data.max_chars, seen)
            print(f"round {rnd}: {task.labels[cls]}"
                  f"{'' if other is None else f' (vs {task.labels[other]})'}: {len(texts)} new texts",
                  flush=True)
            new += [{"id": f"synth:{rnd}:{len(new) + j}", "split": "train", "text": t, "gold": None,
                     "source": "synth", "intended": task.labels[cls]} for j, t in enumerate(texts)]
    if not new:
        raise RuntimeError("augment: the teacher produced no usable texts")
    if label:
        teacher.run(task, run_dir, extra=new)
    return rnd, len(new)
