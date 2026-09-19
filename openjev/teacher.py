"""Label examples with the teacher LLM (vLLM, OpenAI-compatible) via single-letter logprobs.

Writes runs/<name>/teacher.jsonl (append-only, resumable) rows:
  {"id","split","text","gold","probs":[K],"raw":{letter:logprob},"source":"data", ["gold_prob"]}
For K > max_options_per_call, "raw" is {"chunks": [{option_idx: p, "none": p}...], "final": {option_idx: logprob}}.
Also writes runs/<name>/label.json {"calls", "minutes", "n_labeled", "n_failed"} (accumulated over resumes).
"""
import asyncio
import fcntl
import json
import math
import os
import string
import time

import httpx

from openjev.data import append_jsonl, examples, read_jsonl

URL = os.environ.get("OPENJEV_TEACHER_URL", "http://localhost:8000/v1")
LETTERS = string.ascii_uppercase[:19]  # A..S; "Z" is reserved for "none of the above"
NONE = "Z"


def option_lines(task):
    if task.type == "score":
        descs = task.rubric.get("descriptions") or [""] * task.k
        return [f"{lab}: {desc}".rstrip(": ") for lab, desc in zip(task.labels, descs)]
    return list(task.labels)


def system_prompt(task, idx, with_none=False):
    """Prompt over the option subset `idx` (indices into task.labels). Returns (prompt, letters)."""
    if task.type == "noul":  # `question` is optional here; it only overrides the default head line
        head = (task.question or "Is the following statement about the text true?") + \
            f"\nStatement: {task.statement}"
    else:
        head = task.question
    lines = option_lines(task)
    letters = list(LETTERS[: len(idx)])
    body = [f"{l}: {lines[i]}" for l, i in zip(letters, idx)]
    if with_none:
        letters.append(NONE)
        body.append(f"{NONE}: none of the above")
    return "\n".join([head, *body, "Answer with a single letter."]), letters


def parse_logprobs(resp: dict, letters) -> dict:
    """letter -> logprob from a chat completion; tokens are stripped, duplicates log-summed."""
    out = {}
    for t in resp["choices"][0]["logprobs"]["content"][0]["top_logprobs"]:
        tok = t["token"].strip()
        if tok in letters:
            out[tok] = float(t["logprob"]) if tok not in out else float(
                math.log(math.exp(out[tok]) + math.exp(t["logprob"])))
    return out


def softmax_letters(raw: dict, letters) -> list[float]:
    """Softmax over present letters; missing letters get 0."""
    m = max(raw.values())
    e = [math.exp(raw[l] - m) if l in raw else 0.0 for l in letters]
    s = sum(e)
    return [x / s for x in e]


def shortlist(chunk_results, top):
    """chunk_results: list of (option_idx_list, probs over chunk letters + none last).
    Score s_i = p(i|chunk). "none" is one of the letters in that softmax, so a chunk that answers
    "none" already scores all of its options low; multiplying by (1 - p(none)) would count it twice.
    Returns the `top` option indices by score."""
    scores = {}
    for idx, p in chunk_results:
        for i, pi in zip(idx, p[:-1]):
            scores[i] = pi
    return sorted(scores, key=lambda i: -scores[i])[:top]


class Teacher:
    def __init__(self, task, client, model):
        self.task, self.client, self.model = task, client, model
        self.calls = 0

    async def ask(self, system, text, letters):
        body = {"model": self.model, "max_tokens": 1, "temperature": 0, "logprobs": True,
                "top_logprobs": min(len(letters), 20),
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": text}],
                "structured_outputs": {"choice": letters},
                "chat_template_kwargs": {"enable_thinking": False}}
        for attempt in range(4):
            try:
                self.calls += 1
                r = await self.client.post(f"{URL}/chat/completions", json=body)
                if r.status_code < 500:
                    r.raise_for_status()
                    return parse_logprobs(r.json(), letters)
            except (httpx.TimeoutException, httpx.TransportError):
                if attempt == 3:
                    raise
            if attempt < 3:
                await asyncio.sleep(2 ** attempt)
        raise RuntimeError(f"teacher 5xx after retries: {r.status_code} {r.text[:200]}")

    async def label(self, text):
        """Returns (probs over all K labels, raw)."""
        k, m = self.task.k, self.task.teacher.max_options_per_call
        if k <= m:
            system, letters = system_prompt(self.task, range(k))
            raw = await self.ask(system, text, letters)
            return softmax_letters(raw, letters), raw
        n = -(-k // m)  # ceil; balanced chunk sizes (77 -> 5 chunks of 15-16, not 19x4+1)
        chunks = [list(range(j * k // n, (j + 1) * k // n)) for j in range(n)]

        async def one(idx):
            system, letters = system_prompt(self.task, idx, with_none=True)
            return idx, softmax_letters(await self.ask(system, text, letters), letters)

        chunk_results = await asyncio.gather(*(one(c) for c in chunks))
        short = sorted(shortlist(chunk_results, m))  # option order, not score order: letter A must
        # not always be the chunk stage's favourite (that would amplify the teacher's position bias)
        system, letters = system_prompt(self.task, short)
        raw = await self.ask(system, text, letters)
        p = softmax_letters(raw, letters)
        probs = [0.0] * k
        for i, pi in zip(short, p):
            probs[i] = pi
        raw_out = {"chunks": [{**{str(i): pi for i, pi in zip(idx, pr[:-1])}, "none": pr[-1]}
                              for idx, pr in chunk_results],
                   "final": {str(short[LETTERS.index(l)]): lp for l, lp in raw.items()}}
        return probs, raw_out


async def _run(task, todo, out_path):
    stats_path = out_path.parent / "label.json"
    stats = json.loads(stats_path.read_text()) if stats_path.exists() else \
        {"calls": 0, "minutes": 0.0, "n_labeled": 0, "n_failed": 0}
    sem = asyncio.Semaphore(task.teacher.concurrency)
    t0 = time.time()
    async with httpx.AsyncClient(trust_env=False, timeout=120,
                                 limits=httpx.Limits(max_connections=task.teacher.concurrency * 5)) as client:
        model = (await client.get(f"{URL}/models")).json()["data"][0]["id"]
        teacher = Teacher(task, client, model)

        async def work(ex):
            async with sem:
                try:
                    probs, raw = await teacher.label(ex["text"])
                    return {**ex, "probs": probs, "raw": raw, "source": ex.get("source", "data")}
                except Exception as e:  # logged and skipped, not fatal
                    print(f"FAILED {ex['id']}: {e!r}"[:300], flush=True)
                    return None

        done = failed = 0
        buf = []
        with open(out_path, "a") as f:
            for fut in asyncio.as_completed([work(ex) for ex in todo]):
                row = await fut
                if row is None:
                    failed += 1
                    continue
                buf.append(row)
                done += 1
                if len(buf) >= 64:
                    append_jsonl(f, buf)
                    buf = []
                    dt = time.time() - t0
                    print(f"{task.name}: {done}/{len(todo)} {done / dt:.1f} ex/s "
                          f"{teacher.calls / dt:.1f} calls/s failed={failed}", flush=True)
            append_jsonl(f, buf)
    dt = time.time() - t0
    stats = {"calls": stats["calls"] + teacher.calls, "minutes": stats["minutes"] + dt / 60,
             "n_labeled": stats["n_labeled"] + done, "n_failed": failed}
    stats_path.write_text(json.dumps(stats, indent=1))
    print(f"{task.name}: labeled {done} in {dt:.0f}s ({done / max(dt, 1e-9):.1f} ex/s), failed {failed}", flush=True)


def run(task, run_dir, limit=None, extra=None):
    """Label all task examples not yet in run_dir/teacher.jsonl. `extra`: additional example dicts
    (e.g. synthetic, with "source": "synth") labeled the same way."""
    run_dir.mkdir(parents=True, exist_ok=True)
    out_path = run_dir / "teacher.jsonl"
    with open(run_dir / "label.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)  # a second labeler of the same task waits, then finds nothing to do
        have = {r["id"] for r in read_jsonl(out_path)}
        exs = examples(task) + list(extra or [])
        if limit:
            exs = exs[:limit]
        todo = [e for e in exs if e["id"] not in have]
        print(f"{task.name}: {len(exs)} examples, {len(exs) - len(todo)} cached, {len(todo)} to label", flush=True)
        if todo:
            asyncio.run(_run(task, todo, out_path))
