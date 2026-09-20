"""Label examples with a teacher: vLLM (default, single-letter logprobs) or TypeSafe Jev
(`backend="jev"`, POST /alpha/decisions on OpenRouter; label.json then also carries "cost").

Writes runs/<name>/teacher.jsonl (append-only, resumable) rows:
  {"id","split","text","gold","probs":[K],"raw":{letter:logprob},"source":"data", ["gold_prob"]}
For K > max_options_per_call, "raw" is {"chunks": [{option_idx: p, "none": p}...], "final": {option_idx: logprob}}.
Also writes runs/<name>/label.json {"model", "calls", "minutes", "n_labeled", "n_failed"}
("model" is the served teacher id; the counters accumulate over resumes).
"""
import asyncio
import fcntl
import json
import math
import os
import random
import string
import time

from pathlib import Path

import httpx

from openjev.data import append_jsonl, examples, read_jsonl

URL = os.environ.get("OPENJEV_TEACHER_URL", "http://localhost:8000/v1")
JEV_URL = os.environ.get("OPENJEV_JEV_URL", "https://openrouter.ai/api/alpha/decisions")
JEV_MODEL = os.environ.get("OPENJEV_JEV_MODEL", "typesafe/jev-1.13")
JEV_BUDGET = float(os.environ.get("OPENJEV_JEV_BUDGET", "1.0"))  # USD per command, hard stop
JEV_CONCURRENCY = int(os.environ.get("OPENJEV_JEV_CONCURRENCY", "8"))
# ALL_PROXY here is socks5h, which httpx cannot use without extras; the http proxy reaches openrouter.
JEV_PROXY = os.environ.get("OPENJEV_JEV_PROXY") or os.environ.get("HTTPS_PROXY")
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


class BudgetExceeded(Exception):
    pass


class JevTeacher:
    """TypeSafe Jev via OpenRouter's /alpha/decisions. Same interface as Teacher: `.label(text)`
    -> (probs in task.labels order, raw answer). One call per example, no chunking (Jev takes the
    whole criteria set). `choice` criteria use the option text as key *and* description: our task
    YAMLs carry no per-option gloss, and inventing one would change the question."""

    def __init__(self, task, client, key):
        self.task, self.client = task, client
        self.headers = {"Authorization": f"Bearer {key}"}
        self.question = jev_question(task)
        self.calls = 0
        self.cost = 0.0
        self.model = JEV_MODEL  # replaced by the served id from the first response

    def stats(self):
        return {"cost": round(self.cost, 6)}

    async def label(self, text):
        if self.cost > JEV_BUDGET:
            raise BudgetExceeded(f"jev: budget stop, ${self.cost:.4f} > ${JEV_BUDGET:.2f}")
        body = {"model": JEV_MODEL, "state": text, "questions": {"q": self.question}}
        for attempt in range(4):
            self.calls += 1
            try:
                r = await self.client.post(JEV_URL, json=body, headers=self.headers)
                if r.status_code not in (429, 408) and r.status_code < 500:
                    r.raise_for_status()
                    d = r.json()
                    self.model = d.get("model", self.model)
                    self.cost += float(d.get("usage", {}).get("cost", 0.0))
                    if self.calls % 200 == 0:
                        print(f"  jev: {self.calls} calls, ${self.cost:.4f}", flush=True)
                    return jev_probs(self.task, d["answers"]["q"]), d["answers"]["q"]
            except (httpx.TimeoutException, httpx.TransportError):
                if attempt == 3:
                    raise
            if attempt < 3:
                await asyncio.sleep(2 ** attempt + random.random())
            else:
                raise RuntimeError(f"jev {r.status_code}: {r.text[:200]}")


def jev_question(task):
    if task.type == "choice":
        return {"type": "choice", "instructions": task.question,
                "criteria": {o: o for o in task.options}}
    if task.type == "score":
        return {"type": "score", "instructions": task.question, "criteria": option_lines(task)}
    return {"type": "noul", "instructions": task.statement}


def jev_probs(task, a):
    """Jev answer -> probabilities in *our* label order."""
    if a["type"] == "noul":
        p = float(a["noul"])
        return [p, 1 - p]  # labels are ["true", "false"]
    probs = a["probabilities"]
    if a["type"] == "score":  # keys are criteria indices as strings; legend maps them to our lines
        lines, out = option_lines(task), [0.0] * task.k
        for i, p in probs.items():
            out[lines.index(a["legend"][str(i)])] = float(p)
        return out
    return [float(probs.get(l, 0.0)) for l in task.labels]


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


async def _run(task, todo, out_path, backend="vllm"):
    stats_path = out_path.parent / "label.json"
    stats = json.loads(stats_path.read_text()) if stats_path.exists() else \
        {"calls": 0, "minutes": 0.0, "n_labeled": 0, "n_failed": 0}
    conc = JEV_CONCURRENCY if backend == "jev" else task.teacher.concurrency
    sem = asyncio.Semaphore(conc)
    t0 = time.time()
    halted = []
    async with httpx.AsyncClient(trust_env=False, timeout=120,
                                 proxy=JEV_PROXY if backend == "jev" else None,
                                 limits=httpx.Limits(max_connections=conc * 5)) as client:
        if backend == "jev":
            teacher = JevTeacher(task, client, jev_key())
        else:
            model = (await client.get(f"{URL}/models")).json()["data"][0]["id"]
            teacher = Teacher(task, client, model)

        async def work(ex):
            async with sem:
                try:
                    probs, raw = await teacher.label(ex["text"])
                    return {**ex, "probs": probs, "raw": raw, "source": ex.get("source", "data")}
                except BudgetExceeded as e:  # stop spending; the rest of the rows are left unlabeled
                    if not halted:
                        halted.append(print(str(e), flush=True))
                    return None
                except Exception as e:  # logged and skipped, not fatal
                    print(f"FAILED {ex['id']}: {e!r}"[:300], flush=True)
                    return None

        done = failed = 0

        def save_stats():  # B5: written as we go, so a crash mid-run does not lose calls/minutes
            stats_path.write_text(json.dumps(
                {"model": teacher.model, "calls": stats["calls"] + teacher.calls,
                 "minutes": stats["minutes"] + (time.time() - t0) / 60,
                 "n_labeled": stats["n_labeled"] + done, "n_failed": stats["n_failed"] + failed,
                 **(getattr(teacher, "stats", dict)())}, indent=1))

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
                    save_stats()
                    dt = time.time() - t0
                    print(f"{task.name}: {done}/{len(todo)} {done / dt:.1f} ex/s "
                          f"{teacher.calls / dt:.1f} calls/s failed={failed}", flush=True)
            append_jsonl(f, buf)
        save_stats()
    dt = time.time() - t0
    print(f"{task.name}: labeled {done} in {dt:.0f}s ({done / max(dt, 1e-9):.1f} ex/s), failed {failed}", flush=True)


def jev_key():
    """OPENROUTER_API_KEY from the env or .env.local (gitignored). Never printed."""
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        for line in Path(".env.local").read_text().splitlines():
            if line.startswith("OPENROUTER_API_KEY"):
                key = line.split("=", 1)[1].strip().strip("'\"")
    if not key:
        raise SystemExit("no OPENROUTER_API_KEY (env or .env.local)")
    return key


def run(task, run_dir, limit=None, extra=None, split=None, backend="vllm"):
    """Label all task examples not yet in run_dir/teacher.jsonl. `extra`: additional example dicts
    (e.g. synthetic, with "source": "synth") labeled the same way. `split`: only that role
    (`check --probe` labels calib rows into the same append-only file)."""
    run_dir.mkdir(parents=True, exist_ok=True)
    out_path = run_dir / "teacher.jsonl"
    with open(run_dir / "label.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)  # a second labeler of the same task waits, then finds nothing to do
        have = {r["id"] for r in read_jsonl(out_path)}
        if backend == "jev":  # reuse the vLLM run's rows: same ids, same splits, same texts
            base = run_dir.parent / task.name / "teacher.jsonl"
            exs = [{k: r[k] for k in ("id", "split", "text", "gold", "gold_prob", "source") if k in r}
                   for r in read_jsonl(base)]
            if not exs:
                raise SystemExit(f"{base} is missing: label with the vLLM teacher first")
        else:
            exs = examples(task) + list(extra or [])
        if split:
            exs = [e for e in exs if e["split"] == split]
        if limit:
            exs = exs[:limit]
        todo = [e for e in exs if e["id"] not in have]
        print(f"{task.name}: {len(exs)} examples, {len(exs) - len(todo)} cached, {len(todo)} to label", flush=True)
        if todo:
            asyncio.run(_run(task, todo, out_path, backend))
