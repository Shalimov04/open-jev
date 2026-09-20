"""`openjev check task.yaml [--probe N]`: dry-run a task before spending teacher time.

Without --probe nothing is sent to the teacher (one GET /models ping). With --probe N the first N
*calib* rows are labeled into the task's real runs/<name>/teacher.jsonl, so `run` reuses them.
"""
import json
import urllib.request
from collections import Counter
from pathlib import Path

from openjev.data import examples, read_rows
from openjev.spec import load_task

# ex/s measured over the seven labeled runs (see runs/*/label.json); a call is one example unless
# K > max_options_per_call, where the chunked shortlist costs ceil(K/m) + 1 calls.
RATES = [(300, 12.0), (1000, 5.0), (10 ** 9, 3.5)]


def pct(v, q):
    return sorted(v)[min(int(q * len(v)), len(v) - 1)]


def dist(counts, labels, n):
    return "  ".join(f"{lab} {100 * counts.get(i, 0) / max(n, 1):.1f}%" for i, lab in enumerate(labels))


def teacher_ping():
    from openjev.teacher import URL
    try:  # localhost: never through the box's SOCKS proxy
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        body = json.load(opener.open(f"{URL}/models", timeout=10))
        return URL, body["data"][0]["id"], None
    except Exception as e:
        return URL, None, f"{type(e).__name__}: {e}"


def token_lengths(task, texts):
    """Token counts under the student's tokenizer, or None if it is not downloadable."""
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(task.student.model)
        return [len(x) for x in tok(texts, add_special_tokens=True)["input_ids"]]
    except Exception as e:
        print(f"  (student tokenizer {task.student.model} unavailable: {type(e).__name__}) ")
        return None


def mechanism(task, rows):
    """Does this teacher even want to answer with a letter? (idea: r-ms/mini-jev `smoke_letters`)

    Our labeling call is grammar-constrained, so vLLM renormalises the letter logprobs to sum to
    1.0 on every row — a teacher that would have answered "\\n" or "The" looks perfectly confident
    in teacher.jsonl. Only an *unconstrained* call can see it, so this makes its own.
    """
    import math

    from openjev import teacher as T
    try:
        letters, tops = T.probe_open(task, [r["text"] for r in rows])
    except Exception as e:
        print(f"  mechanism probe unavailable ({type(e).__name__}: {e})"[:200])
        return
    emit = [t and t[0]["token"].strip() in letters for t in tops]
    mass = [sum(math.exp(x["logprob"]) for x in t if x["token"].strip() in letters) for t in tops]
    print(f"  letter emission {sum(emit)}/{len(emit)}, candidate mass median "
          f"{pct(mass, .5):.3f} min {min(mass):.3f}  (unconstrained call)")
    for t in tops[:3]:
        print("    top-5 raw: " + "  ".join(f"{x['token']!r} {math.exp(x['logprob']):.2f}" for x in t[:5]))
    if sum(emit) / len(emit) < 0.9 or pct(mass, .5) < 0.5:
        print("  WARNING this teacher does not want to answer with a letter (emission "
              f"{sum(emit) / len(emit):.2f}, median mass {pct(mass, .5):.2f}); the constrained "
              "labels will look confident anyway — the letter logprobs are renormalised to 1.0")


def probe(task, run_dir, n, exs):
    from openjev import teacher as T
    task.teacher.concurrency = min(task.teacher.concurrency, 32)  # :8000 is production
    ids = [e["id"] for e in exs if e["split"] == "calib"][:n]
    T.run(task, run_dir, limit=n, split="calib")
    rows = [r for r in read_rows(run_dir, "calib") if r["id"] in set(ids)]
    if not rows:
        raise SystemExit("probe: no rows labeled")
    pred = [max(range(task.k), key=lambda i: r["probs"][i]) for r in rows]
    maxp = sum(max(r["probs"]) for r in rows) / len(rows)
    stats = json.loads((run_dir / "label.json").read_text()) if (run_dir / "label.json").exists() else {}
    live = T.prompt_sha(task)
    print(f"\nprobe: {len(rows)} calib rows, teacher {stats.get('model') or 'not recorded in label.json'}")
    print(f"  prompt_sha {live} (label.json: {stats.get('prompt_sha', 'not recorded')})")
    mechanism(task, rows)
    from openjev.evaluate import shortlist_stats
    sl = shortlist_stats(rows)
    if sl:
        print(f"  shortlist (K={task.k} > {task.teacher.max_options_per_call}): gold outside the final "
              f"shortlist on {100 * sl['shortlist_miss']:.1f}% of rows — that is the teacher's ceiling; "
              f"median p(none) {sl['p_none_chunk_with_gold']:.2f} in the chunk that holds gold, "
              f"{sl['p_none_chunk_without_gold']:.2f} in the others")
    gold = [r["gold"] for r in rows if r.get("gold") is not None]
    if gold:
        acc = sum(p == r["gold"] for p, r in zip(pred, rows) if r.get("gold") is not None) / len(gold)
        print(f"  teacher accuracy vs gold: {acc:.3f} on {len(gold)} rows")
    pc, gc = Counter(pred), Counter(gold)
    ref = {i: gc[i] / len(gold) for i in range(task.k)} if gold else declared_prior(task)
    print(f"  predicted marginal: {dist(pc, task.labels, len(rows))}")
    if ref:
        src = "gold" if gold else "declared prior"
        print(f"  {src} marginal:      " + "  ".join(f"{lab} {100 * ref[i]:.1f}%" for i, lab in enumerate(task.labels)))
    print(f"  mean max-p: {maxp:.3f}")
    for i, lab in enumerate(task.labels):
        gap = 100 * (ref.get(i, 0) - pc[i] / len(rows)) if ref else 0
        if gap > 10:
            print(f"  WARNING the teacher under-predicts {lab!r} ({100 * pc[i] / len(rows):.0f}% vs "
                  f"{100 * ref[i]:.0f}%): reword that option, or declare `prior:` and let calibration fix it "
                  f"(only if you actually know the deployment prior)")


def declared_prior(task):
    """Top-level `prior:` if the spec has one (added in M1); else nothing."""
    p = getattr(task, "prior", None)
    if p == "uniform":
        return {i: 1 / task.k for i in range(task.k)}
    if isinstance(p, dict):
        return {i: float(p[lab]) for i, lab in enumerate(task.labels) if lab in p}
    return {}


def report(task_path, n_probe=0, runs="runs"):
    task = load_task(task_path)
    print(f"task {task.name}  ({task.type}, K={task.k}, lang={task.lang})  {task.path}")
    print(f"  labels: {task.labels}")
    url, model, err = teacher_ping()
    print(f"teacher {url}: {model if model else 'UNREACHABLE — ' + err}")
    print(f"data source: {task.data.source}")
    exs = examples(task)

    print("\nsplits (gold distribution):")
    for role in ("train", "calib", "eval"):
        rows = [e for e in exs if e["split"] == role]
        g = Counter(e["gold"] for e in rows if e["gold"] is not None)
        print(f"  {role:<5} {len(rows):>5} rows   " +
              (dist(g, task.labels, sum(g.values())) if g else "no gold column"))

    from openjev.teacher import system_prompt
    prompt, letters = system_prompt(task, range(min(task.k, task.teacher.max_options_per_call)))
    print(f"\nsystem prompt (verbatim; {len(letters)} letters"
          f"{', first chunk of %d' % -(-task.k // task.teacher.max_options_per_call) if task.k > len(letters) else ''}):")
    print("-" * 60 + "\n" + prompt + "\n" + "-" * 60)

    chars = [len(e["text"]) for e in exs]
    sample = exs[:: max(1, len(exs) // 1000)]
    toks = token_lengths(task, [e["text"] for e in sample])
    print("\n3 examples exactly as the teacher will see them (after data.text + max_chars):")
    for e in exs[:3]:
        gold = task.labels[e["gold"]] if e["gold"] is not None else "–"
        cut = " TRUNCATED by max_chars" if len(e["text"]) >= task.data.max_chars else ""
        print(f"  [{e['id']} {e['split']} gold={gold} {len(e['text'])} chars{cut}]")
        print("  | " + e["text"][:400].replace("\n", "\n  | ") + ("…" if len(e["text"]) > 400 else ""))

    bind = sum(c >= task.data.max_chars for c in chars) / len(chars)
    print(f"\ntext length: chars p50 {pct(chars, .5)}  p95 {pct(chars, .95)}  max {max(chars)}  "
          f"(max_chars={task.data.max_chars} binds on {100 * bind:.0f}% of rows)")
    if toks:
        over = sum(t > task.student.max_len for t in toks) / len(toks)
        print(f"             tokens ({task.student.model}) p50 {pct(toks, .5)}  p95 {pct(toks, .95)}  "
              f"max {max(toks)}  (student max_len={task.student.max_len}, exceeded by {100 * over:.0f}% of rows)")
        if over > 0.25:
            print(f"  WARNING {100 * over:.0f}% of rows are longer than student.max_len={task.student.max_len} "
                  f"tokens: the teacher reads {task.data.max_chars} chars, the student reads the first "
                  f"{task.student.max_len} tokens. Raise student.max_len (512 costs ~1.6x GPU memory and "
                  f"time) or lower data.max_chars so both see the same text.")
    if bind > 0.25:
        print(f"  WARNING max_chars={task.data.max_chars} truncates {100 * bind:.0f}% of rows — the teacher "
              f"never sees the rest. Raise it if the decision can live at the end of the text.")

    m = task.teacher.max_options_per_call
    calls = 1 if task.k <= m else -(-task.k // m) + 1
    rate = next(r for lim, r in RATES if pct(chars, .5) <= lim) / calls
    secs = len(exs) / rate
    print(f"\nteacher cost: {len(exs)} examples x {calls} call{'s' if calls > 1 else ''} "
          f"≈ {rate:.1f} ex/s ≈ {f'{secs:.0f} s' if secs < 120 else f'{secs / 60:.0f} min'} "
          f"(±2x; measured rate table, concurrency {task.teacher.concurrency})")
    have = len(read_rows(Path(runs) / task.name, "train")) + len(read_rows(Path(runs) / task.name, "calib")) \
        + len(read_rows(Path(runs) / task.name, "eval"))
    if have:
        print(f"              {have} of them are already in {runs}/{task.name}/teacher.jsonl and will be reused")
    if n_probe:
        probe(task, Path(runs) / task.name, n_probe, exs)
    else:
        print(f"\nnext: openjev check {task.path} --probe 100   # what the teacher actually answers, "
              f"≈{100 / rate:.0f} s of teacher time")


def main(task_path, n_probe=0, runs="runs"):
    """Newcomer entry point: every spec/data error comes out as one line, exit 1, no traceback."""
    try:
        report(task_path, n_probe, runs)
    except FileNotFoundError as e:
        raise SystemExit(f"error: {e}\nhint: data.source paths are relative to the current directory "
                         f"— run openjev from the repo root, or use an absolute path")
    except (ValueError, KeyError) as e:
        raise SystemExit(f"error: {e}")
