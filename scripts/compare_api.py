#!/usr/bin/env python
"""Score an external /v1/systemone endpoint on the same eval rows our student was scored on.

    python scripts/compare_api.py --adapter openjev --url http://localhost:8099 --task agnews --limit 200
    JEV_API_KEY=... python scripts/compare_api.py --adapter jev --task agnews --limit 200

Reads runs/<task>/eval_rows.jsonl (ids + gold + our student's calibrated probs) and the texts from
runs/<task>/teacher.jsonl, sends each text to the endpoint, and writes
results/api/api-<name>-<task>.json with accuracy vs gold, argmax agreement with our student, and p50
latency. Latency is *not* comparable across providers (different hardware) — it is here so a slow
endpoint is visible, not so it can be quoted.
"""
import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from openjev.data import read_rows  # noqa: E402

for _v in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
    os.environ.pop(_v, None)


def openjev_adapter(client, url, model, meta, text):
    """Our own server. `model` is the run-dir name; probabilities come back keyed by label."""
    r = client.post(f"{url}/v1/systemone", json={"model": model, "input": text})
    r.raise_for_status()
    return _probs_from_labels(r.json(), meta)


def jev_adapter(client, url, model, meta, text):
    """TypeSafe Jev (https://docs.typesafe.ai/api.md). The request carries the *question* — Jev is
    zero-shot, so the task YAML becomes the schema instead of training data.

    POST https://api.typesafe.ai/v1/systemone, Bearer JEV_API_KEY:
      {"state": <text>, "model": "jev-latest", "questions": {"q": {"type": ..., "instructions": ...,
                                                                   "criteria": ...}}}
    -> {"answers": {"q": {"type": "choice", "choice": ..., "probabilities": {...}, "confidence": ...}}}
       score answers carry "score" + "probabilities" keyed by level number; noul answers carry only
       "noul" (a probability, no confidence). Unverified against the live API: no key was available
       when this was written, so if a field name is off, it is here and nowhere else.
    """
    task, labels = meta["task"], meta["labels"]
    if task["type"] == "choice":
        q = {"type": "choice", "instructions": task["question"],
             "criteria": {l: l for l in labels}}
    elif task["type"] == "score":
        q = {"type": "score", "instructions": task["question"],
             "criteria": list(task["rubric"]["descriptions"])}
    else:
        q = {"type": "noul", "instructions": task["statement"]}
    r = client.post(f"{url}/v1/systemone", json={"state": text, "model": model, "questions": {"q": q}},
                    headers={"Authorization": f"Bearer {os.environ['JEV_API_KEY']}"})
    r.raise_for_status()
    a = r.json()["answers"]["q"]
    if a["type"] == "noul":
        p = float(a["noul"])
        return [p, 1 - p] if labels[0] == "true" else [1 - p, p]
    probs = a["probabilities"]
    if a["type"] == "score":  # keyed by level number (0..K-1), not by our level string
        return [float(probs.get(str(i), probs.get(i, 0.0))) for i in range(len(labels))]
    return [float(probs.get(l, 0.0)) for l in labels]


def _probs_from_labels(body, meta):
    labels = meta["labels"]
    if "probabilities" in body:
        return [float(body["probabilities"].get(l, 0.0)) for l in labels]
    p = float(body["probability"])  # noul view: p(true)
    return [p, 1 - p] if labels[0] == "true" else [1 - p, p]


ADAPTERS = {"openjev": openjev_adapter, "jev": jev_adapter}
DEFAULT_URL = {"openjev": "http://localhost:8099", "jev": "https://api.typesafe.ai"}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--adapter", choices=sorted(ADAPTERS), required=True)
    p.add_argument("--task", required=True, help="run dir name under runs/")
    p.add_argument("--url", help=f"endpoint base; default per adapter {DEFAULT_URL}")
    p.add_argument("--model", help="model id sent in the request; default: the run name (openjev) "
                                   "or jev-latest (jev)")
    p.add_argument("--limit", type=int, default=200, help="eval rows to send (0 = all)")
    p.add_argument("--name", help="results file suffix; default: the adapter name")
    p.add_argument("--runs", default="runs")
    a = p.parse_args(argv)

    run_dir = Path(a.runs) / a.task
    meta = json.loads((run_dir / "openjev.json").read_text())
    labels = meta["labels"]
    texts = {r["id"]: r["text"] for r in read_rows(run_dir, "eval")}
    rows = [json.loads(l) for l in open(run_dir / "eval_rows.jsonl") if l.strip()]
    rows = [r for r in rows if r["id"] in texts][: a.limit or None]
    url = (a.url or DEFAULT_URL[a.adapter]).rstrip("/")
    model = a.model or (a.task if a.adapter == "openjev" else "jev-latest")
    adapter = ADAPTERS[a.adapter]

    hit = agree = n_gold = 0
    lat, failed = [], 0
    with httpx.Client(timeout=60.0) as client:
        for i, r in enumerate(rows):
            t0 = time.perf_counter()
            try:
                probs = adapter(client, url, model, meta, texts[r["id"]])
            except Exception as e:  # one endpoint failure must not throw away 199 measured rows
                failed += 1
                if failed <= 3:
                    print(f"  row {r['id']}: {type(e).__name__}: {e}", file=sys.stderr)
                continue
            lat.append((time.perf_counter() - t0) * 1000)
            pred = max(range(len(labels)), key=probs.__getitem__)
            if r.get("gold") is not None:
                n_gold += 1
                hit += pred == r["gold"]
            agree += pred == max(range(len(labels)), key=r["student_probs"].__getitem__)
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(rows)}", flush=True)
    if not lat:
        raise SystemExit(f"every request failed ({failed}/{len(rows)}) — is {url} up?")
    out = {"adapter": a.adapter, "url": url, "model": model, "task": a.task,
           "n": len(lat), "n_failed": failed,
           "acc": hit / n_gold if n_gold else None, "n_gold": n_gold,
           "agreement_with_student": agree / len(lat),
           "latency_ms_p50": statistics.median(lat),
           "student_acc": meta["metrics"]["student"]["acc"]}
    # results/api/ and not results/: `openjev report` globs results/*.json and builds a table row
    # out of every file it finds, and these are not run results.
    path = ROOT / "results" / "api" / f"api-{a.name or a.adapter}-{a.task}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    acc = f"{out['acc']:.3f}" if out["acc"] is not None else "–"
    print(f"\n| endpoint | task | n | acc | agree w/ student | p50 ms |\n|---|---|---|---|---|---|\n"
          f"| {a.adapter} ({model}) | {a.task} | {out['n']} | {acc} | "
          f"{out['agreement_with_student']:.3f} | {out['latency_ms_p50']:.1f} |\n-> {path}")
    return out


if __name__ == "__main__":
    main()
