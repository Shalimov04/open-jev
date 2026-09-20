#!/usr/bin/env python3
"""AgentRewardBench -> data/arb.jsonl (fields: source_id, text, success, optimality, benchmark).

Only the text-only judge inputs are downloaded (judgments/*/*/gpt-4o-mini-noscreen-noaxtree/*.json,
~20 MB). `cleaned/` (7-24 GB of screenshots) is never touched. Idempotent: the hub cache and the
output file are both reused.
"""
import csv, json, os, random, re, sys
from collections import defaultdict
from urllib.parse import urlparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "arb.jsonl")
REPO = "McGill-NLP/agent-reward-bench"
JUDGE = "gpt-4o-mini-noscreen-noaxtree"

STEP_RE = re.compile(
    r"Step:\s*(\d+)\s*\nURL:\s*(.*?)\s*\nAction:\s*(.*?)\s*\nReasoning:", re.S)
MSG_RE = re.compile(r"send_msg_to_user\((.*)\)\s*$", re.S)


def render(goal, actions_text, err_msg):
    """actions_text is the judge's 'The agent performed the following actions:' block."""
    steps = [(int(n), u, a) for n, u, a in STEP_RE.findall(actions_text)
             if a.strip() and a.strip() != "None"]
    final = "(none)"
    for _, _, a in steps:
        m = MSG_RE.search(a.strip())
        if m:
            final = m.group(1).strip().strip("'\"")[:400]
    lines = [f"goal: {' '.join(goal.split())}", f"steps: {len(steps)}"]
    host = None
    for n, url, a in steps[-12:]:
        h = urlparse(url.strip()).netloc
        a = " ".join(a.split())[:120]
        lines.append(f"{n}. [{h}] {a}" if h and h != host else f"{n}. {a}")
        host = h or host
    lines.append(f"final: {final}")
    if err_msg:
        lines.append(f"error: {' '.join(str(err_msg).split())[:200]}")
    return "\n".join(lines)


def main():
    from huggingface_hub import hf_hub_download, snapshot_download
    ann = csv.DictReader(open(hf_hub_download(REPO, "data/annotations.csv", repo_type="dataset")))
    votes = defaultdict(list)
    for r in ann:
        votes[(r["benchmark"], r["model_name"], r["task_id"])].append(r)

    d = snapshot_download(REPO, repo_type="dataset",
                          allow_patterns=[f"judgments/*/*/{JUDGE}/*.json"])
    rows, skipped = [], defaultdict(int)
    for key, vs in sorted(votes.items()):
        bench, model, tid = key
        succ = [v["trajectory_success"] for v in vs if v["trajectory_success"] != "Unsure"]
        if not succ:
            skipped["unsure"] += 1
            continue
        pos = sum(s == "Successful" for s in succ)
        if pos * 2 == len(succ):
            skipped["tie"] += 1
            continue
        opt = [int(v["trajectory_optimality"][0]) for v in vs if v["trajectory_optimality"][:1].isdigit()]
        path = os.path.join(d, "judgments", bench, model, JUDGE, f"{tid}.json")
        if not os.path.exists(path):
            skipped["no-judgment"] += 1
            continue
        j = json.load(open(path))
        parts = [p["text"] for p in j["chat_messages"]["regular"][1]["content"]]
        acts = next(p for p in parts if p.startswith("The agent performed"))
        err = j["trajectory_info"]["summary_info"].get("err_msg")
        rows.append({
            "source_id": f"{bench}/{model}/{tid}",
            "text": render(j["goal"], acts, err),
            "success": int(pos * 2 > len(succ)),
            "optimality": round(sum(opt) / len(opt)) if opt else 2,
            "benchmark": bench,
        })

    random.Random(0).shuffle(rows)
    with open(OUT, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    n = len(rows)
    lens = sorted(len(r["text"]) for r in rows)
    print(f"wrote {OUT}: {n} rows  (skipped {dict(skipped)})")
    print(f"  success rate {sum(r['success'] for r in rows)/n:.3f}"
          f"  optimality {sorted(set(r['optimality'] for r in rows))}")
    print(f"  chars: median {lens[n//2]}  p95 {lens[int(n*.95)]}  max {lens[-1]}")
    print(f"  YAML slices: one split 'train' ({n} rows); train n=750 / calib n=200 / eval n=300"
          f"  (750+200+300 = 1250 <= {n})")


def _selfcheck():
    txt = ("The agent performed the following actions:\n-----\n"
           "Step: 1\nURL: https://shop.example.com/\nAction: click('12')\nReasoning: go\n-----\n"
           "Step: 2\nURL: https://shop.example.com/cart\nAction: send_msg_to_user('Done: 3 items')\nReasoning: x\n-----\n"
           "Step: 3\nURL: https://shop.example.com/cart\nAction: None\nReasoning: None\n-----\n")
    got = render("Buy  milk\n", txt, "timeout")
    want = ("goal: Buy milk\nsteps: 2\n1. [shop.example.com] click('12')\n"
            "2. send_msg_to_user('Done: 3 items')\nfinal: Done: 3 items\nerror: timeout")
    assert got == want, f"\nGOT:\n{got}\nWANT:\n{want}"
    print("selfcheck ok")


if __name__ == "__main__":
    _selfcheck()
    if "--selfcheck" not in sys.argv:
        main()
