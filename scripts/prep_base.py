"""Build the four S1 jsonl pools that are not a plain HF dataset (PLAN-4 §2).

  python3 scripts/prep_base.py        # -> data/base/{swde-vertical,tweet-stance,rubq-rerank,miracl-rerank}.jsonl

swde-vertical : 2 text nodes per SWDE page, gold = the page's vertical (8 classes).
tweet-stance  : tweet_eval's 5 stance configs as one task, target written into the text.
*-rerank      : mteb reranking sets flattened to (query, passage, relevant) rows, negatives
                downsampled to 3x the positives so the pool is not 93 % "false".
"""
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data/base"
STANCE = ["abortion", "atheism", "climate", "feminist", "hillary"]


def write(name, rows):
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"{name}.jsonl"
    p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print(f"{name}: {len(rows)} rows -> {p}")


def swde_vertical(per_page=2):
    rows, seen = [], {}
    for line in open(ROOT / "data/swde-field.jsonl"):
        r = json.loads(line)
        page = r["source_id"]
        if seen.get(page, 0) >= per_page:
            continue
        seen[page] = seen.get(page, 0) + 1
        rows.append({"id": f"{page}#{seen[page]}", "text": r["text"],
                     "vertical": page.split("/")[0]})
    random.Random(0).shuffle(rows)
    write("swde-vertical", rows)


def tweet_stance():
    from datasets import load_dataset
    names = ["none", "against", "in favour of"]
    rows = []
    for t in STANCE:
        d = load_dataset("cardiffnlp/tweet_eval", f"stance_{t}")
        for split in ("train", "test", "validation"):
            for i, r in enumerate(d[split]):
                rows.append({"id": f"{t}:{split}:{i}", "target": t,
                             "text": f"Target: {t}\nTweet: {r['text']}",
                             "stance": names[r["label"]]})
    random.Random(0).shuffle(rows)
    write("tweet-stance", rows)


def rerank(name, repo, cfgs, neg_per_pos=3):
    """mteb reranking (qrels + queries + corpus) -> (query, passage, relevant) rows."""
    from datasets import load_dataset
    qrels, queries, corpus = (load_dataset(repo, c) for c in cfgs)
    sp = list(qrels)[0]
    q = {r["_id"]: r["text"] for r in queries[list(queries)[0]]}
    c = {r["_id"]: (r["title"] + " " + r["text"]).strip() for r in corpus[list(corpus)[0]]}
    pos, neg = [], []
    for r in qrels[sp]:
        row = {"id": f"{r['query-id']}|{r['corpus-id']}", "query": q[r["query-id"]],
               "passage": c[r["corpus-id"]][:1200], "relevant": int(r["score"] > 0)}
        (pos if row["relevant"] else neg).append(row)
    rng = random.Random(0)
    rng.shuffle(neg)
    rows = pos + neg[: neg_per_pos * len(pos)]
    rng.shuffle(rows)
    print(f"  {name}: {len(pos)} positive, {len(rows) - len(pos)} negative kept")
    write(name, rows)


if __name__ == "__main__":
    swde_vertical()
    tweet_stance()
    rerank("rubq-rerank", "mteb/RuBQReranking", ["default", "queries", "corpus"])
    rerank("miracl-rerank", "mteb/MIRACLReranking", ["ru-qrels", "ru-queries", "ru-corpus"])
