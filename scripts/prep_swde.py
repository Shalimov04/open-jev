#!/usr/bin/env python3
"""SWDE -> data/swde-field.jsonl (choice over 32 `<vertical>.<field>` options + `none`).

Run it with a 7z reader available (no 7z binary on this box):
    VIRTUAL_ENV=.venv uv run --with py7zr scripts/prep_swde.py
Downloads the parquet ground truth plus files/*.7z (205 MB), extracts once into cache/swde/
(~8 GB on disk; gitignored, and a re-run reuses it).
Idempotent: extraction and the hub cache are both reused. Eval = 2 held-out sites per vertical,
written after the train-site rows, so the YAML slice is an unseen-site protocol.
"""
import json, os, random, re, sys
from ast import literal_eval
from html.parser import HTMLParser

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "cache", "swde")
OUT = os.path.join(ROOT, "data", "swde-field.jsonl")
REPO = "abdo-Mansour/SWDE"
VERTICALS = ["auto", "book", "camera", "job", "movie", "nbaplayer", "restaurant", "university"]
SKIP = {"script", "style", "noscript", "head"}
TRAIN_PAGES, EVAL_PAGES, EVAL_SITES = 13, 26, 2


class TextNodes(HTMLParser):
    """Ordered list of text nodes: (text, ancestor chain) — plus the page <title>."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.stack, self.nodes, self.title = [], [], ""

    def handle_starttag(self, tag, attrs):
        self.stack.append((tag, dict(attrs)))

    handle_startendtag = lambda self, tag, attrs: None

    def handle_endtag(self, tag):
        for i in range(len(self.stack) - 1, -1, -1):
            if self.stack[i][0] == tag:
                del self.stack[i:]
                return

    def handle_data(self, data):
        t = " ".join(data.split())
        if not t or any(s in SKIP for s, _ in self.stack):
            if t and self.stack and self.stack[-1][0] == "title":
                self.title = t
            return
        self.nodes.append((t, list(self.stack)))


def path_str(stack, depth=6):
    out = []
    for tag, a in stack[-depth:]:
        s = tag
        if a.get("id", "").split():
            s += "#" + a["id"].split()[0][:20]
        elif a.get("class", "").split():
            s += "." + a["class"].split()[0][:20]
        out.append(s)
    return " > ".join(out)


def render(title, nodes, i):
    before = nodes[i - 1][0][:60] if i else ""
    after = nodes[i + 1][0][:60] if i + 1 < len(nodes) else ""
    return "\n".join([f"title: {title[:80]}", f"path: {path_str(nodes[i][1])}",
                      f'before: "{before}"', f'node: "{nodes[i][0][:120]}"', f'after: "{after}"'])


def parse(html):
    p = TextNodes()
    p.feed(html)
    p.close()
    return p.title, p.nodes


def match(text, value):
    """Node text carries this gt value: equal, or contains it and is at most twice as long."""
    return text == value or (value in text and len(text) <= 2 * len(value))


def page_rows(html, gt, vertical, rng):
    title, nodes = parse(html)
    if not nodes:
        return []
    hits, out = {}, []
    for i, (t, _) in enumerate(nodes):
        for field, vals in gt.items():
            for v in vals:
                if v != "<NULL>" and v and match(t, v):
                    hits.setdefault(i, f"{vertical}.{field}")
    for i, label in hits.items():
        out.append((render(title, nodes, i), label))
    rest = [i for i in range(len(nodes)) if i not in hits]
    for i in rng.sample(rest, min(2, len(rest))):
        out.append((render(title, nodes, i), "none"))
    return out


def main():
    import py7zr
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    os.makedirs(CACHE, exist_ok=True)
    rows, opts, counts = [], [], {}
    for vertical in VERTICALS:
        pqf = hf_hub_download(REPO, f"data/{vertical}-00000-of-00001.parquet", repo_type="dataset")
        recs = pq.read_table(pqf).to_pylist()
        opts += [f"{vertical}.{f}" for f in json.loads(recs[0]["schema"])]
        vdir = os.path.join(CACHE, vertical)
        if not os.path.isdir(vdir):
            a7 = hf_hub_download(REPO, f"files/{vertical}.7z", repo_type="dataset")
            py7zr.SevenZipFile(a7).extractall(CACHE + ".tmp")
            os.replace(os.path.join(CACHE + ".tmp", vertical), vdir)
        dirs = {d.split("-", 1)[1].split("(")[0]: d for d in os.listdir(vdir)}
        sites = sorted(dirs)
        rng = random.Random(0)
        for si, site in enumerate(sites):
            is_eval = si >= len(sites) - EVAL_SITES
            pages = sorted((r for r in recs if r["website_name"] == site),
                           key=lambda r: r["website_id"])
            pages = rng.sample(pages, min(EVAL_PAGES if is_eval else TRAIN_PAGES, len(pages)))
            for r in pages:
                f = os.path.join(vdir, dirs[site], r["website_id"].split("_")[1] + ".htm")
                if not os.path.exists(f):
                    continue
                html = open(f, "rb").read().decode("utf-8", "replace")
                for text, label in page_rows(html, literal_eval(r["gt"]), vertical, rng):
                    rows.append({"source_id": f'{vertical}/{r["website_id"]}',
                                 "split": "eval" if is_eval else "train",
                                 "site": site, "text": text, "field": label})
        counts[vertical] = sum(r["source_id"].startswith(vertical + "/") for r in rows)

    rows.sort(key=lambda r: r["split"] == "eval")   # train-site rows first (stable)
    with open(OUT, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    n_tr = sum(r["split"] == "train" for r in rows)
    lens = sorted(len(r["text"]) for r in rows)
    none = sum(r["field"] == "none" for r in rows) / len(rows)
    print(f"wrote {OUT}: {len(rows)} rows ({n_tr} train-site, {len(rows)-n_tr} held-out-site)")
    print(f"  {len(opts)+1} options; 'none' share {none:.3f}; per-vertical rows {counts}")
    print(f"  chars: median {lens[len(lens)//2]} p95 {lens[int(len(lens)*.95)]} max {lens[-1]}")
    print(f"  YAML slices: train/calib split 'train[:{n_tr}]', eval split 'train[{n_tr}:]'")
    print("  options:", opts + ["none"])


HTML = ("<html><head><title>Kobe Bryant Stats</title></head><body>"
        "<div id='bio' class='card'><table><tr><td>Height:</td>"
        "<td class='v'>6-6</td></tr><tr><td>Team:</td><td>Los Angeles Lakers</td></tr>"
        "</table></div><script>var x='6-6';</script></body></html>")


def _selfcheck():
    title, nodes = parse(HTML)
    assert title == "Kobe Bryant Stats"
    assert [n[0] for n in nodes] == ["Height:", "6-6", "Team:", "Los Angeles Lakers"], nodes
    got = render(title, nodes, 1)
    want = ('title: Kobe Bryant Stats\n'
            'path: html > body > div#bio > table > tr > td.v\n'
            'before: "Height:"\nnode: "6-6"\nafter: "Team:"')
    assert got == want, f"\nGOT:\n{got}\nWANT:\n{want}"
    out = dict(page_rows(HTML, {"height": ["6-6"], "team": ["Los Angeles Lakers"],
                                "weight": ["<NULL>"]}, "nbaplayer", random.Random(0)))
    assert out[want] == "nbaplayer.height", out
    print("selfcheck ok")


if __name__ == "__main__":
    _selfcheck()
    if "--selfcheck" not in sys.argv:
        main()
