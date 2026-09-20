#!/usr/bin/env python3
"""Multimodal-Mind2Web -> data/m2w-element.jsonl + data/m2w-target.jsonl.

Screenshots are 94% of the parquet bytes, so the shards are never downloaded whole: a
column-projected read over HfFileSystem pulls only the text columns, cached under cache/m2w/.
Idempotent: a second run reads the cache and re-renders for free.
"""
import concurrent.futures as cf
import json, os, random, re, sys
from html.parser import HTMLParser

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "cache", "m2w")
REPO = "datasets/osunlp/Multimodal-Mind2Web/data"
COLS = ["annotation_id", "action_uid", "confirmed_task", "action_reprs", "target_action_index",
        "operation", "pos_candidates", "neg_candidates", "cleaned_html", "website", "domain"]
ATTRS = ("type", "role", "aria_label", "placeholder", "title", "alt", "name", "value")
ESCAPED_TAG = re.compile(r"</?text[^<>]*>")
VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link",
        "meta", "param", "source", "track", "wbr"}


class Nodes(HTMLParser):
    """Collect, per backend_node_id: tag, attrs, descendant text, ancestor chain, parent text."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.stack, self.out = [], {}

    def _push(self, tag, attrs):
        a = dict(attrs)
        self.stack.append({"tag": tag, "attrs": a, "text": [],
                           "path": [(n["tag"], n["attrs"]) for n in self.stack[-6:]]})

    def _pop(self):
        n = self.stack.pop()
        # cleaned_html escapes some of its own <text backend_node_id=..> wrappers into the text
        txt = " ".join(ESCAPED_TAG.sub("", " ".join(n["text"])).split())
        if self.stack:
            self.stack[-1]["text"].append(txt)
        bid = n["attrs"].get("backend_node_id")
        if bid:
            self.out[bid] = {"tag": n["tag"], "attrs": n["attrs"], "text": txt,
                             "path": n["path"], "node": n}
        return n

    def handle_starttag(self, tag, attrs):
        self._push(tag, attrs)
        if tag in VOID:
            self._pop()

    def handle_startendtag(self, tag, attrs):
        self._push(tag, attrs)
        self._pop()

    def handle_endtag(self, tag):
        for i in range(len(self.stack) - 1, -1, -1):
            if self.stack[i]["tag"] == tag:
                while len(self.stack) > i:
                    self._pop()
                return

    def handle_data(self, data):
        if self.stack and data.strip():
            self.stack[-1]["text"].append(data.strip())

    def close(self):
        super().close()
        while self.stack:
            self._pop()
        for bid, n in self.out.items():
            p = n["path"][-1] if n["path"] else None
            n["parent_text"] = ""
            if p is not None:
                pb = p[1].get("backend_node_id")
                n["parent_text"] = self.out[pb]["text"] if pb in self.out else ""
            del n["node"]
        return self.out


def parse_html(html):
    p = Nodes()
    p.feed(html)
    return p.close()


def elem_line(cand, node):
    """'<tag> <attr=v ...> "text"' — tag, up to two salient attrs, inner text[:40]."""
    a = json.loads(cand["attributes"]) if isinstance(cand["attributes"], str) else cand["attributes"]
    kv = [f'{k}="{" ".join(a[k].split())[:30]}"' for k in ATTRS if a.get(k, "").strip()][:2]
    text = (node or {}).get("text", "")[:40]
    return " ".join([f"<{cand['tag']}>"] + kv + [f'"{text}"'])


def path_line(node, depth):
    out = []
    for tag, a in (node or {}).get("path", [])[-depth:]:
        s = tag
        if a.get("id", "").split():
            s += "#" + a["id"].split()[0][:20]
        elif a.get("class", "").split():
            s += "." + a["class"].split()[0][:20]
        out.append(s)
    return " > ".join(out)


def done_line(reprs, idx):
    prev = [" ".join(r.split()) for r in list(reprs)[:idx]][-2:]
    return " ; ".join(prev) if prev else "(nothing yet)"


def render_element(task, reprs, idx, cands, nodes):
    lines = [f"task: {task}", f"done: {done_line(reprs, idx)}", "candidates:"]
    for i, c in enumerate(cands):
        lines.append(f"{chr(65+i)}) {elem_line(c, nodes.get(c['backend_node_id']))}")
    return "\n".join(lines)


def render_target(task, reprs, idx, cand, nodes):
    n = nodes.get(cand["backend_node_id"])
    return "\n".join([
        f"task: {task}", f"done: {done_line(reprs, idx)}",
        f"element: {elem_line(cand, n)}",
        f"path: {path_line(n, 4)}",
        f'parent text: "{(n or {}).get("parent_text", "")[:100]}"'])


def shard_cache(fs, path):
    import pyarrow.parquet as pq
    out = os.path.join(CACHE, os.path.basename(path))
    if not os.path.exists(out):
        with fs.open(path) as f:
            t = pq.ParquetFile(f).read(columns=COLS)
        pq.write_table(t, out + ".tmp", compression="zstd")
        os.replace(out + ".tmp", out)
    return out


def main():
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem
    os.makedirs(CACHE, exist_ok=True)
    fs = HfFileSystem()
    shards = {s: sorted(fs.glob(f"{REPO}/{s}-*.parquet")) for s in ("train", "test_website")}
    with cf.ThreadPoolExecutor(6) as ex:
        local = {s: list(ex.map(lambda p: shard_cache(fs, p), v)) for s, v in shards.items()}
    print("cache:", sum(os.path.getsize(p) for v in local.values() for p in v) / 1e6, "MB")

    elem, targ = [], []
    stats = {"actions": 0, "no_pos": 0, "few_neg": 0}
    for split in ("train", "test_website"):
        for p in local[split]:
            for r in pq.read_table(p).to_pylist():
                stats["actions"] += 1
                if not r["pos_candidates"]:
                    stats["no_pos"] += 1
                    continue
                neg = [json.loads(x) for x in r["neg_candidates"]]
                if len(neg) < 15:
                    stats["few_neg"] += 1
                    continue
                pos = [json.loads(x) for x in r["pos_candidates"]]
                pos.sort(key=lambda c: not c.get("is_top_level_target"))
                pos = pos[0]
                nodes = parse_html(r["cleaned_html"])
                rng = random.Random(r["action_uid"])
                idx = int(r["target_action_index"])
                task = " ".join(r["confirmed_task"].split())
                sid = f'{r["annotation_id"]}/{r["action_uid"]}'

                c = rng.sample(neg, 15) + [pos]
                rng.shuffle(c)
                elem.append({"source_id": sid, "split": split, "website": r["website"],
                             "text": render_element(task, r["action_reprs"], idx, c, nodes),
                             "gold": c.index(pos)})
                for cand in [pos] + rng.sample(neg, 3):
                    targ.append({"source_id": f'{sid}/{cand["backend_node_id"]}', "split": split,
                                 "website": r["website"],
                                 "text": render_target(task, r["action_reprs"], idx, cand, nodes),
                                 "target": int(cand is pos)})
    for name, rows in (("m2w-element", elem), ("m2w-target", targ)):
        out = os.path.join(ROOT, "data", name + ".jsonl")
        with open(out, "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        n_tr = sum(r["split"] == "train" for r in rows)
        lens = sorted(len(r["text"]) for r in rows)
        print(f"wrote {out}: {len(rows)} rows ({n_tr} train-source, {len(rows)-n_tr} test_website)")
        print(f"  chars: median {lens[len(lens)//2]} p95 {lens[int(len(lens)*.95)]} max {lens[-1]}")
        print(f"  YAML slices: train/calib split 'train[:{n_tr}]', eval split 'train[{n_tr}:]'")
    print("stats:", stats)


HTML = ('<html><body><div id="main" class="wrap"><form class="search">'
        '<label backend_node_id="7"><input backend_node_id="8" type="text" '
        'placeholder="City" aria_label="Where to?" value=""></label>'
        '<button backend_node_id="9" role="button">Search now</button></form></div></body></html>')


def _selfcheck():
    nodes = parse_html(HTML)
    cand = {"tag": "input", "backend_node_id": "8",
            "attributes": json.dumps({"type": "text", "placeholder": "City",
                                      "aria_label": "Where to?", "backend_node_id": "8"})}
    got = render_target("Book a flight", ["[a] Home -> CLICK", "[b] X -> CLICK", "[c] Y -> CLICK"],
                        3, cand, nodes)
    want = ('task: Book a flight\ndone: [b] X -> CLICK ; [c] Y -> CLICK\n'
            'element: <input> type="text" aria_label="Where to?" ""\n'
            'path: body > div#main > form.search > label\n'
            'parent text: ""')
    assert got == want, f"\nGOT:\n{got}\nWANT:\n{want}"
    btn = {"tag": "button", "backend_node_id": "9",
           "attributes": json.dumps({"role": "button", "backend_node_id": "9"})}
    assert elem_line(btn, nodes["9"]) == '<button> role="button" "Search now"', elem_line(btn, nodes["9"])
    print("selfcheck ok")


if __name__ == "__main__":
    _selfcheck()
    if "--selfcheck" not in sys.argv:
        main()
