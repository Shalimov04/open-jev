"""Dump every finished run's teacher labels as (question, option, text) pairs.

One row per (example, option). Lossless: no rendering, no filtering, no balancing --
a training script decides all of that later.

  python3 scripts/build_pairs.py            # -> data/pairs.jsonl + data/pairs-stats.json
  python3 scripts/build_pairs.py --force    # rebuild from scratch
  python3 scripts/build_pairs.py --selfcheck

Duplicate run dirs (seed/variant copies that symlink the same teacher.jsonl) are collapsed
by realpath, so each unique labeling appears once.
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from openjev.data import read_jsonl  # noqa: E402
from openjev.spec import load_task  # noqa: E402
from openjev.teacher import option_lines  # noqa: E402

# Unique teacher.jsonl files we deliberately leave out, and why.
SKIP = {
    "headlines": "subset of headlines-8k (same seed/source; 4k vs 8k train)",
    "agnews-nogold": "same texts+ids as agnews, gold stripped -- agnews carries the gold",
    "headlines-rev": "reversed-option-order variant, no task yaml to render it faithfully",
    "kinopoisk-rev": "reversed-option-order variant, no task yaml to render it faithfully",
    "example": "template task, not a real run",
}


def run_dirs(runs: Path):
    """{task_name: dir} for unique teacher.jsonl files, plus {kept: [collapsed dirs]}."""
    groups = {}
    for d in sorted(runs.iterdir()):
        f = d / "teacher.jsonl"
        if d.is_dir() and f.exists():
            groups.setdefault(f.resolve(), []).append(d)
    keep, dupes = {}, {}
    for real, dirs in groups.items():
        owner = next((d for d in dirs if d.name == real.parent.name), dirs[0])
        keep[owner.name] = owner
        dupes[owner.name] = [d.name for d in dirs if d != owner]
    return keep, dupes


def pairs(task, rows):
    """Yield one pair row per (example, option)."""
    opts = option_lines(task) if task.type == "score" else list(task.labels)
    question = task.statement if task.type == "noul" else task.question
    for r in rows:
        if len(r["probs"]) != task.k:
            raise ValueError(f"{task.name}/{r['id']}: {len(r['probs'])} probs, task has k={task.k}")
        # Shortlisted tasks (k > max_options_per_call): only the finalists were scored head-to-head;
        # everything else has p == 0.0 because it was *never scored*, not because it scored zero.
        # `p_chunk` keeps the cheap first-stage score so a trainer can tell the two apart.
        chunks = r["raw"].get("chunks") if isinstance(r["raw"], dict) else None
        chunk_p = {int(i): p for c in (chunks or []) for i, p in c.items() if i != "none"}
        final = {int(i) for i in r["raw"].get("final", {})} if chunks is not None else set()
        for i, p in enumerate(r["probs"]):
            row = {"task": task.name, "source_id": r["id"], "pair_id": f"{task.name}|{r['id']}|{i}",
                   "primitive": task.type, "k": task.k, "question": question,
                   "option": opts[i], "option_index": i, "text": r["text"], "p": p,
                   "gold": None if r["gold"] is None else int(r["gold"] == i),
                   "split": r["split"]}
            if chunks is not None:
                row["shortlisted"] = i in final  # not shortlisted => p == 0 means "never scored"
                row["p_chunk"] = chunk_p.get(i)
            if "gold_prob" in r:
                row["gold_prob"] = r["gold_prob"]
            yield row


def build(runs: Path, tasks: Path, out: Path, stats_path: Path, force=False, quiet=False):
    keep, dupes = run_dirs(runs)
    say = (lambda *a: None) if quiet else print
    for name in sorted(keep):
        if dupes[name]:
            say(f"collapsed {name}: {', '.join(dupes[name])}")
    for name in sorted(set(keep) & set(SKIP)):
        say(f"skipped   {name}: {SKIP[name]}")
        del keep[name]

    stats = {} if force or not stats_path.exists() else json.loads(stats_path.read_text())
    if force or not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("")
        stats = {}
    todo = [n for n in sorted(keep) if n not in stats]
    say(f"{len(keep)} tasks, {len(stats)} already in {out}, {len(todo)} to build")

    with open(out, "a") as f:
        for name in todo:
            task = load_task(tasks / f"{name}.yaml")
            rows = read_jsonl(keep[name] / "teacher.jsonl")
            n = hi = 0
            for pr in pairs(task, rows):
                f.write(json.dumps(pr, ensure_ascii=False) + "\n")
                n += 1
                hi += pr["p"] > 0.5
            f.flush()
            stats[name] = {"examples": len(rows), "k": task.k, "primitive": task.type,
                           "pairs": n, "p_gt_half": round(hi / n, 4),
                           "splits": dict(Counter(r["split"] for r in rows)),
                           "shortlisted": any("chunks" in r["raw"] for r in rows
                                              if isinstance(r["raw"], dict)),
                           "run_dir": str(keep[name]), "collapsed": dupes[name]}
            stats_path.write_text(json.dumps(stats, indent=1, sort_keys=True))
            say(f"  {name}: {n} pairs")
    return stats


def table(stats):
    w = max(len(n) for n in stats)
    out = [f"{'task':{w}}  {'prim':6} {'ex':>6} {'K':>3} {'pairs':>8} {'p>.5':>6}  splits"]
    for n, s in sorted(stats.items()):
        sp = " ".join(f"{k}={v}" for k, v in s["splits"].items())
        out.append(f"{n:{w}}  {s['primitive']:6} {s['examples']:6} {s['k']:3} {s['pairs']:8} "
                   f"{s['p_gt_half']:6.3f}  {sp}{'  [shortlist]' if s['shortlisted'] else ''}")
    out.append(f"{'TOTAL':{w}}  {'':6} {sum(s['examples'] for s in stats.values()):6} {'':3} "
               f"{sum(s['pairs'] for s in stats.values()):8}")
    return "\n".join(out)


def selfcheck(tmp: Path):
    """Synthetic run dir -> expected pairs, including a symlinked duplicate and a chunked row."""
    (tmp / "tasks").mkdir(parents=True)
    (tmp / "runs/t1").mkdir(parents=True)
    (tmp / "runs/t1-s1").mkdir(parents=True)
    (tmp / "tasks/t1.yaml").write_text(
        "name: t1\ntype: choice\nquestion: Q?\noptions: [a, b, c]\n"
        "data: {source: {jsonl: x.jsonl}, gold: label}\n")
    rows = [{"id": "train:0", "split": "train", "text": "hello", "gold": 1,
             "probs": [0.2, 0.7, 0.1], "raw": {"A": -1.0}, "source": "data"},
            {"id": "eval:1", "split": "eval", "text": "bye", "gold": None,
             "probs": [0.0, 0.6, 0.4],
             "raw": {"chunks": [{"0": 0.3, "1": 0.5, "none": 0.2}],
                     "final": {"1": -0.5, "2": -1.0}}, "source": "data"}]
    (tmp / "runs/t1/teacher.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (tmp / "runs/t1-s1/teacher.jsonl").symlink_to(tmp / "runs/t1/teacher.jsonl")

    stats = build(tmp / "runs", tmp / "tasks", tmp / "pairs.jsonl", tmp / "stats.json", quiet=True)
    got = [json.loads(l) for l in open(tmp / "pairs.jsonl")]
    assert list(stats) == ["t1"], stats                     # the symlinked copy collapsed
    assert stats["t1"] == {"examples": 2, "k": 3, "primitive": "choice", "pairs": 6,
                           "p_gt_half": 0.3333, "splits": {"train": 1, "eval": 1},
                           "shortlisted": True, "run_dir": str(tmp / "runs/t1"),
                           "collapsed": ["t1-s1"]}, stats
    assert len(got) == 6
    assert got[0] == {"task": "t1", "source_id": "train:0", "pair_id": "t1|train:0|0",
                      "primitive": "choice", "k": 3, "question": "Q?", "option": "a",
                      "option_index": 0, "text": "hello", "p": 0.2, "gold": 0, "split": "train"}
    assert got[1]["gold"] == 1 and got[2]["option"] == "c"
    assert "shortlisted" not in got[0]                      # unchunked row: no flag
    assert [(g["option_index"], g["shortlisted"], g["p_chunk"], g["gold"]) for g in got[3:]] == \
           [(0, False, 0.3, None), (1, True, 0.5, None), (2, True, None, None)]
    # resumable: a second build appends nothing
    build(tmp / "runs", tmp / "tasks", tmp / "pairs.jsonl", tmp / "stats.json", quiet=True)
    assert len(open(tmp / "pairs.jsonl").readlines()) == 6
    print("selfcheck ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="rebuild instead of resuming")
    ap.add_argument("--selfcheck", action="store_true")
    a = ap.parse_args()
    if a.selfcheck:
        import tempfile
        with tempfile.TemporaryDirectory() as t:
            selfcheck(Path(t))
    else:
        st = build(ROOT / "runs", ROOT / "tasks", ROOT / "data/pairs.jsonl",
                   ROOT / "data/pairs-stats.json", force=a.force)
        print(table(st))
