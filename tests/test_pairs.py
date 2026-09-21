"""PLAN-4 §3: pair rendering and the per-epoch mixture."""
import json
import random
from pathlib import Path

from openjev import pairs
from openjev.spec import load_task

ROOT = Path(__file__).parent.parent


def _task(tmp_path, body):
    p = tmp_path / "t.yaml"
    p.write_text(body + "data: {source: {csv: f}}\n")
    return load_task(p)


def test_render_choice(tmp_path):
    t = _task(tmp_path, "name: t\ntype: choice\nquestion: Good or bad?\noptions: [Good, Neutral, Bad]\n")
    assert pairs.render(t, 2, "txt") == (
        "choice | Q: Good or bad? | option: Bad | options: Good | Neutral | Bad", "txt")


def test_render_score_uses_teacher_option_lines(tmp_path):
    t = _task(tmp_path, "name: t\ntype: score\nquestion: How good?\n"
                        "rubric: {levels: [1, 2, 3], descriptions: [bad, ok]}\n")
    a, _ = pairs.render(t, 1, "txt")
    assert a == "score | Q: How good? | option: 2: ok | options: 1: bad | 2: ok | 3"


def test_render_noul_default_head_and_override(tmp_path):
    t = _task(tmp_path, "name: t\ntype: noul\nstatement: It is toxic.\n")
    assert pairs.render(t, 0, "txt") == (
        "noul | Q: Is the following statement about the text true? | statement: It is toxic.", "txt")
    t = _task(tmp_path, "name: t\ntype: noul\nquestion: Judge the step.\nstatement: It is toxic.\n")
    assert pairs.render(t, 1, "txt")[0] == "noul | Q: Judge the step. | statement: It is toxic."


def test_render_m2w_element_takes_the_candidate_line_and_no_options():
    """Placebo: render option 6 as "candidate G" and the option means nothing outside the row."""
    t = load_task(ROOT / "tasks/m2w-element.yaml")
    text = 'task: buy\ncandidates:\nA) <a> "Home"\nB) <li> "Books"\nG) <li> "Rare book"\n'
    a, b = pairs.render(t, 6, text)
    assert a == 'choice | Q: Which candidate element should the agent act on for its next step? | option: G) <li> "Rare book"'
    assert "options:" not in a and b == text


def test_seg_a_never_truncated_long_question_many_options(tmp_path):
    """Placebo: truncation="longest_first" (or no options cut) would eat segment A."""
    q = "Which of these very specific things does the text talk about, in detail? " * 12
    opts = [f"option number {i} with a few words" for i in range(200)]
    t = _task(tmp_path, f"name: t\ntype: choice\nquestion: {q!r}\noptions: {json.dumps(opts)}\n")
    a, b = pairs.render(t, 150, "word " * 3000)
    tok = pairs.tokenizer()
    assert a.endswith(" more)") and "option number 150" in a
    assert pairs.n_tokens(a.split(" | options: ")[1]) <= pairs.OPTIONS_TOKENS + 8  # "…(+N more)"
    ids = pairs.encode(tok, a, b)["input_ids"]
    assert len(ids) == pairs.MAX_LEN
    sep = tok.convert_tokens_to_ids("<eos>")
    assert tok.decode(ids[1:ids.index(sep)]).strip() == a          # segment A survived intact
    assert ids.count(sep) == 2 and ids.index(sep) < len(ids) - 2  # some text is still there


def test_keep_pairs_small_k_keeps_all_and_large_k_keeps_eight():
    rng = random.Random(0)
    assert pairs.keep_pairs([0.1] * 8, rng) == list(range(8))
    p = [0.0] * 77
    p[5], p[9], p[40] = 0.7, 0.2, 0.1          # a banking77 shortlist: three scored, the rest never
    for _ in range(50):
        kept = pairs.keep_pairs(p, rng)
        assert len(kept) == 8 and len(set(kept)) == 8 and kept[0] == 5
        assert {9, 40} <= set(kept[1:5])       # p > 0 is drawn before any p == 0 fills
    assert len({tuple(pairs.keep_pairs(p, rng)[5:]) for _ in range(20)}) > 1  # uniform part redraws


def test_index_and_sample_cap(tmp_path):
    """Placebo: a cap that counts rows instead of pairs, or a K > 8 row that keeps all K."""
    f = tmp_path / "pairs.jsonl"
    (tmp_path / "tasks").mkdir()
    (tmp_path / "tasks/big.yaml").write_text(
        "name: big\ntype: choice\nquestion: Q?\noptions: %s\ndata: {source: {csv: f}}\n"
        % json.dumps([f"o{i}" for i in range(12)]))
    (tmp_path / "tasks/small.yaml").write_text(
        "name: small\ntype: noul\nstatement: S.\ndata: {source: {csv: f}}\n")
    with open(f, "w") as out:
        for name, k, n in (("big", 12, 30), ("small", 2, 30)):
            for r in range(n):
                probs = [1.0 / k] * k
                probs[r % k] = 0.5
                for i in range(k):
                    out.write(json.dumps({"task": name, "source_id": f"train:{r}",
                                          "pair_id": f"{name}|train:{r}|{i}", "primitive": "x",
                                          "k": k, "question": "Q?", "option": f"o{i}",
                                          "option_index": i, "text": f"text {r}", "p": probs[i],
                                          "gold": None, "split": "train" if r % 3 else "calib"}) + "\n")
    rows = pairs.index(f)
    assert {n: len(v) for n, v in rows.items()} == {"big": 20, "small": 20}
    assert pairs.epoch_pairs(rows, 24) == 24 + 24
    tasks = {n: load_task(tmp_path / f"tasks/{n}.yaml") for n in rows}
    got = list(pairs.sample(rows, 24, random.Random(1), path=f, tasks=tasks))
    by = {n: [g for g in got if g["task"] == n] for n in rows}
    assert len(by["big"]) == 24 and len(by["small"]) == 24     # cap is in pairs, not rows
    assert len({g["row_id"] for g in by["big"]}) == 3          # 3 rows x 8 pairs
    assert len({g["row_id"] for g in by["small"]}) == 12       # 12 rows x 2 pairs
    assert all(g["target"] == 0.5 for g in by["big"][::8])      # argmax pair first in each row
    assert by["small"][0]["seg_a"].startswith("noul | Q:") and by["big"][0]["seg_b"].startswith("text ")
    again = {g["row_id"] for g in pairs.sample(rows, 24, random.Random(2), path=f, tasks=tasks)
             if g["task"] == "big"}
    assert again != {g["row_id"] for g in by["big"]}            # rows redrawn per epoch
