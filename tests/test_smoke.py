import json
import math
from pathlib import Path

import pytest

from openjev.spec import load_task
from openjev.teacher import parse_logprobs, shortlist, softmax_letters, system_prompt

ROOT = Path(__file__).parent.parent


def test_spec_parse_all_tasks():
    types = {}
    for p in sorted((ROOT / "tasks").glob("*.yaml")):
        t = load_task(p)
        types[t.name] = (t.type, t.k)
    assert types["agnews"] == ("choice", 4)
    assert types["georeview"] == ("score", 5)
    assert types["toxic"] == ("noul", 2)
    assert types["banking77"] == ("choice", 77)
    geo = load_task(ROOT / "tasks/georeview.yaml")
    assert geo.labels == ["1", "2", "3", "4", "5"] and geo.values == [1.0, 2.0, 3.0, 4.0, 5.0]
    assert load_task(ROOT / "tasks/toxic.yaml").labels == ["true", "false"]


def test_spec_rejects_unknown_key(tmp_path):
    p = tmp_path / "t.yaml"
    p.write_text("name: x\ntype: choice\nquestion: q\noptions: [a, b]\ndata: {source: {csv: f}, bogus: 1}\n")
    with pytest.raises(ValueError):
        load_task(p)


def test_parse_saved_response():
    resp = json.loads((ROOT / "tests/data/response_agnews.json").read_text())
    raw = parse_logprobs(resp, list("ABCD"))
    p = softmax_letters(raw, list("ABCD"))
    assert set(raw) == set("ABCD") and abs(sum(p) - 1) < 1e-9 and p.index(max(p)) == 2


def test_softmax_missing_letter_is_zero():
    p = softmax_letters({"A": math.log(0.3), "B": math.log(0.3)}, list("ABC"))
    assert p == pytest.approx([0.5, 0.5, 0.0])


def test_shortlist_math():
    # each chunk's probs are a softmax over its letters + "none" (last), so they sum to 1
    chunks = [([0, 1, 2], [0.1, 0.8, 0.05, 0.05]), ([3, 4], [0.2, 0.1, 0.7])]
    # score = p(i|chunk): chunk 2 mostly answers "none", so its options already rank low.
    # Penalising them again by (1 - p(none)) would put option 0 (0.1) above option 3 (0.2).
    assert shortlist(chunks, 2) == [1, 3]
    assert shortlist(chunks, 5) == [1, 3, 0, 4, 2]


def test_final_call_letters_options_in_option_order():
    """A3: the shortlist is re-sorted before the final call, so letter A is not always the
    chunk stage's favourite (that would double-count the teacher's letter-position bias)."""
    import asyncio

    from openjev.teacher import Teacher

    task = load_task(ROOT / "tasks/banking77.yaml")
    task.teacher.max_options_per_call = 2
    task.options = task.labels = ["a", "b", "c", "d"]
    weight = {"a": 0.2, "b": 0.1, "c": 0.1, "d": 0.5}  # d wins the chunk stage, a is second
    seen = []

    class Fake(Teacher):
        async def ask(self, system, text, letters):
            opts = [l.split(": ", 1)[1] for l in system.splitlines()[1:-1]]
            seen.append(opts)
            w = [weight.get(o, 0.1) for o in opts]  # "none of the above" -> 0.1
            return {l: math.log(x / sum(w)) for l, x in zip(letters, w)}

    probs, _ = asyncio.run(Fake(task, None, "m").label("text"))
    assert seen[-1] == ["a", "d"]  # not ["d", "a"]
    assert probs[1] == probs[2] == 0 and probs[0] > 0 and probs[3] > 0


def test_noul_bool_and_int_gold_are_not_inverted():
    """A1: labels are ["true", "false"], so a true-ish gold must map to index 0."""
    from openjev.data import gold_index

    t = load_task(ROOT / "tasks/toxic.yaml")
    assert [gold_index(t, g) for g in (True, 1, 0.9, 1.0)] == [0, 0, 0, 0]
    assert [gold_index(t, g) for g in (False, 0, 0.1, 0.0)] == [1, 1, 1, 1]
    assert gold_index(t, None) is None


def test_prompt_letters():
    t = load_task(ROOT / "tasks/banking77.yaml")
    prompt, letters = system_prompt(t, range(16, 31), with_none=True)
    assert letters[-1] == "Z" and len(letters) == 16 and "Z: none of the above" in prompt


def test_split_disjoint():
    from openjev.data import examples
    t = load_task(ROOT / "tasks/agnews.yaml")
    t.data.train.n, t.data.calib.n, t.data.eval.n = 300, 100, 50
    ex = examples(t)
    ids = {s: {e["id"] for e in ex if e["split"] == s} for s in ("train", "calib", "eval")}
    assert [len(v) for v in ids.values()] == [300, 100, 50]
    assert not ids["train"] & ids["calib"]
    assert [e["id"] for e in examples(t)] == [e["id"] for e in ex]  # deterministic
