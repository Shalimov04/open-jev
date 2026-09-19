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
    # chunk 1 confident its option 1 fits; chunk 2 says "none" -> its options are down-weighted
    chunks = [([0, 1, 2], [0.1, 0.8, 0.1, 0.0]), ([3, 4], [0.9, 0.1, 0.9])]
    assert shortlist(chunks, 2) == [1, 0]  # s: 0.1,0.8,0.1 vs 0.09,0.01
    assert shortlist(chunks, 5) == [1, 0, 2, 3, 4]


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
