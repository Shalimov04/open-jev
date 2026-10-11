"""docs/prereg/decision-index-v1.md: the engine's rendering, its refusal rule and its answer math.

No network, no GPU, no harness install: a whitespace tokenizer stands in for mmBERT's and the
model's sigmoids are stubbed, so what is tested is everything around the forward pass.
"""
import importlib.util
import math
from pathlib import Path

import pytest

from openjev import pairs

ROOT = Path(__file__).parent.parent
spec = importlib.util.spec_from_file_location("bench_decision_index",
                                              ROOT / "scripts/bench_decision_index.py")
bdi = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bdi)


class WhitespaceTok:
    def __call__(self, a, b, truncation=False):
        return {"input_ids": [[0] * (len(x.split()) + len(y.split()) + 3) for x, y in zip(a, b)]}


@pytest.fixture
def engine(monkeypatch):
    monkeypatch.setattr(pairs, "n_tokens", lambda s: len(s.split()))
    e = object.__new__(bdi.OpenJevEngine)
    e.pairs, e.tok, e.max_len, e.batch_size = pairs, WhitespaceTok(), 512, 256
    return e


def stub(engine, fn):
    """Sigmoid of each pair := fn(seg_a, seg_b); records what was scored."""
    seen = []

    def sig(flat):
        seen.extend(flat)
        return [fn(a, b) for a, b in flat]
    engine._sigmoids = sig
    return seen


# the kit's own `engines.base.validate`, copied so the test needs no install
def validate(questions, response):
    assert set(response["answers"]) == set(questions)
    for k, q in questions.items():
        a = response["answers"][k]
        assert a["type"] == q["type"]
        if q["type"] == "choice":
            p = a["probabilities"]
            assert a["choice"] in q["criteria"] and set(p) == set(q["criteria"])
            assert all(math.isfinite(v) and 0 <= v <= 1 for v in p.values())
            assert abs(sum(p.values()) - 1) <= 0.01
        else:
            assert 0 <= a["noul"] <= 1


def test_choice_is_renormalised_sigmoids_and_passes_the_kits_validator(engine):
    """Placebo: return the raw sigmoids (no `/ z`) and they sum to 1.2, which the kit rejects."""
    stub(engine, lambda a, b: {"A": 0.9, "B": 0.2, "C": 0.1}[a.split("option: ")[1][0]])
    qs = {"q": {"type": "choice", "instructions": "Which?",
                "criteria": {"A": "alpha", "B": "beta", "C": "gamma"}}}
    resp, raw = engine("some state", qs)
    validate(qs, resp)
    p = resp["answers"]["q"]["probabilities"]
    assert resp["answers"]["q"]["choice"] == "A"
    assert p == pytest.approx({"A": 0.75, "B": 0 + 0.2 / 1.2, "C": 0.1 / 1.2})
    assert raw["sigmoid"]["q"] == [0.9, 0.2, 0.1]


def test_noul_is_p_true_of_one_pair_not_renormalised(engine):
    """Placebo: render noul as a two-option choice and p(true) becomes 0.3/(0.3+0.7*...)."""
    seen = stub(engine, lambda a, b: 0.3)
    qs = {"h": {"type": "noul", "instructions": "Does the answer contain a hallucination?"}}
    resp, _ = engine("ctx", qs)
    validate(qs, resp)
    assert resp["answers"]["h"] == {"type": "noul", "noul": 0.3}
    assert seen == [("noul | Q: " + pairs.NOUL_HEAD
                     + " | statement: Does the answer contain a hallucination?", "ctx")]


def test_segments_are_the_training_render_with_key_colon_description(engine):
    """Placebo: put the state into segment A, or drop the key, and the pair leaves the training format."""
    seen = stub(engine, lambda a, b: 0.5)
    qs = {"q": {"type": "choice", "instructions": {"task": "pick"},
                "criteria": {"yes": "yes", "B": "the other one"}}}
    engine({"doc": "x y"}, qs)
    assert seen == [
        ('choice | Q: {"task":"pick"} | option: yes | options: yes | B: the other one', '{"doc":"x y"}'),
        ('choice | Q: {"task":"pick"} | option: B: the other one | options: yes | B: the other one',
         '{"doc":"x y"}'),
    ]


def test_an_over_long_pair_refuses_the_request_instead_of_truncating(engine):
    """Placebo: delete the `longest > max_len` check and the long state is scored (silently cut at 512)."""
    seen = stub(engine, lambda a, b: 0.5)
    engine.max_len = 40
    qs = {"short": {"type": "noul", "instructions": "ok?"},
          "q": {"type": "choice", "instructions": "Which?", "criteria": {"A": "a", "B": "b"}}}
    with pytest.raises(bdi.Unsupported, match="maximum context length"):
        engine(" ".join(["w"] * 40), qs)
    assert seen == []                                   # nothing scored, nothing partial
    engine.max_len = 512
    validate(qs, engine(" ".join(["w"] * 40), qs)[0])


def test_each_question_reads_its_own_slice_of_one_batched_pass(engine):
    """Placebo: reset the slice offset per question and question 2 gets question 1's sigmoids."""
    seen = stub(engine, lambda a, b: 0.9 if "option: B" in a or "option: y" in a else 0.1)
    qs = {"one": {"type": "choice", "instructions": "1?", "criteria": {"A": "", "B": ""}},
          "two": {"type": "choice", "instructions": "2?", "criteria": {"x": "", "y": "", "z": ""}},
          "three": {"type": "noul", "instructions": "3?"}}
    resp, _ = engine("", qs)
    validate(qs, resp)
    assert len(seen) == 2 + 3 + 1
    assert [resp["answers"][k].get("choice") for k in ("one", "two")] == ["B", "y"]
    assert resp["answers"]["three"]["noul"] == 0.1


def test_unknown_question_type_is_unsupported_not_an_error(engine):
    """Placebo: let `score` through and the runner logs an error and retries it on every resume."""
    stub(engine, lambda a, b: 0.5)
    with pytest.raises(bdi.Unsupported):
        engine("s", {"q": {"type": "score", "criteria": ["lo", "hi"]}})
