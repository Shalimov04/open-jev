"""docs/prereg/bench-v1.md: the metric, the §4 decision rule, the leak rule and the §2 rendering.

Placebo for the whole file: move any threshold in `verdict` and the fixed sentences swap.
No network, no GPU, no `btzsc` install: the harness only supplies datasets and `compute_metrics`.
"""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import f1_score

ROOT = Path(__file__).parent.parent
spec = importlib.util.spec_from_file_location("bench_btzsc", ROOT / "scripts/bench_btzsc.py")
bb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bb)


def test_macro_f1_matches_sklearn_including_absent_classes():
    """Placebo: return tp/(tp+fp+fn) or average only over present classes and it drifts from sklearn."""
    rng = np.random.default_rng(3)
    for k in (2, 5, 21):
        refs = rng.integers(0, k, 500)
        preds = rng.integers(0, max(k - 2, 1), 500)      # two classes never predicted
        assert bb.macro_f1(refs, preds, k) == pytest.approx(
            f1_score(refs, preds, average="macro", zero_division=0), abs=1e-12)


def test_verdict_is_the_three_preregistered_sentences_at_their_thresholds():
    """Placebo: drop the `mean >= 0.581` clause and a CI-clearing-median run reads as competitive."""
    assert bb.verdict(0.70, 0.611) == "Best published model under 200M on BTZSC-17"
    assert bb.verdict(0.60, 0.5749) == "Below the published median at 140M"
    assert bb.verdict(0.590, 0.575) == "Competitive with the published field at 140M"
    assert bb.verdict(0.580, 0.576) == "Below the published median at 140M"   # point below 0.581
    assert bb.verdict(0.640, 0.5749) == "Below the published median at 140M"  # CI does not clear


def test_bootstrap_is_seeded_and_brackets_the_point_estimate():
    """Placebo: reseed inside the draw loop and the interval collapses onto one resample."""
    rng = np.random.default_rng(0)
    refs = rng.integers(0, 3, 400)
    preds = np.where(rng.random(400) < 0.7, refs, rng.integers(0, 3, 400))
    per = {"a": (refs, preds, 3), "b": (refs, preds[::-1].copy(), 3)}
    ci = bb.bootstrap(per, draws=200, seed=0)
    assert ci == bb.bootstrap(per, draws=200, seed=0)
    point = bb.macro_f1(refs, preds, 3)
    assert ci["a"][0] < point < ci["a"][1]
    mean = np.mean([point, bb.macro_f1(refs, preds[::-1], 3)])
    assert ci["__mean__"][0] < mean < ci["__mean__"][1]
    assert ci["__mean__"][1] - ci["__mean__"][0] < ci["a"][1] - ci["a"][0]  # averaging tightens it


def test_choice_task_renders_the_prereg_segment_a():
    """Placebo: change QUESTION or drop `options:` and segment A stops matching prereg §2."""
    from openjev import pairs
    m = object.__new__(bb.OpenJevModel)
    m.render, m.dataset = "choice", "imdb"
    labels = ["This example movie review is negative", "This example movie review is positive"]
    a, b = pairs.render(m._task(labels), 1, "a fine film")
    assert a == ("choice | Q: Which of these statements about the text is true? "
                 "| option: This example movie review is positive "
                 "| options: This example movie review is negative | "
                 "This example movie review is positive")
    assert b == "a fine film"


def test_noul_task_is_one_statement_per_label_with_the_default_head():
    """Placebo: pass the question through and the noul rendering stops being the prereg's diagnostic."""
    from openjev import pairs
    m = object.__new__(bb.OpenJevModel)
    m.render, m.dataset = "noul", "imdb"
    labels = ["This example movie review is negative", "This example movie review is positive"]
    t = m._task(labels, 1)
    assert t.labels == ["true", "false"]
    assert pairs.render(t, 0, "txt")[0] == (
        "noul | Q: Is the following statement about the text true? "
        "| statement: This example movie review is positive")


def test_leak_rule_is_strictly_above_half_a_percent():
    """Placebo: use >= and a dataset at exactly 0.5 % is disqualified against the prereg."""
    assert not bb.is_leak(5, 1000)
    assert bb.is_leak(6, 1000)


def test_our_texts_reads_exact_strings_and_task_names(tmp_path):
    """Placebo: strip or normalise the text and an exact-string leak audit stops finding leaks."""
    p = tmp_path / "pairs.jsonl"
    p.write_text("".join(json.dumps({"task": t, "text": x}) + "\n" for t, x in
                         [("agnews", " Padded text. "), ("agnews", "b"), ("toxic", "b")]))
    texts, tasks = bb.our_texts(p)
    assert texts == {" Padded text. ", "b"} and tasks == {"agnews", "toxic"}


def test_the_primary_set_is_the_17_clean_datasets_and_the_5_are_named():
    """Placebo: let a mixture dataset into CLEAN and the primary mean includes a trained-on source."""
    assert len(bb.CLEAN) == 17 and len(bb.MIXTURE) == 5
    assert not set(bb.CLEAN) & set(bb.MIXTURE)
    assert set(bb.CLEAN) | set(bb.MIXTURE) == set(bb.K_PREREG)


def test_published_17_means_reproduce_the_prereg_field_table():
    """Placebo: average the 22 datasets instead of the 17 and the §4 thresholds move."""
    if not bb.PUBLISHED.exists():
        pytest.skip("published results snapshot not on this box")
    f = bb.published_17()
    assert len(f) == 35
    assert f["Qwen3-Reranker-8B"]["mean17"] == pytest.approx(0.732, abs=5e-4)
    assert f["gte-large-en-v1.5"]["mean17"] == pytest.approx(0.632, abs=5e-4)
    assert f["e5-base-v2"]["mean17"] == pytest.approx(0.611, abs=5e-4)
    assert np.median([v["mean17"] for v in f.values()]) == pytest.approx(0.575, abs=5e-4)
