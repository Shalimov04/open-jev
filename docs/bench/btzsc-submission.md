# BTZSC leaderboard submission — prepared, **not** submitted

Nothing has been pushed anywhere. This file is the PR text for
[`btzsc/btzsc-results`](https://huggingface.co/datasets/btzsc/btzsc-results), to be opened **only**
after the repo owner reads it and says so. The payload is
`results/bench/btzsc/open-jev-base.json` (validated: `btzsc validate-result` → *Validation
passed.*), which goes in the PR at `results/reranker/open-jev-base.json` — `VALID_MODEL_TYPES` has
no `custom`, and a pointwise cross-encoder over (text, label) is a reranker.

Preregistration: `docs/prereg/bench-v1.md` (frozen before the run, three dated amendments).
Full write-up: `docs/experiments.md` §B3.

---

## PR title

    Add open-jev-base (141M pointwise cross-encoder), with 5 of 22 datasets disclosed as trained-on

## PR body

`open-jev-base` is a 141M pointwise cross-encoder (`jhu-clsp/mmBERT-small`, `num_labels=1`)
distilled from teacher soft labels on 40 typed-decision tasks. It scores one (question, option,
text) triple per forward pass and renormalises over the K options offered, so K and the label
wording are free at inference — which is what BTZSC asks for. No BTZSC data was used for training,
no calibration rows were fitted on BTZSC (its protocol forbids labelled examples from its own
tasks), `max_samples: None`, `test` split, harness `btzsc==0.1.2`.

**Overall (22 datasets): macro-F1 0.6011, accuracy 0.6308.**

**Please read this before merging — two disclosures.**

**1. Five of the 22 datasets' text sources are in this model's training mixture, and its numbers on
them must not be read as zero-shot:** `agnews`, `banking77`, `emotiondair`, `massive`,
`financialphrasebank`. They were labelled by a teacher model and trained on, which is a labelled
example from the task under your rule, whatever the labels' origin. We measured the exact-string
overlap between each BTZSC test text and our training file: 64.9 % of banking77's texts, 42.5 % of
financialphrasebank's, 26.3 % of agnews', 25.1 % of emotiondair's and 16.9 % of massive's appear
verbatim in it. We report them because the leaderboard refuses partial submissions; our own
write-up excludes them from every mean and every claim, and uses only the other 17 datasets
(mean macro-F1 0.6049). The worst overlap among those 17 is 2 texts in 8,427 (0.024 %).

**2. We could not reproduce your published `Qwen3-Reranker-0.6B` numbers with the installed
harness.** `btzsc/models/reranker.py` loads every reranker with
`AutoModelForSequenceClassification` but then reads `logits[:, -1, :]` at the `"yes"`/`"no"` vocab
ids, which only exists for a causal-LM head; under `transformers` 5.17 that path raises
`Cannot handle batch sizes > 1 if no padding token is defined` before it scores anything, and the
code at `8a0d52cbe423` is byte-identical to the PyPI sdists. Loading the causal-LM head the
adapter indexes gives 0.7606 / 0.4241 / 0.8842 on rottentomatoes / financialphrasebank /
appreviews against your published 0.7832 / 0.4138 / 0.8868 (−2.26 / +1.03 / −0.26 pt);
left-padding and bfloat16 variants are identical. Our own numbers are produced by our own adapter
and do not depend on that code path, but since we cannot demonstrate an environment that matches
yours, we make **no claim about where this model ranks** in your field, and we would rather you
knew that than not. If you can tell us the `transformers` version the published reranker rows were
produced with, we will rerun the check and report the result either way.

Run details: revision `bf071a696e3a6ea486bfc16c22cedbfb9c3ee207`, weights sha256
`216715a3fba176d19cf650bfaee52b8b11b33258b0e611aca4d3af79ad4a57a4`, one NVIDIA GB10, float32
weights with bf16 autocast, batch 256, `max_len` 512, 2,222,983 pairs in 4.1 h (150 pairs/s under
GPU contention). Each label is rendered as our training format's `choice` prompt with the fixed
question *"Which of these statements about the text is true?"* and your hypothesis strings as the
options; the contrast set is cut at 96 tokens, which is how the model was trained and is why the
large-K datasets (manifesto K=56: 0.041) are weak.

Model: https://huggingface.co/sshalimov04/open-jev-base ·
Code: https://github.com/sshalimov04/open-jev
