# Preregistration — `bench-v1`: `open-jev-base` on an external, already-published benchmark

Written 2026-09-27, **before** `open-jev-base` has been run on any external benchmark. The point
of this round is a number other people can compare against without trusting us: the benchmark,
its subsets, the rendering, the thresholds and the STOP rules are fixed here, and nothing below
may be changed after a number exists except by a dated amendment in §9.

Unlocks: one evaluation script (`scripts/bench_btzsc.py`, not yet written) and the runs in §8.
No training, no teacher calls, no new labels.

## 0. What had been read before the freeze

- **Every number in `docs/experiments.md` §B2 and `runs/base-none-v2/README.md`** — the three
  unseen sets (yahoo-topics acc 0.528 / sst5 MAE 0.775 / ru-inappropriate AUROC 0.619, all at
  `gold500`) and the six leave-one-source-out losses. One of those matters here: the author has
  seen this model's accuracy on **500 rows of the yahoo_answers_topics test split**, which BTZSC
  also evaluates on (§2). The model's weights were never trained on any of it (`SKIP` in
  `scripts/build_pairs.py`, verified in B2), so it stays in the clean set, disclosed.
- **The published BTZSC results, all 35 model JSONs** in `btzsc/btzsc-results` (downloaded
  2026-09-27 into the scratchpad, not the repo), including per-dataset macro-F1. Every threshold
  in §4 is computed from them, which is the point; they are not blind and are not claimed to be.
- **Nothing from `open-jev-base` on any BTZSC dataset, COPA or HellaSwag.** No score matrix, no
  probe, no sample prediction. The BTZSC data itself has not been downloaded here; only its Hub
  metadata (split sizes, schema, three sample rows per config from the datasets-server API).

## 1. The decision

Whether the README and model card may say "on BTZSC (17 of 22 datasets, zero-shot, no
calibration rows) `open-jev-base` is *competitive with / below* the published field of 35
models", with the exact sentence chosen by §4 — and whether the full 22-dataset JSON is submitted
to the BTZSC leaderboard (it is, whatever it says: §7).

## 2. The benchmark, verified

### Candidates checked (primary sources, 2026-09-27)

| candidate | exists / loads under `datasets` 5.0.1 | protocol | published numbers | verdict |
|---|---|---|---|---|
| **BTZSC** ([arXiv 2603.11991](https://arxiv.org/abs/2603.11991), ICLR 2026; code [IliasAarab/btzsc](https://github.com/IliasAarab/btzsc), data [`btzsc/btzsc`](https://huggingface.co/datasets/btzsc/btzsc), results [`btzsc/btzsc-results`](https://huggingface.co/datasets/btzsc/btzsc-results)) | yes: parquet, no loading script, 22 configs + 5 bundles, `test` split only, 2,222,983 pair rows; licence "varies by source dataset" (paper Table 3; the fetch tool could not extract the per-dataset list — read it from the PDF before publishing) | zero-shot single-label: each label verbalised as a fixed hypothesis string (`"This example news text is about business news"`); model scores every (text, hypothesis) pair, argmax; **no labelled examples, no fine-tuning, no model selection on BTZSC**; metric = sklearn macro-F1 (`zero_division=0`) per dataset, overall = flat mean over the 22 | 35 models in the results repo (paper says 38 evaluated), per-dataset macro-F1 + accuracy each; top Qwen3-Reranker-8B 0.722, median 0.555, best ≤200M e5-base-v2 0.597; harness `btzsc` 0.1.1 commit `8a0d52cbe423`, `max_samples: None` on every published run | **primary**: same task shape as ours (K and label wording free at inference), pluggable `BaseModel` interface, per-dataset numbers public so a clean subset can be compared without re-running anyone |
| RAFT ([ought/raft](https://huggingface.co/datasets/ought/raft)) | loads (parquet, 11 subsets), but the **test labels are hidden** (`Label = 0`), and the leaderboard Space [`ought/raft-leaderboard`](https://huggingface.co/spaces/ought/raft-leaderboard) is in **runtime error**; raft.elicit.org redirects to it | 50 labelled rows per task, few-shot | leaderboard unreachable; HELM re-splits the 50 public rows 20/80 (`raft_scenario.py`) | dead: no scorable test set, no live comparison |
| T0 held-out suite ([arXiv 2110.08207](https://arxiv.org/abs/2110.08207), Table 6) | COPA/CB/RTE/WiC/WSC: `aps/super_glue` parquet, **test labels −1**, validation only (COPA val = 100); HellaSwag: `Rowan/hellaswag` parquet, MIT, val 10,042; ANLI: `facebook/anli` parquet, cc-by-nc-4.0; Winogrande: `allenai/winogrande` parquet, val 1,267; **StoryCloze: loading script + manual download → dead** | rank classification over answer choices, median over prompts, validation splits | T5+LM 11B, T0 3B, T0 11B, GPT-3 — **no model under 3B** | secondary only, for the per-row-option tasks (COPA, HellaSwag): no size-comparable number exists, so the only claim possible is "above chance" (§6). RTE is disqualified (in our mixture). |
| MTEB classification / reranking ([arXiv 2210.07316](https://arxiv.org/abs/2210.07316)) | loads | classification = **logistic regression trained on the train-split embeddings**, scored on test; reranking = cosine of query/document embeddings, MAP | many | **mismatch, not bent**: our model has no embedding and trains nothing at test time; there is no zero-shot label-matching track |
| calibration under zero-shot transfer | BTZSC reports no ECE; HELM reports ECE-10 but for 5-shot LLMs on its own splits | — | none comparable | **null finding**: no published leaderboard reports ECE for zero-shot text classification. ECE is reported here as a diagnostic (§3), with no external comparison and no claim |

### The chosen benchmark: BTZSC, and the overlap audit

BTZSC's 22 datasets against `data/pairs-stats.json` (42 run dirs) and `tasks/`, `tasks/base/`,
`tasks/unseen/`. "In mixture" means the text source was labelled by a teacher and trained on;
that is a labelled example from the task under BTZSC's rule, whatever the labels' origin.

| BTZSC dataset | K | pair rows (test) | texts | our run dir / task on the same source | status |
|---|---|---|---|---|---|
| agnews | 4 | 30,400 | 7,600 | `agnews`, `agnews-jev` (fancyzhx/ag_news train) | **disqualified** |
| banking77 | 77 | 221,760 | 2,880 | `banking77` (mteb/banking77 train) | **disqualified** |
| emotiondair | 6 | 12,000 | 2,000 | `emotion-dair-jev` (dair-ai/emotion train) | **disqualified** |
| massive | 59 | 175,466 | 2,974 | `massive-en-jev` (amazon_massive, same utterances, 18-way scenario labels) | **disqualified** (same texts, different label set — still the task) |
| financialphrasebank | 3 | 2,070 | 690 | `fin-sentiment-jev` (FinanceInc/auditor_sentiment, "based on financial_phrasebank") | **disqualified** |
| yahootopics | 10 | 500,000 (harness keeps ≈450k, see below) | 50,000 | `tasks/unseen/yahoo-topics` — calibration/eval only, never in `pairs.jsonl` | clean, disclosed (§0) |
| wikitoxic_insult / obscene / threat / toxicaggregated | 2 | 16,854 / 17,382 / 10,422 / 20,000 | 8,427 / 8,691 / 5,211 / 10,000 | `toxic`, `civil-*-jev` ask the same questions over **google/civil_comments**; BTZSC's are Wikipedia (Wulczyn et al. 2017) | clean, question-adjacent |
| amazonpolarity, imdb, appreviews, yelpreviews, rottentomatoes | 2 | 20,000 / 20,000 / 8,000 / 20,000 / 2,132 | 10,000 / 10,000 / 4,000 / 10,000 / 1,066 | none (sst5 is an unseen set, calibration only) | clean |
| empathetic | 32 | 81,344 | 2,542 | none | clean |
| biasframes_intent / offensive / sex | 2 | 7,296 / 7,676 / 8,808 | 3,648 / 3,838 / 4,404 | none | clean |
| trueteacher | 2 | 17,910 | 8,955 | none | clean |
| manifesto | 56 | 953,008 | 17,018 | none | clean |
| capsotu | 21 | 70,455 | 3,355 | none | clean |

**Primary set = the 17 clean datasets.** Fixed now: `yahootopics, wikitoxic_insult,
wikitoxic_obscene, wikitoxic_threat, wikitoxic_toxicaggregated, amazonpolarity, imdb, appreviews,
yelpreviews, rottentomatoes, empathetic, biasframes_intent, biasframes_offensive, biasframes_sex,
trueteacher, manifesto, capsotu`. The 5 disqualified ones are still **run** (the leaderboard
refuses partial submissions) and **reported**, but never enter the primary mean and are marked
"trained on this source" wherever they appear.

Harness facts that are part of the protocol and are followed as-is: `load_btzsc_dataset` drops
every yahootopics text whose gold is `Business & Finance` except the first 20 rows (so the
published yahootopics numbers are on ≈9 effective classes); label order is taken from the first
grouped sample; `max_samples=None`; metric aggregation is a flat mean over datasets
(`compute_task_summary`), which is what §4 recomputes on the 17.

### The rendering (fixed now)

The harness calls `predict_scores(texts, labels)` with `labels` = the K hypothesis strings of
the dataset. The wrapper builds one `Task` per dataset:

```
type: choice
question: "Which of these statements about the text is true?"
options: <the K hypothesis strings, verbatim, in harness order>
```

and scores with `base.predict_logits(tok, model, task, texts)` (batch 64, `max_len` 512). Segment
A is therefore `choice | Q: Which of these statements about the text is true? | option: <hypothesis>
| options: <all hypotheses, cut at 96 tokens by pairs.options_text>`; segment B is the text. No
code in `openjev/pairs.py` or `openjev/base.py` changes. The 96-token cut means the contrast set
is partial for K ≥ ~6 (manifesto, empathetic, capsotu, banking77, massive): that is how the model
was trained and is a limitation, not a knob. Predicted label = argmax over the K renormalised
sigmoids — identical to the harness's own argmax over scores.

**Calibration rows: none.** BTZSC forbids labelled examples from its tasks, so the `gold500`
bias that our unseen-set claim leaned on is off-protocol; the **primary is the raw model**. No
calibrated variant is run on BTZSC at all — drawing calib rows from the only split would also
change the evaluation set and break comparability, so the variant is not "secondary", it is
absent.

**Secondary rendering, diagnostic only:** the same 22 datasets with a `noul` task per label
(`statement: <hypothesis>`, `question` default), argmax over p(true). Reported next to the
primary; never a claim; run only after the primary is on disk (§7 cut order).

## 3. The metric

| | |
|---|---|
| name | macro-F1 (sklearn `f1_score(average="macro", zero_division=0)`), computed by the harness's `compute_metrics`, plus accuracy |
| numerator / denominator | per dataset: sklearn's macro-F1 over the harness's `references` (gold label index) vs our argmax; the **primary statistic is the flat mean of the 17 per-dataset macro-F1 values** |
| unit | F1 points (×100 in prose) |
| matching rule | the published per-dataset numbers are on the same rows (`max_samples: None`, `split="test"`, harness 0.1.x); we compare on the same 17 datasets by recomputing every published model's 17-mean from its JSON |
| read from | `results/bench/btzsc/open-jev-base.json` (harness `to_json`, all 22) + `results/bench/btzsc/scores/<dataset>.npy` (our score matrices, so the CI below can be computed) |
| CI | per dataset, paired bootstrap over texts (2,000 resamples, seed 0) of macro-F1; the 17-mean CI uses the same 2,000 draws jointly. The published numbers carry no CI; they are treated as fixed points |

Secondary readouts, diagnostic, never a claim: accuracy, the four `by_task` means, ECE (10-bin,
top-probability, raw) per dataset, the 5 disqualified datasets, the `noul` rendering, and the
per-dataset gap to the field median (from the 35 JSONs:
amazonpolarity 0.912, imdb 0.896, appreviews 0.908, yelpreviews 0.951, rottentomatoes 0.814,
empathetic 0.318, biasframes_intent 0.565, yahootopics 0.483, trueteacher 0.409, manifesto 0.162,
capsotu 0.462, biasframes_offensive 0.543, biasframes_sex 0.304, wikitoxic_toxicaggregated 0.655,
wikitoxic_obscene 0.683, wikitoxic_threat 0.290, wikitoxic_insult 0.595).

## 4. What counts as a result

The published field on the **17 clean datasets** (recomputed from the 35 JSONs, 2026-09-27):

| reference | 17-mean macro-F1 | (22-mean) |
|---|---|---|
| Qwen3-Reranker-8B (best overall) | 0.732 | 0.722 |
| gte-large-en-v1.5, 434M (best < 1B) | 0.632 | 0.617 |
| **e5-base-v2, 110M (best ≤ 200M)** | **0.611** | 0.597 |
| median of the 11 models ≤ 200M | 0.577 | — |
| **median of all 35** | **0.575** | 0.555 |
| gte-modernbert-base, 149M (nearest architecture-size peer) | 0.577 | 0.586 |
| ms-marco-MiniLM-L6-v2, 22M (smallest cross-encoder) | 0.477 | 0.422 |

Our size class is ≤ 200M (mmBERT-small ≈ 140M). Exactly one of the following sentences is
earned, decided by the 17-mean **M** and its 95 % CI **[L, U]**:

- **"Best published model under 200M on BTZSC-17"**: L ≥ 0.611.
- **"Competitive with the published field at 140M"**: L ≥ 0.575 (the CI clears the field
  median) **and** M ≥ 0.581 (within 3.0 pts of e5-base-v2). 3.0 pts because the 22-mean vs
  17-mean re-ranking above moves neighbouring models by 1–3 pts, so a gap under that is inside
  the benchmark's own subset noise.
- **"Below the published median at 140M"**: anything else, including U ≥ 0.575 with L < 0.575 —
  that is written as *below the median, CI not excluding it*, not as "competitive".

No per-task-group claim, no per-dataset claim, no "beats X" for any single model other than the
sentences above. Whichever sentence is earned goes into `README.md`, the model card and
`docs/findings.md` verbatim, next to the 5 disqualified datasets' numbers.

## 5. STOP rules (any one voids the primary number)

- **Environment differs from the published one.** Before our model runs, `Qwen/Qwen3-Reranker-0.6B`
  (the one published baseline whose Hub id is unambiguous — the NLI baselines' `url` fields in the
  results repo carry no org) is evaluated with the same installed harness on `rottentomatoes`,
  `financialphrasebank` and `appreviews` (12,202 pairs); any of the three deviates from its
  published macro-F1 (0.783 / 0.414 / 0.887) by more than 0.5 pt → STOP, fix the environment, log an amendment, rerun the
  check. Harness pinned to `btzsc==0.1.2` (published results: 0.1.1 / `8a0d52cbe423`); version and
  commit recorded in our JSON.
- **Text-level leak.** Before scoring, the exact-string intersection of every BTZSC test text
  with the `text` field of `data/pairs.jsonl` is computed; a clean dataset with > 0.5 % of its
  texts present is **disqualified**, not noted, and the primary set shrinks with an amendment.
- Any of the 5 disqualified datasets enters the primary mean, or a calibration variant of any
  kind is fitted on BTZSC rows.
- `max_samples` is not `None`; a dataset yields fewer texts than pair rows ÷ K (after the
  harness's yahootopics filter); the K hypotheses of a dataset are not identical across its rows
  (the harness assumes they are; the wrapper asserts it).
- The weights are not the shipped artefact: `sshalimov04/open-jev-base` at the commit hash
  recorded in `runs/base-none-v2/train.json`'s student (sha logged in our JSON's `revision`).
- The question string, `max_len`, batch size, the 96-token options cut or anything in
  `openjev/pairs.py` / `openjev/base.py` changes after any BTZSC number has been seen. A change
  means a re-run from zero and an amendment.
- Any run on the teacher at :8000 or any OpenRouter/Jev spend in this round (budget 0).

## 6. Secondary: the per-row-option tasks (COPA, HellaSwag)

The shape a fixed-head classifier cannot do and the pair scorer can — the argument the
`m2w-element` result already made — measured on two public sets the T0 paper evaluated.
StoryCloze is excluded: script-based dataset with manual download, dead under `datasets` 5.
Winogrande is excluded by choice: fill-in-the-blank, not "option describes text".

| set | source | rows | K | pairs | rendering (fixed) | chance | T5+LM 11B / T0 3B / T0 11B (Table 6, median over prompts) |
|---|---|---|---|---|---|---|---|
| COPA | `aps/super_glue`, `copa`, **validation** (test labels hidden) | 100 | 2 | 200 | text = premise; question = `"Which alternative is the more plausible {cause/effect} of the premise?"` with the row's `question` field; options = choice1, choice2 | 0.50 | 55.00 / 74.92 / 90.79 |
| HellaSwag | `Rowan/hellaswag`, **validation** | 10,042 | 4 | 40,168 | text = `ctx`; question = `"Which ending most plausibly continues the text?"`; options = the 4 endings | 0.25 | 27.73 / 27.51 / 33.65 |

One `Task` per row (options are per-row), `base.predict_logits` on the single text, raw argmax.
Result rule: accuracy − chance with a bootstrap 95 % CI over rows; **the only claim available is
"above chance (CI excluding 0)"** — there is no published model under 3B here, so no
"competitive" sentence exists for this section and none will be written. COPA at n = 100
resolves about ±10 pts and is reported with that caveat.

## 7. Budget, cut order, publication commitment

Pair counts and time at the task's 6 ms/pair (batched throughput on this box is several times
better; 6 ms is the upper bound):

| run | pairs | ≤ hours |
|---|---|---|
| Qwen3-Reranker-0.6B sanity (§5) | 12,202 | 0.1 |
| primary, all 22 (after the yahootopics filter ≈ −50k) | ≈ 2,173,000 (17 clean ≈ 1,731,000) | 3.7 |
| COPA + HellaSwag | 40,368 | 0.1 |
| `noul` rendering, all 22 | ≈ 2,173,000 | 3.7 |

Cut order if the GPU lane is needed: (1) the `noul` rendering, (2) COPA + HellaSwag. **Never
cut**: the overlap audit, the sanity check, the primary 22-dataset run.

**Publication, whatever the number:** the full-22 JSON is submitted as a PR to
`btzsc/btzsc-results` at `results/reranker/open-jev-base.json` (`VALID_MODEL_TYPES` has no
`custom`; a pointwise cross-encoder over (text, label) is a reranker), with the PR text and the
model card stating that 5 of the 22 datasets' sources were in the training mixture and naming
them. If the maintainers reject it on those grounds, that is their call and the numbers still go
into our docs. The 17-set primary and the earned §4 sentence go into `README.md`, the model card
and `docs/findings.md` in the same commit as the results. A "BTZSC-clean" retrain that excludes
the five sources is a follow-up experiment with its own preregistration, not this one.

## 8. Runnable plan (in order; nothing before step 1 touches the model)

```bash
# 0. isolated env: btzsc pulls sentence-transformers + pandas, keep them out of .venv
python -m venv .venv-bench && .venv-bench/bin/pip install "btzsc==0.1.2" -e .
# 1. overlap audit (data only, no model): writes results/bench/btzsc/overlap.json, exits 1 on a leak
.venv-bench/bin/python scripts/bench_btzsc.py --overlap-audit
# 2. environment sanity on a published model (STOP rule 1)
.venv-bench/bin/python scripts/bench_btzsc.py --model Qwen/Qwen3-Reranker-0.6B --type reranker \
    --datasets rottentomatoes financialphrasebank appreviews --out results/bench/btzsc/sanity.json
# 3. primary: all 22, choice rendering, raw; score matrices cached to results/bench/btzsc/scores/
scripts/gpu_queue.sh .venv-bench/bin/python scripts/bench_btzsc.py --model sshalimov04/open-jev-base \
    --render choice --out results/bench/btzsc/open-jev-base.json
# 4. CI + the §4 decision on the 17 (CPU): results/bench/btzsc/table.md
.venv-bench/bin/python scripts/bench_btzsc.py --table
# 5. secondary per-row sets; 6. noul rendering (cut first if needed)
.venv-bench/bin/python scripts/bench_btzsc.py --t0 copa hellaswag --out results/bench/t0/
scripts/gpu_queue.sh .venv-bench/bin/python scripts/bench_btzsc.py --render noul --out results/bench/btzsc/open-jev-base-noul.json
# 7. validate + PR
.venv-bench/bin/btzsc validate-result results/bench/btzsc/open-jev-base.json
```

`scripts/bench_btzsc.py` is one file: a `btzsc.models.base.BaseModel` subclass whose
`predict_scores(texts, labels)` builds the §2 `Task`, calls `base.predict_logits`, saves the
matrix, and returns it; `--table` recomputes the 35 published 17-means from the results repo and
applies §4. Nothing else is added to `openjev/`.

## 9. Amendments

Append only, each dated, each stating **what had been read when it was written** and what changed.

| date | what had been read | change | why |
|---|---|---|---|
| 2026-09-27 | the prereg as frozen at 3a30b97; `btzsc==0.1.2`'s `data.py`/`benchmark.py`/`metrics.py`/`leaderboard/validate.py`; the 35 published JSONs (§4's field table reproduced exactly: 0.732 / 0.632 / 0.611 / median 0.575, and §3's per-dataset medians); the overlap audit's output through `empathetic` and a shape survey of the raw parquet. **No `open-jev-base` number on any BTZSC dataset, COPA or HellaSwag exists yet.** | **K is taken from the data, not from §2's table.** §2's `K` / `texts` columns came from Hub metadata; the harness derives K from the binary-label period, and for `banking77` the parquet holds **72** distinct hypotheses over **3,080** texts (221,760 = 3,080 × 72, which also happens to equal 2,880 × 77 — that is how the table's 77/2,880 arose). The script therefore asserts only the two invariants §5 actually names — `texts == pair_rows ÷ K` and the K hypotheses identical across every row — and records `k` next to `k_prereg`. Every other dataset's K and text count match §2 as written. | A wrong row in a metadata table is not a protocol change: the harness's own K is what the published numbers were computed with, and `banking77` is disqualified anyway (it is not in the primary 17). Refusing to run would freeze the whole benchmark over a cell that no claim depends on. |
| 2026-09-27 | the same as the row above, plus the completed overlap audit (`results/bench/btzsc/overlap.json`: no leak, primary set = 17) and **three** runs of the §5 sanity check on `Qwen/Qwen3-Reranker-0.6B`. Still **no `open-jev-base` number on any BTZSC dataset**. | **STOP rule 1 fired and could not be cleared.** `btzsc.models.reranker` loads every reranker with `AutoModelForSequenceClassification` but then reads `logits[:, -1, :]` at the `"yes"`/`"no"` vocab ids, which only exists for a causal-LM head; under `transformers` 5.17 the Qwen path raises before it scores anything (`Cannot handle batch sizes > 1 if no padding token is defined`). The published code on GitHub at `8a0d52cbe423` is byte-identical to the PyPI 0.1.1/0.1.2 sdists, so the published numbers came from an environment that cannot be reconstructed from anything published. The environment fix (load the causal-LM head the adapter actually indexes) gives 0.7606 / 0.4241 / 0.8842 against the published 0.7832 / 0.4138 / 0.8868 — **−2.26 / +1.03 / −0.26 pt**, so two of three miss the ±0.5 pt band; left-padding and bfloat16 variants give bit-identical numbers (these datasets' hypotheses tokenise to equal lengths, so padding never bites). Consequence, per §5: **the primary number is voided as a comparison against the published field. No §4 sentence is earned or written anywhere**, and neither the README, the model card nor `docs/findings.md` gets one. The run is still executed and published in full (§7) as an unvalidated-environment number, with this row cited next to it. | The prereg's own answer to "the environment differs" is STOP, not "report anyway". The three attempts are logged instead of hidden, and the failure is in the published baseline's adapter, not in the datasets or the metric — appreviews lands inside the band, which is evidence that the data/metric path is right and only the Qwen adapter's model class is wrong. Fixing it "until it matches" by tuning the baseline would be model selection on the benchmark. |
| 2026-09-27 | as the row above | **§2's yahootopics "harness filter" does not exist.** `load_btzsc_dataset` in 0.1.1 and 0.1.2 contains no Business-&-Finance filter: the config is 500,000 pair rows / 50,000 texts / K = 10, all kept, so the yahootopics numbers are on 10 classes and the 17 clean datasets are **1,781,287** pairs over 161,155 texts (not the ≈1,731,000 §7 budgeted). Nothing is dropped and nothing else changes. | A protocol note that turned out to describe code that is not there. The harness is followed as-is, which here means "no filter". |
