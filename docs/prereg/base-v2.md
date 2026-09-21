# Preregistration — `base-v2`

Written **before** any `base-v2` training runs, and before any number from the three
pre-registered unseen sets was computed. Fixes the three reasons B1 was not publishable:
the fold models never converged (a 45-minute budget cut them), nothing said whether the 27 new
Jev-labelled tasks helped, and "unseen" meant only "unseen text source".

Unlocks: `openjev base train` for `base-F1` / `base-F2` / `base-none` and the diversity ablation.
Nothing in `docs/findings.md` may cite a `base-v2` arm whose `train.json` predates this file's
commit.

## 0. What had been read before the freeze

- **The whole B1 section of `docs/experiments.md` is known to the author.** Every cell of it:
  the leave-one-source-out table (base zero-shot loses to the per-task student on 5 of 6 tasks;
  m2w-element +9.3 pts, toxic −0.007 vs teacher), the calibration-under-transfer table
  (base ECE raw better on 4 of 6, banking77 0.322 → 0.051 under `gold500`), the warm-start table
  (N = 100 Δ = +4.8 pts driven by one collapsed scratch seed; N = 500 no difference), the three
  training logs (`best_calib_bce` 0.3544 / 0.3585 / 0.3550, calib BCE still falling at the last
  checkpoint, 78–103 pairs/s, peak 8.9 GB). This preregistration exists **because** of those
  numbers, so it is not blind to them, and nothing below may be described as a blind result.
  What it *is* blind to: every number the arms below will produce.
- **The three unseen sets: structure only.** They were created and `openjev check`-ed in the same
  job that wrote this file. What was read: they load under `datasets` 5.0.1, their split sizes
  (800/200/500), their gold distributions, their text lengths, and three sample rows per task
  (printed by `openjev check`). **No teacher accuracy, no marginal, no probe, no model output on
  them** — `check --probe` was deliberately not run, and nothing has read
  `runs/{yahoo-topics,sst5,ru-inappropriate}-jev/teacher.jsonl` other than to count its rows and
  their splits (700 = 200 calib + 500 eval each, 0 failed; §6).
- Nothing else. No `base-v2` model exists at the time of writing.

## 1. The decision

Whether `open-jev-base` ships as an artefact with a *zero-shot* story ("answers a typed question
it was never trained on") or only as an *init* story, and whether the 27 small Jev-labelled tasks
stay in the published mixture and in the README sentence about diversity.

## 2. Arms

All arms are pair models (`openjev/base.py`, `AutoModelForSequenceClassification(num_labels=1)`,
BCE against the teacher's probability), trained on `data/pairs.jsonl` as rebuilt by
`scripts/build_pairs.py` at the commit of this file.

| arm | run dir | `base train` flags | seeds | what differs |
|---|---|---|---|---|
| F1 | `runs/base-F1-v2{,-s1,-s2}` | `--fold F1 --tag v2[-s1/-s2] --seed 0/1/2` | 0,1,2 | fold F1 held out (kinopoisk, civil_comments, swde sources) |
| F2 | `runs/base-F2-v2` | `--fold F2 --tag v2` | 0 | fold F2 held out (georeview, banking77, mind2web) |
| none | `runs/base-none-v2` | `--fold none --tag v2` | 0 | nothing held out (the shipped artefact) |
| ABL-orig | `runs/base-F1-v2-orig{,-s1,-s2}` | `--fold F1 --tag v2-orig[-s1/-s2] --only <9 dirs>` | 0,1,2 | fold F1 **and** the 27 new tasks dropped: the held-in original run dirs only |
| ABL-all | = the F1 arm | — | 0,1,2 | identical to ABL-orig except the 27 new tasks are in the mixture |

The diversity ablation is **ABL-all − ABL-orig**, same fold, same seeds, same everything else;
the F1 arm is reused as ABL-all rather than retrained (it is bit-for-bit the same configuration).

ABL-orig's mixture, named now so it cannot be chosen later (`--only`, i.e. the 13 original run
dirs in `data/pairs.jsonl` minus the four fold F1 holds out — 9 dirs, 8 distinct tasks; PLAN-4 M6
calls them "the 8 held-in originals"):
`agnews`, `arb-quality`, `arb-success`, `banking77`, `georeview`, `georeview-jev`, `headlines-8k`,
`m2w-element`, `m2w-target`. ABL-all adds the 22 remaining `-jev` run dirs (27 new tasks minus
the five dropped by fold F1: `civil-{insult,threat,obscene,identity}-jev`, `swde-vertical-jev`),
for 31 run dirs — the same 31 B1's `base-F1` trained on. (Fold `none` would be 13 vs 40; the
ablation is run on F1 only, because only there is a held-out task with a per-task student.)

**Convergence (the fix for B1's (a)).** No wall-clock budget. Per arm:

- **max epochs = 6** over the capped mixture (`cap = 12000` pairs per task per epoch; fold F1 =
  171,116 pairs/epoch = 5,347 steps at `batch = 32`, so the ceiling is ~32,000 steps ≈ 3 h at the
  measured 78–103 pairs/s). `--minutes` is not used.
- **Stop rule**: mean BCE on the held-in tasks' *calib* pairs (`calib_cap = 120` per task),
  evaluated every 1,500 steps; a checkpoint counts as an improvement only if it beats the best by
  **min_delta = 0.0010** BCE; **patience = 3** consecutive non-improving evaluations stops the run.
  The saved `student/` is the best-BCE checkpoint, not the last.
- A run that reaches 6 epochs without the patience rule firing is **not converged** → STOP rule §5.

Held fixed across all arms: teacher labels and their run dirs (nothing re-labelled), split ids,
`init = jhu-clsp/mmBERT-small`, `max_len = 512`, `lr = 5e-5`, `batch = 32`, `cap = 12000`,
`eval_every = 1500`, `calib_cap = 120`, and the calibration variant used for each readout.

Code the next job must add before running anything (and nothing else in `base.py`):
`--max-epochs`, `--patience`, `--min-delta`, `--only` on `openjev base train`, and the
`min_delta` / epoch-ceiling logic in `base.train`. The existing `only=` / `limit=` parameters
already exist; only the CLI surface and `min_delta` are new.

## 3. The metric

| | |
|---|---|
| name | the task's `metric_fn` (`openjev/evaluate.py`): **acc** for `choice`, **MAE** for `score`, **AUROC** for `noul` |
| numerator | acc: eval rows whose *calibrated* argmax over the K offered options equals gold. MAE: Σ over eval rows of \|Σ_k p_k·v_k − value(gold)\|. AUROC: correctly-ordered (positive, negative) eval-row pairs by calibrated p(true) |
| denominator | acc: eval rows carrying gold and scored by **both** arms. MAE: the same count. AUROC: all (positive, negative) pairs of those rows |
| unit | pts (acc, AUROC ×100) / MAE units (sst5: levels 1–5; georeview: stars) |
| matching rule | rows are paired by `id` across arms, variants and seeds; a row absent from either arm is dropped from both; seeds are pooled by averaging per-row calibrated probabilities, and the per-seed metric is also reported |
| read from | `results/base/*.json` written by `openjev base eval`, compared with `openjev compare A B` (paired bootstrap, 2000 resamples); tables via `scripts/base_table.py` |
| calibration | fitted on the task's **calib** rows only, all four variants (`raw`, `prior`, `teacher500`, `gold500`); the **`gold500` variant is the pre-registered primary** — it is what the per-task students get. The variant names are the code's; on the unseen sets both `gold500` and `teacher500` fit on that task's **200** calib rows, not 500 |

Two readouts, both primary:

1. **Zero-shot on genuinely unseen tasks** — `base-none-v2`, seed 0, on the eval splits of
   `tasks/unseen/{yahoo-topics,sst5,ru-inappropriate}.yaml` (n = 500 each), against `chance`
   (uniform) and against the Jev teacher on the same rows. The labels live in `runs/<task>-jev/`,
   so the command carries `--src runs/<task>-jev`.
2. **Diversity ablation** — Δ = ABL-all − ABL-orig on the fold-F1 held-out tasks
   `kinopoisk` (acc, n = 1500), `swde-field` (acc, n = 2000), `toxic` (AUROC, n = 3000), 3 seeds
   per arm. Secondary (reported, no claim): the same Δ on the three unseen sets.

Secondary readouts, diagnostic, never a claim: ECE raw/cal, NLL, Brier, agreement with the
teacher, the per-variant spread, the leave-one-source-out table for F2 / none, `best_calib_bce`
and the step at which early stopping fired.

## 4. What counts as a result

**Diversity ablation** (the claim "the 27 small tasks bought zero-shot transfer"). All three must hold:

- the pooled paired-bootstrap 95 % CI on Δ excludes 0, **and**
- |Δ| ≥ **2.0 pts** acc / **2.0 pts** AUROC / **0.04** MAE units — the improvement threshold, set
  here because a paired comparison of two same-family models on 1500–3000 rows resolves to about
  ±2.4 pts (B1's warm-start N = 500 row), so anything smaller is inside the noise this repo can
  see, and because below 2 pts no downstream decision about the mixture changes, **and**
- **all three per-seed Δ have the same sign** on that task. B1's only "win" (+4.8 pts at N = 100)
  was one seed out of three; sign agreement is what would have caught it.

…on **at least 2 of the 3** fold-F1 held-out tasks. Otherwise there is no diversity claim.

- **Non-inferiority** ("the 27 tasks did not hurt", which is what keeps them in the artefact):
  the lower bound of the 95 % CI on Δ is above **−2.0 pts** (−0.04 MAE) on all three tasks. Same
  margin as the improvement threshold, for the same reason: this repo cannot resolve less.
- **Inconclusive**: the CI contains 0 *and* `openjev compare`'s resolvable half-width exceeds
  2.0 pts (0.04 MAE). This is **not** a null result and must not be written as one; the sentence
  is "the comparison could not resolve a 2-pt difference at n = <n>, 3 seeds".

**Zero-shot on the unseen sets** (the claim "answers a typed question it was never trained on"):

- required on **all three** sets: the 95 % CI on (metric − chance) excludes 0, where chance is a
  uniform `[N, K]` (AUROC chance 0.5; MAE chance = the uniform-prediction MAE), **and**
- on **at least 2 of the 3**: the teacher-normalised score
  (metric − chance) / (teacher − chance) ≥ **0.5** — i.e. the model recovers at least half of what
  the teacher recovers over chance on the same rows (for MAE the sign is flipped:
  (chance − MAE) / (chance − teacher_MAE)).
- Anything less: the table is published as a negative/partial result, and neither "general" nor
  "any question" may appear next to it (PLAN-4 §6).

**Convergence** is a precondition, not a claim: each arm reports the step and epoch at which
early stopping fired and its `best_calib_bce`. No comparison may be drawn between a converged and
a non-converged arm.

## 5. STOP rules (any one voids the comparison)

- **Mechanism probe.** For a vLLM-labelled task in the mixture, `openjev check --probe` showing
  letter emission < 0.90 or candidate-mass median < 0.5. The three unseen sets are Jev-labelled,
  where there is no letter emission to probe: their equivalent guard is `label.json`'s
  `n_failed` > 1 % of the rows, or the teacher's answers collapsing onto one option on more than
  95 % of the **calib** split (checked on calib only — eval stays shut until the arms are run).
- `label.json`'s `model` (and, for vLLM-labelled tasks, `prompt_sha`) differs between the arms, or
  differs from what B1's mixture was built on. The Jev backend writes no `prompt_sha`, so for the
  `-jev` run dirs the guard is: the `question:` / `statement:` / `options:` / `rubric:` block of
  the task YAML is unchanged in git since the labels were written.
- Any run is resumed under a changed prompt (`openjev label` refuses; **do not `--force`**).
- **A fold or ablation model reaches 6 epochs without early stopping** — not converged, and the
  whole point of this round was convergence.
- The arms share fewer than **450** of the 500 unseen eval rows, or fewer than 1400 / 1900 / 2800
  rows on kinopoisk / swde-field / toxic.
- Any eval-fitted quantity leaks into an arm (all four calibration variants fit on calib only);
  or `openjev base eval` is made to score a task that is in the model's mixture (it refuses —
  do not work around it).
- Any of `tasks/unseen/*` appears in `data/pairs.jsonl` or in any arm's `train_tasks`.
- The mixture cap, `lr`, `batch`, `max_len` or the stop rule is tuned after seeing a held-out or
  unseen number. If one of them has to change, the arms are re-run from scratch and an amendment
  is appended below.
- Teacher spend for the unseen sets exceeds $0.50 (hard cap `OPENJEV_JEV_BUDGET`).

## 6. The pre-registered unseen sets

Three sets, none opened by anyone here before this file, each verified to load under
`datasets` 5.0.1 and to pass `openjev check` (exit 0, no warning). They exist to be **evaluated
on**: only their **calib (200)** and **eval (500)** splits are labelled — calib because the
`teacher500` / `gold500` calibration variants need it, eval because it is the readout. The
`train: n = 800` row stays in each YAML unlabelled so that the calib/eval ids never move if a
later experiment wants the train split.

| task | file | source | type / K | n cal/eval | note |
|---|---|---|---|---|---|
| `yahoo-topics` | `tasks/unseen/yahoo-topics.yaml` | `community-datasets/yahoo_answers_topics` | choice / 10 | 200 / 500 | PLAN-4 named `yahoo_answers_topics`; the canonical id is a **loading-script** dataset and raises under `datasets` 5.0.1, so the community Parquet mirror of the same data is used instead |
| `sst5` | `tasks/unseen/sst5.yaml` | `SetFit/sst5` | score / 5 | 200 / 500 | as planned |
| `ru-inappropriate` | `tasks/unseen/ru-inappropriate.yaml` | `ai-forever/inappropriateness-classification` | noul / 2 | 200 / 500 | PLAN-4 named `apanc/russian-inappropriate-messages`; **that repo no longer exists on the Hub** (`DatasetNotFoundError`). Replacement: the same Skoltech-lineage Russian inappropriateness data, with train/validation/test already split and a balanced test half |

None of the three shares a text source with any run dir in any mixture, so they are unseen in the
strong sense (unseen question *and* unseen source), which the leave-one-source-out folds are not.
They are named in `scripts/build_pairs.py`'s `SKIP` so a rebuild of `data/pairs.jsonl` cannot
sweep them into a mixture by accident; after adding them the rebuild is still 42 tasks /
1,183,247 pairs, byte-identical to B1's.

**Labelling record**, written before any of it was read as a metric (`openjev label
tasks/unseen/<t>.yaml --teacher jev --split calib|eval`, `OPENJEV_JEV_BUDGET=0.15`):

| task | run dir | rows | teacher model | failed | cost |
|---|---|---|---|---|---|
| `yahoo-topics` | `runs/yahoo-topics-jev/` | 700 (200 calib + 500 eval) | `typesafe/jev-1.13-20260917` | 0 | $0.010246 |
| `sst5` | `runs/sst5-jev/` | 700 (200 + 500) | same | 0 | $0.007251 |
| `ru-inappropriate` | `runs/ru-inappropriate-jev/` | 700 (200 + 500) | same | 0 | $0.007762 |
| | | 2,100 | | 0 | **$0.025259** |

## 7. Budget and cut order (preregistered, so a cut cannot become a claim)

8 trainings at ≤ 3 h = ≤ 24 h of GPU lane, expected ~12 h if early stopping fires around epoch 3.
If the lane runs short, cut in this order: (1) ABL-orig seeds 1 and 2 — which by §4 makes the
diversity ablation **inconclusive by rule**, never a win; (2) the `prior` and `teacher500`
calibration variants on the unseen sets; (3) `base-F2-v2`. **Never cut**: `base-none-v2`, the
three unseen sets at `gold500`, the convergence criterion, seed 0 of both ablation arms.

## 8. Amendments

Append only, each dated, each stating **what had been read when it was written** and what changed.

| date | what had been read | change | why |
|---|---|---|---|
