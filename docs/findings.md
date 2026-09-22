# Findings

What the runs on one teacher showed, what the columns of the [results table](../README.md#results)
mean, and where the method stops working. Every table, command and bootstrap behind a number here
is in [`experiments.md`](experiments.md).

Every Δ is a paired bootstrap (`openjev compare`, 2000 resamples) over the eval rows two arms
share; a Δ whose 95 % CI includes 0 is written as *no measurable difference*, never as a gain.
Training-side deltas are pooled over three seeds per arm; the epochs comparison is the one
single-seed exception and is labelled.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="img/calibration-dark.svg">
  <img alt="ECE before and after temperature scaling, every run" src="img/calibration-light.svg" width="900">
</picture>

## The calibration bias

**500 gold labels are worth more as a per-class bias than as training signal.** The calibration
step fits `logits / T + b`, `b ∈ R^K` (`method: vector`), on the 500-row calib split, accepted only
when it wins a 2-fold held-out NLL test against a plain temperature. CPU, seconds, no teacher
calls, no retraining, and the largest single win in the repo: headlines 0.767 → **0.843** (+7.5 pts,
95 % CI +6.0..+9.1), kinopoisk 0.609 → **0.660** (+5.1, +2.9..+7.5), georeview MAE 0.783 →
**0.609** (−0.17, −0.19..−0.16). Refit on all three seeds: headlines +7.0..+7.5, kinopoisk
+4.5..+5.1, every CI excluding 0 ([R1](experiments.md#r1--a-k-dim-bias-on-the-students-logits-recovers-part-of-the-teacher-prior-gap)).
On agnews (+0.95, CI −0.10..+1.95) and banking77 (+0.20, −1.35..+1.65) it does nothing. toxic is
excluded: its metric is AUROC, which a constant per-class shift cannot move, and its calib split has
the wrong prior (see [toxic](#what-the-numbers-mean) below).

**With the bias, two students beat their own teacher and two match it.** headlines 0.843 vs the
teacher's 0.783 (+6.0 pts, 95 % CI +4.3..+7.7) and headlines-8k 0.8515 vs 0.783 (+6.9, +5.2..+8.6)
are measurable wins; kinopoisk 0.660 vs 0.652 (+0.8, −1.8..+3.5) and georeview MAE 0.609 vs 0.619
(−0.0095, −0.027..+0.005) are *no measurable difference*. Before the bias all four were behind.
Nothing about the teacher changed; the bias only removes the marginal the teacher shifted.

**Why it works: the teacher's dominant error on the hard tasks is a marginal shift, not
per-example noise.** On kinopoisk (gold an even 33/33/33) the teacher answers *Good* for 56.5 % of
rows and *Neutral* for 12.7 %, and the student reproduces that faithfully. The fitted bias is
`[+0.80 Bad, +0.59 Neutral, −1.39 Good]` — it pushes down exactly the class the teacher
over-predicts and lifts the one it starves. The [option-order permutation
diagnostic](experiments.md#the-option-order-permutation-diagnostic-readout-and-decision) shows the
Bad↔Good split is positional but the *Neutral* collapse survives both orders and gets worse when
they are averaged: the shift is semantic, so averaging over permutations was not built — it would
double the teacher bill to fix what the post-hoc bias fixes for free. openjev labels each row once.

**The caveat that ships with it: you have to know your prior.** The no-gold variant — `prior:`
declared in the YAML, gold ignored entirely (`--ignore-gold`) — lands within half a point end to
end: kinopoisk 0.6553, georeview MAE 0.5986. That reads like "free debiasing with zero labels", and
it is not: `uniform` is the right prior on these benchmarks *because their eval sets are balanced by
construction*. A real deployment gets whatever prior it can actually declare, and a wrong
declaration moves the marginal to the wrong place. The honest claim: **if you know your class prior
you can have this without labelling anything; if you do not, spend 500 rows from your own
distribution and fit it.**

**Where to spend 500 gold labels, measured.** Same task, same seeds, same teacher labels, one
factor: 500 gold rows in the CE loss (`--gold-n 500 --gold-weight 1.0`) gives kinopoisk
0.603 / 0.622 / 0.623 over three seeds; the same 500 rows used *only* for the calibration bias gives
0.660 / 0.654 / 0.657. Calibration wins by **Δ = +4.1 ± 1.7 pts (95 % CI +2.4..+5.7)**. All 4000
gold labels in the loss reach 0.672 (one seed): 8× the labels buys 1.2 pts over the 500-row bias.
Hypothesis, not a claim: a few hundred hard labels fight 4000 soft ones inside the loss, while the
same rows spent post-hoc correct the one thing that is actually broken. Two points on one task, not
an N-labels curve.

**More teacher labels help, a little.** headlines at 4000 vs 8000 teacher-labeled train rows, three
seeds each, synth-free, identical eval ids: **+1.3 pts (95 % CI +0.6..+2.1)**, 0.838 → 0.851.
Doubling the teacher bill buys about a sixth of what the 500-row bias buys on the same task.

## The cascade

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="img/cascade-dark.svg">
  <img alt="Escalation rate vs accuracy for agnews, headlines and kinopoisk" src="img/cascade-light.svg" width="900">
</picture>

`eval` sweeps "student answers when confidence ≥ τ, else the row goes to the teacher" over
probabilities already on disk — zero new teacher calls — and writes the parity point into
`openjev.json` as `escalate_below`. **Four of the nine runs reach teacher parity at 0 %
escalation**: after the bias the student already matches or beats its teacher, so the cheapest
cascade is "never escalate" (on headlines escalating actively *costs* accuracy, all the way down to
the teacher's 0.783). Where the student is behind, escalating the least confident rows pays:
banking77 0.751 → 0.776 at 27 %, kinopoisk 0.660 → 0.673 at 9 %, agnews 0.889 → 0.897 at 10 %,
georeview MAE 0.609 → 0.583 at 26 %. What this may **not** claim is any gain in *quality*:
escalation routes to the same zero-shot teacher that produced the training labels, scored on the
rows it labelled. `serve` returns `"escalate": true/false` and never calls the teacher itself.

## Two knobs that did not matter

Both were believed to work before they were measured with a control arm.

- **Epochs: 5 vs 12 is no measurable difference** (one seed per arm; kept because it decides the
  default). agnews Δ +0.0 ± 0.6 pts, headlines +0.7 ± 0.9, kinopoisk +0.0 ± 1.5 — all three CIs
  include 0, and the sign on headlines favours the *shorter* run. All three 12-epoch runs early-stop
  (patience 2) at 9, 7 and 5 epochs, so what `--epochs` really changes is the LR-decay horizon.
  **The default stays 5.**
- **The augment (PGKD-lite) round on headlines is no measurable difference.** Three seeds per arm,
  the only factor whether the 522 synthetic rows are in the training set: **Δ = +0.0 ± 0.5 pts
  (95 % CI −0.5..+0.6)**. Seed 0 showed +0.8 pts and seed 1 −0.9. An earlier README claimed a
  +0.5 pt gain from the round; that was one seed, exactly the size of this noise, and it is
  withdrawn. This does not show PGKD fails — one task, one round, 522 rows against 4000 — only that
  nothing here measured it working.

**The noise floor every claim is measured against.** Three seeds, identical data and labels, only
the head init and batch order changing, move accuracy 0.6–1.2 pts peak to peak: agnews
0.888 ± 0.006, headlines 0.838 ± 0.005, kinopoisk 0.657 ± 0.003. One seed is not a measurement,
which is why `compare` exists and why `report` prints an sd column.

## The rest of the table

**The student ranks better than its teacher on toxic** — AUROC 0.856 vs 0.817, Brier against the
annotator fraction 0.035 vs 0.048. Ignore the accuracy column there: the eval split is 8.1 %
positive, so always answering "not toxic" scores 0.919, above both. The win is not free: toxic is
the one task whose training rows were drawn 50/50 *by gold* (`balance: true`), so gold paid for the
row selection even though every label is the teacher's. A plausible mechanism is that 4000 soft
labels over a balanced sample average out the teacher's per-example noise; it is one task and it
did not generalise to any other here.

**banking77 is where the teacher cost shows up.** 77 classes go through the chunked shortlist at 6
teacher calls per example: 33,000 calls for 5,500 rows, 61 minutes, 42 % of the night's teacher
budget for one task. The student lands at 0.751 against the teacher's 0.764 (agreement 0.766) and
the bias does not move it. This is the one task where escalation is the tool that helps (0.776 at
27 %).

**Model size is not the bottleneck at either end.** mmBERT-base (307M vs 140M) buys agnews 0.891
vs 0.889 and georeview MAE 0.594 vs 0.609, for half the batch-64 throughput (435 vs 869 ex/s on
agnews; 115 vs 203 on georeview), 9.8–9.9 GB of training memory instead of 5.5, and 7.4 minutes of
training instead of 1–3.5 on agnews (on georeview both took ~10 minutes on a busy box, not a clean
number). On the easy task the small student had already reached the teacher; on the hard one the
teacher is what was wrong, and a bias on 500 rows fixed more of it than 2.2× the parameters did.

**A task with no gold at all gets the same student.** `tasks/agnews-nogold.yaml` is
`tasks/agnews.yaml` with the `data.gold` line deleted. Read its table row carefully: with no gold,
"teacher acc" is the teacher scored against itself (1.000 by construction) and "student acc" 0.944
is the agreement column under another name. Scored offline against the real ag_news test labels on
the same 2000 rows: **0.8820** against the teacher's **0.8815**, within noise of the gold-calibrated
run. Removing gold cost nothing measurable *here*, on a task where the teacher is strong and already
well calibrated; on the hard tasks the gold-free path is the `prior:` one above. (`gold_acc_offline`
lives in `results/variants/agnews-nogold-goldacc.json`, from `scripts/nogold_gold_acc.py`, because
`openjev eval` rewrites `results/agnews-nogold.json` and would drop a key it did not write.)

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="img/cost-dark.svg">
  <img alt="Teacher labeling cost paid once vs student inference cost per call" src="img/cost-light.svg" width="900">
</picture>

**Teacher cost.** 66,522 teacher calls and 144 minutes (2.4 h) of labeling wall-clock across the
six tasks, of which banking77 alone is 61 minutes; headlines-8k added 4,434 rows in 9 more. Student
training is 1.5–12 minutes per task, and every post-hoc result above is CPU seconds on logits
already on disk. The teacher is the budget; everything else is rounding.

## A better teacher: Jev's labels against our Qwen teacher's

`openjev label --teacher jev` labels the same rows through TypeSafe's hosted Jev, so the two
teachers can be scored on identical eval rows against the same gold (`results/api/jev-teacher.json`,
5800 rows, $0.156):

| task | n | metric | Jev teacher | Qwen teacher | our Qwen student |
|---|---|---|---|---|---|
| agnews | 2000 | acc ↑ | **0.897** | 0.880 | 0.889 |
| kinopoisk | 1500 | acc ↑ | **0.694** | 0.652 | 0.660 |
| georeview | 2000 | mae ↓ | **0.502** | 0.619 | 0.609 |
| arb-success | 300 | auroc ↑ | 0.822 | 0.837 | **0.843** |

Jev beats the local teacher on argmax everywhere and beats the students on the two Russian tasks.
On agent-run success it fixes the teacher's degenerate marginal (macro-F1 0.630 vs 0.426) but ranks
no better. Its probabilities are spiky: 71 % of agnews rows come back as exactly 1.0 on one option
and 0 on the rest (kinopoisk 61 %, georeview 18 %, arb-success 0 %).

**Then the teacher was swapped inside the pipeline** on the two tasks where the gap was real. Both
arms share everything — task YAML, split ids, texts, eval rows, student checkpoint, `max_len`,
`lr`, `batch_size`, epochs, early stopping, calibration rule — except who produced the training
labels. Three seeds per arm. Labeling the two remaining splits cost **$0.1966** for 9000 rows at
23 ex/s with 0 failures ($0.0617 georeview, $0.1348 kinopoisk; $0.313 including the eval rows the
head-to-head already paid for); training was 8 queue jobs of 5–7 minutes.

| task | metric | n | Qwen teacher | Qwen student | Jev teacher | Jev student |
|---|---|---|---|---|---|---|
| georeview | mae ↓ | 2000 | 0.619 | 0.607 ± 0.003 | **0.502** | 0.555 ± 0.002 |
| kinopoisk | acc ↑ | 1500 | 0.652 | 0.657 ± 0.003 | **0.694** | 0.659 ± 0.003 |

Students are mean ± sd over three seeds; teachers are their own `teacher.jsonl` eval rows
(`results/api/jev-student.json`, from `scripts/jev_student_report.py`).

**Only one of the two teacher wins survives distillation.** georeview: **Δ MAE = −0.052 ± 0.010
(95 % CI −0.062..−0.042)**, consistent across seeds (−0.056, −0.048, −0.051). kinopoisk: **no
measurable difference**, Δ = +0.2 ± 1.2 pts (95 % CI −1.0..+1.4), seeds straddling zero (−0.13,
+0.27, +0.53). The teacher was 4.2 points better and none of it reached the student.

**Why: the 500-gold bias had already bought what the better teacher offers.** Before calibration
the Jev student wins on *both* tasks — georeview MAE −0.111 (CI −0.135..−0.085), kinopoisk
**+4.7 pts** (CI +3.0..+6.3), the teacher gap almost exactly. The vector fit then lifts the Qwen
student by 4.9 pts on kinopoisk and the Jev student by 0.5, and they meet:

| task | arm | uncalibrated | calibrated |
|---|---|---|---|
| georeview (mae ↓) | Qwen student | 0.674 | 0.607 |
| georeview (mae ↓) | Jev student | 0.564 | 0.555 |
| kinopoisk (acc ↑) | Qwen student | 0.608 | 0.657 |
| kinopoisk (acc ↑) | Jev student | 0.654 | 0.659 |

On kinopoisk both teachers fail the same way — both starve *Neutral* (Qwen 12.7 %, Jev 7.7 %,
gold 33.3 %) and split the rest between *Bad* and *Good* (Qwen 30.7/56.5, Jev 43.4/48.9). That is
a marginal shift, and a per-class bias removes a marginal shift whoever caused it. georeview is the
case where the better teacher is better *per example*: its MAE gap is 0.117 and about 45 % of it
(0.052) survives into the student. There the money buys something the bias cannot.

**Near-hard labels change what calibration has to do, not which method it picks.** Jev's
probabilities are rounded to 0.01 and often a single spike: on the training splits 17.6 % of
georeview rows and **60.3 % of kinopoisk rows** carry a max probability above 0.999, against 0.0 %
for both Qwen splits (mean label entropy 0.38 vs 0.75 and 0.17 vs 0.49). All six arms still
selected `method: vector` and passed the 2-fold test. What moved is the temperature: the Jev
students come out badly overconfident and need roughly twice the temperature — georeview
T = 1.78/2.06/2.03 against 1.09/1.09/1.09, kinopoisk 2.19/2.26/2.20 against 1.21/1.28/1.20.
Calibrated ECE lands in the same place either way (georeview 0.029–0.057 vs 0.035–0.062; kinopoisk
0.042–0.054 vs 0.041–0.060), so the spikiness cost nothing here — only because a calibration step
was there to undo it.

**Near-hard targets are measurably harder to fit, and it did not hurt.** The KL floor is roughly
double: best calib KL 0.32 vs 0.13 (georeview) and 0.22 vs 0.11 (kinopoisk), training-loss floor
0.047 vs 0.021 and 0.037 vs 0.018; georeview's Jev arm early-stopped at epoch 2 instead of 4.
Argmax agreement with its own teacher is unchanged (georeview 0.647 vs 0.649, kinopoisk 0.702 vs
0.693) while mean KL to the teacher is higher (0.49 vs 0.33, 0.48 vs 0.39): the student copies the
spiky teacher's *decisions* as faithfully and its *distribution* less so.

**What limits it.** Jev is a closed hosted model, so nothing here explains *why* it is better and
the result is not reproducible from weights; its probabilities are quantised to 0.01, a property of
the API and not of the model; and two tasks, both Russian, both ordinal-ish, is two tasks. The one
arm where Jev ranks *worse* than our student (arb-success) was not distilled, so nothing here says
what a worse-but-softer teacher does to a student.

## One pair scorer over 40 tasks

`open-jev-base` drops the fixed K-way head for a pointwise cross-encoder over
`(question, option, text)`, trained on the pooled soft labels of 40 run dirs; K and the option
wording are then inputs, not architecture. The shipped artefact is `base-none-v2`
([sshalimov04/open-jev-base](https://huggingface.co/sshalimov04/open-jev-base)). It was
preregistered before it was run ([`prereg/base-v2.md`](prereg/base-v2.md)); the arms, the
convergence log and every table are
[B2](experiments.md#b2--open-jev-base-converged-one-unseen-question-claim-survives-the-diversity-ablation-is-voided-by-its-own-stop-rule).

**On three questions and three text sources it never saw, it beats chance and recovers part of its
teacher's margin.** yahoo-topics acc 0.528 against a 0.088 chance floor and a 0.718 teacher
(advantage over chance +0.440, 95 % CI +0.386..+0.492); sst5 MAE 0.775 against chance 1.152 and
teacher 0.493 (+0.377, +0.324..+0.435); ru-inappropriate AUROC 0.619 against 0.500 and 0.891
(+0.119, +0.072..+0.168). Teacher-normalised that is **0.70 / 0.57 / 0.30**, with 200 gold
calibration rows per task — the preregistered rule (CI excludes 0 on all three, normalised ≥ 0.5 on
2 of 3) is met. The rule's own logic bans the words *general* and *any question*: it passed 2 of 3
at the primary variant, its weakest set is the safety-shaped one, and three sets on one seed do not
make a claim about questions in general.

**The counterweights are the same size as the claim.** It loses to every per-task student except the
collapsed 16-way `m2w-element` head — kinopoisk −13.8 pts, banking77 −37.8, swde-field −55.2,
georeview +0.161 MAE, toxic −0.027 AUROC, every CI excluding 0. Training it to the minimum of
held-in calib BCE made transfer *worse* on 5 of those 6 tasks than the 45-minute models that
preceded it: converging overfits the mixture's own tasks at the expense of a held-out one. And the
zero-shot result leans on its gold calibration: under `teacher500` (no gold, temperature fitted to
the teacher's soft probabilities) sst5 falls to 0.48 and the rule would have been met on 1 of 3.
Raw, its renormalised sigmoids are also unusable at K = 77 (ECE 0.307 on banking77, repaired
post-hoc to 0.055 by the 500-gold vector fit).

### The ablation that passed and does not count

The round's other question was whether the 27 small Jev-labelled tasks in the mixture help at all.
The ablation answered *yes* on 3 of 3 held-out tasks — kinopoisk +2.2 pts, swde-field +10.3, toxic
+14.9 AUROC, every CI excluding 0, every per-seed sign agreeing, all three parts of the
preregistered threshold met — and **it is void**, because 2 of the 3 control-arm seeds hit the
6-epoch ceiling with their loss still falling and the preregistration forbids comparing a converged
arm with a non-converged one. An under-trained control biases the difference upward, which is
exactly the direction these numbers point; the per-seed spread (swde-field +0.005 / +0.055 / +0.248)
says the same thing. Written after the fact, this is the paragraph where a favourable result gets
kept "with a caveat". Written before it, the STOP rule simply deletes it: **the 27 tasks are neither
shown to help nor shown not to**, the README may claim nothing from them, and the fix is a rerun of
the control arm under a matched step budget — an amendment, not a re-analysis of these files. That
is the whole return on the paperwork, and it is the best argument for preregistration this repo has.

## What the numbers mean

- **The teacher** for every row is Qwen3.8-27B zero-shot (README, Requirements). Runs from before
  2026-09-20 do not record the model id in `runs/<task>/label.json`; newer ones do.
- **The student sees zero gold labels.** `gold_weight` is 0 by default, so training uses only the
  teacher's distributions. Gold is used for the eval and, when present, for the calibration fit.
- **toxic is the exception, twice.** It sets `balance: true` on its training split, which picks
  50/50 rows *by gold* from an 8 %-positive dataset: the labels are still the teacher's, but the row
  selection spent gold. That balanced draw also came out of the same 60k-row pool as calib, and it
  went first, so toxic's calib split ended up **11/500 = 2.2 % positive against 8.1 % on eval** —
  T = 0.25 was fitted on 11 positives. It still cut eval ECE 0.116 → 0.037, but it is a temperature
  fitted under the wrong prior on very few rows, not a clean result, and it is why toxic is
  excluded from the calibration comparison. The sampler now draws natural-rate splits before any
  balanced one; the numbers in the table predate that fix.
- **The baseline is the teacher, not the state of the art.** The claim being tested is "a 140M
  encoder can keep the teacher's accuracy at a fraction of the cost", not "this beats a supervised
  model".
- **ECE**: max-probability bucketed into 15 equal-width bins, mean over bins of |mean confidence −
  accuracy|, weighted by bin mass; `raw→cal` is before and after calibration (`calib.json` says
  whether a bias was fitted). It is a summary, never the headline: a bias that fixes a shifted
  marginal can raise ECE on the calib split while raising accuracy by 7 points.
- **Brier** is the multiclass sum-of-squares against one-hot gold, on calibrated probabilities.
- **agree** is argmax agreement between student and teacher on the eval split — the metric that
  matters when a task has no gold.
- **Latency** is batch-1 on the GB10 GPU, p50 of 200 requests after 20 warmups, measured with vLLM
  resident but **idle**, so rows are comparable to each other; a box without the teacher loaded
  would be a little faster. `ex/s` is batch-64 throughput on the same hardware. `results/*.json`
  also carries a CPU batch-1 p50.
- **Distillation cannot beat its teacher's *systematic* errors.** Where the teacher is consistently
  wrong the student inherits it, and the agreement column says how much. Independent per-example
  noise is the one thing averaging can wash out — that is the most it did on toxic.

## Limitations and follow-ups

- **The calibration bias needs a prior you actually know.** `method: vector` fits a per-class bias
  on ~500 gold rows *from the deployment distribution*; the no-gold path needs a `prior:` you
  declare. Every benchmark here is balanced by construction, so `uniform` happens to be correct — a
  property of the benchmarks, not of the method. Fewer than ~500 calib rows makes the fit noise, and
  rows drawn at the wrong rate fit the wrong prior (toxic, above).
- **A per-class bias cannot fix a per-example error.** On agnews and banking77, where the teacher's
  marginal is already right, it buys nothing, and on a two-class AUROC task (toxic) it cannot move
  the metric at all.
- **The cascade is measured against the teacher that made the labels.** It is a cost/latency
  trade-off, not evidence of any gain in answer quality, and it says nothing about escalating to a
  *better* model.
- **Option position bias is measured, not fixed.** Reversing the option order moves the teacher's
  accuracy and its Bad↔Good split on kinopoisk, but not the *Neutral* collapse, and headlines goes
  the other way; picking the better order would itself need gold. The numbers and the decision rule
  are in [`experiments.md`](experiments.md#the-option-order-permutation-diagnostic-readout-and-decision).
- **K > 19 is approximate.** `top_logprobs` caps at 20, so large label sets go through a chunked
  shortlist: softness is exact only within the shortlist, and if every chunk misses the true class
  the label is simply wrong. Cost is `ceil(K/19) + 1` calls per example. Measured
  (`teacher.shortlist_miss` in `results/*.json`, printed by `openjev check --probe`): on banking77
  the gold option is outside the final shortlist on **4.6 % of eval rows** (272 / 5500 = 4.9 % over
  all splits), which caps the teacher at 0.954 before it answers; on swde-field it is 0 / 2000. The
  chunk that holds gold answers `Z: none of the above` with median p 0.02, the chunks that do not
  with median 0.92 — the `none` letter does its job; the misses are chunks that were confidently
  wrong.
- **No ONNX / quantized export.** `serve` runs the PyTorch model; the export bundle is `student/` +
  `openjev.json` (+ `conformal.json`), which is what `openjev push` uploads.
- **No recalibration under drift.** One `T` and one `b` per task, fitted once. Nothing detects that
  the deployment prior has moved, which is exactly the failure mode the bias is sensitive to.
- **Calibration target depends on the data.** With gold on the calib split the fit targets gold;
  without it, the temperature targets the teacher's *soft* probabilities and the bias targets the
  declared `prior:` — the student is calibrated to the teacher's opinion plus your belief, not to
  the truth. `calib_target` in `results/*.json` says which path ran.
