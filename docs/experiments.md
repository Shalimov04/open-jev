# Experiments

Everything here is reproducible from the repo: every row names the command that produced it and the
`results/*.json` or `runs/*/` file it was read from. Claim rule (PLAN-2 §5): a delta is an
*improvement* only when the paired bootstrap 95 % CI excludes 0; otherwise the sentence is "no
measurable difference". Anything fitted on eval is labelled **diagnostic** and is never a claim.

From now on a comparison enters `findings.md` only if it has a preregistration file
(`docs/prereg-template.md` → `docs/prereg/<id>.md`) whose commit predates the arms' `label.json` /
`train.json` timestamps, and every amendment to it states what had been read when it was written.
Not retrofitted: the sections below were written before the rule.

What is held fixed in every comparison below: the teacher labels (`runs/<task>/teacher.jsonl` — the
same ids and the same probabilities as the original runs, nothing was re-labelled), the split ids,
the student checkpoint (`jhu-clsp/mmBERT-small`), `max_len`, `lr`, `batch_size`, and the calibration
method-selection rule. Exactly one factor varies per comparison.

## R1 — a K-dim bias on the student's logits recovers part of the teacher-prior gap

### Where the gap comes from

The teacher's dominant error on the Russian and Russian-adjacent tasks is a *marginal* shift, not a
per-example one. The M0 permutation diagnostic (`runs/*-rev/perm_check.json`, reversing the option
order and re-labelling calib+eval through the same teacher) separates the two:

| task | class | gold | teacher, original order | teacher, reversed | averaged |
|---|---|---|---|---|---|
| kinopoisk | Bad | 0.330 | 0.308 | 0.462 | 0.407 |
| kinopoisk | Neutral | 0.337 | 0.129 | 0.120 | 0.090 |
| kinopoisk | Good | 0.334 | 0.564 | 0.419 | 0.504 |
| headlines | политика | 0.162 | 0.241 | 0.310 | 0.275 |
| headlines | наука | 0.161 | 0.067 | 0.087 | 0.084 |

Teacher accuracy: kinopoisk 0.646 original / 0.693 reversed / 0.693 averaged; headlines 0.784 /
0.768 / 0.791. The Bad↔Good split on kinopoisk moves with the option order, so *that* part is
positional. The Neutral collapse does not — it is 12.9 % against a gold 33.7 % in one order and
12.0 % in the other, and averaging the two orders makes it *worse* (9.0 %). headlines is the same
shape: политика is over-predicted and наука under-predicted in both orders. The student is
distilled from these probabilities and faithfully inherits the shift.

Two more readouts from the same two runs, no new teacher calls (`python scripts/perm_check.py
kinopoisk`): **500 / 2000 kinopoisk rows (25.0 %) and 297 / 2500 headlines rows (11.9 %) change
their argmax when the option list is reversed** — far from a near-tie effect (mini-jev measures
3 / 50 on 15-token English utterances at k = 4). Accuracy per class, scored on the same rows in
both orders, with the position that class occupied:

| task | class | position → acc, original | position → acc, reversed |
|---|---|---|---|
| kinopoisk | Bad | 0 → 0.750 | 2 → 0.954 |
| kinopoisk | Neutral | 1 → 0.209 | 1 → 0.242 |
| kinopoisk | Good | 2 → 0.985 | 0 → 0.889 |
| headlines | наука | 3 → 0.414 | 2 → 0.526 |
| headlines | экономика | 5 → 0.780 | 0 → 0.626 |

Bad and Good swing by 10–20 pts when their letter moves, which is positional; Neutral sits at the
middle position in *both* orders and stays at 0.21 / 0.24, which is not. A chosen-position
histogram cannot tell those two apart — the per-class readout can.

A shift in the predicted marginal is exactly what a constant per-class bias on the logits removes,
which is why vector scaling (`logits / T + b`, `b ∈ R^K`) is the right post-hoc tool here and an
ordinal head or a bigger student is not.

### Method

Four calibrations, all fitted on the **calib** split only, all applied to the same stored eval
logits (`runs/<run>/eval_rows.jsonl`, written by `openjev eval`; no re-inference):

- **T-only** — `fit_temperature` on the 500 gold calib rows. This is what the runs shipped before M1.
- **vector** — `fit_vector` (LBFGS over `log T` and `b ∈ R^K`, CE loss) on the same 500 rows.
  Selection guard: the calib rows are split 2-fold, each half is fitted and scored by NLL on the
  other, and `vector` is accepted only if it wins on **both** held-out halves. Otherwise the run
  ships `temperature`. `calib.json` records `method` and `bias`, and `calibrate.apply(logits, calib)`
  is the single place calibration is applied — `calibrate` and `evaluate` call it, and
  `serve` reads the same `calib` block from `runs/<run>/openjev.json`.
- **prior-declared** — **no gold at all**: `T` is fitted to the teacher's *soft* labels (what the
  student was trained on), then `b` is fitted so that the mean calibrated probability over the calib
  rows equals a prior the user declares (`prior: uniform` here), by iterating
  `b_k += log(prior_k / mean_p_k)` 50 times. `calib_target: prior-declared`.
- **oracle (diagnostic, fitted on eval)** — the same fit with the *eval gold marginal* as the
  target. It says how much of the gap is marginal at all. It is not a bound on accuracy: matching
  the marginal is not the accuracy-optimal bias, so a declared prior can land slightly above it.
  These numbers never enter the README.

| run | metric | teacher | raw | T-only | vector | prior-declared (no gold) | oracle (fitted on eval) | 2-fold verdict | Δ vector (95 % CI) | Δ prior-declared (95 % CI) |
|---|---|---|---|---|---|---|---|---|---|---|
| agnews | acc | 0.880 | 0.879 | 0.879 | 0.889 | 0.895 | 0.890 | vector | +0.95 (-0.10..+1.95) | +1.55 (+0.55..+2.50) * |
| agnews-mmBERT-base | acc | 0.880 | 0.883 | 0.883 | 0.891 | 0.894 | 0.894 | vector | +0.80 (-0.30..+1.90) | +1.05 (+0.00..+2.10) |
| agnews-nogold | acc | 1.000 | 0.944 | – | – | 0.924 | – | – (no gold) | – | -2.00 (-3.00..-1.00) * |
| agnews-s1 | acc | 0.880 | 0.882 | 0.882 | 0.893 | 0.898 | 0.896 | vector | +1.05 (+0.05..+2.05) * | +1.60 (+0.65..+2.55) * |
| agnews-s2 | acc | 0.880 | 0.875 | 0.875 | 0.881 | 0.885 | 0.885 | vector | +0.65 (-0.35..+1.70) | +1.05 (+0.10..+2.00) * |
| banking77 | acc | 0.764 | 0.749 | 0.749 | 0.750 | 0.755 | 0.786 | vector | +0.20 (-1.35..+1.65) | +0.70 (-0.70..+2.15) |
| georeview | mae | 0.619 | 0.675 | 0.783 | 0.609 | 0.599 | 0.599 | vector | -0.17 (-0.19..-0.16) * | -0.18 (-0.20..-0.17) * |
| georeview-mmBERT-base | mae | 0.619 | 0.664 | 0.771 | 0.594 | 0.583 | 0.585 | vector | -0.18 (-0.19..-0.16) * | -0.19 (-0.20..-0.17) * |
| headlines | acc | 0.783 | 0.767 | 0.767 | 0.843 | 0.842 | 0.838 | vector | +7.50 (+5.95..+9.05) * | +7.45 (+5.95..+8.90) * |
| headlines-s1 | acc | 0.783 | 0.763 | 0.763 | 0.833 | 0.836 | 0.832 | vector | +7.00 (+5.55..+8.50) * | +7.30 (+5.75..+8.75) * |
| headlines-s2 | acc | 0.783 | 0.764 | 0.764 | 0.839 | 0.840 | 0.838 | vector | +7.40 (+5.90..+8.85) * | +7.55 (+6.10..+8.95) * |
| kinopoisk | acc | 0.652 | 0.609 | 0.609 | 0.660 | 0.655 | 0.659 | vector | +5.13 (+2.87..+7.47) * | +4.67 (+2.33..+7.07) * |
| kinopoisk-s1 | acc | 0.652 | 0.609 | 0.609 | 0.654 | 0.657 | 0.662 | vector | +4.53 (+2.33..+6.67) * | +4.87 (+2.53..+7.13) * |
| kinopoisk-s2 | acc | 0.652 | 0.605 | 0.605 | 0.657 | 0.663 | 0.662 | vector | +5.13 (+2.80..+7.53) * | +5.73 (+3.27..+8.27) * |
| toxic | auroc | 0.817 | 0.856 | 0.856 | 0.856 | 0.856 | 0.856 | temperature | +0.00 (-0.00..+0.00) | +0.00 (-0.00..+0.00) |

Reading the table:

- The bias is doing exactly what the permutation diagnostic predicted. kinopoisk's fitted bias is
  `[+0.80 Bad, +0.59 Neutral, −1.39 Good]` (`runs/kinopoisk/calib.json`) — it pushes down the class
  the teacher over-predicts and lifts the one it collapses, and accuracy moves 0.609 → 0.660.
- **Where it works**: headlines +7.5 pts, kinopoisk +5.1 pts, georeview −0.17 MAE, all with a 95 %
  CI excluding 0. These are the three tasks whose teacher marginal is visibly shifted.
- **Where it does nothing**: banking77 (+0.2 pts, CI spans 0) and toxic. The 2-fold guard *accepted*
  vector on banking77 — the plan expected it to refuse at K=77 — but accepting it buys nothing, so
  the guard was not the binding constraint there; the bias simply has no prior shift to remove.
- **toxic's Δ is 0.000 by construction, not by measurement.** Its metric is AUROC, and AUROC ranks by
  `p(true)`, which a constant per-class bias transforms monotonically. A K-dim bias can never move a
  two-class AUROC. Its accuracy/ECE would move, but toxic is excluded from R1 anyway (below).
- **agnews-nogold's row is not an accuracy row.** That task declares no `data.gold`, so eval scores
  the student against the *teacher's* argmax — which is why the "teacher" column reads 1.000. The
  −2.0 pts there is the prior bias moving the student *away* from the teacher it is measured
  against, not away from truth. `scripts/nogold_gold_acc.py` is the only honest scorer for that run.
- **Temperature alone makes georeview worse** (MAE 0.675 raw → 0.783). For a `score` task the metric
  is the MAE of the *expectation*, and a T > 1 flattens the distribution toward the middle level. The
  bias is what actually helps there. This is an argument for reporting the task's own metric rather
  than ECE.
- **ECE and accuracy come apart under vector scaling.** The pre-M1 guard ("never let calibration
  raise ECE on the calib split") was written for temperature, which cannot change an argmax. Applied
  to a bias it vetoed a +7 pt accuracy gain on headlines seed 1 because calib-split ECE moved
  0.038 → 0.051. The guard now covers the temperature path only; a `vector` fit is gated by the
  2-fold held-out NLL test instead, which is the stricter of the two. ECE is reported, never the
  headline.

### End-to-end, with the gold deliberately ignored

The `prior-declared` column above is refitted offline. To check that the *pipeline* produces the
same number when it never sees a gold label, `scripts/prior_variant.py` makes a `-prior` run dir
(student and teacher labels symlinked to the base run — nothing is retrained) and calibrates with
`ignore_gold=True`:

| run | calib_target | metric, end-to-end | same, refitted offline |
|---|---|---|---|
| kinopoisk-prior | prior-declared | acc 0.6553 | 0.655 |
| georeview-prior | prior-declared | MAE 0.5986 | 0.599 |

Two independent code paths, same answer. `results/variants/{kinopoisk,georeview}-prior.json`.

### What may be claimed, and what may not

- The Δ columns are paired bootstraps over eval rows at one seed per run. They are an honest
  statement about *these students on these eval rows*: "500 gold calib rows spent on a K-dim bias
  move headlines accuracy by Δ". They say nothing about the teacher's ability.
- **The declared prior is a leak on these benchmarks.** `uniform` is right for agnews, headlines,
  kinopoisk, georeview and banking77 *because those eval sets are balanced by construction* — the
  gold marginals are 0.24–0.26, 0.16–0.18, 0.333 and 0.199–0.202. A real deployment does not get
  that for free; it gets whatever prior the user can actually declare, and a wrong declaration moves
  the marginal to the wrong place. The honest reading is "if you know your class prior, you can have
  this without labelling anything", not "no-gold calibration is free".
- **toxic is excluded from R1.** Its calib split is 2.2 % positive and its eval split 8.1 % (the
  depleted-pool bug, fixed in code after that run was made). A bias fitted on that calib split
  calibrates to the wrong prior. Re-drawing and re-labelling its calib split (500 rows, ~1.2 min of
  teacher) is what would make it comparable.
- Each Δ above is a paired bootstrap within one run. For the three tasks that have seed arms the
  refit is in the table for **all three seeds**: headlines +7.50 / +7.00 / +7.40, kinopoisk
  +5.13 / +4.53 / +5.13 — every CI excludes 0 — and agnews +0.95 / +1.05 / +0.65, inside noise on
  two of three. That is what backs the calibration claim in the findings in `findings.md`; the single-seed
  rows (banking77, georeview, toxic, the `-mmBERT-base` arms) stay `docs/` numbers under PLAN-2 §5.
- `-mmBERT-base` rows are a different student and are not part of any comparison; they are here only
  to show the effect is not an artifact of the small model.

## R2 — cascade / abstention curve

The student answers when its confidence (max calibrated probability) is ≥ τ, otherwise the row is
escalated to the teacher. τ sweeps 0.30 → 1.00 in steps of 0.05. This is a pure offline sweep over
rows we already hold both probability matrices for, so it costs **zero** new teacher calls; an
escalated row takes the teacher's probabilities, which for accuracy is exactly "the teacher's
argmax". The curve and the parity point (the lowest escalation rate whose metric is within 0.005 of
the teacher's) are stored in `results/<run>.json` as `cascade` and `cascade_parity`, and the parity
τ is exported to `runs/<run>/openjev.json` as `escalate_below` for `serve`.

| run | metric | student alone | 10 % escalated | 25 % | 50 % | teacher (100 %) | parity: escalation | tau |
|---|---|---|---|---|---|---|---|---|
| agnews | acc | 0.889 | 0.897 | 0.885 | 0.880 | 0.880 | 0.0% | 0.3 |
| agnews-mmBERT-base | acc | 0.891 | 0.897 | 0.888 | 0.880 | 0.880 | 0.0% | 0.3 |
| agnews-nogold | acc | 0.944 | 0.983 | 0.994 | 1.000 | 1.000 | 36.9% | 0.8 |
| banking77 | acc | 0.751 | 0.771 | 0.776 | 0.776 | 0.764 | 6.9% | 0.35 |
| georeview | mae | 0.609 | 0.594 | 0.583 | 0.574 | 0.619 | 0.9% | 0.3 |
| georeview-mmBERT-base | mae | 0.594 | 0.580 | 0.570 | 0.571 | 0.619 | 1.0% | 0.3 |
| headlines | acc | 0.842 | 0.840 | 0.814 | 0.792 | 0.783 | 1.0% | 0.3 |
| kinopoisk | acc | 0.660 | 0.671 | 0.673 | 0.659 | 0.652 | 0.0% | 0.3 |
| toxic | auroc | 0.856 | 0.857 | 0.817 | 0.817 | 0.817 | 0.0% | 0.3 |
| kinopoisk-prior | acc | 0.655 | 0.673 | 0.671 | 0.666 | 0.652 | 0.0% | 0.3 |
| georeview-prior | mae | 0.599 | 0.587 | 0.582 | 0.574 | 0.619 | 0.8% | 0.3 |

Reading the curve:

- **Four of the nine runs are already at parity with 0 % escalation** — after vector scaling the
  student matches or beats its own teacher on the eval split, so the cheapest cascade is "never
  escalate". (Seven of the nine are at or above their teacher with no escalation at all; the
  recorded parity point sits at the sweep floor τ = 0.30, which already routes ~1 % of rows on
  headlines, georeview and headlines-8k.) headlines (student 0.843 vs teacher 0.783) is the
  extreme case: escalating *costs*
  accuracy all the way up, which is why its curve falls from 0.843 to 0.783.
- Escalation buys something on the tasks where the student is behind or the teacher disagrees
  usefully: banking77 0.751 → 0.776 at 27 % escalation (above both student and teacher), kinopoisk
  0.660 → 0.673 at 9 %, georeview MAE 0.609 → 0.583 at 26 %.
- agnews and agnews-mmBERT-base peak at ~10 % escalation (0.889 → 0.897) and then fall back to the
  teacher's 0.880: a small, real "route the unsure ones" gain.
- The `agnews-nogold` row is definitional again — it is scored against the teacher, so escalating to
  the teacher walks it to 1.000 by construction. It is not a quality measurement.
- The two `-prior` rows are the no-gold calibration's own curve; it lands within a point of the
  gold-fitted one everywhere.

What this may claim: "at escalation e, the cascade reaches metric m". What it may **not** claim: any
saving in teacher *quality* — escalation goes back to the same zero-shot teacher that produced the
training labels, measured on the rows that teacher labelled. A run whose parity is at 0 % escalation
is one where the distilled student already matches or beats its own teacher on the eval split.

## `openjev compare` — the paired-bootstrap readout

`compare A B` pairs the two arms seed by seed (`A`, `A-s1`, `A-s2` …), scores each seed on the eval
rows both arms share, and bootstraps the pooled per-row delta 2000 times with the *same* resampled
rows in both arms. It prints one verdict line, and that line is the only form a delta is allowed to
take in the README. Example — the augment control arm, three seeds each, labels and epochs held
fixed, only the 522 synthetic train rows varying:

```
$ openjev compare runs/headlines runs/headlines-nosynth
A = runs/headlines   B = runs/headlines-nosynth   metric: acc (higher is better), 2000 shared eval rows, 3 seed(s), 2000 bootstrap resamples
| seed | A | B | Δ |
|---|---|---|---|
| 0 | 0.8425 | 0.8350 | +0.0075 |
| 1 | 0.8330 | 0.8420 | -0.0090 |
| 2 | 0.8385 | 0.8365 | +0.0020 |
verdict: no measurable difference (Δ = +0.0 ± 0.5 pts (95 % CI -0.5..+0.6))
```

The per-seed column is the point: seed 1 has the *opposite* sign to seeds 0 and 2. A single-seed
run of this pair would have "shown" either a +0.8 pt gain or a −0.9 pt loss.

## What the M1 GPU queue produced (input for M2)

`runs/queue2.jobs`, run through `scripts/gpu_queue.sh`, 13:47 → 14:35, no failures
(`runs/queue2.log`). One factor varies in each group; teacher labels are the cached ones.

| group | run dirs | varies |
|---|---|---|
| E1 epochs | `agnews-e12`, `headlines-e12`, `kinopoisk-e12` | `--epochs 12` vs the YAML's 5 |
| E2 seeds | `headlines-s{1,2}`, `agnews-s{1,2}`, `kinopoisk-s{1,2}` | seed only |
| E3 augment | `headlines-nosynth{,-s1,-s2}` vs `headlines{,-s1,-s2}` | the 522 synth train rows |
| M1.5 re-eval | the nine baseline run dirs | calibration method + cascade curve added |

Note for E1: all three 12-epoch runs early-stop on calib KL (patience 2) before epoch 12 —
`agnews-e12` best at epoch 6 of 9 run, `headlines-e12` best at 4 of 7, `kinopoisk-e12` best at 2 of 5. The knob that actually changed is the LR schedule's horizon, not the number of epochs trained.

## E1 — epochs: 12 vs the YAML's 5

One factor varies: `--epochs 12`. Same seed 0, same cached teacher labels, same LR, same batch
size, same calibration rule. `openjev compare runs/<task> runs/<task>-e12`:

| pair | metric | 5 epochs | 12 epochs | Δ (95 % CI) | verdict |
|---|---|---|---|---|---|
| agnews vs agnews-e12 | acc | 0.8890 | 0.8890 | +0.0 ± 0.6 pts (95 % CI −0.6..+0.6) | no measurable difference |
| headlines vs headlines-e12 | acc | 0.8425 | 0.8360 | +0.7 ± 0.9 pts (95 % CI −0.2..+1.6) | no measurable difference |
| kinopoisk vs kinopoisk-e12 | acc | 0.6600 | 0.6600 | +0.0 ± 1.5 pts (95 % CI −1.5..+1.6) | no measurable difference |

(Δ is A − B, so a positive Δ favours the 5-epoch default. One seed per arm: `docs/` numbers, never
`findings.md`.)

**Decision: the default stays at 5 epochs.** The rule in PLAN-2 §3 was "raise it to 10 if the CI
excludes 0 on ≥ 2 of 3 tasks". It excludes 0 on **0 of 3**, and on headlines the sign favours the
*shorter* run.

### Why — the epoch count is not the knob that binds

The early-stopping traces (`runs/<run>/train.json`, `history`) say what actually happened:

| run | epochs asked | epochs run | best epoch | best calib KL |
|---|---|---|---|---|
| agnews | 5 | 5 | 5 of 5 | 0.0496 |
| agnews-e12 | 12 | 9 | 7 | 0.0463 |
| headlines | 5 | 5 | 5 of 5 | 0.1690 |
| headlines-e12 | 12 | 7 | 5 | 0.1755 |
| kinopoisk | 5 | 5 | 5 of 5 | 0.1130 |
| kinopoisk-e12 | 12 | 5 | 3 | 0.1350 |

Two things follow, and they point the same way.

1. **All three 5-epoch baselines are best at their *last* epoch** — calib KL was still falling when
   the budget ran out. Read alone, that curve says "train longer".
2. **No 12-epoch run ever reached epoch 12.** Patience-2 early stopping fired at 9, 7 and 5 epochs,
   and two of the three stopped at a *worse* calib KL than the 5-epoch run had reached
   (headlines 0.1755 vs 0.1690; kinopoisk 0.1350 vs 0.1130) — while scoring the same accuracy.

`--epochs` sets the length of the linear LR decay (`steps = epochs × batches`), so a 12-epoch run at
epoch 4 has a much higher learning rate than a 5-epoch run at epoch 4. The two arms are not "the
same training, one longer": they are two different LR horizons, and epoch-for-epoch the long-horizon
arm is *less* converged, which is what makes the KL curve noisy enough for patience 2 to trip. The
binding knob is the LR horizon, not the epoch count, and moving the horizon by 2.4× buys nothing
measurable on any of the three tasks.

kinopoisk is the cleanest statement of it: `kinopoisk-e12` stopped at epoch 3 with calib KL 0.1350
against the baseline's 0.1130 — a visibly worse fit to the calib targets — and scored **exactly the
same 0.6600 accuracy**. On these tasks calib KL is not a proxy for accuracy over this range, so
spending GPU time to minimise it further is not an improvement, and neither is the extra epoch
budget that lets early stopping chase it.

What this may not claim: anything about patience (never varied), or about epochs on a task not in
the table. A budget that starts *below* 5 was not tested either.

## E2 — seed variance: the noise floor for everything else

Three seeds per arm, everything else fixed. Accuracy on the shared eval rows
(`results/<task>{,-s1,-s2}.json`):

| task | seed 0 | seed 1 | seed 2 | mean ± sd | spread |
|---|---|---|---|---|---|
| agnews | 0.8890 | 0.8930 | 0.8815 | 0.888 ± 0.006 | 1.2 pts |
| headlines | 0.8425 | 0.8330 | 0.8385 | 0.838 ± 0.005 | 0.9 pts |
| headlines-nosynth | 0.8350 | 0.8420 | 0.8365 | 0.838 ± 0.004 | 0.7 pts |
| kinopoisk | 0.6600 | 0.6540 | 0.6567 | 0.657 ± 0.003 | 0.6 pts |

Seeds change the classifier-head init and the batch order only (the teacher labels and the splits
are identical). That alone moves accuracy by 0.6–1.2 pts peak-to-peak. **Every one-seed delta in
the E1 table above is inside this band**, which is the whole reason the epochs verdict is "no
measurable difference" rather than "12 epochs is 0.7 pts worse on headlines". This is the sd column
in `openjev report` and the noise floor the claim rule in PLAN-2 §5 is calibrated against.

## E3 — the augment (PGKD-lite) control arm

Three seeds per arm; the only factor varying is whether the 522 synthetic train rows written by
`openjev augment` are in the training set. Teacher labels, epochs and calibration rule are fixed —
the synth rows' *labels* are in both run dirs, `--no-synth` just drops them at load time.

```
$ openjev compare runs/headlines runs/headlines-nosynth
A = runs/headlines   B = runs/headlines-nosynth   metric: acc (higher is better), 2000 shared eval rows, 3 seed(s), 2000 bootstrap resamples
| seed | A | B | Δ |
|---|---|---|---|
| 0 | 0.8425 | 0.8350 | +0.0075 |
| 1 | 0.8330 | 0.8420 | -0.0090 |
| 2 | 0.8385 | 0.8365 | +0.0020 |
verdict: no measurable difference (Δ = +0.0 ± 0.5 pts (95 % CI -0.5..+0.6))
```

**Verdict, and it is the README sentence: no measurable difference (Δ = +0.0 ± 0.5 pts, 95 % CI
−0.5..+0.6).** Seed 1 has the opposite sign to seeds 0 and 2; the pre-M1 "+0.5 pt augment gain" was
a one-seed artefact of exactly this size. What this may not claim: that PGKD does not work — one
task, one round, 522 rows against 4000.

## The option-order permutation diagnostic: readout and decision

The M0 diagnostic (`scripts/perm_check.py`, results in `runs/{kinopoisk,headlines}-rev/perm_check.json`)
re-labelled the calib+eval rows of both tasks through the *same* teacher with the option list
reversed and the gold remapped `g → K−1−g`, then scored the original order, the reversed order and
their probability average. The numbers are in the R1 table at the top of this file.

PLAN-2 §3 M2.2 set the decision rule before the numbers were in: build
`teacher: {permutations: 2}` **only if** averaging moves the kinopoisk marginal ≥ 5 pts closer to
gold on the *worst* class **and** averaged teacher accuracy ≥ original + 1 pt.

| test | measured | passes? |
|---|---|---|
| worst class (kinopoisk *Neutral*, gold 0.337): distance to gold, original → averaged | 0.209 → 0.248 — averaging moves it **3.9 pts further away** | **no** |
| teacher accuracy, original → averaged | 0.646 → 0.693 (+4.7 pts) | yes |

The rule is an `and`, so: **the feature is not built.** `openjev` keeps labelling each row once.

Averaging fails the marginal test for a concrete reason. The Bad↔Good split *is* positional —
reversing the order moves Good 0.564 → 0.419 and Bad 0.308 → 0.462 — and averaging two orders
splits that difference. The *Neutral* collapse is not positional: it is 12.9 % in one order and
12.0 % in the other against a gold 33.7 %, and averaging two distributions that both starve the
middle class starves it further (9.0 %), because the class that is never the argmax in either order
gains nothing from the mean. Doubling the teacher bill would buy an accuracy gain the cheaper tool
already gets: the M1 calibration bias fitted on kinopoisk is
`[+0.80 Bad, +0.59 Neutral, −1.39 Good]` — it lifts exactly the class averaging starves and pushes
down exactly the class averaging over-predicts, and it moves the *student* 0.609 → 0.660 for zero
teacher calls. Post-hoc calibration fixes the marginal that permutation averaging could not.

(The reversed order on its own scores +4.7 pts over the original. That is not a free win either:
choosing it needs gold to know which order is the good one, and the headlines pair goes the other
way — 0.784 original vs 0.768 reversed. There is no order-agnostic rule here.)

**The measured sentence for the limitations section of `findings.md`**, which is what the rule says to write
instead of the feature:

> Reversing the option order moves the teacher's *Good* rate on kinopoisk from 56.4 % to 41.9 %
> (gold 33.4 %) and its accuracy from 0.646 to 0.693 — but the *Neutral* collapse survives both
> orders (12.9 % and 12.0 % against a gold 33.7 %), and averaging the two orders makes it worse
> (9.0 %). The bias is semantic, not positional, so openjev labels each row once and removes the
> marginal shift post-hoc with the calibration bias instead — on kinopoisk that is
> `[+0.80 Bad, +0.59 Neutral, −1.39 Good]`, fitted on 500 calib rows, for zero extra teacher calls.


## headlines-8k — does doubling the teacher labels move the student?

`tasks/headlines-8k.yaml` is `tasks/headlines.yaml` with `data.train.n: 8000` and nothing else
changed (same source, same prompt, same options, same `max_chars`, same calib and eval `n`, same
student config).

### Split-id check, before spending teacher time

PLAN-2 §3 M2.4 predicted "calib/eval ids stay identical because roles are drawn train-first from a
seeded pool". **Half of that is true, and the check is what caught the other half:**

| role | headlines | headlines-8k | identical? | overlap |
|---|---|---|---|---|
| train | 4000 | 8000 | no (by design) | 4000 — headlines' train set is an exact **subset** |
| calib | 500 | 500 | **no** | 7 of 500 |
| eval | 2000 | 2000 | **yes** | 2000 |

`data.examples()` draws roles in order from the not-yet-used rows of each source split. eval comes
from the `test` split, which neither train nor calib touches, so its 2000 ids are bit-identical —
that is the id set every metric in this comparison is computed on, and it is what makes the pair a
one-factor comparison *of the metric*. calib, however, is drawn from the **same `train` split** as
train, from the pool left over after the train draw: growing train from 4000 to 8000 rows consumes
4000 more rows of that pool, 59 of headlines' old calib rows are now 8k train rows, and the fresh
calib draw shares only 7 ids with the old one.

So the honest framing is: **the eval rows are identical and the train set is a strict superset; the
500-row calib split is re-drawn from the same split and the same distribution.** The re-draw is
nuisance variation in the fitted calibration, not a systematic difference — but it is a second thing
that changed, and the claim below is stated with that in it rather than around it.

Teacher labels are frozen per PLAN-2 §2: `scripts/seed_cache.py` copied the 6066 rows the two tasks
share out of `runs/headlines/teacher.jsonl` with their probabilities untouched (rewriting only the
`split` field for the 59 promoted rows), so only the 4434 genuinely new rows were labelled —
9 min at 8.2 ex/s, `runs/headlines-8k/label.log`. Every row both arms contain carries the same
teacher probabilities in both.


### Result

Three seeds per arm, both arms synth-free, metrics on the 2000 identical eval ids.

| seed | headlines (4000) | headlines-8k (8000) | Δ |
|---|---|---|---|
| 0 | 0.8350 | 0.8515 | +0.0165 |
| 1 | 0.8365 | 0.8530 | +0.0165 |
| 2 | 0.8365 | 0.8490 | +0.0125 |

`openjev compare runs/headlines-nosynth runs/headlines-8k` →
**Δ = +1.3 ± 0.8 pts (95 % CI +0.6..+2.1)**, sign consistent across seeds. Against the
augment-carrying `runs/headlines` arm the delta is the same to a tenth of a point.

Doubling the teacher bill — 4434 new rows, 9 minutes — buys +1.3 pts, about a sixth of what the
500-row calibration bias buys on this same task (+7.5). The calib re-draw described above is inside
this interval and cannot be separated from it; the claim is "8000 labels and a re-drawn calib beat
4000 labels", which is the choice a user actually faces.


## Where to spend 500 gold labels: in the loss, or on the calibration bias?

One task (kinopoisk), one factor, three seeds per arm, identical teacher labels and identical eval
rows. Arm A is the normal run: gold never enters the loss, and the 500 calib rows fit `logits / T + b`.
Arm B puts the same 500 rows into the CE term (`--gold-weight 1.0 --gold-n 500`) *and* still gets the
vector calibration, so B is A's calibration plus supervision, not an alternative to it.

| seed | A — bias only | B — 500 gold in the loss | Δ (A − B) |
|---|---|---|---|
| 0 | 0.6600 | 0.6033 | +0.0567 |
| 1 | 0.6540 | 0.6220 | +0.0320 |
| 2 | 0.6567 | 0.6233 | +0.0333 |

`openjev compare runs/kinopoisk runs/kinopoisk-gold-n500` →
**A is better: Δ = +4.1 ± 1.7 pts (95 % CI +2.4..+5.7)**.

Adding the labels to the loss does not merely fail to help, it costs 4 points. All 4000 gold labels
in the loss (`--gold-weight 1.0`, one seed) reach **0.672**, i.e. 8× the labels buy 1.2 pts over the
500-row bias, and the teacher itself scores 0.652.

Hypothesis, not a claim: a few hundred hard labels pull against 4000 soft ones inside the same loss
(both gold-weighted arms early-stop at epoch 2, against epoch 4 for pure distillation), while the same
rows spent post-hoc correct the marginal, which is the error that actually dominates here. Two points
on one task is not an N-labels curve.


## Reproducing

```bash
openjev calibrate tasks/<task>.yaml && openjev eval tasks/<task>.yaml --no-latency
python scripts/r1_table.py                       # both tables above, from runs/*/eval_rows.jsonl
python scripts/prior_variant.py tasks/kinopoisk.yaml uniform   # the no-gold arm, into runs/<task>-prior
openjev compare runs/A runs/B                    # paired per-seed deltas + pooled 95 % CI
```

`scripts/r1_table.py` refits all four calibrations from `runs/<run>/calib_logits.pt` on the CPU, so
the table can be rebuilt without touching the GPU. `results/variants/` holds the run dirs that are
not part of a completed comparison (the `-prior` arms); they are kept out of `results/` so they can
never reach the README table.
