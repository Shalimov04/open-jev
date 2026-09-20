# Experiments

Everything here is reproducible from the repo: every row names the command that produced it and the
`results/*.json` or `runs/*/` file it was read from. Claim rule (PLAN-2 §5): a delta is an
*improvement* only when the paired bootstrap 95 % CI excludes 0; otherwise the sentence is "no
measurable difference". Anything fitted on eval is labelled **diagnostic** and is never a claim.

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
| banking77 | acc | 0.764 | 0.749 | 0.749 | 0.750 | 0.755 | 0.786 | vector | +0.20 (-1.35..+1.65) | +0.70 (-0.70..+2.15) |
| georeview | mae | 0.619 | 0.675 | 0.783 | 0.609 | 0.599 | 0.599 | vector | -0.17 (-0.19..-0.16) * | -0.18 (-0.20..-0.17) * |
| georeview-mmBERT-base | mae | 0.619 | 0.664 | 0.771 | 0.594 | 0.583 | 0.585 | vector | -0.18 (-0.19..-0.16) * | -0.19 (-0.20..-0.17) * |
| headlines | acc | 0.783 | 0.767 | 0.767 | 0.843 | 0.842 | 0.838 | vector | +7.50 (+5.95..+9.05) * | +7.45 (+5.95..+8.90) * |
| kinopoisk | acc | 0.652 | 0.609 | 0.609 | 0.660 | 0.655 | 0.659 | vector | +5.13 (+2.87..+7.47) * | +4.67 (+2.33..+7.07) * |
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
- Every Δ above is **one seed per run**. Under PLAN-2 §5 that makes them `docs/` numbers; the README
  gets them only with the 3-seed arms (`runs/{headlines,agnews,kinopoisk}-s{1,2}`, trained in the M1
  queue) behind them.
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
| banking77 | acc | 0.758 | 0.771 | 0.776 | 0.776 | 0.764 | 6.9% | 0.35 |
| georeview | mae | 0.606 | 0.594 | 0.583 | 0.574 | 0.619 | 0.9% | 0.3 |
| georeview-mmBERT-base | mae | 0.590 | 0.580 | 0.570 | 0.571 | 0.619 | 1.0% | 0.3 |
| headlines | acc | 0.845 | 0.840 | 0.814 | 0.792 | 0.783 | 1.0% | 0.3 |
| kinopoisk | acc | 0.660 | 0.671 | 0.673 | 0.659 | 0.652 | 0.0% | 0.3 |
| toxic | auroc | 0.856 | 0.857 | 0.817 | 0.817 | 0.817 | 0.0% | 0.3 |
| kinopoisk-prior | acc | 0.655 | 0.673 | 0.671 | 0.666 | 0.652 | 0.0% | 0.3 |
| georeview-prior | mae | 0.596 | 0.587 | 0.582 | 0.574 | 0.619 | 0.8% | 0.3 |

Reading the curve:

- **Four of the nine runs are already at parity with 0 % escalation** — after vector scaling the
  student matches or beats its own teacher on the eval split, so the cheapest cascade is "never
  escalate". headlines (student 0.845 vs teacher 0.783) is the extreme case: escalating *costs*
  accuracy all the way up, which is why its curve falls from 0.845 to 0.783.
- Escalation buys something on the tasks where the student is behind or the teacher disagrees
  usefully: banking77 0.758 → 0.776 at 25 % escalation (above both student and teacher), kinopoisk
  0.660 → 0.673 at 25 %, georeview MAE 0.606 → 0.583 at 25 %.
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

## Reproducing

```bash
openjev calibrate tasks/<task>.yaml && openjev eval tasks/<task>.yaml --no-latency
python scripts/r1_table.py                       # both tables above, from runs/*/eval_rows.jsonl
python scripts/prior_variant.py tasks/kinopoisk.yaml uniform   # the no-gold arm, into runs/<task>-prior
openjev compare runs/A runs/B                    # paired per-seed deltas + pooled 95 % CI
```

(`compare` is `openjev.evaluate.compare`; until the subcommand is wired in `cli.py`, call it as
`python -c 'from openjev.evaluate import compare; compare("runs/A", "runs/B")'`.)

`scripts/r1_table.py` refits all four calibrations from `runs/<run>/calib_logits.pt` on the CPU, so
the table can be rebuilt without touching the GPU. `results/variants/` holds the run dirs that are
not part of a completed comparison (the `-prior` arms); they are kept out of `results/` so they can
never reach the README table.
