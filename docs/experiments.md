# Experiments

The lab notebook. Every row names the command that produced it and the `results/*.json` or
`runs/*/` file it was read from. Claim rule (PLAN-2 §5): a delta is an *improvement* only when the
paired bootstrap 95 % CI excludes 0; otherwise the sentence is "no measurable difference". Anything
fitted on eval is labelled **diagnostic** and is never a claim. From here on a comparison enters
`findings.md` only with a preregistration file (`prereg-template.md` → `docs/prereg/<id>.md`) whose
commit predates the arms' `label.json` / `train.json`; the sections below predate the rule and were
not retrofitted.

Held fixed in every comparison: the teacher labels (`runs/<task>/teacher.jsonl`, same ids and
probabilities, nothing re-labelled), the split ids, the student checkpoint (`jhu-clsp/mmBERT-small`),
`max_len`, `lr`, `batch_size`, and the calibration method-selection rule. One factor varies.

## R1 — a K-dim bias on the student's logits recovers part of the teacher-prior gap

### Where the gap comes from

The M0 permutation diagnostic (`runs/*-rev/perm_check.json`: the option order reversed, calib+eval
re-labelled through the same teacher, gold remapped `g → K−1−g`) separates a *marginal* shift from
per-example noise:

| task | class | gold | teacher, original order | teacher, reversed | averaged |
|---|---|---|---|---|---|
| kinopoisk | Bad | 0.330 | 0.308 | 0.462 | 0.407 |
| kinopoisk | Neutral | 0.337 | 0.129 | 0.120 | 0.090 |
| kinopoisk | Good | 0.334 | 0.564 | 0.419 | 0.504 |
| headlines | политика | 0.162 | 0.241 | 0.310 | 0.275 |
| headlines | наука | 0.161 | 0.067 | 0.087 | 0.084 |

Teacher accuracy: kinopoisk 0.646 original / 0.693 reversed / 0.693 averaged; headlines 0.784 /
0.768 / 0.791. The Bad↔Good split moves with the order, so that part is positional. The Neutral
collapse does not — 12.9 % against a gold 33.7 % in one order, 12.0 % in the other — and averaging
makes it *worse* (9.0 %). headlines has the same shape: политика over-predicted and наука
under-predicted in both orders. The student inherits the shift faithfully.

Per-row readouts from the same runs (`python scripts/perm_check.py kinopoisk`, no new teacher
calls): **500 / 2000 kinopoisk rows (25.0 %) and 297 / 2500 headlines rows (11.9 %) change their
argmax when the option list is reversed** — far from a near-tie effect (mini-jev measures 3 / 50 on
15-token English utterances at k = 4). Accuracy per class with the position the class occupied:

| task | class | position → acc, original | position → acc, reversed |
|---|---|---|---|
| kinopoisk | Bad | 0 → 0.750 | 2 → 0.954 |
| kinopoisk | Neutral | 1 → 0.209 | 1 → 0.242 |
| kinopoisk | Good | 2 → 0.985 | 0 → 0.889 |
| headlines | наука | 3 → 0.414 | 2 → 0.526 |
| headlines | экономика | 5 → 0.780 | 0 → 0.626 |

Bad and Good swing 10–20 pts when their letter moves; Neutral sits in the middle in *both* orders
and stays at 0.21 / 0.24. A chosen-position histogram cannot tell those apart; the per-class readout
can. A shifted marginal is exactly what a constant per-class bias on the logits removes, which is
why vector scaling is the right post-hoc tool and an ordinal head or a bigger student is not.

### Method

Four calibrations, all fitted on the **calib** split only, all applied to the same stored eval
logits (`runs/<run>/eval_rows.jsonl`, written by `openjev eval`; no re-inference):

- **T-only** — `fit_temperature` on the 500 gold calib rows. What the runs shipped before M1.
- **vector** — `fit_vector` (LBFGS over `log T` and `b ∈ R^K`, CE loss) on the same 500 rows.
  Guard: the calib rows are split 2-fold, each half fitted and scored by NLL on the other, and
  `vector` is accepted only if it wins on **both** halves; otherwise the run ships `temperature`.
  `calib.json` records `method` and `bias`; `calibrate.apply(logits, calib)` is the single place
  calibration is applied, and `serve` reads the same block from `runs/<run>/openjev.json`.
- **prior-declared** — **no gold at all**: `T` fitted to the teacher's *soft* labels, then `b`
  fitted so that the mean calibrated probability over the calib rows equals a declared prior
  (`prior: uniform` here), by iterating `b_k += log(prior_k / mean_p_k)` 50 times.
  `calib_target: prior-declared`.
- **oracle (diagnostic, fitted on eval)** — the same fit with the *eval gold marginal* as target.
  It says how much of the gap is marginal at all; it is not an accuracy bound (matching the marginal
  is not the accuracy-optimal bias, so a declared prior can land above it). Never enters the README.

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

- kinopoisk's fitted bias is `[+0.80 Bad, +0.59 Neutral, −1.39 Good]` (`runs/kinopoisk/calib.json`):
  it pushes down the class the teacher over-predicts and lifts the one it collapses, exactly as the
  permutation diagnostic predicted.
- **Where it works**: headlines +7.5 pts, kinopoisk +5.1 pts, georeview −0.17 MAE, every CI
  excluding 0 — the three tasks whose teacher marginal is visibly shifted. For the three tasks with
  seed arms the refit is in the table for all three seeds: headlines +7.50 / +7.00 / +7.40,
  kinopoisk +5.13 / +4.53 / +5.13, every CI excluding 0; agnews +0.95 / +1.05 / +0.65, inside noise
  on two of three. That is what backs the calibration claim in `findings.md`; the single-seed rows
  stay `docs/` numbers under PLAN-2 §5.
- **Where it does nothing**: banking77 (+0.2 pts, CI spans 0) and toxic. The 2-fold guard
  *accepted* vector on banking77 — the plan expected it to refuse at K=77 — but there is no prior
  shift to remove.
- **toxic's Δ is 0.000 by construction.** AUROC ranks by `p(true)`, which a constant per-class bias
  transforms monotonically. toxic is excluded from R1 anyway: its calib split is 2.2 % positive
  against 8.1 % on eval ([findings.md](findings.md#what-the-numbers-mean)); re-drawing and
  re-labelling that split (500 rows, ~1.2 min of teacher) is what would make it comparable.
- **agnews-nogold's row is not an accuracy row.** With no `data.gold`, eval scores the student
  against the *teacher's* argmax (hence "teacher" 1.000); the −2.0 pts is the prior bias moving the
  student *away* from the teacher it is measured against, not away from truth.
  `scripts/nogold_gold_acc.py` is the only honest scorer for that run.
- **Temperature alone makes georeview worse** (MAE 0.675 raw → 0.783): for a `score` task the
  metric is the MAE of the *expectation*, and a T > 1 flattens the distribution toward the middle
  level. The bias is what helps — an argument for reporting the task's own metric, not ECE.
- **ECE and accuracy come apart under vector scaling.** The pre-M1 guard ("never let calibration
  raise ECE on the calib split") was written for temperature, which cannot change an argmax; applied
  to a bias it vetoed a +7 pt gain on headlines seed 1 because calib-split ECE moved 0.038 → 0.051.
  The guard now covers the temperature path only; `vector` is gated by the 2-fold NLL test, the
  stricter of the two.
- **The declared prior is a leak on these benchmarks.** `uniform` is right for agnews, headlines,
  kinopoisk, georeview and banking77 because their eval sets are balanced by construction — gold
  marginals 0.24–0.26, 0.16–0.18, 0.333 and 0.199–0.202.
- `-mmBERT-base` rows are a different student and part of no comparison; they show the effect is
  not an artifact of the small model.

### End-to-end, with the gold deliberately ignored

The `prior-declared` column is refitted offline. `scripts/prior_variant.py` makes a `-prior` run
dir (student and teacher labels symlinked to the base run, nothing retrained) and calibrates with
`ignore_gold=True`, to check the *pipeline* produces the same number when it never sees gold:

| run | calib_target | metric, end-to-end | same, refitted offline |
|---|---|---|---|
| kinopoisk-prior | prior-declared | acc 0.6553 | 0.655 |
| georeview-prior | prior-declared | MAE 0.5986 | 0.599 |

Two code paths, same answer. `results/variants/{kinopoisk,georeview}-prior.json`.

## R2 — cascade / abstention curve

The student answers when its confidence (max calibrated probability) is ≥ τ, otherwise the row is
escalated to the teacher; τ sweeps 0.30 → 1.00 in steps of 0.05 over probability matrices already
on disk (zero new teacher calls). An escalated row takes the teacher's probabilities, i.e. its
argmax. The curve and the parity point (the lowest escalation rate whose metric is within 0.005 of
the teacher's) are stored in `results/<run>.json` as `cascade` and `cascade_parity`; the parity τ
is exported to `runs/<run>/openjev.json` as `escalate_below` for `serve`.

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

- **Four of the nine runs are at parity with 0 % escalation**; seven of the nine are at or above
  their teacher with no escalation at all (the recorded parity point sits at the sweep floor
  τ = 0.30, which already routes ~1 % of rows on headlines, georeview and headlines-8k). headlines
  (0.843 vs 0.783) is the extreme case: escalating *costs* accuracy all the way up.
- Escalation pays where the student is behind or the teacher disagrees usefully: banking77
  0.751 → 0.776 at 27 % (above both), kinopoisk 0.660 → 0.673 at 9 %, georeview MAE 0.609 → 0.583
  at 26 %. agnews and agnews-mmBERT-base peak at ~10 % (0.889 → 0.897) and fall back to 0.880.
- `agnews-nogold` is scored against the teacher, so escalating to the teacher walks it to 1.000 by
  construction. The two `-prior` rows land within a point of the gold-fitted curve everywhere.

What this may claim: "at escalation e, the cascade reaches metric m". Not: any saving in teacher
*quality* — escalation goes back to the zero-shot teacher that produced the training labels.

## E1 — epochs: 12 vs the YAML's 5

One factor: `--epochs 12`. Seed 0, cached teacher labels, same LR, batch size and calibration rule.
`openjev compare runs/<task> runs/<task>-e12` (Δ is A − B, so positive favours the 5-epoch default;
one seed per arm, so `docs/` numbers, never `findings.md` claims):

| pair | metric | 5 epochs | 12 epochs | Δ (95 % CI) | verdict |
|---|---|---|---|---|---|
| agnews vs agnews-e12 | acc | 0.8890 | 0.8890 | +0.0 ± 0.6 pts (95 % CI −0.6..+0.6) | no measurable difference |
| headlines vs headlines-e12 | acc | 0.8425 | 0.8360 | +0.7 ± 0.9 pts (95 % CI −0.2..+1.6) | no measurable difference |
| kinopoisk vs kinopoisk-e12 | acc | 0.6600 | 0.6600 | +0.0 ± 1.5 pts (95 % CI −1.5..+1.6) | no measurable difference |

**Decision: the default stays at 5 epochs.** The PLAN-2 §3 rule was "raise it to 10 if the CI
excludes 0 on ≥ 2 of 3 tasks". It excludes 0 on **0 of 3**, and on headlines the sign favours the
*shorter* run. The early-stopping traces (`runs/<run>/train.json`, `history`) say why:

| run | epochs asked | epochs run | best epoch | best calib KL |
|---|---|---|---|---|
| agnews | 5 | 5 | 5 of 5 | 0.0496 |
| agnews-e12 | 12 | 9 | 7 | 0.0463 |
| headlines | 5 | 5 | 5 of 5 | 0.1690 |
| headlines-e12 | 12 | 7 | 5 | 0.1755 |
| kinopoisk | 5 | 5 | 5 of 5 | 0.1130 |
| kinopoisk-e12 | 12 | 5 | 3 | 0.1350 |

All three 5-epoch baselines are best at their *last* epoch (calib KL still falling), which read
alone says "train longer". But no 12-epoch run reached epoch 12: patience-2 early stopping fired at
9, 7 and 5, and two of the three stopped at a *worse* calib KL than the 5-epoch run (headlines
0.1755 vs 0.1690; kinopoisk 0.1350 vs 0.1130) while scoring the same accuracy. `--epochs` sets the
length of the linear LR decay, so the two arms are two different LR horizons, not "the same
training, one longer"; epoch-for-epoch the long-horizon arm is *less* converged, which is what makes
the KL curve noisy enough for patience 2 to trip. kinopoisk is the cleanest statement: `-e12`
stopped at epoch 3 with calib KL 0.1350 against 0.1130 and scored **exactly the same 0.6600** — over
this range calib KL is not a proxy for accuracy. Not claimed: anything about patience (never
varied), epochs on other tasks, or a budget below 5.

## E2 — seed variance: the noise floor for everything else

Three seeds per arm, everything else fixed; seeds change the classifier-head init and the batch
order only. Accuracy on the shared eval rows (`results/<task>{,-s1,-s2}.json`):

| task | seed 0 | seed 1 | seed 2 | mean ± sd | spread |
|---|---|---|---|---|---|
| agnews | 0.8890 | 0.8930 | 0.8815 | 0.888 ± 0.006 | 1.2 pts |
| headlines | 0.8425 | 0.8330 | 0.8385 | 0.838 ± 0.005 | 0.9 pts |
| headlines-nosynth | 0.8350 | 0.8420 | 0.8365 | 0.838 ± 0.004 | 0.7 pts |
| kinopoisk | 0.6600 | 0.6540 | 0.6567 | 0.657 ± 0.003 | 0.6 pts |

0.6–1.2 pts peak-to-peak from the seed alone. **Every one-seed delta in E1 is inside this band**,
which is why its verdict is "no measurable difference" and not "12 epochs is 0.7 pts worse on
headlines". This is the sd column in `openjev report` and the floor the claim rule is calibrated
against.

## E3 — the augment (PGKD-lite) control arm

Three seeds per arm; the only factor is whether the 522 synthetic train rows written by
`openjev augment` are in the training set (their *labels* are in both run dirs; `--no-synth` drops
them at load time). `compare` pairs the arms seed by seed, scores each on the eval rows both share,
and bootstraps the pooled per-row delta 2000 times with the *same* resampled rows in both arms:

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

The per-seed column is the point: seed 1 has the *opposite* sign to seeds 0 and 2, so a one-seed
run would have "shown" either +0.8 or −0.9 pts — the pre-M1 "+0.5 pt augment gain" was a one-seed
artefact of exactly this size. Not claimed: that PGKD does not work — one task, one round, 522 rows
against 4000.

The M1 queue that produced E1–E3 (`runs/queue2.jobs` through `scripts/gpu_queue.sh`, 13:47 → 14:35,
no failures, `runs/queue2.log`): `agnews-e12`, `headlines-e12`, `kinopoisk-e12` (E1);
`{agnews,headlines,kinopoisk}-s{1,2}` (E2); `headlines-nosynth{,-s1,-s2}` (E3); plus the nine
baseline dirs re-evaluated with the calibration method and cascade curve added.

## The option-order permutation diagnostic: readout and decision

The numbers are in the R1 tables above. PLAN-2 §3 M2.2 set the decision rule before they were in:
build `teacher: {permutations: 2}` **only if** averaging moves the kinopoisk marginal ≥ 5 pts closer
to gold on the *worst* class **and** averaged teacher accuracy ≥ original + 1 pt.

| test | measured | passes? |
|---|---|---|
| worst class (kinopoisk *Neutral*, gold 0.337): distance to gold, original → averaged | 0.209 → 0.248 — averaging moves it **3.9 pts further away** | **no** |
| teacher accuracy, original → averaged | 0.646 → 0.693 (+4.7 pts) | yes |

The rule is an `and`, so **the feature is not built**; `openjev` labels each row once. Averaging
splits the positional Bad↔Good difference but starves the middle class further (9.0 %), because a
class that is never the argmax in either order gains nothing from the mean. The M1 calibration bias
lifts exactly that class for zero teacher calls (findings.md). The reversed order alone scores
+4.7 pts over the original, but choosing it needs gold to know which order is the good one, and
headlines goes the other way (0.784 original vs 0.768 reversed): there is no order-agnostic rule.

## headlines-8k — does doubling the teacher labels move the student?

`tasks/headlines-8k.yaml` is `tasks/headlines.yaml` with `data.train.n: 8000` and nothing else.

### Split-id check, before spending teacher time

PLAN-2 §3 M2.4 predicted "calib/eval ids stay identical because roles are drawn train-first from a
seeded pool". Half of that is true:

| role | headlines | headlines-8k | identical? | overlap |
|---|---|---|---|---|
| train | 4000 | 8000 | no (by design) | 4000 — headlines' train set is an exact **subset** |
| calib | 500 | 500 | **no** | 7 of 500 |
| eval | 2000 | 2000 | **yes** | 2000 |

eval comes from the `test` split, which neither train nor calib touches, so its 2000 ids are
bit-identical — the id set every metric here is computed on. calib is drawn from the same `train`
split as train, from the pool left after the train draw: growing train to 8000 consumed 4000 more
rows of that pool, 59 of headlines' old calib rows became 8k train rows, and the fresh calib draw
shares only 7 ids with the old one. So: **eval rows identical, train a strict superset, the 500-row
calib split re-drawn from the same distribution** — nuisance variation in the fitted calibration,
stated with the claim rather than around it.

Teacher labels frozen per PLAN-2 §2: `scripts/seed_cache.py` copied the 6066 rows the two tasks
share out of `runs/headlines/teacher.jsonl` with their probabilities untouched (rewriting only the
`split` field for the 59 promoted rows), so only the 4434 new rows were labelled — 9 min at
8.2 ex/s, `runs/headlines-8k/label.log`.

### Result

Three seeds per arm, both synth-free, metrics on the 2000 identical eval ids:

| seed | headlines (4000) | headlines-8k (8000) | Δ |
|---|---|---|---|
| 0 | 0.8350 | 0.8515 | +0.0165 |
| 1 | 0.8365 | 0.8530 | +0.0165 |
| 2 | 0.8365 | 0.8490 | +0.0125 |

`openjev compare runs/headlines-nosynth runs/headlines-8k` →
**Δ = +1.3 ± 0.8 pts (95 % CI +0.6..+2.1)**, sign consistent across seeds; against the
augment-carrying `runs/headlines` arm the delta is the same to a tenth of a point. Doubling the
teacher bill (4434 rows, 9 minutes) buys +1.3 pts, about a sixth of what the 500-row bias buys on
the same task (+7.5). The calib re-draw is inside this interval and cannot be separated from it;
the claim is "8000 labels and a re-drawn calib beat 4000 labels", which is the choice a user faces.

## Where to spend 500 gold labels: in the loss, or on the calibration bias?

One task (kinopoisk), one factor, three seeds per arm, identical teacher labels and eval rows. Arm A
is the normal run: gold never enters the loss, the 500 calib rows fit `logits / T + b`. Arm B puts
the same 500 rows into the CE term (`--gold-weight 1.0 --gold-n 500`) *and* still gets the vector
calibration, so B is A's calibration plus supervision, not an alternative to it.

| seed | A — bias only | B — 500 gold in the loss | Δ (A − B) |
|---|---|---|---|
| 0 | 0.6600 | 0.6033 | +0.0567 |
| 1 | 0.6540 | 0.6220 | +0.0320 |
| 2 | 0.6567 | 0.6233 | +0.0333 |

`openjev compare runs/kinopoisk runs/kinopoisk-gold-n500` →
**A is better: Δ = +4.1 ± 1.7 pts (95 % CI +2.4..+5.7)**. Adding the labels to the loss costs 4
points. All 4000 gold labels in the loss (`--gold-weight 1.0`, one seed) reach **0.672**: 8× the
labels buy 1.2 pts over the 500-row bias; the teacher itself scores 0.652. Hypothesis, not a claim:
a few hundred hard labels pull against 4000 soft ones inside the same loss (both gold-weighted arms
early-stop at epoch 2, against epoch 4 for pure distillation), while the same rows spent post-hoc
correct the marginal, which is the error that dominates here.

## Reproducing

```bash
openjev calibrate tasks/<task>.yaml && openjev eval tasks/<task>.yaml --no-latency
python scripts/r1_table.py                       # both R1 tables, from runs/*/eval_rows.jsonl
python scripts/prior_variant.py tasks/kinopoisk.yaml uniform   # the no-gold arm, into runs/<task>-prior
openjev compare runs/A runs/B                    # paired per-seed deltas + pooled 95 % CI
```

`scripts/r1_table.py` refits all four calibrations from `runs/<run>/calib_logits.pt` on the CPU.
`results/variants/` holds run dirs that are not part of a completed comparison (the `-prior` arms),
kept out of `results/` so they can never reach the README table.

## B1 — `open-jev-base`: one pair scorer, six tasks it was never trained on

**What was trained.** `AutoModelForSequenceClassification(num_labels=1)` on
`(decision, option, text)` pairs, BCE against the teacher's probability of that option
(`openjev/base.py`, rendering from `openjev/pairs.py`). Inference: sigmoid per option, renormalised
over the K offered, handed to `calibrate.apply` / `evaluate.run` as an `[N, K]` matrix of
renormalised log-probabilities — so `cascade`, `selective`, `compare` and `views.view` are the same
code as for a per-task student. `noul` is one pair scored once and returned as `[log p, log(1−p)]`
(the option index is not in the prompt; training kept both of its pairs until this milestone, which
was fitting one input against `p` and `1−p` at the same time — fixed in `pairs.sample`).

**Mixture.** `data/pairs.jsonl` rebuilt from every finished run dir: 42 labelled task files,
1,183,247 pairs, 129,226 examples. That is the 11 original tasks (+4 `-jev` relabels) and **all 27
new Jev-labelled tasks** under `tasks/base/` — boolq, civil-{identity,insult,obscene,threat}, cola,
emotion-dair, fin-sentiment, massive-{en,ru}, miracl-rerank, mrpc, paws, rte, ru-reviews,
rubq-rerank, sib200-ru, swde-vertical, trec-coarse, tweet-{emotion,hate,irony,offensive,sentiment,
stance}, xnli-{en,ru} — 49,376 rows for $0.84, 0 of 27 failed `scripts/base_gate.py`. Two are short
of their YAML (`tweet-irony` 100 rows, `sib200-ru` 749) because the per-task budget cap hit first.

**Folds are by text source** (`tasks/base/folds.yaml`), never by task name: F1 drops kinopoisk,
swde-field + swde-vertical and toxic + the four civil_comments sub-tasks; F2 drops georeview,
banking77 and m2w-element + m2w-target. `openjev base eval` refuses to score a task that is not in
the model's recorded `excluded` list.

| model | train tasks | pairs/epoch (cap 12k) | steps | minutes | peak GPU | best calib BCE |
|---|---|---|---|---|---|---|
| base-F1 | 31 | 171,116 | 6302 | 32.5 | 8.93 GB | 0.3544 |
| base-F2 | 35 | 176,316 | 6302 | 43.1 | 8.91 GB | 0.3585 |
| base-none | 40 | 234,316 | 6302 | 42.5 | 8.91 GB | 0.3550 |

78–103 pairs/s measured. Calib BCE was **still falling at the last checkpoint in all three runs** —
the 45-minute budget stopped them, early stopping never fired. Every number below is a
one-pass-over-the-mixture model, not a converged one.

### Leave-one-source-out (`python scripts/base_table.py --variant gold500`)

Chance = a uniform `[N, K]`; student = the existing per-task run's 3 seeds averaged; deltas are
paired bootstrap over the shared eval rows, 2000 resamples.

| task        | fold | metric | n | chance | teacher | student | base-zs | Δ vs student (95 % CI) | Δ vs teacher (95 % CI) |
|-------------|---|---|---|---|---|---|---|---|---|
| kinopoisk   | F1 | acc | 1500 | 0.333 | 0.652 | 0.657 | 0.525 | -0.132 (-0.157..-0.107) | -0.127 (-0.158..-0.095) |
| swde-field  | F1 | acc | 2000 | 0.052 | 0.803 | 0.869 | 0.353 | -0.516 (-0.538..-0.493) | -0.450 (-0.476..-0.424) |
| toxic       | F1 | auroc | 3000 | 0.500 | 0.817 | 0.856 | 0.809 | -0.047 (-0.074..-0.021) | -0.007 (-0.036..+0.022) |
| georeview   | F2 | mae | 2000 | 1.201 | 0.619 | 0.607 | 0.751 | +0.144 (+0.127..+0.162) | +0.132 (+0.113..+0.152) |
| banking77   | F2 | acc | 2000 | 0.015 | 0.764 | 0.750 | 0.452 | -0.299 (-0.321..-0.275) | -0.312 (-0.337..-0.288) |
| m2w-element | F2 | acc | 900 | 0.049 | 0.649 | 0.059 | 0.152 | +0.093 (+0.064..+0.122) | -0.497 (-0.533..-0.460) |

The same table for `raw` / `prior` / `teacher500` is in `results/base/base-loto-*.json`; the
calibration variant moves base-zs accuracy by up to 12 pts (swde-field 0.231 raw → 0.353 gold500)
and never changes the verdict.

**Read it plainly: zero-shot loses to the per-task student on 5 of 6 tasks, every CI excluding 0**
— by 13 pts on kinopoisk, 30 pts on banking77, 52 pts on swde-field, 0.14 MAE on georeview and
0.047 AUROC on toxic. The two results that are not losses:

- **m2w-element: base zero-shot 0.152 vs the fixed 16-way head's 0.059, Δ = +9.3 pts
  (95 % CI +6.4..+12.2).** This is the PLAN-4 §0 argument reproduced on a model that never saw a
  Mind2Web page: a head that must mean the same thing on every row cannot learn "candidate G",
  a pair scorer reads the candidate's own line. It is still 50 pts below the teacher's 0.649 and
  only 3× a 0.049 chance floor — the argument is about primitives, not about this model being useful
  here.
- **toxic: AUROC 0.809 vs the teacher's 0.817, Δ = −0.007 (95 % CI −0.036..+0.022) — no measurable
  difference from the teacher**, on a `noul` question over texts no fold-F1 task had seen. The
  per-task student still beats both (0.856).

### Calibration under transfer (`--calib`)

| task | arm                | acc | ECE raw | ECE cal | NLL |
|---|--------------------|---|---|---|---|
| kinopoisk | per-task student   | 0.660 | 0.1608 | 0.0378 | 0.744 |
| kinopoisk | base-F1 raw        | 0.515 | 0.0760 | 0.0760 | 1.032 |
| kinopoisk | base-F1 prior      | 0.535 | 0.0760 | 0.0542 | 1.008 |
| kinopoisk | base-F1 teacher500 | 0.515 | 0.0760 | 0.0759 | 0.997 |
| kinopoisk | base-F1 gold500    | 0.525 | 0.0760 | 0.0484 | 0.983 |
| swde-field | per-task student   | 0.863 | 0.0488 | 0.0361 | 0.597 |
| swde-field | base-F1 raw        | 0.230 | 0.1164 | 0.1164 | 2.678 |
| swde-field | base-F1 prior      | 0.278 | 0.1164 | 0.1571 | 2.761 |
| swde-field | base-F1 teacher500 | 0.230 | 0.1164 | 0.1268 | 2.456 |
| swde-field | base-F1 gold500    | 0.353 | 0.1164 | 0.1239 | 2.167 |
| toxic | per-task student   | 0.917 | 0.1164 | 0.0373 | 0.252 |
| toxic | base-F1 raw        | 0.618 | 0.0825 | 0.0825 | 0.612 |
| toxic | base-F1 prior      | 0.489 | 0.0825 | 0.1944 | 0.761 |
| toxic | base-F1 teacher500 | 0.618 | 0.0825 | 0.0775 | 0.611 |
| toxic | base-F1 gold500    | 0.919 | 0.0825 | 0.0558 | 0.275 |
| georeview | per-task student   | 0.558 | 0.2527 | 0.0246 | 1.048 |
| georeview | base-F2 raw        | 0.344 | 0.0661 | 0.0661 | 1.331 |
| georeview | base-F2 prior      | 0.416 | 0.0661 | 0.0678 | 1.257 |
| georeview | base-F2 teacher500 | 0.344 | 0.0661 | 0.0277 | 1.312 |
| georeview | base-F2 gold500    | 0.326 | 0.0661 | 0.0918 | 1.257 |
| banking77 | per-task student   | 0.751 | 0.0171 | 0.0422 | 1.008 |
| banking77 | base-F2 raw        | 0.389 | 0.3222 | 0.3222 | 3.330 |
| banking77 | base-F2 prior      | 0.419 | 0.3222 | 0.3377 | 3.076 |
| banking77 | base-F2 teacher500 | 0.389 | 0.3222 | 0.2370 | 3.024 |
| banking77 | base-F2 gold500    | 0.451 | 0.3222 | 0.0512 | 2.376 |
| m2w-element | per-task student   | 0.059 | 0.1017 | 0.0036 | 2.773 |
| m2w-element | base-F2 raw        | 0.152 | 0.0550 | 0.0550 | 2.736 |
| m2w-element | base-F2 prior      | 0.141 | 0.0550 | 0.0439 | 2.736 |
| m2w-element | base-F2 teacher500 | 0.152 | 0.0550 | 0.0622 | 2.736 |
| m2w-element | base-F2 gold500    | 0.152 | 0.0550 | 0.0578 | 2.735 |

**The §6 question answered with numbers, not adjectives.** Out of the box (ECE raw) the pair scorer
is *better* calibrated than the per-task student on 4 of 6 tasks — kinopoisk 0.076 vs 0.161,
georeview 0.066 vs 0.253, toxic 0.083 vs 0.116, m2w-element 0.055 vs 0.102 — and *much worse* on the
two large-K tasks: swde-field 0.116 vs 0.049 and banking77 **0.322 vs 0.017**. Renormalising K
independent sigmoids is what §6 predicted would break at large K, and it does. The fix is cheap and
post-hoc: the 500-gold vector fit takes banking77 from ECE 0.322 to 0.051 and NLL 3.33 to 2.38.
`teacher500` (no gold, one minute of teacher) gets a third of the way there (0.237 / 3.02).
`prior:` on unlabelled calib texts is a *marginal* correction only and can hurt: with no `prior:`
declared in any YAML the variant assumes uniform, which is right for kinopoisk (+2.0 pts accuracy,
ECE 0.076→0.054) and wrong for toxic, where forcing a 50 % positive rate on an 8 %-positive task
drives ECE to 0.194 (AUROC is unaffected — a constant bias is monotone in p(true)).

### Warm start on kinopoisk (`--warm`)

Both arms are the same pair architecture on the same N teacher-labelled rows with the same 500-row
vector calibration and the same eval ids; the only factor is the init.

| task | N | init | seeds | metric | mean | Δ vs scratch (95 % CI) |
|---|---|---|---|---|---|---|
| kinopoisk | 100 | base fold model | 3 | acc | 0.5627 | +0.0478 (+0.0242..+0.0707) |
| kinopoisk | 100 | mmBERT-small | 3 | acc | 0.5149 | — |
| kinopoisk | 500 | base fold model | 3 | acc | 0.5933 | -0.0142 (-0.0304..+0.0020) |
| kinopoisk | 500 | mmBERT-small | 3 | acc | 0.6076 | — |

At **N = 100** the pooled CI excludes 0, but the whole effect is one seed: per-seed accuracy is
0.557/0.569/0.562 warm against 0.561/**0.415**/0.569 scratch. Warm-starting bought a *variance*
reduction (it stopped one from-scratch run collapsing), not a level. Claiming "+4.8 pts at N = 100"
without that sentence would be the overstatement the reviewers have twice caught.
At **N = 500** there is no measurable difference (Δ = −1.4 ± 1.6 pts, 95 % CI −3.0..+0.2, resolvable
to ±2.4 pts) — as §6 expected, from-scratch has already saturated. Reference lines on the same eval
rows: the 4000-row per-task student 0.657, the 500-gold-in-loss run 0.616, base-F1 zero-shot 0.525.

### Not done

No unseen pre-registered set (yahoo_answers_topics, inappropriateness, sst5) was opened: they were
never labelled, and the teacher budget was closed for this run. No fold F3, no diversity ablation
(M6), so **nothing here says whether the 27 new tasks helped** — that question is untouched.

```bash
python scripts/build_pairs.py && python scripts/base_gate.py
scripts/gpu_queue.sh runs/queue-base.jobs          # base train --fold F1 | F2 | none
scripts/gpu_queue.sh runs/queue-base-eval.jobs     # base eval, four calib variants per task
scripts/gpu_queue.sh runs/queue-base-warm.jobs     # scripts/warm_start.py + the agnews regression
python scripts/base_table.py [--variant V | --calib | --warm]
```


## B2 — `open-jev-base`, converged: one unseen-question claim survives, the diversity ablation is voided by its own STOP rule

Preregistration: [`docs/prereg/base-v2.md`](prereg/base-v2.md), commit `317e01e`, written before any arm
existed and before any number below was computed. Every threshold, fold, seed, run-dir name and STOP
rule in this section is quoted from that file; nothing here was chosen after seeing a number. The
mixture is `data/pairs.jsonl` as B1 left it — 42 tasks, 1,183,247 pairs, byte-identical, with the
three unseen sets excluded by name in `scripts/build_pairs.py`'s `SKIP`.

Code added first, exactly the four things §2 names: `--max-epochs`, `--patience`, `--min-delta`,
`--only` on `openjev base train`, plus the `min_delta` rule and the epoch ceiling in `base.train`
(`converged` is now a field of `train.json`). Patience counts only evaluations that beat the best by
`min_delta`; `student/` is still the best-BCE checkpoint. Tests: `tests/test_base.py` (4 new) and
`tests/test_base_v2.py` (4 new); `pytest -q tests` = **60 passed**.

### Convergence (a precondition, not a claim)

Stop rule: mean BCE on the held-in tasks' calib pairs every 1,500 steps, `min_delta` 0.0010,
patience 3, ceiling 6 epochs over the capped mixture. No wall-clock budget.

| arm | run dirs in mixture | pairs/epoch | ceiling (steps) | steps run | best step (epoch) | best calib BCE | converged | min | peak GPU |
|---|---|---|---|---|---|---|---|---|---|
| `base-F1-v2` (= ABL-all s0) | 31 | 171,116 | 32,085 | 15,000 | 10,500 (3) | 0.3562 | yes | 76 | 8.93 GB |
| `base-F1-v2-s1` | 31 | 171,116 | 32,085 | 27,000 | 22,500 (5) | 0.3385 | yes | 138 | 8.92 GB |
| `base-F1-v2-s2` | 31 | 171,116 | 32,085 | 18,000 | 13,500 (3) | 0.3594 | yes | 92 | 8.92 GB |
| `base-F2-v2` | 35 | 176,316 | 33,060 | 16,500 | 12,000 (3) | 0.3637 | yes | 113 | 8.91 GB |
| `base-none-v2` | 40 | 234,316 | 43,935 | 15,000 | 10,500 (2) | 0.3569 | yes | 100 | 8.91 GB |
| `base-F1-v2-orig` (ABL-orig s0) | 9 | 86,500 | 16,219 | 16,219 | 16,219 (6) | 0.2641 | **no** | 94 | 8.92 GB |
| `base-F1-v2-orig-s1` | 9 | 86,500 | 16,219 | 16,219 | 12,000 (5) | 0.2587 | at the ceiling | 93 | 8.93 GB |
| `base-F1-v2-orig-s2` | 9 | 86,500 | 16,219 | 16,219 | 13,500 (6) | 0.2611 | **no** | 94 | 8.94 GB |

78–105 pairs/s, 13.3 GPU-hours for the eight arms. B1's complaint is fixed for the five large-mixture
arms: early stopping fired on its own, between epoch 3 and epoch 5, 2–4× past the 45-minute budget
that cut B1. Calib BCE is **not comparable across arms with different mixtures** — ABL-orig's 0.26 is
a mean over 960 calib pairs from 9 tasks, ABL-all's 0.36 over 2,939 pairs from 31.

`base-F1-v2-orig-s1` is the awkward one: its patience rule fired on the very last evaluation the
ceiling allowed (best at 12,000, three non-improvements at 13,500 / 15,000 / 16,219). It is counted
as converged because the rule fired, and flagged because one more epoch would have been needed to
know.

### STOP rules that fired

- **§5, "a fold or ablation model reaches 6 epochs without early stopping" — fired on
  `base-F1-v2-orig` (seed 0) and `base-F1-v2-orig-s2` (seed 2).** Both ran the full 16,219-step
  ceiling with calib BCE still falling (seed 0: 0.2644 → 0.2641 on the last two evaluations; seed 2's
  best was at 13,500 of 16,219). Their arm is compared below only to say that the comparison is void:
  §4 says *"no comparison may be drawn between a converged and a non-converged arm"*, and 2 of the 3
  ABL-orig seeds are non-converged while all 3 ABL-all seeds converged. **The diversity ablation is
  therefore not a result of this round**, whatever its numbers say — and the numbers below say
  "improvement", which is exactly the direction a half-trained control arm would fake.
- No other STOP rule fired. Checked and clean: no `tasks/unseen/*` in `data/pairs.jsonl` or in any
  arm's `train_tasks` (42 tasks, 1,183,247 pairs, `SKIP` holds); `label.json`'s `model` identical
  across the mixture and unchanged since B1 (`typesafe/jev-1.13-20260917` on the 31 `-jev` dirs,
  `Qwen/Qwen3.8-27B-FP8` on 6, none re-labelled; the Jev backend writes no `prompt_sha`, and
  `git diff f52aa0c HEAD -- tasks/` touches no `-jev` task YAML, only `folds.yaml` and the new
  `tasks/unseen/*`); the three unseen sets have `n_failed` 0/700 and no teacher collapse on calib
  (top option 16.0 % / 27.5 % / 55.5 %, against the 95 % guard); shared eval rows 500 / 500 / 500
  (guard 450) and 1500 / 2000 / 3000 on kinopoisk / swde-field / toxic (guards 1400 / 1900 / 2800);
  all four calibration variants fitted on calib rows only; no arm scored on a task in its own
  mixture; no teacher call and no spend in this round (the unseen labels cost $0.0253 before the
  freeze, against the $0.50 cap).

### Leave-one-source-out, converged (`python scripts/base_table.py --tag v2 --variant gold500`)

Chance = a uniform `[N, K]` scored by the task's own metric; student = the per-task run's 3 seeds
averaged; paired bootstrap over the shared eval rows, 2,000 resamples. `gold500` is the
preregistered primary variant.

| task | fold | metric | n | chance | teacher | student | base-zs v2 | Δ vs student (95 % CI) | Δ vs teacher (95 % CI) | B1 (unconverged) |
|---|---|---|---|---|---|---|---|---|---|---|
| kinopoisk | F1 | acc | 1500 | 0.333 | 0.652 | 0.657 | 0.519 | −0.138 (−0.164..−0.112) | −0.133 (−0.164..−0.101) | 0.525 |
| swde-field | F1 | acc | 2000 | 0.052 | 0.803 | 0.869 | 0.317 | −0.552 (−0.574..−0.529) | −0.487 (−0.511..−0.460) | 0.353 |
| toxic | F1 | auroc | 3000 | 0.500 | 0.817 | 0.856 | 0.829 | −0.027 (−0.055..−0.000) | +0.012 (−0.017..+0.042) | 0.809 |
| georeview | F2 | mae | 2000 | 1.201 | 0.619 | 0.607 | 0.767 | +0.161 (+0.143..+0.179) | +0.149 (+0.129..+0.170) | 0.751 |
| banking77 | F2 | acc | 2000 | 0.015 | 0.764 | 0.750 | 0.372 | −0.378 (−0.403..−0.355) | −0.392 (−0.418..−0.366) | 0.452 |
| m2w-element | F2 | acc | 900 | 0.049 | 0.649 | 0.059 | 0.136 | +0.077 (+0.051..+0.103) | −0.513 (−0.550..−0.476) | 0.152 |

**Convergence did not buy transfer.** B1's headline — zero-shot loses to the per-task student on 5 of
6 tasks, every CI excluding 0 — survives unchanged, and the converged model is *worse* on 5 of the 6
than the one the 45-minute budget cut short (banking77 −8.0 pts, swde-field −3.6 pts, kinopoisk
−0.6 pts, georeview +0.016 MAE, m2w-element −1.6 pts; only toxic improves, +2.0 pts AUROC). Training
to the minimum of held-in calib BCE overfits the mixture's tasks at the expense of a held-out one.
That is a diagnostic reading of a secondary comparison across two training recipes, not a
preregistered claim — but it kills the one explanation B1 offered for its own result ("these models
were never converged").

The two non-losses survive in the same shape: **m2w-element 0.136 vs the fixed 16-way head's 0.059,
Δ = +7.7 pts (CI +5.1..+10.3)**, the PLAN-4 §0 primitive argument again; and **toxic 0.829 vs the
teacher's 0.817, Δ = +0.012 (CI −0.017..+0.042) — still no measurable difference from the teacher**,
now from a converged model. The per-task student still beats base zero-shot on toxic (0.856,
CI −0.055..−0.000, i.e. the CI only just excludes 0).

### The three pre-registered unseen sets (`python scripts/base_v2.py --unseen`)

`base-none-v2`, seed 0, the shipped artefact, on the eval splits of `tasks/unseen/*` (n = 500 each),
never in any mixture, Jev-labelled before the freeze. Teacher-normalised =
(metric − chance) / (teacher − chance), sign-flipped for MAE. `gold500`, fitted on each task's own
200 calib rows.

| task | metric | n | K | chance | teacher | base-none-v2 | advantage over chance (95 % CI) | teacher-normalised |
|---|---|---|---|---|---|---|---|---|
| yahoo-topics | acc | 500 | 10 | 0.088 | 0.718 | 0.528 | +0.440 (+0.386..+0.492) | 0.70 |
| sst5 | mae | 500 | 5 | 1.152 | 0.493 | 0.775 | +0.377 (+0.324..+0.435) | 0.57 |
| ru-inappropriate | auroc | 500 | 2 | 0.500 | 0.891 | 0.619 | +0.119 (+0.072..+0.168) | 0.30 |

**The preregistered rule is met**: the CI on (metric − chance) excludes 0 on all three, and the
teacher-normalised score is ≥ 0.5 on 2 of 3 (0.70, 0.57; ru-inappropriate 0.30). So the §4 sentence
is earned: on three questions it was never trained on, over three text sources it never saw, the pair
scorer recovers 70 % / 57 % / 30 % of what the Jev teacher recovers over chance.

Two things that keep it honest, both preregistered as diagnostics:

- **It is variant-sensitive at the margin.** The same table at `teacher500` (no gold: temperature
  fitted to the teacher's soft probabilities on 200 calib rows) puts sst5 at 0.48 and the rule would
  read 1 of 3 — i.e. the claim leans on the 200 *gold* calib labels, which a real deployment on a new
  question would have to supply. `raw` 0.70 / 0.57 / 0.30, `prior` 0.72 / 0.62 / 0.30 (all three beat
  chance under every variant; only the ≥ 0.5 count moves).
- **The weakest set is the one closest to the model's reason for existing.** ru-inappropriate is a
  Russian `noul` safety question at 0.619 AUROC — 30 % of the teacher — while the model's *best*
  unseen result is a 10-way English topic choice. Whatever this model is, it is not a safety
  classifier for a new language.

### Diversity ablation — **VOID by STOP §5** (`python scripts/base_v2.py --ablation`)

Δ = ABL-all − ABL-orig on fold F1's held-out tasks, 3 seeds each, per-row paired across all six run
dirs, seeds pooled by averaging the per-row calibrated probabilities. ABL-all is `base-F1-v2{,-s1,-s2}`
(31 run dirs); ABL-orig is the same configuration with `--only` the 9 preregistered original run dirs
(`agnews arb-quality arb-success banking77 georeview georeview-jev headlines-8k m2w-element
m2w-target`). The difference is exactly the 22 new Jev-labelled run dirs that fold F1 leaves in.

| task | metric | n | seeds | ABL-all | ABL-orig | Δ (95 % CI) | per-seed Δ | signs agree | rule |
|---|---|---|---|---|---|---|---|---|---|
| kinopoisk | acc | 1500 | 3v3 | 0.524 | 0.501 | +0.0224 (+0.0064..+0.0376) | +0.020, +0.016, +0.031 | yes | would pass |
| swde-field | acc | 2000 | 3v3 | 0.348 | 0.245 | +0.1030 (+0.0898..+0.1170) | +0.055, +0.248, +0.005 | yes | would pass |
| toxic | auroc | 3000 | 3v3 | 0.812 | 0.663 | +0.1492 (+0.1227..+0.1735) | +0.217, +0.042, +0.189 | yes | would pass |

All three parts of the §4 rule are met on 3 of 3 tasks (CI excludes 0, |Δ| ≥ 2.0 pts, per-seed signs
agree), and non-inferiority holds on all three. **It still does not count.** Two of the three
ABL-orig seeds hit the 6-epoch ceiling with calib BCE falling, which is the §5 STOP rule, and §4
forbids comparing a converged arm with a non-converged one. An under-trained control arm biases Δ
upward; the rule exists precisely so that a favourable void cannot be cashed as a finding. The
preregistered wording for this round is therefore: **the diversity ablation was not completed; the 27
new tasks are neither shown to help nor shown not to.**

What the numbers are good for is diagnosis, and two features of them survive the void because they
are about mechanism, not size:

- The per-seed spread is enormous on the two tasks whose Δ is large: swde-field +0.005 / +0.055 /
  +0.248, toxic +0.042 / +0.189 / +0.217. A "+10 pt" pooled Δ that ranges over 24 points between
  seeds is not a measurement of a 2-pt threshold, whatever the pooled CI says.
- The gap is largest on **toxic**, and the 22 added run dirs include `tweet-hate`, `tweet-offensive`,
  `tweet-irony` and seven other `noul` tasks, against ABL-orig's two `noul` tasks (`arb-success`,
  `m2w-target`, both agent/web questions). The honest hypothesis is "the mixture gained near-neighbour
  toxicity questions", not "diversity in general transfers" — and a fold that holds out the text
  source cannot separate those two.

The preregistered *secondary* read-out (the same Δ on the three unseen sets, explicitly "no claim")
was **not run**: `base.zs_eval` refuses to score a fold model on any task outside its recorded
`excluded` list, and §2 froze `base.py` to the four additions named there, so the guard could not be
relaxed without breaking the freeze. That trade — keep the freeze, lose a no-claim diagnostic — is the
one the prereg's own wording forces.

### Calibration under transfer, converged (`python scripts/base_table.py --calib --tag v2`)

| task | arm | acc | ECE raw | ECE cal | NLL |
|---|---|---|---|---|---|
| kinopoisk | per-task student | 0.660 | 0.1608 | 0.0378 | 0.744 |
| kinopoisk | base-F1-v2 raw | 0.466 | 0.1794 | 0.1794 | 1.106 |
| kinopoisk | base-F1-v2 prior | 0.522 | 0.1794 | 0.0572 | 0.995 |
| kinopoisk | base-F1-v2 teacher500 | 0.466 | 0.1794 | 0.0631 | 1.033 |
| kinopoisk | base-F1-v2 gold500 | 0.519 | 0.1794 | 0.0622 | 0.987 |
| swde-field | per-task student | 0.863 | 0.0488 | 0.0361 | 0.597 |
| swde-field | base-F1-v2 raw | 0.212 | 0.0419 | 0.0419 | 2.440 |
| swde-field | base-F1-v2 prior | 0.338 | 0.0419 | 0.2041 | 2.580 |
| swde-field | base-F1-v2 teacher500 | 0.212 | 0.0419 | 0.1483 | 2.272 |
| swde-field | base-F1-v2 gold500 | 0.317 | 0.0419 | 0.1232 | 1.949 |
| toxic | per-task student | 0.917 | 0.1164 | 0.0373 | 0.252 |
| toxic | base-F1-v2 raw | 0.666 | 0.0588 | 0.0588 | 0.581 |
| toxic | base-F1-v2 prior | 0.511 | 0.0588 | 0.1972 | 0.790 |
| toxic | base-F1-v2 teacher500 | 0.666 | 0.0588 | 0.0436 | 0.582 |
| toxic | base-F1-v2 gold500 | 0.919 | 0.0588 | 0.0559 | 0.266 |
| georeview | per-task student | 0.558 | 0.2527 | 0.0246 | 1.048 |
| georeview | base-F2-v2 raw | 0.412 | 0.0160 | 0.0160 | 1.295 |
| georeview | base-F2-v2 prior | 0.414 | 0.0160 | 0.0688 | 1.251 |
| georeview | base-F2-v2 teacher500 | 0.412 | 0.0160 | 0.0573 | 1.292 |
| georeview | base-F2-v2 gold500 | 0.412 | 0.0160 | 0.0487 | 1.288 |
| banking77 | per-task student | 0.751 | 0.0171 | 0.0422 | 1.008 |
| banking77 | base-F2-v2 raw | 0.368 | 0.3074 | 0.3074 | 3.521 |
| banking77 | base-F2-v2 prior | 0.349 | 0.3074 | 0.2709 | 3.234 |
| banking77 | base-F2-v2 teacher500 | 0.368 | 0.3074 | 0.2443 | 3.344 |
| banking77 | base-F2-v2 gold500 | 0.372 | 0.3074 | 0.0545 | 2.722 |
| m2w-element | per-task student | 0.059 | 0.1017 | 0.0036 | 2.773 |
| m2w-element | base-F2-v2 raw | 0.136 | 0.0569 | 0.0569 | 2.735 |
| m2w-element | base-F2-v2 prior | 0.140 | 0.0569 | 0.0643 | 2.735 |
| m2w-element | base-F2-v2 teacher500 | 0.136 | 0.0569 | 0.0441 | 2.721 |
| m2w-element | base-F2-v2 gold500 | 0.136 | 0.0569 | **0.8550** | **inf** |

B1's calibration finding survives convergence in the same shape: out of the box the pair scorer is
better calibrated than the per-task student on georeview (0.016 vs 0.253), toxic (0.059 vs 0.116),
m2w-element (0.057 vs 0.102) and swde-field (0.042 vs 0.049), and catastrophically worse on
banking77 (**0.307 vs 0.017**) — renormalised independent sigmoids at K = 77 — where the 500-gold
vector fit repairs it post-hoc to 0.055 (NLL 3.52 → 2.72). It is now worse than the student on
kinopoisk too (0.179 vs 0.161), where B1's shorter run was better (0.076): more training sharpened
the probabilities past the point where the raw sigmoids were accidentally well spread.

One new failure, diagnostic and unexplained: **m2w-element `gold500` has ECE 0.855 and NLL `inf`** —
the vector fit on 500 calib rows drove one option's calibrated probability to 0 on a row that carries
it as gold. Accuracy (0.136) is unaffected, since argmax is. Anything that reads `nll` off that file
is reading a degenerate fit.

### Verdict on publishing the model

Measured against what §1 of the preregistration asked *before* any of this existed — ship with a
zero-shot story, or ship as an init only — the answer is: **ship it, with the zero-shot sentence the
unseen sets actually earned and no sentence about diversity at all.** The earned sentence is narrow:
on three questions and three text sources the model never saw, it beats chance with the CI excluding
0 on all three and recovers 70 % / 57 % / 30 % of the teacher's margin over chance, with 200 gold
calib rows per task; under a no-gold calibration the middle number drops to 48 % and the rule is met
on 1 of 3. Everything else in this round argues for modesty: the converged model is *worse* at
leave-one-source-out transfer than B1's unconverged one on 5 of 6 tasks, it still loses to every
per-task student except the 16-way m2w-element head, and its raw probabilities at K = 77 are
unusable without a calibration fit. The diversity question the round was half-built to answer is
**void by its own STOP rule** — two of three control seeds never converged — so the README may not
say the 27 small tasks bought anything, and the honest next round is ABL-orig re-run under a step
budget matched to ABL-all rather than an epoch budget, which is an amendment, not a re-analysis of
these files. The words "general" and "any question" stay out of the write-up by the §4 rule's own
logic even though the rule passed: it passed on 2 of 3 at the primary variant, with the weakest
result on the safety-shaped question, and a claim of generality from three sets and one seed would be
exactly the overstatement this preregistration was written to prevent.

```bash
# the eight arms (13.3 GPU-h) and the 24 zero-shot evaluations, all under the memory gate
scripts/gpu_queue.sh runs/queue-base-v2.jobs
scripts/gpu_queue.sh runs/queue-base-v2-eval.jobs
python scripts/base_table.py --tag v2 --variant gold500      # LOTO   -> results/base/base-loto-v2-*.json
python scripts/base_table.py --tag v2 --calib                # calibration under transfer
python scripts/base_v2.py --unseen                           # -> results/base/unseen-base-none-v2-*.json
python scripts/base_v2.py --ablation                         # -> results/base/ablation-*.json (VOID, see §5)
```
