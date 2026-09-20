# REVIEW-session2 — M6.2 honesty review (README Findings, docs/, results/)

Read-only check of README.md Findings + Results table, docs/experiments.md, docs/task-spec.md,
docs/img/README.md and make_figures.py against results/*.json, results/variants/*.json,
runs/*/{train,calib,label}.json and runs/*/eval_rows.jsonl. CPU checks run: `pytest -q tests`
(33 passed), `openjev --help` for every subcommand, `python scripts/r1_table.py <seed dirs>`
(refits calibration from calib_logits.pt, prints only), and a paired bootstrap of student vs
teacher on eval_rows.jsonl. `openjev report` and the figure script were NOT run (both write).

Bottom line: the numbers are almost all right and the tone is honest. The four things that
must change before push are (1) the README's own description of its evidence base is wrong
in two places (epochs is one seed, not three; the 3-seed calibration evidence exists but is
not on disk anywhere), (2) "four students beat their teacher" fails the repo's own claim rule
for two of the four, (3) the agnews-nogold 0.8825 number no longer exists in results/ and the
figure script crashes on it, (4) task-spec.md still tells newcomers to raise epochs, the
opposite of the README's verdict. Everything else is stale prose or wording.

Rows quoted from the CPU refit (`scripts/r1_table.py`, one line per seed, T-only → vector,
paired 95 % CI over eval rows):

| run | T-only | vector | Δ vector |
|---|---|---|---|
| headlines / -s1 / -s2 | 0.767 / 0.763 / 0.764 | 0.843 / 0.833 / 0.839 | +7.50 (+5.95..+9.05) / +7.00 (+5.55..+8.50) / +7.40 (+5.90..+8.85) |
| kinopoisk / -s1 / -s2 | 0.609 / 0.609 / 0.605 | 0.660 / 0.654 / 0.657 | +5.13 (+2.87..+7.47) / +4.53 (+2.33..+6.67) / +5.13 (+2.80..+7.53) |
| agnews / -s1 / -s2 | 0.879 / 0.882 / 0.875 | 0.889 / 0.893 / 0.881 | +0.95 (−0.10..+1.95) / +1.05 (+0.05..+2.05) / +0.65 (−0.35..+1.70) |
| headlines-8k / -s1 / -s2 | 0.777 / 0.787 / 0.775 | 0.852 / 0.853 / 0.849 | +7.40 / +6.55 / +7.35, all CIs exclude 0 |

Paired bootstrap, student (calibrated) vs teacher argmax, same eval rows:

| run | Δ student − teacher | 95 % CI |
|---|---|---|
| headlines | +5.95 pts | +4.25..+7.65 |
| headlines-8k | +6.85 pts | +5.15..+8.55 |
| kinopoisk (s0 / s1 / s2 / pooled) | +0.80 / +0.20 / +0.47 / +0.49 | −1.80..+3.47 / −2.33..+2.80 / −2.07..+3.20 / −1.89..+2.96 |
| georeview MAE | −0.0095 | −0.0227..+0.0047 |
| agnews (s0 / s1 / s2) | +0.90 / +1.30 / +0.15 | −0.30..+2.05 / +0.20..+2.45 / −1.10..+1.40 |
| banking77 | −1.35 pts | −3.10..+0.35 |

---

## A. Confirmed errors — must fix before push

### A1. README says the epochs delta is 3-seed pooled; it is one seed per arm
- **README.md:269-270** — "Training-side deltas (epochs, augment, more labels, where gold is
  spent) are pooled over **three seeds per arm**."
- **Evidence**: only `runs/{agnews,headlines,kinopoisk}-e12` exist (no -s1/-s2);
  docs/experiments.md:229 says so explicitly: "One seed per arm: `docs/` numbers, never README
  Findings." PLAN-2 §5 forbids one-seed pairs in README Findings.
- **Fix**: drop "epochs" from that list, and in the bullet at README.md:339-343 add "(one seed
  per arm — a null result on a 12-epoch run that early-stopped anyway, kept here because it
  decides the default; the CIs are over eval rows only)". Alternatively move the bullet to
  docs/ and leave one sentence: "5 vs 12 epochs: no measurable difference on one seed; the
  default stays 5."

### A2. The one-seed calibration deltas are argued around instead of backed — and the backing exists
- **README.md:270-273** — "The post-hoc calibration deltas are over eval rows at one seed per run
  — the tasks they matter on move 5–8 points, far outside the 0.6–1.2 pt seed band, and the
  3-seed table rows carry the same conclusion."
- **docs/experiments.md:130-132** contradicts it: "Every Δ above is **one seed per run**. Under
  PLAN-2 §5 that makes them `docs/` numbers; the README gets them only with the 3-seed arms
  behind them." The 3-seed arms were trained but never refit into the R1 table, so nothing on
  disk supports "carry the same conclusion" (the README table shows only post-bias means).
- **Is the argument true?** Yes — see the refit table at the top: headlines +7.0/+7.4/+7.5,
  kinopoisk +4.5/+5.1/+5.1, every CI excludes 0; agnews stays inside noise on 2 of 3 seeds.
  The comparison of a paired within-run delta against the across-seed accuracy sd is not quite
  the right argument (the sd measures the student's accuracy, not the bias's effect), but the
  per-seed refit makes it moot.
- **Fix**: append the seed rows to the R1 table in docs/experiments.md (they come straight from
  `python scripts/r1_table.py runs/headlines runs/headlines-s1 runs/headlines-s2 runs/kinopoisk
  runs/kinopoisk-s1 runs/kinopoisk-s2 runs/agnews runs/agnews-s1 runs/agnews-s2`), delete the
  "far outside the seed band" sentence in README and replace with "refit on all three seeds:
  headlines +7.0..+7.5, kinopoisk +4.5..+5.1, every CI excludes 0 (`docs/experiments.md`)".
  Update experiments.md:130-132 to say the 3-seed rows are now in the table.

### A3. "four students beat their own teacher" — two of the four do not, by the repo's own rule
- **README.md:285-288** — "headlines 0.843 vs 0.783, headlines-8k 0.8515 vs 0.783, kinopoisk
  0.660 vs 0.652, georeview MAE 0.609 vs 0.619. Before it, three of those four were behind."
- **Evidence**: paired bootstrap above. kinopoisk +0.8 pts (CI −1.8..+3.5; pooled over 3 seeds
  +0.5, CI −1.9..+3.0) and georeview −0.0095 MAE (CI −0.023..+0.005) both include 0. PLAN-2 §5:
  a delta whose CI includes 0 is "no measurable difference", and one-seed deltas under 2 pts
  never go in the README. headlines and headlines-8k clearly beat (+6.0 and +6.9, CIs exclude 0).
- Also **"three of those four were behind" is wrong: all four were.** headlines-8k T-only is
  0.777 < 0.783 (refit table), headlines 0.767, kinopoisk 0.609, georeview 0.783 MAE.
- **Fix**: "With the bias, two students beat their teacher (headlines +6.0 pts, 95 % CI
  +4.3..+7.7; headlines-8k +6.9, +5.2..+8.6) and two match it (kinopoisk 0.660 vs 0.652,
  georeview MAE 0.609 vs 0.619 — CIs include 0). Before it, all four were behind."

### A4. The agnews-nogold headline number is no longer on disk; the figure script crashes on it
- **README.md:381-388** — "**0.8825** against the teacher's **0.8815** ... (`gold_acc_offline` in
  `results/agnews-nogold.json` comes from `scripts/nogold_gold_acc.py` ... re-running `openjev
  eval` overwrites the file without it.)"
- **Evidence**: `results/agnews-nogold.json` has no `gold_acc_offline` / `teacher_gold_acc_offline`
  key (verified: key list). The M1.5 re-eval overwrote it, exactly as the parenthesis warns.
  `docs/img/make_figures.py:196` does `ng["gold_acc_offline"]` → KeyError, so the documented
  command `python docs/img/make_figures.py` (docs/img/README.md:7) is dead. (It would then also
  KeyError at make_figures.py:267 `c["acc"]` on georeview/toxic cascades, which carry `mae`/`auroc`.)
- **Fix**: re-run `python scripts/nogold_gold_acc.py` (student forward on 2000 rows, CPU is fine,
  no teacher), and make the number survive the next eval: have the script write
  `results/variants/agnews-nogold-goldacc.json` (or have `evaluate.run` preserve keys it did not
  write). Point README:385 at wherever it lands. Then fix make_figures.py (A7) or drop it.

### A5. task-spec.md tells users the opposite of the README's epochs verdict
- **docs/task-spec.md:137** — "early stopping (patience 2) on calib KL **never triggered in the
  seven runs in this repo** — the best epoch was the last one and calib KL was still falling
  every time, so `epochs` is the binding knob. Raise it, especially for large label sets."
- **Evidence**: README.md:339-343 and docs/experiments.md:213-217, 240-262: all three 12-epoch
  runs early-stopped (9, 7, 5 epochs), `--epochs` changes the LR-decay horizon, "the default
  stays 5". kinopoisk-s1 best epoch is 3 of 5 (runs/kinopoisk-s1/train.json), so "best epoch
  was the last one" is no longer true for the seed runs either.
- **Fix**: "Upper bound with early stopping (patience 2) on calib KL. 5 vs 12 was measured on
  three tasks: no measurable difference — `--epochs` mostly changes the LR-decay horizon (see
  README Findings). banking77 (77 classes) uses 12 in its YAML."

### A6. README Results table is stale relative to results/ (bench ran after `report`)
- **README.md:201** headlines-8k (n=3) shows `– | –` for GPU p50 / ex/s; **README.md:206**
  kinopoisk-gold shows `– | –`.
- **Evidence**: results/headlines-8k.json has `latency_ms.gpu_b1_p50 = 5.28`, `throughput_gpu_b64
  = 2561`; results/kinopoisk-gold.json has 7.00 / 186. `report()` (openjev/evaluate.py:325-352)
  aggregates over the seeds that have latency, so it would print 5.3 / 2561 and 7.0 / 186.
- **Fix**: `openjev report` before commit (already M6.3 step 3). PLAN-2 §6: "README and
  `results/` must agree at every commit".

### A7. docs/experiments.md R2 table: the "student alone" column is the τ=0.30 row (the known hazard)
- **docs/experiments.md:146-158** and the prose at **:164-168** — "headlines (student 0.845 vs
  teacher 0.783)", "banking77 0.758 → 0.776 at 25 %", "georeview MAE 0.606 → 0.583".
- **Evidence**: `scripts/r1_table.py` `r2()` prints `c[0][m]`, i.e. the τ=0.30 point, which
  already escalates 1.0 % (headlines, true student 0.8425), 4.7 % (banking77, true 0.7505),
  0.95 % (georeview, true 0.609). The README (330-331) uses the right numbers; the docs table
  and prose do not. docs/findings-v2-draft.md:22-25 carries the same wrong numbers.
- **Fix**: in `r2()` read the column from `res["student"]["acc"]` / `res["score"]["mae"]` /
  `res["noul"]["auroc"]`, or rename the header to "τ = 0.30 (0–5 % escalated)"; regenerate the
  table; change :164 to "student 0.843", :167-168 to "banking77 0.751 → 0.776 at 27 %,
  kinopoisk 0.660 → 0.673 at 9 %, georeview MAE 0.609 → 0.583 at 26 %" (the README's rows).

### A8. "9.0 minutes of training instead of 1.7" pairs numbers from two different tasks
- **README.md:370-373** — "mmBERT-base ... 9.9 GB of training memory instead of 5.5, and 9.0
  minutes of training instead of 1.7."
- **Evidence** (results/*.json `train_minutes`): georeview-mmBERT-base 9.02 vs georeview 10.24
  (base was *faster*); agnews-mmBERT-base 7.42 vs agnews 3.54 (seed 0; s1 1.62, s2 0.97). No run
  pairs 9.0 with 1.7. Peak memory is fine (9.75 / 9.87 vs 5.5).
- **Fix**: "9.8–9.9 GB of training memory instead of 5.5, and 7.4 minutes instead of 1–3.5 on
  agnews (on georeview the two took the same ~10 minutes — the box was busy, training time here
  is not a clean number)". Or drop the minutes clause; the throughput numbers already make the point.

---

## B. Stale or inconsistent prose — should fix (nice-to-have, 5 minutes each)

### B1. Figure captions and figure script text are pre-M1
Figures are not embedded in README.md (no `docs/img` reference), so this is docs-only, but
docs/img/README.md:3 calls them "candidate illustrations for the top-level README".
- **docs/img/README.md:39** and **make_figures.py:144,153**: "→ one temperature →" — every run
  except toxic/agnews-nogold ships `method: vector` (T + per-class bias).
- **docs/img/README.md:40** and **make_figures.py:191,236**: "headlines is the mean of its two
  seeds", footnote "(0.768, 0.763)" hard-coded — those are the pre-M1 T-only values; there are
  three seeds now at 0.8425 / 0.833 / 0.8385.
- **docs/img/README.md:41** and **make_figures.py:263-265,317**: "every student already matches
  or beats its teacher at 0% escalation, and escalating buys +1.0 to +1.8 points" — banking77
  is 1.35 pts behind; on headlines escalation buys +0.3 at most and then costs. "Source:
  cascade[] in results/*-e12.json" / "(12-epoch runs)" is wrong twice: `by_task.setdefault`
  over `sorted(ALL)` picks `agnews` before `agnews-e12`, so the baseline runs are plotted.
- **docs/img/README.md:42** and **make_figures.py:330-355**: "ECE before → after temperature
  scaling", "grey = no-op (headlines, fit rejected)", "one scalar fitted on 500 held-out rows" —
  no run has T = 1.0 any more; headlines ECE went 0.025 → 0.029 (red, not grey) because the
  bias is fitted for NLL/accuracy, which the README explains at :245-249.
- **Fix**: either regenerate after A4 and rewrite the four captions to match, or delete
  docs/img/ from the push (`git rm`) until the figures are rebuilt. Half-stale figures shipped
  next to an honest README are worse than no figures.

### B2. docs/findings-v2-draft.md is scratch and contradicts the final text
"Not the README. M4 merges these." M4 is done; it still carries the R2 hazard numbers (0.758,
0.606, 0.845) and "1 seed, docs-only" notes. **Fix**: delete it.

### B3. docs/experiments.md:449-450 — `compare` "until the subcommand is wired in cli.py"
`openjev compare` exists (`openjev --help`). Delete the parenthesis.

### B4. Epoch indexing is mixed inside docs/experiments.md
- :216 "best at epoch 6 of 9 run, 4 of 7, 2 of 5" is 0-indexed (train.json `best_epoch`);
  the table at :244-251 ("7", "5", "3", "5 of 5") is 1-indexed; :435 "early-stop at epoch 2,
  against epoch 4 for pure distillation" is 0-indexed again and "early-stop" is wrong — the
  gold-n500 arms ran all 5 epochs (history has 5 rows) and *selected* epoch index 2 on
  `calib_loss` (KL + CE), not on calib KL (whose minimum is the last epoch).
- **Fix**: pick 1-indexed everywhere; :435 → "(the gold-weighted arms select their 3rd epoch on
  calib loss; pure distillation selects its 5th)".

### B5. README.md:320-323 — headlines-8k claim omits the calib re-draw
docs/experiments.md:364-384 makes a point of it: only 7 of 500 calib ids are shared, "the claim
below is stated with that in it rather than around it". The README sentence says only "identical
eval ids". **Fix**: add "(same 2000 eval ids; the 500-row calib split is re-drawn from the same
pool, so the fitted bias is not identical between arms)".

### B6. README.md:326-329 — "Four of the nine runs reach teacher parity at 0 % escalation"
Literally what `cascade_parity` says, but it undercounts because the τ sweep starts at 0.30
(`openjev/evaluate.py:54`), where headlines, georeview, headlines-8k already escalate ~1 %. The
student alone is at or above the teacher on 7 of 9 (all but banking77 and the definitional
agnews-nogold) — and the same sentence uses headlines as its example of "never escalate" while
headlines' recorded parity is 1.0 % / `escalate_below: 0.3` in runs/headlines/openjev.json, so
`serve` flags 1 % of headlines rows for a teacher that is 6 pts worse.
**Fix (docs)**: "Seven of the nine runs are at or above their teacher with no escalation at all
(the recorded parity point is at the sweep floor τ = 0.30, which already routes ~1 % of rows)".
**Fix (code, optional)**: `TAUS = [0.0] + [...]` so parity can be "never"; re-eval is CPU-only
(`openjev eval --no-latency` on the run dirs; do not use `run`).

### B7. README.md:367-368 — "This is the one task where escalation is the tool that helps"
Contradicts :329-331, which lists kinopoisk, agnews and georeview as also gaining from
escalation. **Fix**: "the one task where the bias does nothing and escalation is the only lever".

### B8. README.md:314-316 — "8× the labels buys 1.2 pts over the 500-row bias"
0.672 vs 0.660 is one seed vs one seed; against the 3-seed mean (0.657) it is 1.5; the kinopoisk
seed band is 0.6 pts and the paired CI was not run. Labelled "(one seed)" so not dishonest, but
"buys" is a claim. **Fix**: "reaches 0.672 (one seed, ~1 pt above the bias-only seeds — within
noise of them)".

### B9. README.md:421-425 — "mmBERT-base ... does not fit beside a resident 84 GB vLLM on this box"
Two mmBERT-base rows sit in the table above it. **Fix**: "did not fit during this session
(13.5 GB free next to vLLM); the two base rows were trained on the first night when more was".

### B10. README.md:79 — "30 ms on CPU for mmBERT-small"
CPU batch-1 p50 in results/*.json is 19–43 ms at `max_len` 256 and 174 ms on kinopoisk (512).
**Fix**: "5 ms on GPU and 20–45 ms on CPU at `max_len` 256 (174 ms on the 512-token task)".

### B11. README.md:390 — "headlines-8k added 4,434 rows in 9 more [minutes]"
runs/headlines-8k/label.json: 4466 calls, 8.10 min. experiments.md:389 says "9 min at 8.2 ex/s"
(4434 / 8.2 / 60 = 9.0 — the two numbers were computed from each other). **Fix**: "8 more".

### B12. `openjev check --probe` warning nudges toward free debiasing
openjev/check.py:75: "reword that option, or declare `prior:` and let calibration fix it". The
README (:156, :300-310) and task-spec (:40) carry the "only if you know it" clause; the CLI line a
newcomer actually sees does not. **Fix**: append " (only if you know the deployment prior)".

---

## C. Checked and found correct (no action)

- Every Δ and CI in README Findings matches docs/experiments.md and the refit: headlines
  +7.5 (+6.0..+9.1), kinopoisk +5.1 (+2.9..+7.5), georeview −0.17 (−0.19..−0.16), agnews +0.95
  (−0.10..+1.95), banking77 +0.20 (−1.35..+1.65); gold-spend +4.1 ± 1.7 (+2.4..+5.7) with per-seed
  0.603/0.622/0.623 vs 0.660/0.654/0.657 = results/*.json; headlines-8k +1.3 (+0.6..+2.1) with
  0.8515/0.853/0.849 vs 0.835/0.8365/0.8365; epochs +0.0 ± 0.6 / +0.7 ± 0.9 / +0.0 ± 1.5;
  augment +0.0 ± 0.5 (−0.5..+0.6) with the sign flip on seed 1; noise floor 0.888 ± 0.006 /
  0.838 ± 0.005 / 0.657 ± 0.003.
- Cascade pairs in README:330-331 are read off the correct rows: banking77 0.751 → 0.776 at
  τ=0.65 (27.4 %), kinopoisk 0.660 → 0.6727 at τ=0.50 (9.1 %), agnews 0.889 → 0.897 at τ=0.75
  (10.6 %), georeview 0.609 → 0.583 at τ=0.45 (26.0 %); headlines falls monotonically to 0.783.
- kinopoisk bias `[+0.80, +0.59, −1.39]` = runs/kinopoisk/calib.json. Permutation numbers
  (0.646/0.693, Good 56.4 → 41.9 %, Neutral 12.9/12.0/9.0 %) consistent between README:292-299,
  :447-455 and experiments.md:21-33 (perm_check.json not re-derived).
- prior-declared: kinopoisk 0.6553, georeview 0.5986 = results/variants/*.json; refitted
  0.655 / 0.599 match. "within half a point" holds.
- Teacher cost: six base runs sum to 66,522 calls / 144.3 min / 2.40 h; banking77 33,000 calls,
  60.9 min = 42.2 %; 5,500 rows × 6 calls. toxic AUROC 0.856 / 0.817, Brier-vs-fraction
  0.035 / 0.048, majority baseline 0.919 all match results/toxic.json.
- The `prior:` caveat is present and correctly hedged everywhere it matters: README:300-310,
  :438-446, :460-463; task-spec.md:40; experiments.md:118-125. Nothing in README or docs
  implies free debiasing (the only soft spot is the CLI string, B12). The model card
  (openjev/hub.py:50-54) states method, bias and target factually.
- toxic exclusion reasoning (AUROC invariant under a K=2 bias; calib 2.2 % vs eval 8.1 %) is
  correct and consistent across README, experiments.md and task-spec.md:259-266.
- CLI: every flag documented in README and task-spec exists (`--seed --tag --no-synth --gold-n
  --ignore-gold --no-latency --force`, `check --probe`, `compare`, `report`, `bench`, `serve
  --device`, `push --repo --dry-run --public`, `augment --rounds --per-class`). Every referenced
  file exists: scripts/{compare_api,nogold_gold_acc,prior_variant,r1_table,perm_check,
  seed_cache}.py, scripts/gpu_queue.sh, .github/workflows/test.yml, tasks/{example,agnews-nogold,
  headlines-8k}.yaml, results/api/*.json, results/variants/*.json, runs/*-rev/perm_check.json,
  LICENSE. End-to-end latency 21 / 160 ms = results/api. `pytest -q tests`: 33 passed.

## D. Suspicions (not verified, low stakes)

- README.md:9-11 / 122-127: the Jev adapter "has not been run against the live API" — fine, but
  the README line 9 says "none of [Jev's] claims are repeated here"; docs/task-spec.md and
  scripts/compare_api.py were not read for Jev numbers. Probably clean.
- headlines ex/s 2644 and banking77 ex/s 2644 look copy-pasted but are two distinct measurements
  (2644.46 vs 2644.01 in results/). Coincidence; no action.
- experiments.md:389 "6066 rows the two tasks share" and "59 promoted rows" were not re-derived
  from the split ids.
