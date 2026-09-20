# Open-Jev — session 2 plan (12 h, mixed research / framework)

> Written 2026-09-20 after reading README (Findings), PLAN.md, REVIEW.md, REVIEW-final.md,
> docs/task-spec.md, openjev/*.py, tasks/*.yaml, results/*.json, runs/*/train.json and the label logs.
> Implementation is by Opus agents; every milestone has a command that proves it. No code in this file.

## 0. Spine — the three results that make the session worth it

**R1. Teacher-prior debiasing recovers part of the hard-task gap, and it costs nothing.**
Measured today from `runs/*/teacher.jsonl` (eval split, *oracle* correction fitted on eval itself — a
diagnostic, never a claim): reweighting the teacher's probabilities by gold-prior / teacher-prior moves
teacher accuracy kinopoisk 0.652 → 0.684, georeview 0.470 → 0.547, headlines 0.783 → 0.817,
agnews 0.880 → 0.892, toxic 0.893 → 0.919. The teacher's dominant error on the Russian tasks is a
*marginal* shift (kinopoisk predicts Good 56.5 % / Neutral 12.7 % against 33/33/33), and a marginal
shift is exactly what a K-dim bias on the student's logits fixes. That bias can be fitted (a) on the 500
gold calib rows (vector scaling: `logits / T + b`), or (b) with **no gold at all** against a prior the
user declares in the YAML (`prior: uniform`). Both are post-hoc on the *existing* students — CPU,
minutes, no teacher, no retraining. This is the session's headline experiment.

**R2. A cascade / abstention curve on every run, shipped as a serve option.**
We already hold student probs, teacher probs and gold for the same eval rows. "Student answers when
`confidence ≥ τ`, else escalate to the teacher" is a pure offline sweep: escalation rate vs end-to-end
accuracy, for all nine runs, zero new teacher calls. It turns "the student lags the teacher by 4 pts on
kinopoisk" into "at X % escalation you get the teacher's accuracy at (1−X) of the cost". Served as
`"escalate": true/false` in the response.

**R3. Honesty becomes structural, and the newcomer loop gets a dry run.**
Reviewers caught overstated claims twice. This session: `--seed` runs with a grouped README table
(mean ± sd, n), per-row eval predictions saved, and `openjev compare A B` printing a paired-bootstrap CI.
No delta goes into the README without it. And `openjev check task.yaml [--probe 100]` shows a newcomer
exactly what the teacher will see, what it costs, and — for 100 calls — how biased the teacher is on
*their* prompt, before they spend an hour of teacher time. That is the low-barrier feature: the teacher
is the ceiling, so iterating on the prompt must be cheap.

Everything else is ordered around those three. Secondary, cheap, and worth doing: seed variance of the
augment gain (with a real control arm), the epochs default, "500 gold labels: spend them on calibration
or on a CE term?", and the option-order diagnostic.

## 1. Candidate directions: verdicts

| direction | verdict | why |
|---|---|---|
| teacher debiasing vs known prior / vector scaling | **do (R1)** | oracle headroom measured above; post-hoc, CPU-only |
| cascade router | **do (R2)** | offline from cached probs; ships as a serve flag |
| seed variance for augment | **do** | 6 trainings × 1.5 min; makes or kills the +0.5 pt claim |
| gold_weight sweep | **do, narrowed** | one question only: 500 gold on calib-bias vs 500 gold in CE, kinopoisk (+georeview if time) |
| more epochs | **do** | 3 runs; 5 was never the best epoch |
| option-order permutation | **do the diagnostic, not the feature** | relabel calib+eval of kinopoisk and headlines with reversed options (12 min teacher); implement averaging only if it moves the marginal |
| more unlabeled data | **do, headlines only** | +4000 rows = 6 min teacher, 3 × 1.5 min GPU; kinopoisk version optional |
| conformal sets | **small, optional** | 30 lines from calib.json; ships as `set` when asked for; cut first |
| few-shot in the labeling prompt | **drop** | spends gold inside the prompt; `check --probe` lets the user iterate on the prompt themselves, which is the same thing done honestly |
| self-consistency / prompt ensembles | **drop** | N× the one bottleneck; the permutation test is the cheapest member of this family — run it first, decide later |
| ordinal head for `score` | **drop** | georeview student already tracks the teacher's argmax at 0.80–0.82; a head change cannot fix what it faithfully copies. Debiasing attacks the actual error |
| mmBERT-base anything | **banned this session** | 9.9 GB next to an 84 GB vLLM with 13.5 GB free; a run was already killed. Not a research question anymore either (README: "not capacity-limited") |
| ONNX / CPU-quantised export | **drop to cut list** | CPU batch-1 p50 is already measured (19–43 ms for mmBERT-small); an aarch64 ORT + ModernBERT export spike is 1.5 h of wheel wrangling for a number users already have. Do `serve --device cpu` instead |
| PyPI package | **wheel yes, publish no** | `uv build` + install in a fresh venv is a 15 min proof; publishing is the user's call once the repo is public |
| `openjev init` scaffold / template gallery | **drop** | `cp tasks/example.yaml` is the scaffold; `check` is what a newcomer actually lacks. Add a 7-row index table of the task YAMLs to docs (10 min) |
| batch endpoint / multi-task serving | **batch yes (10 lines)**; multi-task already exists | |
| CI on GitHub Actions | **do** | 45 min; the tests are CPU-only already |
| metrics/report page | **drop** | `report` already regenerates the table; print it to stdout too |
| HF Hub push helper | **do, late, token-gated** | `upload_folder` is one call; `serve hf:<repo>` makes "try it without training" real |
| Jev head-to-head | **adapter + harness, late, key-gated** | our `/v1/systemone` mirrors the shape, so the harness smoke-tests against `openjev serve` itself |
| B5 / B6 / C11 | **do in M0** | 20 min total |

## 2. Hard rules (memory, teacher, honesty)

- **vLLM on :8000 is production. Never kill, restart, or exceed `concurrency: 32` against it.** The
  teacher lane in this plan totals under one hour of calls.
- **Host OOM kills the biggest process, and that is vLLM (84 GB).** Every training process runs under
  `choom -n 1000 --` (util-linux) so the OOM killer picks the trainer, and `scripts/gpu_queue.sh` refuses
  to start a job unless `torch.cuda.mem_get_info()` free ≥ 12 GB **and** `free -g` available ≥ 10 GB.
  One training at a time (`flock runs/.gpu.lock`). mmBERT-small only. len-512 tasks stay at bs 32
  (measured peak 8.9 GB); if free memory is below the gate, the queue waits, it does not shrink the batch
  (a different batch size is a different experiment).
- **Latency is measured only on an idle teacher.** `openjev bench` polls `:8000/metrics` for
  `vllm:num_requests_running == 0` before measuring, and every eval in the queue runs `--no-latency`.
  Existing latency numbers stay (same architecture → same latency); only seed-0 rows of *new* task
  variants get benched in M6.
- **Teacher labels are frozen.** Every comparison this session uses the existing `teacher.jsonl` files
  (same ids, same probs), except the permutation diagnostic, which writes to its own run dir.
- **Nothing is fitted on eval.** The oracle numbers in §0 are diagnostics; they may appear in
  `docs/` as "oracle (fitted on eval)" and never in the README table.

## 3. Milestones (wall clock; two agents from hour 1, one before and after)

Hour budgets are wall-clock; agent effort is listed per item. Commands assume `cd ~/projects/open-jev`,
`.venv` on PATH, proxy vars unset (`env -u ALL_PROXY -u all_proxy -u HTTP_PROXY -u HTTPS_PROXY`).

### M0 — prep and hygiene (0:00–1:00, one agent, 1 h)

Small, mechanical changes that both lanes need. Do them first, in one commit, so the two agents never
edit `cli.py` at the same time afterwards (after M0, agent F owns `cli.py`, `serve.py`, `spec.py`,
tests, docs; agent R owns `calibrate.py`, `evaluate.py`, `train.py`, `scripts/`).

1. `cli.py`: `--seed N` (run dir suffix `-sN` for N > 0; passed to `train.run(seed=)`), `--tag X`
   (free-form suffix, replaces ad-hoc suffix rules), `--no-synth` (train.py drops `source == "synth"`
   rows), `--gold-n N` (CE term only on the first N train rows in id order), `eval --no-latency`.
2. `evaluate.py`: write `runs/<run>/eval_rows.jsonl` — one line per eval row: `id, gold, gold_prob?,
   student_probs (calibrated), student_logits, teacher_probs`. Save `calib_logits.pt` in `calibrate.py`.
   Everything post-hoc in M1 reads these; no re-inference.
3. `evaluate.report()`: group `results/<name>-s<k>.json` with `results/<name>.json`; show
   `mean ± sd (n=3)` in the acc/F1/ECE columns when n > 1; `–` for missing latency; also print the table.
4. B5 (`label.json` stats written inside the flush loop, `n_failed` summed), B6 (`read_rows` sorts by
   id), C11 (`labels.index("true")` in both places).
5. `scripts/gpu_queue.sh`: reads a job list file, one `openjev ...` command per line, runs them
   serially under `flock runs/.gpu.lock` + `choom -n 1000`, memory gate as in §2, appends to
   `runs/queue2.log`. `openjev bench RUN_DIR`: latency + throughput only, idle-teacher check, writes
   `latency_ms`/`throughput_gpu_b64` into the existing `results/<run>.json`.
6. `scripts/perm_check.py`: builds an in-memory copy of a task with `options` reversed (gold remapped
   `g → K−1−g`), reads the calib+eval rows of `runs/<task>/teacher.jsonl`, labels them through
   `teacher._run` into `runs/<task>-rev/teacher.jsonl`, then prints for original, reversed and their
   average: teacher acc, predicted marginal, mean max-p. Start it on kinopoisk then headlines the moment
   it works (teacher lane job T1, ~12 min).
7. `hf auth whoami` and `gh auth status`: record whether an HF token exists (gates M3b.3).

Accept:
```
pytest -q tests                                                     # green, ≥ 19 tests
openjev train tasks/headlines.yaml --seed 1 && openjev calibrate tasks/headlines.yaml --seed 1 \
  && openjev eval tasks/headlines.yaml --seed 1 --no-latency      # -> runs/headlines-s1, results/headlines-s1.json
openjev report | grep headlines                                    # one grouped row, "n=2"
ls runs/headlines-s1/eval_rows.jsonl runs/headlines-s1/calib_logits.pt
python scripts/perm_check.py kinopoisk                             # prints the 3-way table
```

### M1 — post-hoc research suite on the existing nine runs (1:00–4:00, agent R, 3 h)

All CPU except the GPU queue running in the background.

1. **Vector scaling** in `calibrate.py`: `fit_vector(logits, y)` = LBFGS over `(log T, b ∈ R^K)`, CE
   loss. Method selection with a held-out check, because a K-dim bias fitted on 500 rows can overfit
   (K=77): split calib 2-fold, fit on each half, score NLL on the other; accept `vector` only if it
   beats `temperature` on both folds, then refit on all 500. Keep the existing `T=1.0` guard.
   `calib.json` gains `method: temperature|vector|none`, `bias: [...]`. One function
   `calibrate.apply(logits, calib)` used by calibrate, evaluate and serve (serve.py:49 currently does
   `logits / T` inline — replace).
2. **Declared prior, no gold**: top-level YAML `prior: uniform | {label: p}`. When the calib split has
   no gold (or `calib_target: prior` is forced), fit T to teacher-soft as now, then `b` so that the mean
   calibrated probability over the calib rows equals the prior (iterate `b_k += log(prior_k / mean_p_k)`
   ~50 times). `calib_target: prior-declared`. Test it on kinopoisk and georeview by running calibrate
   with the gold *ignored* (`--tag prior` variant dir) so the number is a genuine no-gold number.
3. **Cascade curve** in `evaluate.py` from `eval_rows.jsonl`: for τ in 0.30..1.00 step 0.05, route
   `conf < τ` to the teacher's argmax; store `[{tau, escalation, acc (or mae/auroc)}]` in results and
   the *parity point*: the lowest escalation rate at which cascade metric ≥ teacher metric − 0.005.
4. **`openjev compare A B`**: for seeds present in both (`A`, `A-s1`, `A-s2` …), per-seed paired delta
   on the task's metric (acc / MAE / AUROC) from `eval_rows.jsonl`, pooled 95 % bootstrap CI over rows
   (2000 resamples, delta averaged over seeds), printed as a table plus one verdict line:
   `Δ = +0.4 ± 0.7 pts (95 % CI −0.3..+1.1): no measurable difference`.
5. Re-run `calibrate` + `eval --no-latency` on all nine existing run dirs (≈ 1 min each; GPU inference
   only, fits inside the memory gate — run them through the queue). Write `docs/experiments.md` with the
   R1 table (per task: T-only ECE/acc/MAE, vector, prior-declared, oracle) and the R2 curves.

GPU queue (job file `runs/queue2.jobs`, started at 1:00, ~1.5 h serial, unattended):
```
# E1 epochs: same seed, same labels, only epochs differs from the existing run
openjev run tasks/agnews.yaml    --tag e12 --epochs 12 --no-latency      # 7 min
openjev run tasks/headlines.yaml --tag e12 --epochs 12 --no-latency      # 3 min
openjev run tasks/kinopoisk.yaml --tag e12 --epochs 12 --no-latency      # 25 min
# E2 seeds of the existing baselines (epochs as in the YAML)
openjev run tasks/headlines.yaml --seed 1 ; --seed 2                    # 3 min
openjev run tasks/agnews.yaml    --seed 1 ; --seed 2                    # 7 min
openjev run tasks/kinopoisk.yaml --seed 1 ; --seed 2                    # 24 min
# E3 augment control: same run dir labels, synth rows dropped, 3 seeds each arm
openjev run tasks/headlines.yaml --no-synth --tag nosynth --seed 0 ; 1 ; 2   # 5 min
```
(`run` skips label — the labels are cached — and goes train → calibrate → eval.) Note the headlines seed-0
arm *with* synth already exists as `runs/headlines`.

Accept:
```
openjev calibrate tasks/kinopoisk.yaml && cat runs/kinopoisk/calib.json      # method + bias present
openjev eval tasks/kinopoisk.yaml --no-latency && python -c "import json;print(json.load(open('results/kinopoisk.json'))['cascade'][:3])"
openjev compare runs/headlines runs/headlines-nosynth                        # table + verdict line
pytest -q tests                                                             # + tests for fit_vector guard, prior fit, cascade, compare
```
Allowed claims after M1: see §5.

### M3a — the newcomer loop (1:00–4:00, agent F, 3 h, in parallel with M1)

1. **`openjev check task.yaml`** (no teacher cost): loads the spec, pings `OPENJEV_TEACHER_URL/models`,
   loads the data source, prints: split sizes and gold distribution per split; the system prompt
   verbatim; 3 formatted examples; text length stats and a warning when `max_chars` binds on > 25 % of
   rows or `max_len` tokens is far below `max_chars` (the kinopoisk trap); estimated teacher cost
   (calls × a rate table from the seven measured runs: ≤ 300 chars ≈ 12 ex/s, ≤ 1000 ≈ 5, longer ≈ 3.5,
   divided by calls per example — "±2×"). Every existing newcomer error message goes through this path.
2. **`--probe N`**: labels N calib rows (cached into the real `teacher.jsonl`, so they are reused by
   `run`), prints teacher accuracy vs gold (if any), predicted marginal vs gold marginal (or vs
   `prior:` if declared), mean max-p, and one warning line when a class is under-predicted by > 10 pts:
   "the teacher under-predicts *Neutral* (13 % vs 33 %): reword the option, or declare `prior:`".
   This is R1 made usable at prompt-writing time.
3. **CI**: `.github/workflows/test.yml` — ubuntu, python 3.12, torch CPU wheel
   (`--index-url https://download.pytorch.org/whl/cpu`), `actions/cache` on `~/.cache/huggingface`,
   `pytest -q tests`. Repo is private; Actions minutes are fine at this size.
4. **LICENSE** (user picks MIT or Apache-2.0 — ask once; default MIT), `pyproject.toml` license /
   classifiers, `uv build` and `pip install dist/*.whl` into a throwaway venv, `openjev --help` from
   there. No publish.
5. README "Add your own task" becomes: write YAML → `check` → `check --probe 100` → `run` → `serve`.
   Task index table in `docs/task-spec.md` (7 rows: name, type, K, lang, what it demonstrates).

Accept:
```
openjev check tasks/kinopoisk.yaml                       # prompt, 3 examples, "max_chars binds on 78 %", cost estimate
openjev check tasks/kinopoisk.yaml --probe 100           # teacher acc + marginal + the Neutral warning
openjev check tasks/example.yaml                         # fails on the missing CSV with a one-line hint, exit 1
gh workflow run test.yml && gh run watch                  # green
(cd /tmp && python -m venv v && v/bin/pip install ~/projects/open-jev/dist/*.whl && v/bin/openjev --help)
```

### M2 — training-side experiments and readouts (4:00–7:00, agent R, 3 h)

1. **Read E1/E2/E3 out with `compare`**, write the numbers into `docs/experiments.md` as they land.
   Decisions: if 12 epochs beats 5 with the CI excluding 0 on ≥ 2 of 3 tasks, the default becomes 10
   (patience 2); otherwise the default stays and the doc says so. Augment: the verdict line from
   `compare runs/headlines runs/headlines-nosynth` is the README sentence, whatever it says.
2. **Permutation diagnostic readout** (T1 finished in M0). Decision rule: implement
   `teacher: {permutations: 2}` (label each row under both orders, average, store both `raw`s) **only if**
   the averaged marginal on kinopoisk is closer to gold by ≥ 5 pts on the worst class *and* averaged
   teacher acc ≥ original + 1 pt. If not, the README limitation gets the measured sentence ("reversing
   the option order moved the teacher's Good rate from 56.5 % to X %; the bias is semantic, not
   positional") and no feature is built.
3. **Gold spend, one question**: with 500 gold labels, is it better to (a) fit a bias on them (M1.1,
   already measured) or (b) put them in the loss? Jobs:
   ```
   openjev run tasks/kinopoisk.yaml --gold-weight 1.0 --gold-n 500  --no-latency   # -> kinopoisk-gold-n500
   openjev run tasks/kinopoisk.yaml --gold-weight 1.0                --no-latency   # all 4000 gold: the ceiling of "just supervise it"
   ```
   (24 min GPU; georeview pair +20 min if the queue is idle.) The (b) runs are then *also* vector-scaled,
   so the comparison is (a) vs (a)+(b). Seeds 1–2 only if the seed-0 delta is > 2 pts.
4. **Headlines with 8000 train rows**: `tasks/headlines-8k.yaml` (train n 8000, everything else
   identical; calib/eval ids stay identical because roles are drawn train-first from a seeded pool —
   verify by diffing eval ids before spending). Teacher job T2 (≈ 6 min), then 3 seeds (5 min GPU).
   Claim: "doubling teacher labels on headlines moves the student Δ = … (CI)".
5. Push all results through `openjev report`; draft the Findings v2 bullets with CIs in a scratch file.

Accept:
```
openjev compare runs/kinopoisk runs/kinopoisk-e12          # verdict line exists
openjev compare runs/headlines runs/headlines-8k           # verdict line exists
grep -c "95 % CI" docs/experiments.md                     # every claimed delta carries one
```

### M3b — serve polish, hub, harness (4:00–7:00, agent F, 3 h)

1. **serve**: `input` may be a list (batch, one forward pass, list response); `--device cpu|cuda`;
   `escalate_below: τ` per request or as a default in `openjev.json` (set by eval to the parity point
   from M1.3) → `"escalate": true/false` in the response — serve never calls the teacher itself, the
   README shows the two-line client pattern; `coverage: 0.9` → `"set": [...]` from a conformal
   quantile stored in `calib.json` (LAC score `1 − p_gold` on calib, teacher-argmax pseudo-gold when
   there is no gold, labelled as such). Conformal is the first thing cut if M3b runs long.
2. **`openjev push RUN_DIR --repo user/name`**: `huggingface_hub.upload_folder` of `student/` +
   `openjev.json` + a generated model card (task YAML, teacher id, the results row, the calibration
   method, "distilled with no gold in the loss"). **`openjev serve hf:user/name`**: `snapshot_download`
   into `runs/hf--user--name` then the normal path. Push is gated on the token found in M0.7; the
   pull path is tested against whatever public repo the push created, or skipped with a note.
3. **Jev harness** `scripts/compare_api.py`: reads `runs/<task>/eval_rows.jsonl`, sends each text to an
   external `/v1/systemone`-shaped endpoint with an adapter function (`--adapter openjev|jev`), writes
   `results/api-<name>-<task>.json` with acc / agreement-with-our-student / p50 latency, and a README
   stub row. The agent fetches Jev's public docs for the request shape; if the exact shape is
   undocumented, the `jev` adapter is a clearly marked 10-line function to fill in, and the harness is
   still fully exercised against `openjev serve runs/agnews --port 8099` (same endpoint shape). Needs
   `JEV_API_KEY` only at the very end, if the user creates the account. Never a blocker.

Accept:
```
openjev serve runs/agnews runs/kinopoisk --port 8099 &   # then:
curl -s localhost:8099/v1/systemone -d '{"model":"agnews","input":["a","b"],"escalate_below":0.7}'   # list of 2, each with "escalate"
openjev serve runs/agnews --device cpu --port 8098 &      # works, ~30 ms
python scripts/compare_api.py --adapter openjev --url http://localhost:8099 --task agnews --limit 200   # writes results/api-openjev-agnews.json
openjev push runs/agnews --repo <user>/openjev-agnews --dry-run                                        # lists files + card; real push only with token
```

### M4 — write-up (7:00–9:00, both agents, 2 h)

README **Findings v2**: rewrite around R1/R2/R3 with the CIs from `compare`; every number traceable to a
`results/*.json` or `docs/experiments.md` row; the grouped table; refreshed Requirements (memory numbers:
small len 256 5.5 GB, len 512 8.9 GB; base banned next to vLLM; the OOM-killer note); Limitations updated
(prior must be *known*; benchmark priors are balanced by construction, which is a leak a real user
does not get; vector scaling needs ≥ 500 rows from the deployment distribution; cascade is measured
against the same teacher that produced the labels). `docs/task-spec.md`: `prior`, `--seed/--tag/--no-synth/
--gold-n`, `check`, `compare`, `bench`, `push`. Commit after each file.

Accept: `openjev report` leaves README byte-identical (table already fresh); every "Δ" in Findings has a
CI; `grep -n oracle README.md` returns nothing in the table section.

### M5 — buffer / optional (9:00–10:30, 1.5 h)

Only if M1–M4 are accepted, in this order: (1) kinopoisk n = 8000 (T3 ≈ 18 min teacher, 25 min GPU;
answers "should I label more?" on the hard task); (2) augment round 2 on headlines with the control arm
and 3 seeds; (3) pre-distillation debiasing (reweight train `probs` by the calib-fitted prior ratio,
retrain kinopoisk, compare with post-hoc on the same seed — expected to tie); (4) conformal if cut from M3b.

### M6 — bench, review, ship (10:30–12:00, 1.5 h)

1. `openjev bench` on every seed-0 run dir that is new this session (idle teacher enforced by the tool).
2. **Honesty review**: spawn the `oj-planner` agent with the same brief as REVIEW-final.md on the new
   README + docs/experiments.md + results; 45 min to fix what it finds. This is scheduled, not optional —
   it is the step that caught the last two overstatements.
3. `pytest -q tests`, `openjev report`, commit, `git push`.

Budget: M0 1.0 + M1/M3a 3.0 + M2/M3b 3.0 + M4 2.0 + M5 1.5 + M6 1.5 = 12.0 h wall clock;
≈ 18 agent-hours with two agents over hours 1–7.

## 4. What runs where

| lane | serial? | jobs (in order) | total |
|---|---|---|---|
| **Teacher** (:8000, the bottleneck) | yes, one at a time, concurrency ≤ 32 | T1 perm relabel calib+eval: kinopoisk (2000 rows, ~9 min) then headlines (~3 min) · `check --probe 100` smoke tests (≈ 1 min) · T2 headlines +4000 train rows (~6 min) · T3 kinopoisk +4000 (optional, ~18 min) · augment round 2 (optional, ~10 min generation) | ≈ 20 min required, ≈ 50 min with options |
| **GPU** (training, one job, memory-gated) | yes, `flock` | E1 epochs (35 min) · E2 seeds (34 min) · E3 augment control (5 min) · nine re-evals for M1 (10 min) · gold spend (24–44 min) · headlines-8k seeds (5 min) · M5 options (≈ 60 min) · M6 bench (10 min) | ≈ 2.2 h required, ≈ 3.3 h with options |
| **CPU / agents** | two in parallel | everything in M1.1–1.4, M3a, M3b, M4 | the actual constraint |

The teacher is idle > 90 % of the session; labeling and training never wait on each other except that
headlines-8k training waits for T2. The GPU queue file is appended to as decisions land (E1/E2/E3 first,
gold-spend after M1.1 exists, 8k after T2).

## 5. Experiment design: what is controlled, what each run may claim

**Protocol for every comparison.** Held fixed: `teacher.jsonl` (same ids, same probs), split ids, student
checkpoint (`mmBERT-small`), `max_len`, `lr`, `batch_size`, epochs as in the task YAML, calibration
method selection rule. Exactly one factor varies per pair. Metric per type: accuracy (choice), MAE of
the expectation (score), AUROC (noul); macro-F1 alongside for choice. ECE is reported, never the
headline.

**Noise floor.** A single accuracy on n = 2000 has a 95 % CI of ± 1.8 pts (± 2.4 on kinopoisk's 1500).
Paired deltas on the same rows are tighter, which is why `compare` bootstraps the *per-row* delta. Seeds
change the classifier-head init and the batch order (CUDA is not bit-exact; three seeds is the floor for
any claim, not a nicety).

**Claim rule.** A delta is stated as an improvement only when the pooled paired-bootstrap 95 % CI
excludes 0 with 3 seeds per arm. Otherwise the sentence is "no measurable difference (Δ = x ± y)". A
seed-0-only pair may be reported as "preliminary, one seed" in `docs/experiments.md` and never in README
Findings. The oracle prior numbers are diagnostics: they say how much *could* be recovered, never how
much *was*.

| run(s) | varies | fixed | may claim | may not claim |
|---|---|---|---|---|
| vector scaling vs T on 9 runs (M1.1) | calibration method (post-hoc) | student weights, calib rows | "500 gold rows spent on a bias recover Δ of the teacher-prior gap on task X" with CI over rows (1 seed per existing run, + seeds for kinopoisk/headlines/agnews) | anything about the teacher's ability; the bias is fitted to the eval *distribution's* prior via calib |
| prior-declared, no gold (M1.2) | calib target | same | "with no labels and a declared prior, Δ" | that the prior is knowable in general — on these benchmarks it is known *because* they are balanced by construction |
| cascade (M1.3) | τ | same | "at escalation e, cascade metric m" | any saving in teacher *quality* — escalation goes to the same zero-shot teacher that made the labels |
| E1 epochs | epochs 5 vs 12 | seed 0 | default change if CI excludes 0 on ≥ 2/3 tasks | anything about patience |
| E2 seeds | seed | all | the sd column; the noise floor for everything else | — |
| E3 augment | synth rows in/out, 3 seeds each | labels, epochs | the verdict line, either way | "PGKD works" — one task, one round |
| perm diagnostic | option order, teacher only | rows | "order reversal moved the marginal by …" | anything about the student unless the feature is built and rerun |
| gold spend | where 500 gold go | same seed; vector scaling applied to both | "CE on 500 gold adds Δ on top of the bias" | a general N-labels curve — two points, one task |
| headlines-8k | n_train 4000 vs 8000, 3 seeds | calib/eval ids (verified identical) | "doubling labels on headlines: Δ" | extrapolation to other tasks |
| Jev harness | — | — | nothing until a key exists; then acc/agreement on the same 2000 rows | cost or latency comparisons (different hardware) |

**Never in the README:** numbers from `--tag` variant dirs that were not part of a completed comparison,
oracle numbers, one-seed deltas under 2 pts.

## 6. Cut order (first cut first) and the 4 am fallback

1. Jev harness (M3b.3) 2. HF push/pull (M3b.2) 3. conformal sets 4. kinopoisk-8k and every M5 item
5. georeview half of gold spend 6. `permutations: 2` feature (keep the diagnostic) 7. headlines-8k
8. wheel/LICENSE/CI (CI last of these) 9. serve `--device`/batch.

**Never cut:** M0 (seeds, `eval_rows.jsonl`, grouped report), `compare`, vector scaling + prior-declared
on the existing runs, cascade curve, `check`, README Findings v2 with CIs, the M6 honesty review.

**If it is 4 am and something is broken:**
- Code broken: `git stash`, go back to the last green commit (`pytest -q tests` decides), keep only
  the M0 flags + `compare`. The M1 results do not need the pipeline — `eval_rows.jsonl` +
  `calib_logits.pt` from the existing nine run dirs are enough for a standalone
  `scripts/posthoc.py` to produce the R1/R2 tables. Ship those, note the rest as "not done".
- GPU queue stuck / OOM: kill *only* the trainer (`pkill -f "openjev (run|train)"`), check vLLM is still
  answering `/v1/models`, lower nothing, restart the queue after the memory gate passes. Do not start new
  variants; finish the seeds.
- Teacher down: everything except T1–T3 continues. Do not restart vLLM; write it in the summary for the
  user. The perm diagnostic becomes "not run".
- README and `results/` must agree at every commit: `openjev report` before every commit, no hand-edited
  table cells.

## 7. Risks

- **Host OOM → vLLM dies.** Mitigation in §2 (`choom`, memory gate, single job, small model only).
  The gate numbers come from today's `free -g` (22 GB available) and `mem_get_info` (13.5 GB free).
- **Teacher contention.** Production clients share :8000. The teacher lane is short and daytime; keep
  concurrency at 32 max, and no labeling during `bench`. If the user's other clients are active, T2/T3
  can wait — they are optional.
- **Vector scaling overfits at K=77.** The 2-fold guard exists for this; expect `temperature` (or
  `none`) to be selected on banking77 and headlines, and say so.
- **Declared prior is a leak on benchmarks.** Stated in Limitations and in the results note for every
  `prior-declared` row.
- **Calib ≠ eval distribution (toxic).** Calib is 2.2 % positive, eval 8.1 % (the depleted-pool bug,
  fixed in code but the run predates it). A bias fitted on that calib split calibrates to the wrong prior;
  toxic is excluded from R1 unless its calib split is re-drawn and re-labelled (500 rows, 1.2 min teacher —
  cheap, do it as T1b if R1 on toxic is wanted; otherwise report toxic as "n/a: calib prior mismatch").
- **Seed runs multiply results files.** Grouping in `report` handles the table; `results/` grows by ~15
  files — acceptable, they are small. Keep them committed; they are the evidence.
- **Agents editing the same file.** Ownership split after M0 (§3 M0 preamble); `cli.py` is agent F's
  after M0, agent R requests flags through F rather than editing.
- **Time.** The research lane's real risk is analysis paralysis over marginal deltas. The claim rule in
  §5 is the stop condition: run `compare`, write the verdict line, move on.
