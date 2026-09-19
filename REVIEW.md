# Open-Jev review (2026-09-20, code at 025fa84 + working tree)

Scope: PLAN.md, openjev/*.py, tasks/*.yaml, tests/, scripts/, results/agnews.json, runs/agnews (read-only).
`pytest -q tests` (CPU): 10 passed. Nothing under runs/ was touched; no GPU jobs started.

**Bottom line.** The core math is right: soft labels are a proper renormalised softmax over the letter
logprobs (verified against the saved vLLM response: post-grammar-mask logprobs, the four letters sum to 1.0000;
0 of 6500 agnews rows and 0 of 1000 kinopoisk rows have a missing letter); the loss is
`KL(teacher || student)` in the correct direction; temperature is fit by LBFGS on the calib split only and
converges (matches a 4000-point grid search to 3 decimals) and is applied identically in eval and serve;
ECE/Brier/AUROC/MAE are standard implementations. **The agnews numbers are not inflated**: eval ids and
texts are disjoint from train and calib (checked by id and by exact text), eval is drawn from the HF `test`
split, early stopping uses calib teacher-KL, temperature uses calib gold, and the teacher accuracy
(0.880 on eval, 0.880 on train, 0.896 on calib) is consistent across splits. If anything the latency
numbers are pessimistic (measured while vLLM was labeling kinopoisk on the same GPU).

Severity legend: **bug** = wrong output today or on a path a user will hit; **risk** = wrong under a
realistic condition (crash, no-gold data, K>19); **polish** = clarity/claims/cuts.

---

## A. Confirmed bugs

### A1. noul: boolean or 0/1 integer gold columns are silently inverted — bug
`openjev/data.py:28-32`. `gold_index` only applies the `>= gold_threshold` rule when `raw` is a `float`.
`True`/`1` fall through to `int(raw)` → index 1 → `"false"`. Verified: `gold_index(toxic, True) -> "false"`,
`gold_index(toxic, 1) -> "false"`, `gold_index(toxic, 0.9) -> "true"`. Tonight's toxic task uses a float
column so it is unaffected, but any user with an `is_toxic: true/false` column gets inverted accuracy/AUROC
with no error. Fix (one branch):
```python
if task.type == "noul" and isinstance(raw, (bool, int, float)):
    return 0 if raw >= d.gold_threshold else 1
```
and delete the float-only branch.

### A2. Crash mid-label leaves a poisoned line that `train` cannot read — bug (crash path)
`openjev/train.py:16-18` parses `teacher.jsonl` with strict `json.loads`; `openjev/data.py:78-90`
(`read_jsonl`, used for resume) tolerates a truncated last line. After a kill during the 64-row flush
(`teacher.py:162-164`), the next resume appends straight after the partial line, gluing two rows into one
invalid line. Verified: `read_jsonl` skips it (and the lost id is relabeled on the next resume, good), but
`train.read_rows` raises `JSONDecodeError` → `openjev run` dies at the train stage with no hint.
Fix: `train.read_rows = lambda run_dir, split: [r for r in read_jsonl(Path(run_dir)/"teacher.jsonl") if r["split"]==split]`
(also removes the duplicate JSONL reader). Optional one-liner in `teacher.run`: if the file exists and does
not end in `\n`, write `\n` before appending, so the partial line is isolated instead of glued.

### A3. Shortlist final call letters the candidates in score order → position bias amplified — bug (K>19 path, banking77)
`openjev/teacher.py:76` returns the shortlist sorted by score; `teacher.py:120` builds the final prompt
from that order, so letter `A` is always the chunk-stage favourite and `S` the weakest. LLMs have a known
prior for early letters; the final soft distribution therefore double-counts the chunk-stage score. PLAN §8
explicitly lists letter-position bias as a known risk, and this code path makes it worse in the exact
place it matters. Fix: `short = sorted(shortlist(chunk_results, m))` at `teacher.py:119` (keep original
option order in the final call). Also: `shortlist()` multiplies `p_i` (already a softmax that includes
`Z`) by `(1 - p_Z)` again — a double "none" penalty; harmless as a within-chunk ranking but not what the
docstring says. Either drop the factor or renormalise `p[:-1]` first. `tests/test_smoke.py:44` feeds a
chunk that sums to 1.9, so the test does not pin down the intended formula.

### A4. `--gold-weight 0` creates a `-gold` variant dir — bug (minor)
`openjev/cli.py:18`: `is not None` → suffix `gold` even for `0.0`, giving a directory whose results claim
"zero gold labels" in the README note (`evaluate.py:135`). Fix: `if args.gold_weight:`.

---

## B. Risks (correct today, wrong under a realistic condition)

### B1. No-gold calibration re-sharpens the student toward the teacher's argmax — risk (the framework's headline use case)
`openjev/calibrate.py:41-45`. When the calib split has no gold, T is fit by cross-entropy against
**teacher argmax as hard labels**. The student was trained to match the teacher's *soft* distribution, so
this systematically drives T < 1 and produces probabilities sharper than the teacher's, i.e. calibrated to
a label the teacher itself does not believe. None of tonight's tasks hit this (all have gold), but "zero
gold labels" users will. Fix: let `fit_temperature` take soft targets and minimise
`-(p_teacher * log_softmax(logits/T)).sum(-1).mean()` when `target == "teacher"` (3 lines); report
`calib_target: teacher-soft`.

### B2. Student and teacher see different text on long-review tasks — risk (kinopoisk, georeview)
PLAN §8 says "truncate to max_chars for both so they see the same input", but `student.max_len: 256`
tokens is the binding constraint: on kinopoisk **76 % of texts exceed 256 mmBERT tokens** (median 394
tokens; median text is exactly `max_chars`=1500 chars). The student is asked to reproduce a label the
teacher derived from text the student never sees. Not a bug, but it caps agreement and it contradicts the
plan's stated invariant. Fix: `max_len: 512` for kinopoisk/georeview (ModernBERT handles it; ~2x train
time), or lower `max_chars` to ~900 for Russian so the two truncations coincide; add one doc line
explaining the two knobs (`max_chars` = chars for both, `max_len` = tokens for the student).

### B3. toxic: temperature fit on a balanced calib split, evaluated on the natural (8 % positive) split — risk
`tasks/toxic.yaml:13`. A single scalar T cannot correct a prior shift; `ece_cal` on eval will look worse
than it should and may exceed `ece_raw`. Fix: `calib: {split: "train[:60000]", n: 500}` (natural
distribution; train can stay balanced). Also note that `balance: true` uses **gold to select the 4000
training rows** (`data.py:40-43`), so the README note "distilled … with zero gold labels" is not quite true
for toxic; `report()` should say "gold used for balanced sampling" when any split has `balance: true`.

### B4. queue.sh blocks forever if the label lane fails — risk
`scripts/queue.sh:13-15`. `.labeled` is only touched on success; a vLLM outage during `openjev label`
(the `/models` call at `teacher.py:140` is the only hard failure) leaves the train lane in an infinite
`sleep 30` loop for that task **and every task after it**. Fix: record the label-lane PID and wait on
`.labeled || ! kill -0 $pid` — `openjev run` resumes labeling itself, so the train lane can just proceed
once the label lane has exited.

### B5. `label.json` stats are only written on clean exit; `n_failed` is not accumulated — risk (minor)
`teacher.py:170-172`. A crash loses `calls`/`minutes` for that session, so `teacher_minutes` in
results is undercounted after a resume; `n_failed` is overwritten rather than summed (inconsistent with
the other three fields). Fix: write stats inside the flush loop (same place as the progress print), and
sum `n_failed`.

### B6. Training order depends on teacher completion order — risk (reproducibility)
`teacher.py:155` appends rows in `as_completed` order, so two labelings of the same task give different
`teacher.jsonl` orders and `train.py:74` `random.shuffle` (seeded) yields different batches. Fix: sort
rows by `id` in `read_rows`. (CUDA/sdpa nondeterminism remains; document "seeded, not bit-exact".)

### B7. `--limit` on `openjev run` labels only train rows, then train crashes on an empty calib split — risk (minor)
`cli.py:55` exposes `--limit` on every subcommand; `teacher.py:185-186` slices the role-ordered list so
`--limit 50` yields 50 train rows, 0 calib, 0 eval; `train.py:97` then `torch.cat([])`. Fix: register
`--limit` only on `label`, or apply it per role.

### B8. `results/*.json` `git` field is HEAD even when the tree is dirty — risk (provenance)
`evaluate.py:37-39`. `results/agnews.json` says `0a61cf8`, a commit that did not contain train/eval code.
Fix: append `-dirty` when `git status --porcelain` is non-empty.

### B9. Latency/throughput numbers are measured with vLLM active on the same GPU — risk (claims)
`evaluate.py:27-34, 81-88`. agnews eval ran at 00:36 while kinopoisk labeling (started 00:30) was
saturating the teacher. 20.7 ms batch-1 for mmBERT-small is ~3-4x what an idle GB10 should give. Not
inflated, but not the "edge story" number either. Say so in the README note, or re-run eval once the queue
is idle (`openjev eval` re-measures without retraining).

---

## C. Low barrier to entry: concrete cuts and renames (the user's top priority)

The YAML is close to minimal already; the confusion is at the edges.

1. **`question` is required but ignored for `noul`** (`spec.py:61`, `teacher.py:36-39`); `tasks/toxic.yaml:3`
   carries a dead line that a user will assume matters. Make `question` optional with a per-type default
   (`noul` → "Is the following statement about the text true?"). Cut the dead line from toxic.yaml.
2. **`teacher.system_prompt` is not a system prompt** — it replaces the question line and options are still
   appended (`teacher.py:34-35`). It duplicates `question`. Delete it; `question` is the override.
3. **Per-split `seed`** (`spec.py:24`) is misleading: splits are drawn sequentially, so changing
   `train.seed` also changes calib. One `data.seed` (default 0). Same for `balance` on eval (never wanted).
4. **`labels`, `values`, `path` are accepted from YAML and silently overwritten** (`spec.py:70-72`,
   `_build` treats them as known keys). Verified: a YAML with `labels: [zzz]` loads without error. Move
   them out of the dataclass fields that `_build` validates (or reject them in `load_task`).
5. **`gold_map` + `gold_threshold` + `gold_prob`**: three gold knobs on one line each in the docstring
   only. `gold_map` is unused by every task tonight (georeview uses the int-index rule); consider cutting
   it until a task needs it, or add a one-line example in the `spec.py` docstring.
6. **No validation of `rubric.descriptions` length** (`teacher.py:27-28` zips silently). One assert.
7. **`serve --host` defaults to `0.0.0.0` in the CLI (`cli.py:60`) and `127.0.0.1` in `serve.main`.** Pick
   `127.0.0.1` in both.
8. **`--student` suffix heuristic** (`cli.py:17`): any model with "base" in its name gets the same `-base`
   dir (mmBERT-base, ModernBERT-base collide). Always use the last path component (`-mmBERT-base`).
9. **Docs lines the YAML needs** (put them as comments in `tasks/agnews.yaml`, which is the de-facto
   template): `max_chars` vs `max_len` (see B2); "no distillation temperature, teacher probs used as-is";
   "temperature is fit to gold on `calib` when gold exists, else to the teacher"; "Brier is the K-class
   sum form (range 0-2)".
10. **Brier definition inconsistency** (`calibrate.py:23-24` vs `evaluate.py:78`): the README `Brier`
    column is sum-form (2x binary for noul) while `brier_vs_gold_prob` is the binary form on the same
    task. Use the binary form for `noul` in `_cls`, or document the sum form once.
11. `evaluate.py:74-75` hardcodes column 0 for "true" while `:71` uses `labels.index("true")`. Use one.
12. The README note "distilled from the teacher with zero gold labels" should read "no gold in
    training; 500 gold labels for temperature" — that is what actually happened and it is a stronger,
    more honest sentence.

---

## D. Over-engineering / delete

- `scripts/label_chain.sh` — a strict subset of `queue.sh`'s label lane. Delete.
- `data.examples(task, roles=ROLES)` — the `roles` parameter has no caller that passes it. Delete.
- `train.read_rows` — duplicate of `data.read_jsonl` with worse crash behaviour (A2). Delete.
- `evaluate.run:54-55` computes `has_gold` and then `targets_for` computes it again. Keep the second.
- `TeacherSpec.system_prompt` (C2), per-split `seed` (C3), `gold_map` (C5) — all speculative knobs.
- `tests/test_smoke.py::test_split_disjoint` and `test_core.py::test_end_to_end_tiny` download ag_news and
  mmBERT-small; fine for this repo but note in the README that `pytest` needs the HF cache warm.

---

## E. Verified non-issues (so nobody re-checks them)

- Letter logprobs are post-grammar-mask (sum to 1.0000 in `tests/data/response_agnews.json`); the
  re-softmax in `softmax_letters` is idempotent and correct; the "missing letter → 0" branch never fired
  on 7500 real rows. `top_logprobs = min(len(letters), 20)` never exceeds the vLLM cap (max 17 for
  banking77 chunks, 19 for the final call).
- Loss `F.kl_div(log_softmax(student), teacher, batchmean)` = KL(teacher || student); targets clamped at
  1e-6 and renormalised (shortlist zeros → 1e-6, negligible mass); gold CE only on rows with gold.
- Temperature: LBFGS on `log T` converges (T matches grid search at scales 0.3-10x); applied in eval
  (`evaluate.py:51`) and serve (`serve.py:49`); early stopping and T both use calib, never eval.
- ECE: 15 equal-width bins on max-prob, standard. AUROC via sklearn on p(true). MAE on the expectation.
- Split disjointness: `test_split_disjoint` passes; agnews `teacher.jsonl` has 6500 unique ids,
  train/calib/eval = 4000/500/2000, zero exact-text overlap across any pair.
- Variant dirs: symlinks `../<task>/teacher.jsonl` are correct relative paths; the label stage always
  writes to the base dir (`cli.py:33-34`); `flock` serialises the queue's label lane and `run`'s label stage.
- Training: agnews calib KL was still falling at epoch 4/5 (0.0513 → 0.0496), so early stopping never
  triggered; `epochs: 8` would likely gain a little. Not a bug.
- Kinopoisk teacher (first 1000 rows): acc 0.62, predicts "Good" 58 % vs 31 % gold and "Neutral" 11 % vs
  36 % gold — a teacher prompt bias the student will inherit. PLAN §8 anticipates "modest"; worth one
  sentence in the README row rather than a code change tonight.
