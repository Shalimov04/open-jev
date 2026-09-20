# Open-Jev — track 4: `open-jev-base`, one model for a typed question it was never trained on

> Written 2026-09-20 after reading README, docs/{findings,use-cases,task-spec}.md, PLAN-2/3, openjev/*.py,
> the task YAMLs, `runs/*/teacher.jsonl` (sizes and record shape) and `results/*.json`. Implementation by
> the oj-coder agent; every milestone has a command that proves it. No code in this file.

## 0. The claim being tested, and why this shape

Today every task is one teacher pass plus one fixed-head student, and the README says so ("not a general
model that takes arbitrary options at request time"). The proposition: **one cross-encoder that scores a
`(question, one option, text)` pair and returns a probability.** A `choice` is K pairs renormalised, a
`noul` is one pair, a `score` is one pair per rubric level. The option wording, count and meaning are all
inference-time inputs, so a question never seen in training is answerable — how *well* is the question.

The shape is forced by our own measured failure, not by taste: `m2w-element` (16 per-row candidates, fixed
16-way head) collapsed to **0.059 accuracy on a 0.0625 chance floor**, agreement with the teacher 0.133,
while the identical decision expressed per candidate (`m2w-target`) reached **AUROC 0.899–0.904**. A fixed
head can only learn a label that means the same thing on every row; a pair scorer reads the option text.

Everything downstream of the logits stays as it is: if the base model's inference returns an `[N, K]`
logit matrix per task, then `calibrate.apply`, `fit_prior_bias`, `evaluate.run`, `cascade`, `selective`,
`compare`, `views.view` and `serve` all work unchanged. **The base model is a different `predict_logits`,
not a different pipeline.** That is the design constraint for every item below.

On disk at zero teacher cost: ~917k pairs from 11 unique `teacher.jsonl` files (banking77 424k, swde-field
215k, m2w-element 102k, the other eight 2.5k–63k each; `headlines` ⊂ `headlines-8k`, use the 8k file; `*-rev`
dirs are relabels, not data). Two tasks are 70 % of the pool. The scarce resource is **distinct decisions**:
11 questions is a tiny universe for zero-shot transfer, so teacher hours buy ~30–50 *new small tasks* (§2).

## 1. Hard rules (unchanged, plus two new ones)

- vLLM on :8000 is production: never killed or restarted, `concurrency ≤ 32`; teacher lane ≈ 1 h 10 (12 h)
  / ≈ 2 h (24 h). One training at a time through `scripts/gpu_queue.sh`; mmBERT-small only, `max_len 512`,
  batch 32 (8.9 GB peak measured, 22 GB available today); mmBERT-base stays banned.
- **Folds are by text source, not by task name.** civil_comments sub-column tasks share texts with `toxic`,
  invented questions over kinopoisk reviews share texts with `kinopoisk`, `arb-*` share rows, `m2w-*` share
  pages. A held-out task is held out together with every task over the same texts, or the number is a leak.
- **Unseen tasks are pre-registered here and opened last.** `community-datasets/yahoo_answers_topics`
  (en, 10 topics), `ai-forever/inappropriateness-classification` (ru, noul), `SetFit/sst5` (en, score 1–5).
  M0 verifies they load; a substitute is named *before* any base model is evaluated, never after.
- Claim rule from PLAN-2 §5 holds: three seeds where a seed exists, `compare` CI on every delta, a CI that
  includes 0 is "no measurable difference".

## 2. Diversity: where 30–50 new decisions come from, and what they cost

**S1 — ~24 public sets with gold, small, through the unchanged YAML → `check` → `label` path**
(`tasks/base/<name>.yaml`, 800 train / 200 calib / 500 eval, natural rate): tweet_eval {emotion, hate, irony,
offensive, sentiment, stance ×5 as one task with the target in `data.text`}; trec-coarse; dair-ai/emotion;
financial_phrasebank; boolq, xnli en + ru, paws (all noul); civil_comments insult / threat / obscene /
identity_attack (noul, `gold_prob`); swde page vertical (from `cache/swde`); clinc_oos domain; massive
scenario en + ru; sib200 ru; ru-reviews-classification; RuBQ and MIRACL reranking as noul "this passage
answers the question" (both in the HF cache). Quality check = the existing `check --probe 100`; a task where
the teacher is at chance is dropped and listed in `docs/base.md`. Cost ≈ 36k calls, short texts at the
measured 12–14 ex/s, boolq/xnli/rerank at ~5–6 → **≈ 60 min**.

**S2 — invented tasks over texts already on disk** (24 h plan; 12 h: at most 6). `scripts/invent_tasks.py`
shows the teacher 12 texts from one source (agnews, headlines, kinopoisk, georeview, banking77, arb logs,
m2w element lines, swde nodes, civil_comments) and asks for 6 questions per source: 2 `choice` with 3–6
options, 1 `choice` with 8–12, 2 `noul` statements, 1 `score` rubric with 3–4 levels, each answerable from
the text alone and "not obvious for most texts". Output is written as ordinary YAMLs under `tasks/base/inv-*`
with `{jsonl: data/base/<source>.jsonl}` (700 rows drawn from the source's *train* split, never its eval
ids), so labeling is `openjev label`, nothing new. **Cheap gate, no gold:** label 150 rows; keep the task
only if (a) the argmax marginal's top class ≤ 80 %, (b) mean max-p ≥ 0.55, (c) `scripts/perm_check.py`-style
reversed-option relabel of 100 rows agrees on argmax ≥ 0.75. Failing tasks are deleted, counted, reported.
Cost: 54 candidates × 150 gate rows + ~30 survivors × 550 + perm 5k ≈ 30k calls → **≈ 45 min**. Invented
tasks are *training diversity only*: no gold, never an evaluation task.

**Unseen + warm-start labels:** the three pre-registered sets, 2000 train + 500 calib + 1000 eval each
(teacher baseline on eval, warm-start pool on train) ≈ 10k calls ≈ **15 min**.

Teacher lane ≈ 1 h 10 (S1 + unseen) in 12 h; ≈ 2 h with S2 in 24 h. The teacher is idle > 90 % of the
track; **the GPU is the bottleneck this time** (§4).

## 3. Design decisions (pair rendering, mixture, loss, head)

**Rendering — one function, `openjev/pairs.py:render(task, option_idx, text) -> (seg_a, seg_b)`**, used
by derive, train, eval and serve. Segment A is the decision and is never truncated; segment B is the text
and takes the rest of the 512-token window (`tok(seg_a, seg_b, truncation="only_second", max_length=512)`).
Segment A, built from the existing `teacher.option_lines(task)` so it reads exactly like the teacher prompt:

```
choice | Q: <question> | option: <option line> | options: <all option lines joined by " | ", cut at 96 tokens, "…(+N more)">
score  | Q: <question> | option: <level>: <description> | options: <all levels>
noul   | Q: <head line, default "Is the following statement about the text true?"> | statement: <statement>
```

The primitive tag is the first token; the options list is the contrast set (p("Good") depends on whether
"Neutral" is offered). `m2w-element` is the one special case (`ponytail:` comment, keyed on task name): the
option is the candidate's own `G) <tag> …` line parsed out of the text and the options list is omitted
because the text already holds all 16. Target for a pair = the teacher's `probs[k]` for that row (0 outside
a banking77 shortlist is a legitimate target).

**Loss and head.** `AutoModelForSequenceClassification(num_labels=1)` — CLS pooling + one scalar, no custom
module — trained **pointwise: BCE-with-logits against the teacher's probability of that option.** Inference:
`noul` = sigmoid of the one pair; `choice`/`score` = sigmoid per option, renormalised over the K options
offered, and the *renormalised log-probabilities* are what goes into `calibrate.apply` as the `[N, K]`
logit matrix. Why pointwise and not listwise: one loss for all three primitives, one head, negatives are
"drop pairs" with no group bookkeeping, and a single pair returns a probability, which is the product
promise. Listwise (softmax over a row's kept pairs, KL to the teacher — the existing `train.py` loss) is a
one-training ablation on fold F1 in the 24 h plan, decided by `compare`.

**Mixture.** Per epoch, per task: ≤ 12k pairs, rows sampled fresh each epoch. Rows with K > 8 keep the
teacher-argmax pair + 7 negatives (4 drawn ∝ teacher p, i.e. the shortlist's hard ones, 3 uniform), redrawn
every epoch; K ≤ 8 keeps all pairs. So banking77 contributes 12k of its 424k pairs, the same as agnews —
volume is capped, the pool's 70 % skew is gone, and ~250k pairs/epoch cover 11 + ~24 + ~30 tasks. One
epoch, lr 5e-5, 6 % warmup, bf16, batch 32, dynamic padding as now. Training time is the unknown that sets
every cap: measured 95 ex/s at len 256 and 28 ex/s at len 512 on the fixed head; mixed pairs are assumed
~60/s → ≈ 70 min per base training. **M0 measures pairs/s on a 300-step smoke run and rescales the per-task
cap so one base training is ≤ 60 min.** That number, not this paragraph, is the budget.

**Folds** (`tasks/base/folds.yaml`: run dir → text-source group; the trainer excludes a group, not a name):
F1 = {kinopoisk, swde-field (+swde vertical), toxic (+civil_comments sub-tasks)}, F2 = {georeview,
banking77, m2w-element + m2w-target}, F3 = {agnews, headlines-8k, arb-success + arb-quality}. Invented tasks
over a held-out source leave with it. Three fold models + one full model = 4 base trainings; 12 h runs F1,
F2 and full (6 held-out tasks cover every primitive, both languages and the K = 33/77 cases).

## 4. Milestones (wall clock; agent A = code, agent B = tasks/teacher lane; GPU queue in the background)

Times are the 24 h plan; the 12 h subset is marked. Commands assume the repo root, `.venv`, proxy vars unset.

### M0 — pair core (0:00–1:30, agent A) — 12 h: yes
`openjev/pairs.py`: `render`, `derive(run_dir, task, cap, rng) -> pair rows {seg_a, seg_b, target, row_id,
k}`, `predict_logits(tok, model, task, texts) -> [N, K]` (renormalised log-probs; `noul` → `[logit(p),
logit(1-p)]` so `views.view` and `evaluate` need no change). `train.py`: `pair=True` path (num_labels 1, BCE,
per-epoch re-derive, same early stopping on calib — calib metric is mean BCE on the held-in tasks' calib
pairs). `spec.py`: `task_from_dict` split out of `load_task` (serve needs it in M5). `cli.py`: `openjev base
train --fold F1|F2|F3|none [--tag]` → `runs/base-<fold>/`. 300-step smoke on agnews pairs, printing pairs/s.
Accept:
```
pytest -q tests                                   # + test_pairs: render ≤ 512 tokens on kinopoisk's longest row, seg_a intact; derive counts; logits [N,K]
openjev base train --fold none --tag smoke --max-steps 300   # prints pairs/s -> per-task cap set so pairs_per_epoch / pairs_per_s <= 3600
```

### M1 — new tasks and the teacher lane (1:00–3:30, agent B; teacher runs until ≈ 4:00) — 12 h: S1 + unseen only
S1 YAMLs (§2), `openjev check` on each, `--probe 100`, then `openjev label` in one serial job file
(`runs/queue8.jobs`-style, teacher lane, concurrency 32). Unseen sets labelled last. 24 h: `scripts/invent_tasks.py`
+ gate, run after S1. Accept:
```
for t in tasks/base/*.yaml; do openjev check $t | grep -E "^task|teacher cost"; done   # every task loads, no max_len warning
ls runs/base-*/teacher.jsonl | wc -l                                                    # ≥ 24 (12 h) / ≥ 50 (24 h)
python scripts/invent_tasks.py --report        # 24 h: candidates / gated out (a,b,c) / kept, per source
```

### M2 — base trainings on the queue (3:30–8:30 GPU, agent A 3:00–4:30) — 12 h: F1, F2, full
`runs/queue-base.jobs`: `openjev base train --fold F1`, `--fold F2`, `--fold F3` (24 h), `--fold none`
(the artefact). `openjev base eval --fold F1 --task tasks/kinopoisk.yaml` runs the fold model zero-shot on
the task's *existing* eval ids and writes `runs/base-F1-zs-kinopoisk/{eval_rows.jsonl, openjev.json,
calib.json}` + `results/base-F1-zs-kinopoisk.json`, with four calibration variants as `--calib
raw|prior|teacher500|gold500` (raw; `fit_prior_bias` on the 500 calib *texts*, no labels; T fitted to the
teacher's soft probs on the 500 calib rows the task already has — "1 minute of teacher"; the full vector fit
on 500 gold — the per-task student got the same). `compare runs/kinopoisk runs/base-F1-zs-kinopoisk`
works unchanged because the ids and the file shape are the same. Accept:
```
openjev base eval --fold F1 --task tasks/kinopoisk.yaml --calib prior && openjev compare runs/kinopoisk runs/base-F1-zs-kinopoisk-prior
tail -3 runs/queue2.log                            # base trainings OK, each ≤ 60 min, peak_gpu_gb ≤ 9.5
```

### M3 — leave-source-out readout, unseen tasks, calibration table (5:00–10:00, both agents) — 12 h: 6 tasks, 2 unseen
Per held-out original task: chance, teacher zero-shot, per-task student (existing 3 seeds), base zero-shot
under the four calib variants, `compare` vs student and vs teacher. Unseen tasks: full model vs teacher vs
chance (no student exists — M4's warm-start student is the reference). Calibration-under-transfer table:
ECE_raw and NLL of base zero-shot vs the per-task student's `ece_raw` on the same rows, plus what each calib
variant buys — that answers "does `prior:` cover it" (mechanically yes: `fit_prior_bias` only needs logits)
and "is a pair scorer better calibrated out of the box" (measured, not assumed). Bench: `openjev bench` on
`runs/base-none-zs-<task>` for K = 2, 4, 6, 16, 33, 77 as ms per *decision* and decisions/s. Accept:
```
grep -c "95 % CI" docs/base.md         # ≥ one per held-out task per variant
python scripts/base_table.py           # prints the LOTO table from results/base-*.json, no hand-typed cells
```

### M4 — warm start (8:30–13:00 GPU, agent A) — 12 h: kinopoisk only, N ∈ {100, 500}
**How many labelled rows does a new task need when it starts from `open-jev-base`?** Arms: init ∈
{mmBERT-small, the fold model that never saw the task} × N ∈ {100, 250, 500, 1000} teacher-labelled train
rows × 3 seeds; same pair architecture in both arms (the only factor is the init), same 500 calib rows
(vector fit), same eval ids as the existing student. Tasks: kinopoisk (F1 model; the 4000-row student
0.657 ± 0.003 and the 500-gold-in-loss run 0.616 are measured reference lines) and sst5 (full model, unseen,
score). `openjev run tasks/kinopoisk.yaml --student runs/base-F1/student --train-n N --tag warm-n$N --seed s`
(`--train-n`: first N train rows in id order, as `--gold-n`). 24 runs per task at 1–4 min ≈ 1.5 h per task.
Publishable: the curve with CIs and one sentence "at N = 100 (≈ 10 s of teacher time) base-init reaches x,
from-scratch y, the 4000-row student z" — only where the CI excludes 0; "no measurable difference" at
N = 1000 is the expected, acceptable outcome. Accept:
```
openjev compare runs/kinopoisk-warm-n100 runs/kinopoisk-scratch-n100      # 3 seeds each, verdict line
python scripts/base_table.py --warm     # 2 tasks × 4 N × 2 inits, mean ± sd, CI columns
```

### M5 — the artefact (13:00–16:00 24 h / 9:30–11:00 12 h, agent B)
`openjev serve runs/base-none`: the request carries the task fragment (`"question"` + `"options"`, or
`"statement"`, or `"rubric"`) → `spec.task_from_dict` → K pairs, one forward pass at batch K, `view()`
unchanged; `prior` and `escalate_below` per request. `openjev push runs/base-none --repo
sshalimov04/open-jev-base`: the card is the LOTO table, unseen rows, warm-start curve, K-cost table and the
§6 "may not claim" list; `openjev serve hf:sshalimov04/open-jev-base` is the try-it path. Accept:
```
openjev serve runs/base-none --port 8097 &  # then:
curl -s localhost:8097/v1/systemone -d '{"model":"base","input":"Shares fell 8% after the profit warning.","question":"Is this good or bad for the company?","options":["good","bad","unclear"]}'
openjev push runs/base-none --repo sshalimov04/open-jev-base --dry-run
```

### M6 — ablations (24 h only, 16:00–20:00 GPU): does diversity actually buy anything?
The §0 argument is a hypothesis until this runs: fold F1 retrained on (a) the 8 held-in originals only,
(b) + S1, (c) + S1 + S2 (= the M2 model). Zero-shot on kinopoisk / swde-field / toxic, `compare` (a) vs (b)
vs (c). Plus the listwise-loss arm on F1. Three trainings ≈ 3 h GPU. If (b) and (c) do not beat (a) with the
CI excluding 0 on at least 2 of 3 tasks, the README sentence is "adding 50 small tasks did not measurably
improve zero-shot transfer" and S2 is not part of the artefact's story.

### M7 — write-up, review, ship (20:00–24:00 / 11:00–12:00)
`docs/base.md` (design, tables, cost, failure modes as measured); README section with the claim exactly as
§6 allows; base rows stay out of the `openjev report` table (a different experiment); `oj-planner` honesty
review with the REVIEW-final brief, 45 min to fix, commit, push.

Budget, 24 h: ≈ 22 h wall, GPU lane ≈ 4 base + 3 warm + 3 ablation ≈ 10 h serial. **12 h:** M0, M1-S1, M2
(F1, F2, full ≈ 3 h GPU), M3 on 6 tasks + 2 unseen, M4 kinopoisk {100, 500} (≈ 40 min GPU), M5, M7 — no S2,
no F3, no ablations: the 12 h write-up may say "zero-shot on 6 held-out tasks" and "warm start on one task"
and *nothing* about whether diversity helped.

## 5. Inference cost, quantified

K forward passes per decision. From the measured numbers (b = 1 p50 5.2–6.2 ms; b = 64 869 ex/s at agnews
length ≈ 1.15 ms/example amortised): K = 4 ≈ 8 ms, ~200 decisions/s; K = 16 ≈ 25 ms, ~55/s; K = 33 ≈ 45 ms,
~26/s; K = 77 ≈ 95 ms, ~11/s — against 5 ms / 2644 decisions/s for the fixed banking77 head, **~240× at
K = 77**, plus ~4 tokens per option in segment A. Rule to publish: right tool for K ≤ ~10 in an inner loop
and K ≤ ~33 offline; above that the per-task student, or the base as a re-ranker over a shortlist. M3's
bench numbers replace these estimates.

## 6. What will probably not work, and what may be claimed

- **Zero-shot will be clearly below the per-task student on most held-out tasks.** 140M parameters and
  ~60 questions is not a world model. Expect kinopoisk's *Neutral*, georeview's stars and arb-quality near
  the teacher's marginal; banking77 (77 unseen intents) and the Russian tasks (S1 is mostly English) worst.
  The honest headline may be "x % of the student on n of 6 tasks, and a better *init*" — not "any question".
- **Pointwise sigmoids renormalised over a large K may be badly calibrated raw** (everything near 0). The
  calib-variant table measures it; if `teacher500` fixes it, "zero-shot plus one minute of teacher" is the recipe.
- **The pair scorer inherits the teacher's marginal shift exactly like the K-way head.** No head shape fixes
  a wrong prior; `prior:` on balanced benchmarks is the leak PLAN-2 named. Say it again.
- **Leakage is the easy way to fake this.** Folds by text source (§1), invented tasks leave with their
  source, unseen sets opened last, no cap or mixture knob tuned on a held-out task's number.
- **Warm start may show nothing at N = 1000** — from-scratch already saturates at 200–500 in this repo. The
  interesting region is 100–250, where the CI is widest; three seeds is the floor, and "no measurable
  difference" is a permitted result.
- **Never claim:** "general", "any question" without the LOTO table beside it; a zero-shot number on a task
  whose source was in training; a diversity effect without M6; agent accuracy of any kind.

README claim, if M3 and M4 land: *"`open-jev-base` answers a typed question it was never trained on. On six
held-out tasks it reaches [table] zero-shot against the per-task students, and warm-starting a new task from
it needs N = [100/250] teacher-labelled rows to match what [z] took 4000 — every delta a paired-bootstrap CI."*
If only M4 lands: *"a better init, not a zero-shot oracle: N rows instead of 4000."* If neither: the LOTO
table is published as a negative result with the m2w-element argument intact, because the argument was
about primitives, not about this model.

## 7. Cut order (first cut first) and the 4 am fallback

1. M6 listwise arm 2. M6 diversity arms 3. S2 invented tasks 4. sst5 warm-start task 5. fold F3
6. `bench` at K 7. `serve` request-time task fragment (push the checkpoint anyway) 8. warm-start N = 250, 1000
9. the third unseen set. **Never cut:** folds by text source, F1 + F2 + full, the six-task LOTO table with
CIs and all four calib variants, kinopoisk warm start at N = 100 with 3 seeds, the honesty review.
If a base training exceeds 60 min, halve the per-task cap and restart — do not raise `max_len`, do not
change the batch. If the teacher is busy, S1 order is: xnli en/ru, boolq, tweet_eval, civil_comments
sub-tasks, then the rest; whatever is unlabelled at hour 4 is out of the mixture and listed as such. If the
queue OOMs, kill only the trainer, confirm `/v1/models` answers, restart after the gate passes. Every
committed number comes from `results/base-*.json` via `scripts/base_table.py`; no hand-edited table cells.
