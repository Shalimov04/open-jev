# open-jev

Turn a prompt into a small, fast, calibrated classifier. You describe a decision in a YAML file — a
choice between options, a score on a rubric, or the truth of a statement — and `openjev` has a local
LLM teacher (here Qwen3.8-27B on vLLM) label a few thousand examples with *soft* labels (per-option probabilities read off the
logprobs of a single constrained letter token), distills those into a ~140M encoder, fits a
temperature on a held-out split, and serves the result at `POST /v1/systemone`. The output is a typed
decision with probabilities you can threshold on, not a string you have to parse. It is an open
reimplementation of the idea behind TypeSafe's Jev — Jev itself is a closed hosted API, so its
published claims cannot be checked from outside and none of them are repeated here; the only numbers
in this repo are the ones produced by the code in this repo.

## How it works

1. **Task** — one YAML: type (`choice` / `score` / `noul`), the question, the option set, where the
   text comes from, split sizes. `docs/task-spec.md` is the full reference.
2. **Teacher** — a local vLLM endpoint (`OPENJEV_TEACHER_URL`, default `http://localhost:8000/v1`)
   is asked one constrained question per example; the request is exactly:

   ```json
   {"model": "<first id from /v1/models>", "max_tokens": 1, "temperature": 0,
    "logprobs": true, "top_logprobs": 4,
    "messages": [{"role": "system", "content": "<question>\nA: World\nB: Sports\nC: Business\nD: Sci/Tech\nAnswer with a single letter."},
                 {"role": "user", "content": "<the text>"}],
    "structured_outputs": {"choice": ["A", "B", "C", "D"]},
    "chat_template_kwargs": {"enable_thinking": false}}
   ```

   Letters come back as bare tokens; `softmax` over their logprobs is the soft label. `finish_reason:
   "length"` is expected. More than 19 options → chunked shortlist (`top_logprobs` caps at 20).
3. **Student** — `AutoModelForSequenceClassification` (default `jhu-clsp/mmBERT-small`) trained on
   `KL(teacher ‖ student)`, optionally plus a gold CE term; early stop on calibration-split KL.
4. **Calibrate** — one temperature fitted by LBFGS on the calib split's NLL.
5. **Serve** — `openjev serve runs/<task>`: the calibrated probability vector goes through one view
   per type (`openjev/views.py`) and comes back as a typed JSON decision.

Everything lands in `runs/<task>/`: append-only `teacher.jsonl`, `student/`, `calib.json`,
`eval.json`, and `openjev.json` (spec + labels + temperature + metrics — the exportable bundle).

## Quickstart

```bash
uv venv && uv pip install -e . && . .venv/bin/activate   # ./.venv; python -m venv .venv works too
cp tasks/example.yaml tasks/mytask.yaml                  # your own task: edit question, options, data.source
OPENJEV_TEACHER_URL=http://localhost:8000/v1 openjev run tasks/agnews.yaml   # label -> train -> calibrate -> eval
openjev serve runs/agnews --port 8099                    # blocks; curl from another shell
curl -s localhost:8099/v1/systemone -H 'Content-Type: application/json' \
  -d '{"model":"agnews","input":"Shares of the airline fell 8% after it cut its full-year profit forecast."}'
```

That last line, run against `runs/agnews`, prints:

```json
{"model":"agnews","choice":"Business","probabilities":{"World":0.005825810134410858,
 "Sports":0.005110130645334721,"Business":0.9872164130210876,"Sci/Tech":0.0018476743716746569},
 "confidence":0.9872164130210876}
```

`GET /v1/models` lists the loaded runs. Each served model is one trained student with a fixed type
and a fixed label set — this is not a general model that takes arbitrary options at request time.

Run everything from the repo root (`results/` and this README are resolved relative to it). Tests:
`pytest -q tests` — under a minute on CPU, no GPU and no teacher needed (also run on every push by
`.github/workflows/test.yml`); two of them want
`jhu-clsp/mmBERT-small` and `fancyzhx/ag_news` in the Hugging Face cache.

The tasks here are English and Russian, but nothing is language-specific: `question`, `options` and
`statement` are free text, so write them in whatever language you want the teacher prompted in, as
long as the teacher and the student checkpoint (mmBERT is multilingual) both cover it. `lang` itself
is only a report column and the language `openjev augment` generates in.

## Serving

`openjev serve runs/agnews runs/kinopoisk --port 8099` loads every run dir it is given and routes on
the `model` field. `--device cpu` forces CPU (the default is CUDA when it is there). The forward
pass is 5 ms on GPU and 30 ms on CPU for mmBERT-small (the table below); measured end to end through
HTTP on a busy box, one request is ~21 ms on GPU and ~160 ms on CPU, so most of the CPU number is
tokenisation and framing, not the model. Either way CPU-only serving is a real option, not a
fallback.

**Batch.** `input` may be a list. One forward pass, one list back, same order:

```bash
curl -s localhost:8099/v1/systemone -H 'Content-Type: application/json' \
  -d '{"model":"agnews","input":["Shares of the airline fell 8% after it cut its forecast.",
                                 "Manchester United beat Arsenal 2-1."]}'
```

**Escalation.** `eval` writes the cascade *parity point* into `openjev.json` as `escalate_below`: the
lowest confidence threshold at which "student answers when confident, teacher answers the rest"
matches the teacher's own metric. Serve returns `"escalate": true/false` against it, and
`escalate_below` in the request overrides it. **The server never calls the teacher** — that is your
decision to make, and it keeps the teacher's URL, key and concurrency out of this process:

```python
r = httpx.post(f"{OPENJEV}/v1/systemone", json={"model": "kinopoisk", "input": text}).json()
answer = ask_the_llm(text) if r["escalate"] else r["choice"]
```

**Prediction sets.** `"coverage": 0.9` adds `"set": [...]`, a split-conformal (LAC) set built from
the calib split's `1 − p_true` quantile — it contains the right label ~90 % of the time, and it is
wider exactly where the model is unsure. `"set_target"` says what "right" was measured against:
`gold`, or `teacher` when the calib split had no gold labels, in which case the set covers *the
teacher's* answer and not the truth.

```bash
curl -s localhost:8099/v1/systemone -H 'Content-Type: application/json' \
  -d '{"model":"kinopoisk","input":"Ну такое. Актёры старались, но сценарий провальный.","coverage":0.9}'
# {"model":"kinopoisk","choice":"Bad",
#  "probabilities":{"Bad":0.6697,"Neutral":0.2994,"Good":0.0309},"confidence":0.6697,
#  "escalate":false,"set":["Bad","Neutral"],"set_target":"gold"}
```

`GET /v1/models` lists what is loaded, with each model's `escalate_below` and whether it has a
conformal calibration.

### Share a trained student

```bash
openjev push runs/agnews --repo you/openjev-agnews --dry-run   # prints the file list and the card
openjev push runs/agnews --repo you/openjev-agnews             # private; --public to publish
openjev serve hf:you/openjev-agnews --port 8099                # snapshot_download, then as usual
```

`push` uploads `student/`, `openjev.json`, `conformal.json` and a generated model card — the task
YAML, the teacher id, the results row, the calibration method, and the "no gold in the loss"
sentence when that is what the run actually did. It does **not** upload `teacher.jsonl`, so the
teacher's labels stay yours. Needs a Hugging Face token with write access (`hf auth login`).

### Comparing against another endpoint

`scripts/compare_api.py` scores any `/v1/systemone`-shaped endpoint on the same eval rows this
repo's numbers come from, and writes `results/api/api-<name>-<task>.json` with accuracy, argmax
agreement with our student, and p50 latency:

```bash
python scripts/compare_api.py --adapter openjev --url http://localhost:8099 --task agnews --limit 200
JEV_API_KEY=... python scripts/compare_api.py --adapter jev --task agnews --limit 200
```

The `jev` adapter targets [TypeSafe's Jev](https://docs.typesafe.ai/api.md), whose request shape
(`state` + a `questions` map of Choice/Score/Noul) this project's own endpoint deliberately mirrors.
It is written from the published docs and has not been run against the live API — there is no key
here. Latency across two providers is two different machines and is not a comparison; accuracy and
agreement on identical rows are.

## Add your own task

Write the YAML → `check` → `check --probe 100` → `run` → `serve`. The teacher is the ceiling of
everything downstream, so the two `check` steps are there to make iterating on the *prompt* cheap:
the first costs nothing, the second costs 100 calls.

```bash
cp tasks/example.yaml tasks/mytask.yaml        # or the closest task in docs/task-spec.md
openjev check tasks/mytask.yaml                # no teacher calls
openjev check tasks/mytask.yaml --probe 100    # ~1 min of teacher time; the rows are cached for `run`
openjev run   tasks/mytask.yaml                # label -> train -> calibrate -> eval
openjev serve runs/mytask
```

`check` prints the splits and their gold distribution, the system prompt verbatim, three examples
exactly as the teacher will see them (after `data.text` and `max_chars`), text length in characters
and in student tokens, and an estimated teacher cost (±2×). It warns when `max_chars` truncates more
than a quarter of the rows, and when more than a quarter of them are longer than `student.max_len` —
that second one is the trap kinopoisk fell into: the teacher reads 1500 characters, and a student
left at the default `max_len: 256` would read about half of that.

`--probe 100` labels the first 100 calib rows into the real `runs/mytask/teacher.jsonl` (so `run`
reuses them) and prints what the teacher actually answers — accuracy vs gold, its predicted marginal
next to the gold marginal, mean max-p, and a warning for any class it under-predicts by more than 10
points. On kinopoisk that warning is the whole story of the task:

```
probe: 100 calib rows, teacher Qwen/Qwen3.8-27B-FP8   # the id is read back from label.json
  teacher accuracy vs gold: 0.710 on 100 rows
  predicted marginal: Bad 30.0%  Neutral 9.0%  Good 61.0%
  gold marginal:      Bad 34.0%  Neutral 24.0%  Good 42.0%
  mean max-p: 0.812
  WARNING the teacher under-predicts 'Neutral' (9% vs 24%): reword that option, or declare `prior:` ...
```

Spec and data errors come out of `check` as one line with a hint and exit 1 — it is also the fastest
way to debug a `data.text` template. Full field reference: `docs/task-spec.md`.

## Results

<!-- results -->
| task | type | K | lang | n_train | student acc / F1 | teacher acc / F1 | agree | ECE raw→cal | Brier | GPU p50 ms | ex/s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| agnews (n=3) | choice | 4 | en | 4000 | 0.888 ± 0.006 / 0.888 ± 0.006 | 0.880 / 0.881 | 0.916 ± 0.004 | 0.075 ± 0.005→0.030 ± 0.004 | 0.177 ± 0.006 | 5.3 | 869 |
| agnews-e12 | choice | 4 | en | 4000 | 0.889 / 0.889 | 0.880 / 0.881 | 0.914 | 0.079→0.023 | 0.174 | – | – |
| agnews-mmBERT-base | choice | 4 | en | 4000 | 0.891 / 0.892 | 0.880 / 0.881 | 0.916 | 0.081→0.023 | 0.175 | 6.3 | 435 |
| agnews-nogold | choice | 4 | en | 4000 | 0.944 / 0.944 | 1.000 / 1.000 | 0.944 | 0.131→0.129 | 0.115 | 5.7 | 863 |
| banking77 | choice | 77 | en | 3000 | 0.751 / 0.741 | 0.764 / 0.749 | 0.766 | 0.017→0.042 | 0.363 | 5.2 | 2644 |
| georeview | score | 5 | ru | 4000 | 0.558 / 0.567 | 0.470 / 0.455 | 0.649 | 0.253→0.025 | 0.556 | 6.2 | 203 |
| georeview-mmBERT-base | score | 5 | ru | 4000 | 0.564 / 0.573 | 0.470 / 0.455 | 0.648 | 0.242→0.027 | 0.544 | 7.1 | 115 |
| headlines (n=3) | choice | 6 | ru | 4522 | 0.838 ± 0.005 / 0.838 ± 0.005 | 0.783 / 0.775 | 0.816 ± 0.002 | 0.025 ± 0.001→0.029 ± 0.007 | 0.248 ± 0.001 | 5.5 | 2644 |
| headlines-8k (n=3) | choice | 6 | ru | 8000 | 0.851 ± 0.002 / 0.851 ± 0.002 | 0.783 / 0.775 | 0.818 ± 0.002 | 0.031 ± 0.004→0.036 ± 0.004 | 0.231 ± 0.002 | – | – |
| headlines-e12 | choice | 6 | ru | 4522 | 0.836 / 0.836 | 0.783 / 0.775 | 0.813 | 0.034→0.023 | 0.250 | – | – |
| headlines-nosynth (n=3) | choice | 6 | ru | 4000 | 0.838 ± 0.004 / 0.838 ± 0.004 | 0.783 / 0.775 | 0.818 ± 0.006 | 0.024 ± 0.006→0.030 ± 0.005 | 0.247 ± 0.004 | – | – |
| kinopoisk (n=3) | choice | 3 | ru | 4000 | 0.657 ± 0.003 / 0.656 ± 0.004 | 0.652 / 0.609 | 0.693 ± 0.006 | 0.154 ± 0.006→0.040 ± 0.005 | 0.460 ± 0.001 | 7.0 | 186 |
| kinopoisk-e12 | choice | 3 | ru | 4000 | 0.660 / 0.657 | 0.652 / 0.609 | 0.681 | 0.173→0.033 | 0.463 | – | – |
| kinopoisk-gold | choice | 3 | ru | 4000 | 0.672 / 0.669 | 0.652 / 0.609 | 0.655 | 0.075→0.030 | 0.435 | – | – |
| kinopoisk-gold-n500 (n=3) | choice | 3 | ru | 4000 | 0.616 ± 0.011 / 0.614 ± 0.012 | 0.652 / 0.609 | 0.624 ± 0.021 | 0.086 ± 0.025→0.044 ± 0.014 | 0.490 ± 0.012 | – | – |
| toxic | noul | 2 | en | 4000 | 0.917 / 0.626 | 0.893 / 0.624 | 0.925 | 0.116→0.037 | 0.129 | 5.7 | 437 |

- **agnews**: jhu-clsp/mmBERT-small distilled from the teacher with zero gold labels; argmax accuracy vs gold on 2000 eval examples; calibrated to gold (T=0.64).
- **agnews-e12**: jhu-clsp/mmBERT-small distilled from the teacher with zero gold labels; argmax accuracy vs gold on 2000 eval examples; calibrated to gold (T=0.62).
- **agnews-mmBERT-base**: jhu-clsp/mmBERT-base distilled from the teacher with zero gold labels; argmax accuracy vs gold on 2000 eval examples; calibrated to gold (T=0.63).
- **agnews-nogold**: jhu-clsp/mmBERT-small distilled from the teacher with zero gold labels; argmax accuracy vs teacher on 2000 eval examples; calibrated to teacher-soft (T=0.99).
- **banking77**: jhu-clsp/mmBERT-small distilled from the teacher with zero gold labels; argmax accuracy vs gold on 2000 eval examples; calibrated to gold (T=0.82).
- **georeview**: jhu-clsp/mmBERT-small distilled from the teacher with zero gold labels; expected level; MAE vs gold on 2000 eval examples; MAE 0.609 (teacher 0.619); calibrated to gold (T=1.09).
- **georeview-mmBERT-base**: jhu-clsp/mmBERT-base distilled from the teacher with zero gold labels; expected level; MAE vs gold on 2000 eval examples; MAE 0.594 (teacher 0.619); calibrated to gold (T=1.07).
- **headlines**: jhu-clsp/mmBERT-small distilled from the teacher with zero gold labels; argmax accuracy vs gold on 2000 eval examples; calibrated to gold (T=0.81).
- **headlines-8k**: jhu-clsp/mmBERT-small distilled from the teacher with zero gold labels; argmax accuracy vs gold on 2000 eval examples; calibrated to gold (T=0.75).
- **headlines-e12**: jhu-clsp/mmBERT-small distilled from the teacher with zero gold labels; argmax accuracy vs gold on 2000 eval examples; calibrated to gold (T=0.78).
- **headlines-nosynth**: jhu-clsp/mmBERT-small distilled from the teacher with zero gold labels; argmax accuracy vs gold on 2000 eval examples; calibrated to gold (T=0.81).
- **kinopoisk**: jhu-clsp/mmBERT-small distilled from the teacher with zero gold labels; argmax accuracy vs gold on 1500 eval examples; calibrated to gold (T=1.28).
- **kinopoisk-e12**: jhu-clsp/mmBERT-small distilled from the teacher with zero gold labels; argmax accuracy vs gold on 1500 eval examples; calibrated to gold (T=1.27).
- **kinopoisk-gold**: jhu-clsp/mmBERT-small distilled from the teacher with gold CE weight 1.0; argmax accuracy vs gold on 1500 eval examples; calibrated to gold (T=1.03).
- **kinopoisk-gold-n500**: jhu-clsp/mmBERT-small distilled from the teacher with gold CE weight 1.0; argmax accuracy vs gold on 1500 eval examples; calibrated to gold (T=1.43).
- **toxic**: jhu-clsp/mmBERT-small distilled from the teacher with no gold in the loss (but train rows picked 50/50 by gold); p(true); AUROC vs gold on 3000 eval examples; AUROC 0.856 (teacher 0.817); calibrated to gold (T=0.25).
<!-- results -->

Teacher for every row: **Qwen3.8-27B**, 3-bit GSQ quantisation (ISTA-DASLab) with MTP, served by vLLM
on the same box under the id `Qwen/Qwen3.8-27B-FP8`, called with `max_tokens: 1` and
`enable_thinking: false`. Every "teacher" number in the table is that model zero-shot; the students
cannot do better than it except by denoising it. Runs from before 2026-09-20 do not record the model
id in `runs/<task>/label.json` — newer ones do.

## What the numbers mean

- **The student sees zero gold labels.** `gold_weight` is 0 by default, so training uses only the
  teacher's distributions. Gold is used for the eval and (when present) for fitting the temperature.
  One exception: **toxic** sets `balance: true` on its training split, which picks 50/50 rows *by gold*
  from an 8%-positive dataset. Its labels are still the teacher's, but the row selection spent gold, so
  it is not a zero-gold row; every other task is. That balanced draw also came out of the same 60k-row
  pool as calib, and it went first, so toxic's calib split ended up **11/500 = 2.2% positive against
  8.1% on eval** — T = 0.25 was fitted on 11 positives. It still cut eval ECE 0.116 → 0.037, but it is
  a temperature fitted under the wrong prior on very few rows, not a clean result. The sampler now
  draws the natural-rate splits before any balanced one; the numbers in the table predate that fix.
- **The baseline is the teacher, not the state of the art.** Teacher zero-shot accuracy on the same
  eval split is in the table beside the student's. The claim being tested is "a 140M encoder can keep
  the teacher's accuracy at a fraction of the cost", not "this beats a supervised model".
- **ECE** is expected calibration error: max-probability bucketed into 15 equal-width bins, mean over
  bins of |mean confidence − accuracy|, weighted by bin mass. `raw→cal` is before and after
  temperature scaling. Lower is better; it is a summary, not a guarantee about any single prediction.
- **Brier** is the multiclass sum-of-squares against the one-hot gold, on calibrated probabilities.
- **agree** is argmax agreement between student and teacher on the eval split. It is the metric that
  matters when a task has no gold at all.
- **Latency** is batch-1 on the GB10 GPU, p50 of 200 requests after 20 warmups. Every row in the table
  was measured with vLLM resident but **idle**, so the rows are comparable to each other; a box without
  the teacher loaded at all would be a little faster still. `ex/s` is batch-64 throughput on the same
  hardware. `results/*.json` also carries a CPU batch-1 p50.
- **Distillation cannot beat its teacher's *systematic* errors.** Where the teacher is consistently
  wrong the student inherits it, and the agreement column tells you how much. Independent per-example
  noise is the one thing averaging can wash out — that is the most it did on toxic.

## Findings

One night, seven runs, one teacher. What it showed:

**Distillation reaches the teacher on easy tasks, and stops there.** On agnews the 140M student scores
0.879 against the teacher's 0.880 — parity, at 5.3 ms batch-1 and 869 ex/s instead of one LLM call per
example. The same task with `mmBERT-base` (307M params vs 140M) buys 0.883: +0.4 points for half the
batch-64 throughput (435 vs 869 ex/s) and 9.8 GB of training memory instead of 5.5. On a task this
easy the student size is not the bottleneck; the teacher is.

**The student ranks better than its teacher on toxic** — AUROC 0.856 vs 0.817, and Brier against the
annotator fraction 0.035 vs 0.048. Ignore the accuracy column here: the eval split is 8.1% positive
(244/3000), so always answering "not toxic" scores 0.919 — above both the student's 0.917 and the
teacher's 0.893 — and macro-F1 is a tie (0.626 vs 0.624). Accuracy on a task this skewed says nothing;
the ranking and probability metrics are where the student is ahead. That win is not free either: toxic
is the one task whose training rows were selected 50/50 *by gold* (`balance: true` over an 8%-positive
dataset), so gold paid for the row selection even though every label is still the teacher's. One
plausible mechanism is that averaging 4000 soft labels over a balanced sample denoises the teacher's
per-example noise; it is one task, and it does not generalise to the ones below.

**On the hard Russian tasks the framework did not deliver.** kinopoisk: 0.609 against a teacher at
0.652. georeview: MAE 0.783 against a teacher at 0.619. The teacher is the ceiling and the teacher is
weak — on the kinopoisk eval split (gold is an even three-way split) it answers "Good" for 56.5% of
rows and "Neutral" for 12.7%, a prompt-level prior the student reproduces faithfully (agreement 0.817).
Distillation transfers the bias along with the signal. Without gold there is no mechanism here that can
exceed a weak teacher, and none of the knobs in this repo change that.

**banking77 is where the teacher cost shows up.** 77 classes go through the chunked shortlist, which is
6 teacher calls per example: 33,000 calls for 5,500 rows and 61 minutes of teacher time — 42% of the
night's entire teacher budget for one task. The result is 0.748 against the teacher's 0.764 at agreement
0.836, i.e. the shortlist's approximation survives distillation but does not improve under it. The first
attempt at 5 epochs was underfit (calib KL 0.407 and still falling); 12 epochs moved accuracy 0.7425 →
0.7485, +0.6 points, with calib KL at 0.347 and *still* falling. The default of 5 epochs is too low,
full stop: in all seven runs the best epoch was the *last* one and calib KL was still going down, so
early stopping never fired and `epochs` — not patience — is the binding knob.

**Temperature scaling is worth it exactly where the model is overconfident, and nowhere else.**
georeview 0.253 → 0.072 and toxic 0.116 → 0.037 are large, real wins. Where the raw model is already
calibrated it is a no-op or slightly harmful: headlines 0.024 → 0.024 (the fit was rejected), and
banking77 0.017 → 0.028, where a temperature that improved ECE on the 500-row calib split made it worse
on eval. A single scalar fitted on 500 rows is noise at that scale, which is why `calibrate` now keeps
T = 1.0 whenever the fit does not improve ECE on the calib split.

**The augment round is a small gain, not a result.** One round on headlines produced 522 synthetic
Russian headlines from the teacher, aimed at the weakest classes and the most frequent confusion pairs;
accuracy went 0.762 → 0.767 and macro-F1 0.755 → 0.760. That is one round, one seed, no control arm and
no repeat — it is consistent with the method working and equally consistent with variance.

**Teacher cost.** 66,522 teacher calls, 144 minutes (2.4 h) of labeling wall-clock across the six tasks,
of which banking77 alone is 61 minutes. Student training is 1.5–12 minutes per task. The teacher is the
budget; everything else is rounding.

**A task with no gold at all gets the same student.** `tasks/agnews-nogold.yaml` is `tasks/agnews.yaml`
with the `data.gold` line deleted, so nothing in the pipeline ever sees a label: training was already
gold-free, and now the temperature is fitted to the teacher's *soft* probabilities
(`calib_target: teacher-soft`, T=0.99) and eval is reported against the teacher. Read its table row
carefully — with no gold, "teacher acc" is the teacher scored against itself (1.000 by construction)
and "student acc" 0.944 is just the agreement column under another name. The number that actually
answers "what do you get with no labels at all" is not in the pipeline: scored offline against the
real ag_news test labels, this student is **0.8825** against the teacher's **0.8815** on the same 2000
rows — i.e. within noise of the gold-supervised-calibration run (`agnews` student 0.879 / teacher
0.880). Removing gold from the loop cost nothing measurable here, but that is on a task where the
teacher is strong and already well calibrated; the temperature it produced is ~1.0, so on a task where
temperature scaling matters (georeview, kinopoisk) fitting to the teacher's soft labels is untested and
would be calibrating to the teacher's confidence, not to the truth. `gold_acc_offline` and
`teacher_gold_acc_offline` in `results/agnews-nogold.json` are written by `scripts/nogold_gold_acc.py`,
which joins the eval rows back to `fancyzhx/ag_news` by row id — it is a hand-run measurement outside
the pipeline, and re-running `openjev eval` on that task overwrites the file without them.

**A bigger student does not close the georeview gap.** mmBERT-base (307M) at batch 16 gets MAE 0.771
against mmBERT-small's 0.783, with the teacher at 0.619; accuracy 0.4335 vs 0.421 (teacher 0.4705) and
agreement with the teacher 0.820 vs 0.797. That is 8% of a 0.164 MAE gap for 2.2x the parameters, 9.0
minutes of training instead of 1.7, 9.9 GB of GPU memory instead of 5.5, and batch-64 throughput of 115
ex/s instead of 203. The student is not capacity-limited on this task — it is already reproducing the
teacher more faithfully than the small one does, and the teacher is what is wrong. Same conclusion as
agnews-mmBERT-base, from the opposite end: on the easy task the bigger student had nothing left to
learn, and on the hard one there is nothing better to learn from.

## Targeted synthetic data (`openjev augment`)

`openjev augment tasks/<task>.yaml --rounds 1 --per-class 100` is a PGKD-lite active loop. It reads
the per-class recall and the confusion matrix that `eval` measured **on the calib split** (never on
eval), picks the weakest classes and the most frequent confusion pairs, and asks the teacher — normal
generation, `temperature 0.9`, `enable_thinking: false`, JSON-list output — for new texts in the
task's language, few-shot-prompted with three real train examples of that class. Confusion pairs get
the "make half of them look like *B* but really be *A*" variant. The parser is lenient (JSON array if
one can be found, else one text per line); duplicates (against the whole train split and within the
batch) and texts over `max_chars` are dropped.

The survivors are appended to the same append-only `teacher.jsonl` as `{"source": "synth", "split":
"train"}` rows with ids `synth:<round>:<i>`, which the dataset can never produce (dataset ids are
`<source_split>:<row_idx>`), and then labeled by the **normal** label stage. The teacher often
disagrees with the class the text was generated for; that label is kept as-is — the point is targeted
data near the decision boundary, not more hard labels. Train, calibrate and eval then re-run, and
`results/<task>.json` carries `augment_round` and `n_synth`.

Ceiling: no hard-negative mining beyond the confusion pairs, and no filter on whether a synthetic
text is actually useful; the upgrade path is the full PGKD selection loop.

## Requirements

- A vLLM (or other OpenAI-compatible) endpoint serving the teacher, reachable at
  `OPENJEV_TEACHER_URL`, supporting `logprobs` + `top_logprobs` and the vLLM `structured_outputs`
  field. Developed against Qwen3.8-27B (3-bit GSQ + MTP, alias `Qwen/Qwen3.8-27B-FP8`) served by vLLM
  on the same box. Any instruct model that returns letter logprobs works; it sets the ceiling.
- One NVIDIA GPU for training. mmBERT-small at batch 32 / len 256 peaks around 5.5 GB, so it fits
  next to a resident vLLM; mmBERT-base is roughly 8 GB. CPU-only works for `serve` and `eval`.
- Python ≥ 3.10, and the deps in `pyproject.toml` (torch, transformers, datasets, httpx, pyyaml,
  numpy, scikit-learn, fastapi, uvicorn). No other runtime dependencies.
- Developed on a DGX Spark (GB10, 20 cores, 128 GB unified memory), Ubuntu 24.04, torch 2.13/cu130.
- If `HTTP_PROXY`/`HTTPS_PROXY`/`ALL_PROXY` are set, unset them: they break localhost calls to the
  teacher and slow Hugging Face downloads. The teacher client itself uses `trust_env=False`.

## Limitations and follow-ups

- **Option position bias.** The teacher sees one fixed option order, and LLMs are known to prefer
  certain letter positions. Averaging over two or more permutations per example (`shuffle_options`)
  is the obvious fix and is not implemented.
- **K > 19 is approximate.** `top_logprobs` caps at 20, so large label sets go through a chunked
  shortlist: softness is exact only within the shortlist, and if every chunk misses the true class
  the label is simply wrong. Cost is `ceil(K/19) + 1` calls per example.
- **No conformal prediction.** You get a calibrated probability, not a set with a coverage guarantee.
  Abstention is left to you (threshold on `confidence`).
- **No ONNX / quantized export.** `serve` runs the PyTorch model; the export bundle is
  `student/` + `openjev.json`.
- **One temperature per task.** No per-class or vector scaling, no recalibration under drift.
- **Calibration target depends on the data.** With gold on the calib split the temperature is fitted
  to gold; without it, to the teacher's *soft* probabilities (`calib_target: teacher-soft`) — which
  calibrates the student to the teacher's opinion, not to the truth. `calib_target` in
  `results/*.json` says which. A temperature also cannot correct a prior shift, so fit it on a split
  drawn at the same rate as the eval split — toxic is the cautionary example above (calib 2.2%
  positive vs 8.1% on eval, because the balanced train draw went first and depleted the pool).
- **The teacher is the cost.** Labeling dominates wall-clock; the student trains in minutes.
