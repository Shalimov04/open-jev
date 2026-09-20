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
`pytest -q tests` — 19 tests, ~20 s on CPU, no GPU and no teacher needed; two of them want
`jhu-clsp/mmBERT-small` and `fancyzhx/ag_news` in the Hugging Face cache.

The tasks here are English and Russian, but nothing is language-specific: `question`, `options` and
`statement` are free text, so write them in whatever language you want the teacher prompted in, as
long as the teacher and the student checkpoint (mmBERT is multilingual) both cover it. `lang` itself
is only a report column and the language `openjev augment` generates in.

## Results

<!-- results -->
| task | type | K | lang | n_train | student acc / F1 | teacher acc / F1 | agree | ECE raw→cal | Brier | GPU p50 ms | ex/s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| agnews-mmBERT-base | choice | 4 | en | 4000 | 0.883 / 0.884 | 0.880 / 0.881 | 0.945 | 0.081→0.053 | 0.194 | 6.3 | 435 |
| agnews-nogold | choice | 4 | en | 4000 | 0.944 / 0.944 | 1.000 / 1.000 | 0.944 | 0.131→0.129 | 0.115 | 5.7 | 863 |
| agnews | choice | 4 | en | 4000 | 0.879 / 0.880 | 0.880 / 0.881 | 0.938 | 0.071→0.042 | 0.198 | 5.3 | 869 |
| banking77 | choice | 77 | en | 3000 | 0.748 / 0.736 | 0.764 / 0.749 | 0.836 | 0.017→0.028 | 0.357 | 5.2 | 2644 |
| georeview-mmBERT-base | score | 5 | ru | 4000 | 0.433 / 0.409 | 0.470 / 0.455 | 0.820 | 0.242→0.065 | 0.661 | 7.1 | 115 |
| georeview | score | 5 | ru | 4000 | 0.421 / 0.391 | 0.470 / 0.455 | 0.797 | 0.253→0.072 | 0.670 | 6.2 | 203 |
| headlines | choice | 6 | ru | 4522 | 0.767 / 0.760 | 0.783 / 0.775 | 0.860 | 0.024→0.024 | 0.336 | 5.5 | 2644 |
| kinopoisk | choice | 3 | ru | 4000 | 0.609 / 0.564 | 0.652 / 0.609 | 0.817 | 0.161→0.109 | 0.542 | 7.0 | 186 |
| toxic | noul | 2 | en | 4000 | 0.917 / 0.626 | 0.893 / 0.624 | 0.925 | 0.116→0.037 | 0.129 | 5.7 | 437 |

- **agnews-mmBERT-base**: jhu-clsp/mmBERT-base distilled from the teacher with zero gold labels; argmax accuracy vs gold on 2000 eval examples; calibrated to gold (T=0.73).
- **agnews-nogold**: jhu-clsp/mmBERT-small distilled from the teacher with zero gold labels; argmax accuracy vs teacher on 2000 eval examples; calibrated to teacher-soft (T=0.99).
- **agnews**: jhu-clsp/mmBERT-small distilled from the teacher with zero gold labels; argmax accuracy vs gold on 2000 eval examples; calibrated to gold (T=0.74).
- **banking77**: jhu-clsp/mmBERT-small distilled from the teacher with zero gold labels; argmax accuracy vs gold on 2000 eval examples; calibrated to gold (T=1.09).
- **georeview-mmBERT-base**: jhu-clsp/mmBERT-base distilled from the teacher with zero gold labels; expected level; MAE vs gold on 2000 eval examples; MAE 0.771 (teacher 0.619); calibrated to gold (T=1.91).
- **georeview**: jhu-clsp/mmBERT-small distilled from the teacher with zero gold labels; expected level; MAE vs gold on 2000 eval examples; MAE 0.783 (teacher 0.619); calibrated to gold (T=1.95).
- **headlines**: jhu-clsp/mmBERT-small distilled from the teacher with zero gold labels; argmax accuracy vs gold on 2000 eval examples; temperature kept at 1.0 (fitting it did not improve ECE on the calib split).
- **kinopoisk**: jhu-clsp/mmBERT-small distilled from the teacher with zero gold labels; argmax accuracy vs gold on 1500 eval examples; calibrated to gold (T=2.35).
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
