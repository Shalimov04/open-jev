# open-jev

Turn a prompt into a small, fast, calibrated classifier. You describe a decision in a YAML file — a
choice between options, a score on a rubric, or the truth of a statement — and `openjev` has a local
LLM teacher (here Qwen3.8-27B on vLLM) label a few thousand examples with *soft* labels (per-option probabilities read off the
logprobs of a single constrained letter token), distills those into a ~140M encoder, calibrates it
on a held-out split, and serves the result at `POST /v1/systemone`. The output is a typed
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
4. **Calibrate** — `logits / T` fitted by LBFGS on the calib split's NLL, or `logits / T + b` with a
   per-class bias `b ∈ R^K` when that wins a 2-fold held-out NLL test on the same rows. The bias is
   what removes the teacher's shifted marginal, and on the hard tasks it is worth more than anything
   else in this pipeline (see [Findings](#findings)). With no gold it targets a declared `prior:`.
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
| headlines-8k (n=3) | choice | 6 | ru | 8000 | 0.851 ± 0.002 / 0.851 ± 0.002 | 0.783 / 0.775 | 0.818 ± 0.002 | 0.031 ± 0.004→0.036 ± 0.004 | 0.231 ± 0.002 | 5.3 | 2561 |
| headlines-e12 | choice | 6 | ru | 4522 | 0.836 / 0.836 | 0.783 / 0.775 | 0.813 | 0.034→0.023 | 0.250 | – | – |
| headlines-nosynth (n=3) | choice | 6 | ru | 4000 | 0.838 ± 0.004 / 0.838 ± 0.004 | 0.783 / 0.775 | 0.818 ± 0.006 | 0.024 ± 0.006→0.030 ± 0.005 | 0.247 ± 0.004 | – | – |
| kinopoisk (n=3) | choice | 3 | ru | 4000 | 0.657 ± 0.003 / 0.656 ± 0.004 | 0.652 / 0.609 | 0.693 ± 0.006 | 0.154 ± 0.006→0.040 ± 0.005 | 0.460 ± 0.001 | 7.0 | 186 |
| kinopoisk-e12 | choice | 3 | ru | 4000 | 0.660 / 0.657 | 0.652 / 0.609 | 0.681 | 0.173→0.033 | 0.463 | – | – |
| kinopoisk-gold | choice | 3 | ru | 4000 | 0.672 / 0.669 | 0.652 / 0.609 | 0.655 | 0.075→0.030 | 0.435 | 7.0 | 186 |
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
  calibration (temperature, or temperature plus the per-class bias — `calib.json` says which).
  Lower is better; it is a summary, not a guarantee about any single prediction, and where the two
  disagree the task's own metric is the headline, never ECE: a bias that fixes a shifted marginal
  can raise ECE on the calib split while raising accuracy by 7 points.
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

Nine runs, one teacher, two nights. Every "Δ" below is a paired bootstrap (`openjev compare`, 2000
resamples) over the eval rows the two arms share; a Δ whose 95 % CI includes 0 is written as *no
measurable difference*, never as a gain. Training-side deltas (augment, more labels, where gold is
spent) are pooled over **three seeds per arm**; the epochs bullet below is the one exception and is
labelled. The post-hoc calibration deltas below are quoted at one seed per run, but the fit was
refit on all three seeds: headlines +7.0..+7.5 pts, kinopoisk +4.5..+5.1, every CI excludes 0
(R1 table in `docs/experiments.md`). The full tables are in `docs/experiments.md`.

**500 gold labels are worth more as a per-class bias than as training signal.** The calibration step
fits `logits / T + b`, `b ∈ R^K` (`method: vector`), on the 500-row calib split, accepted only when
it wins a 2-fold held-out NLL test against a plain temperature. It is CPU, seconds, no teacher calls
and no retraining, and it is the largest single win in the repo: headlines 0.767 → **0.843**
(+7.5 pts, 95 % CI +6.0..+9.1), kinopoisk 0.609 → **0.660** (+5.1 pts, +2.9..+7.5), georeview MAE
0.783 → **0.609** (−0.17, −0.19..−0.16). On agnews it is +0.95 pts (CI −0.10..+1.95) and on
banking77 +0.20 (CI −1.35..+1.65) — both CIs include 0, so on those two tasks it does nothing.
toxic is excluded: its metric is AUROC, which a constant per-class shift transforms monotonically
and therefore cannot move at all, and its calib split is 2.2 % positive against 8.1 % on eval.

**With the bias, two students beat their own teacher and two match it.** headlines 0.843 vs the
teacher's 0.783 (+6.0 pts, 95 % CI +4.3..+7.7) and headlines-8k 0.8515 vs 0.783 (+6.9, +5.2..+8.6)
are measurable wins; kinopoisk 0.660 vs 0.652 (+0.8, −1.8..+3.5) and georeview MAE 0.609 vs 0.619
(−0.0095, −0.027..+0.005) are *no measurable difference* — both CIs include 0. Before it, all four
were behind. Nothing about the teacher changed — the student is distilled from
the same probabilities; the bias only removes the marginal the teacher shifted.

**Why it works: the teacher's dominant error on the hard tasks is a marginal shift, not per-example
noise.** On kinopoisk (gold is an even 33/33/33 split) the teacher answers *Good* for 56.5 % of rows
and *Neutral* for 12.7 %, and the student reproduces that faithfully. The fitted bias is
`[+0.80 Bad, +0.59 Neutral, −1.39 Good]` — it pushes down exactly the class the teacher
over-predicts and lifts the one it starves. The permutation diagnostic says how much of that is the
prompt and how much is the model: reversing the option order moves teacher accuracy 0.646 → 0.693
and its *Good* rate 0.564 → 0.418, so the Bad↔Good split is positional — but the *Neutral* collapse
survives both orders (12.9 % and 12.0 %) and averaging the two orders makes it **worse** (9.0 %),
because a class that is never the argmax in either order gains nothing from the mean. The shift is
semantic, so averaging over permutations was **not** implemented: it would double the teacher bill
to fix the part the cheap post-hoc bias already fixes for free. openjev labels each row once.

**The caveat that ships with it: you have to know your prior.** The bias above is fitted on 500 gold
rows. The no-gold variant — `prior:` declared in the YAML, gold ignored entirely (`--ignore-gold`) —
lands within half a point of it end to end: kinopoisk 0.6553, georeview MAE 0.5986. That reads like
"free debiasing with zero labels", and it is not. `uniform` is the right prior on these benchmarks
*because their eval sets are balanced by construction*; a real deployment does not get that, it gets
whatever prior it can actually declare, and a wrong declaration moves the marginal to the wrong
place. The honest claim is: **if you know your class prior you can have this without labelling
anything; if you do not, spend 500 rows from your own distribution and fit it.**

**Where to spend 500 gold labels, measured.** Same task, same seeds, same teacher labels, one factor:
500 gold rows in the CE loss (`--gold-n 500 --gold-weight 1.0`) gives kinopoisk 0.603 / 0.622 / 0.623
over three seeds; the same 500 rows used *only* to fit the calibration bias gives 0.660 / 0.654 /
0.657. Calibration wins by **Δ = +4.1 ± 1.7 pts (95 % CI +2.4..+5.7)**. Putting *all* 4000 gold
labels in the loss reaches 0.672 (one seed), i.e. 8× the labels buys 1.2 pts over the 500-row bias.
A plausible mechanism — offered as a hypothesis, not a claim — is that a few hundred hard labels
fight 4000 soft ones inside the loss, while the same rows spent post-hoc correct the one thing that
is actually broken. Two points on one task; it is not a general N-labels curve.

**More teacher labels do help, a little.** headlines at 4000 vs 8000 teacher-labeled train rows,
three seeds each, both synth-free, identical eval ids: **+1.3 pts (95 % CI +0.6..+2.1)**, 0.838 →
0.851. Doubling the teacher bill buys about a sixth of what the 500-row bias buys on the same task.

**The cascade is offline and often unnecessary.** `eval` sweeps "student answers when confidence ≥ τ,
else the row goes to the teacher" over the probabilities we already hold — zero new teacher calls —
and writes the parity point into `openjev.json` as `escalate_below`. **Four of the nine runs reach
teacher parity at 0 % escalation**: after the bias the student already matches or beats its teacher,
so the cheapest cascade is "never escalate" (on headlines escalating actively *costs* accuracy, all
the way down to the teacher's 0.783). Where the student is behind, escalating the least confident
rows pays: banking77 0.751 → 0.776 at 27 %, kinopoisk 0.660 → 0.673 at 9 %, agnews 0.889 → 0.897 at
10 %, georeview MAE 0.609 → 0.583 at 26 %. What this may **not** claim is any saving in teacher
*quality*: escalation routes to the same zero-shot teacher that produced the training labels, scored
on the rows that teacher labelled. `serve` returns `"escalate": true/false` and never calls the
teacher itself — that call is yours.

**Two knobs that turned out not to matter.** Both were believed to work before they were measured
with a control arm:

- **Epochs: 5 vs 12 is no measurable difference** (one seed per arm — the CIs are over eval rows
  only; kept here because it decides the default). agnews Δ +0.0 ± 0.6 pts, headlines +0.7 ± 0.9,
  kinopoisk +0.0 ± 1.5 — all three CIs include 0, and the sign on headlines favours the *shorter*
  run. All three 12-epoch runs early-stop (patience 2) at 9, 7 and 5 epochs, so what `--epochs`
  really changes is the LR-decay horizon, not the training length. **The default stays 5.**
- **The augment (PGKD-lite) round on headlines is no measurable difference.** Three seeds per arm,
  the only factor being whether the 522 synthetic rows are in the training set: **Δ = +0.0 ± 0.5 pts
  (95 % CI −0.5..+0.6)**. Seed 0 showed +0.8 pts and seed 1 showed −0.9 — opposite signs on the same
  comparison. An earlier version of this README claimed a +0.5 pt gain from the round; that claim
  was one seed, exactly the size of this noise, and it is withdrawn. This does not show PGKD fails —
  one task, one round, 522 rows against 4000 — only that nothing here measured it working.

**The noise floor that every claim above is measured against.** Three seeds, identical data and
labels, only the head init and batch order changing, move accuracy 0.6–1.2 pts peak to peak: agnews
0.888 ± 0.006, headlines 0.838 ± 0.005, kinopoisk 0.657 ± 0.003. One seed is not a measurement, which
is why `compare` exists and why `report` prints an sd column.

**The student ranks better than its teacher on toxic** — AUROC 0.856 vs 0.817, Brier against the
annotator fraction 0.035 vs 0.048. Ignore the accuracy column there: the eval split is 8.1 % positive,
so always answering "not toxic" scores 0.919, above both. That win is not free either — toxic is the
one task whose training rows were drawn 50/50 *by gold* (`balance: true`), so gold paid for the row
selection even though every label is the teacher's. A plausible mechanism is that averaging 4000 soft
labels over a balanced sample denoises the teacher's per-example noise; it is one task and it did not
generalise to any other here.

**banking77 is where the teacher cost shows up.** 77 classes go through the chunked shortlist at 6
teacher calls per example: 33,000 calls for 5,500 rows, 61 minutes of teacher time, 42 % of the
night's entire teacher budget for one task. The student lands at 0.751 against the teacher's 0.764
(agreement 0.766) and the calibration bias does not move it — the shortlist's approximation survives
distillation but nothing here improves on it. This is the one task where escalation is the tool that
helps (0.776 at 27 %).

**Model size is not the bottleneck at either end.** mmBERT-base (307M vs 140M) buys agnews
0.891 vs 0.889 and georeview MAE 0.594 vs 0.609, for half the batch-64 throughput (435 vs 869 ex/s on
agnews; 115 vs 203 on georeview), 9.8–9.9 GB of training memory instead of 5.5, and 7.4 minutes of
training instead of 1–3.5 on agnews (on georeview the two took the same ~10 minutes — the box was
busy, so training time here is not a clean number). On the easy task the small student had already reached the teacher; on the
hard one it was already reproducing the teacher more faithfully than the big one — the teacher is
what was wrong, and a bias on 500 rows fixed more of it than 2.2× the parameters did.

**A task with no gold at all gets the same student.** `tasks/agnews-nogold.yaml` is `tasks/agnews.yaml`
with the `data.gold` line deleted, so nothing in the pipeline ever sees a label. Read its table row
carefully: with no gold, "teacher acc" is the teacher scored against itself (1.000 by construction)
and "student acc" 0.944 is the agreement column under another name. The number that answers "what do
you get with no labels at all" is measured offline against the real ag_news test labels:
**0.8820** against the teacher's **0.8815** on the same 2000 rows, within noise of the
gold-calibrated run. Removing gold from the loop cost nothing measurable *here* — on a task where the
teacher is strong and already well calibrated. On the hard tasks the gold-free path is the `prior:`
one above, and it is only as good as the prior you declare. (`gold_acc_offline` in
`results/variants/agnews-nogold-goldacc.json` comes from `scripts/nogold_gold_acc.py`, a hand-run
measurement outside the pipeline; it lives in `results/variants/` because `openjev eval` rewrites
`results/agnews-nogold.json` and would drop a key it did not write.)

**Teacher cost.** 66,522 teacher calls and 144 minutes (2.4 h) of labeling wall-clock across the six
tasks, of which banking77 alone is 61 minutes; headlines-8k added 4,434 rows in 9 more. Student
training is 1.5–12 minutes per task, and every post-hoc result above — the bias, the prior variant,
the cascade curves — is CPU seconds on logits already on disk. The teacher is the budget; everything
else is rounding.

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
- One NVIDIA GPU for training. Measured peaks for mmBERT-small at batch 32: **5.5 GB at
  `max_len: 256`, 8.9 GB at `max_len: 512`**. mmBERT-base is **9.9 GB**, which does not fit beside a
  resident 84 GB vLLM on this box — a run was killed that way, so base is not trained next to the
  teacher here. CPU-only works for `serve`, `calibrate`, `eval` and `compare`.
- **Do not let a training job OOM the teacher.** On a unified-memory box the trainer and vLLM share
  the same pool, and the host OOM killer will happily take the 84 GB process. Every training run in
  this repo goes through `scripts/gpu_queue.sh`: one job at a time behind `flock`, each under
  `choom -n 1000` so the killer picks the trainer, and a memory gate that *waits* before starting a
  job instead of shrinking the batch size (a different batch size is a different experiment).
- Python ≥ 3.10, and the deps in `pyproject.toml` (torch, transformers, datasets, httpx, pyyaml,
  numpy, scikit-learn, fastapi, uvicorn). No other runtime dependencies.
- Developed on a DGX Spark (GB10, 20 cores, 128 GB unified memory), Ubuntu 24.04, torch 2.13/cu130.
- If `HTTP_PROXY`/`HTTPS_PROXY`/`ALL_PROXY` are set, unset them: they break localhost calls to the
  teacher and slow Hugging Face downloads. The teacher client itself uses `trust_env=False`.

## Limitations and follow-ups

- **The calibration bias needs a prior you actually know.** `method: vector` fits a per-class bias on
  ~500 gold rows *from the deployment distribution*; the no-gold path needs a `prior:` you declare.
  Every benchmark here is balanced by construction, so `uniform` happens to be correct — that is a
  property of the benchmarks, not a property of the method. A wrong prior moves the marginal to the
  wrong place, and fewer than ~500 calib rows makes the fit noise. Fit it on rows drawn at the same
  rate as the traffic you will serve: toxic is the cautionary example (calib 2.2 % positive against
  8.1 % on eval, because a balanced train draw went first and depleted the pool).
- **A per-class bias cannot fix a per-example error.** It removes a marginal shift and nothing else:
  on agnews and banking77, where the teacher's marginal is already right, it buys nothing, and on a
  two-class AUROC task (toxic) it cannot move the metric at all.
- **The cascade is measured against the teacher that made the labels.** `escalate_below` and the
  curves in `results/*.json` route the unsure rows back to the same zero-shot teacher that produced
  the training labels, scored on the rows it labelled. That is a cost/latency trade-off, not evidence
  of any gain in answer quality, and it says nothing about escalating to a *better* model.
- **Option position bias is measured, not fixed.** Reversing the option order moves the teacher's
  *Good* rate on kinopoisk from 56.4 % to 41.9 % (gold 33.4 %) and its accuracy from 0.646 to 0.693 —
  but the *Neutral* collapse survives both orders (12.9 % and 12.0 % against a gold 33.7 %), and
  averaging the two orders makes it worse (9.0 %). The bias is semantic, not positional, so openjev
  labels each row once and removes the marginal shift post-hoc with the calibration bias instead — on
  kinopoisk that is `[+0.80 Bad, +0.59 Neutral, −1.39 Good]`, fitted on 500 calib rows, for zero extra
  teacher calls. Picking the better order would itself need gold, and headlines goes the other way
  (0.784 original vs 0.768 reversed).
- **K > 19 is approximate.** `top_logprobs` caps at 20, so large label sets go through a chunked
  shortlist: softness is exact only within the shortlist, and if every chunk misses the true class
  the label is simply wrong. Cost is `ceil(K/19) + 1` calls per example.
- **No ONNX / quantized export.** `serve` runs the PyTorch model; the export bundle is
  `student/` + `openjev.json` (+ `conformal.json`), which is what `openjev push` uploads.
- **No recalibration under drift.** One `T` and one `b` per task, fitted once. Nothing here detects
  that the deployment prior has moved, which is exactly the failure mode the bias is sensitive to.
- **Calibration target depends on the data.** With gold on the calib split the fit targets gold;
  without it, the temperature targets the teacher's *soft* probabilities and the bias targets the
  declared `prior:` — which calibrates the student to the teacher's opinion plus your belief, not to
  the truth. `calib_target` in `results/*.json` says which path ran.
- **The teacher is the cost.** Labeling dominates wall-clock; the student trains in minutes and every
  post-hoc step is CPU seconds.
