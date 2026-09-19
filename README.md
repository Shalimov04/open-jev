# open-jev

Turn a prompt into a small, fast, calibrated classifier. You describe a decision in a YAML file — a
choice between options, a score on a rubric, or the truth of a statement — and `openjev` has a local
Qwen teacher label a few thousand examples with *soft* labels (per-option probabilities read off the
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

## Results

<!-- results -->
| task | type | K | lang | n_train | student acc / F1 | teacher acc / F1 | agree | ECE raw→cal | Brier | GPU p50 ms | ex/s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| agnews | choice | 4 | en | 4000 | 0.879 / 0.880 | 0.880 / 0.881 | 0.938 | 0.071→0.042 | 0.198 | 20.7 | 382 |

- **agnews**: jhu-clsp/mmBERT-small distilled from the teacher with zero gold labels; argmax accuracy vs gold on 2000 eval examples; calibrated to gold (T=0.74).
<!-- results -->

## What the numbers mean

- **The student sees zero gold labels.** `gold_weight` is 0 by default, so training uses only the
  teacher's distributions. Gold is used for the eval and (when present) for fitting the temperature.
  One exception: **toxic** sets `balance: true` on its training split, which picks 50/50 rows *by gold*
  from an 8%-positive dataset. Its labels are still the teacher's, but the row selection spent gold, so
  it is not a zero-gold row; every other task is.
- **The baseline is the teacher, not the state of the art.** Teacher zero-shot accuracy on the same
  eval split is in the table beside the student's. The claim being tested is "a 140M encoder can keep
  the teacher's accuracy at a fraction of the cost", not "this beats a supervised model".
- **ECE** is expected calibration error: max-probability bucketed into 15 equal-width bins, mean over
  bins of |mean confidence − accuracy|, weighted by bin mass. `raw→cal` is before and after
  temperature scaling. Lower is better; it is a summary, not a guarantee about any single prediction.
- **Brier** is the multiclass sum-of-squares against the one-hot gold, on calibrated probabilities.
- **agree** is argmax agreement between student and teacher on the eval split. It is the metric that
  matters when a task has no gold at all.
- **Latency** is batch-1 on the GB10 GPU, p50 of 200 requests after 20 warmups, measured *while vLLM
  was resident on the same GPU* — a dedicated GPU would be faster. `ex/s` is batch-64 throughput on
  the same hardware. `results/*.json` also carries a CPU batch-1 p50.
- **Distillation cannot beat its teacher's errors.** Where the teacher is systematically wrong the
  student inherits it, and the agreement column is what tells you how much.

## Requirements

- A vLLM (or other OpenAI-compatible) endpoint serving the teacher, reachable at
  `OPENJEV_TEACHER_URL`, supporting `logprobs` + `top_logprobs` and the vLLM `structured_outputs`
  field. Developed against a local Qwen3 served by vLLM on the same box.
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
  drawn at the same rate as the eval split.
- **The teacher is the cost.** Labeling dominates wall-clock; the student trains in minutes.
