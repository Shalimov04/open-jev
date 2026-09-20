# open-jev

*[Русская версия](README.ru.md)*

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/img/pipeline-dark.svg">
  <img alt="open-jev pipeline: text to teacher soft labels to distilled student to typed JSON" src="docs/img/pipeline-light.svg" width="900">
</picture>

Turn a prompt into a small, fast, calibrated classifier. You describe a decision in a YAML file — a
choice between options, a score on a rubric, or the truth of a statement — and `openjev` has a local
LLM teacher (here Qwen3.8-27B on vLLM) label a few thousand examples with *soft* labels (per-option probabilities read off the
logprobs of a single constrained letter token), distills those into a ~140M encoder, calibrates it
on a held-out split, and serves the result at `POST /v1/systemone`. The output is a typed
decision with probabilities you can threshold on, not a string you have to parse. It is an open
reimplementation of the idea behind TypeSafe's Jev — Jev itself is a closed hosted API, so its
published claims cannot be checked from outside and none of them are repeated here; the only numbers
in this repo are the ones produced by the code in this repo.

**Task** (one YAML) → **teacher** (one constrained letter per example, softmax over its logprobs)
→ **student** (`mmBERT-small` trained on `KL(teacher ‖ student)`) → **calibrate** (`logits / T + b`
on a 500-row split) → **serve** (typed JSON at `/v1/systemone`). Everything lands in
`runs/<task>/`. Stage-by-stage detail, the exact teacher request and `openjev augment`:
[`docs/pipeline.md`](docs/pipeline.md). The task YAML reference and a walkthrough for your own task:
[`docs/task-spec.md`](docs/task-spec.md).

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/img/response-card-dark.svg">
  <img alt="The typed JSON decision returned by POST /v1/systemone" src="docs/img/response-card-light.svg" width="900">
</picture>

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

![openjev run tasks/agnews.yaml](docs/img/terminal.svg)

`GET /v1/models` lists the loaded runs. Each served model is one trained student with a fixed type
and a fixed label set — this is not a general model that takes arbitrary options at request time.
Batching, the escalation flag, conformal prediction sets, `openjev push` and comparing against
another endpoint: [`docs/serving.md`](docs/serving.md).

Run everything from the repo root (`results/` and this README are resolved relative to it). Tests:
`pytest -q tests` — under a minute on CPU, no GPU and no teacher needed (also run on every push by
`.github/workflows/test.yml`); two of them want
`jhu-clsp/mmBERT-small` and `fancyzhx/ag_news` in the Hugging Face cache.

The tasks here are English and Russian, but nothing is language-specific: `question`, `options` and
`statement` are free text, so write them in whatever language you want the teacher prompted in, as
long as the teacher and the student checkpoint (mmBERT is multilingual) both cover it. `lang` itself
is only a report column and the language `openjev augment` generates in.

## Results

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/img/student-vs-teacher-dark.svg">
  <img alt="Student vs teacher accuracy on seven tasks" src="docs/img/student-vs-teacher-light.svg" width="900">
</picture>

<!-- results -->
| task | type | K | lang | n_train | student acc / F1 | teacher acc / F1 | agree | ECE raw→cal | Brier | GPU p50 ms | ex/s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| agnews (n=3) | choice | 4 | en | 4000 | 0.888 ± 0.006 / 0.888 ± 0.006 | 0.880 / 0.881 | 0.916 ± 0.004 | 0.075 ± 0.005→0.030 ± 0.004 | 0.177 ± 0.006 | 5.3 | 869 |
| agnews-e12 | choice | 4 | en | 4000 | 0.889 / 0.889 | 0.880 / 0.881 | 0.914 | 0.079→0.023 | 0.174 | – | – |
| agnews-mmBERT-base | choice | 4 | en | 4000 | 0.891 / 0.892 | 0.880 / 0.881 | 0.916 | 0.081→0.023 | 0.175 | 6.3 | 435 |
| agnews-nogold | choice | 4 | en | 4000 | 0.944 / 0.944 | 1.000 / 1.000 | 0.944 | 0.131→0.129 | 0.115 | 5.7 | 863 |
| arb-quality | score | 4 | en | 750 | 0.343 / 0.240 | 0.360 / 0.234 | 0.540 | 0.197→0.129 | 0.709 | – | – |
| arb-success (n=3) | noul | 2 | en | 750 | 0.780 ± 0.015 / 0.705 ± 0.018 | 0.743 / 0.426 | 0.761 ± 0.007 | 0.201 ± 0.020→0.068 ± 0.007 | 0.286 ± 0.007 | 6.0 | 182 |
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
| m2w-element | choice | 16 | en | 5000 | 0.059 / 0.007 | 0.649 / 0.652 | 0.133 | 0.102→0.004 | 0.937 | – | – |
| m2w-target (n=3) | noul | 2 | en | 5000 | 0.857 ± 0.006 / 0.795 ± 0.009 | 0.783 / 0.562 | 0.837 ± 0.004 | 0.120 ± 0.003→0.019 ± 0.007 | 0.207 ± 0.004 | 5.8 | 714 |
| swde-field (n=3) | choice | 33 | en | 4000 | 0.868 ± 0.007 / 0.924 ± 0.010 | 0.803 / 0.876 | 0.805 ± 0.003 | 0.053 ± 0.005→0.028 ± 0.007 | 0.206 ± 0.004 | 6.2 | 1054 |
| toxic | noul | 2 | en | 4000 | 0.917 / 0.626 | 0.893 / 0.624 | 0.925 | 0.116→0.037 | 0.129 | 5.7 | 437 |

- **agnews**: jhu-clsp/mmBERT-small distilled from the teacher with zero gold labels; argmax accuracy vs gold on 2000 eval examples; calibrated to gold (T=0.62).
- **agnews-e12**: jhu-clsp/mmBERT-small distilled from the teacher with zero gold labels; argmax accuracy vs gold on 2000 eval examples; calibrated to gold (T=0.62).
- **agnews-mmBERT-base**: jhu-clsp/mmBERT-base distilled from the teacher with zero gold labels; argmax accuracy vs gold on 2000 eval examples; calibrated to gold (T=0.63).
- **agnews-nogold**: jhu-clsp/mmBERT-small distilled from the teacher with zero gold labels; argmax accuracy vs teacher on 2000 eval examples; calibrated to teacher-soft (T=0.99).
- **arb-quality**: jhu-clsp/mmBERT-small distilled from the teacher with zero gold labels; expected level; MAE vs gold on 300 eval examples; MAE 0.776 (teacher 0.809); calibrated to gold (T=1.44).
- **arb-success**: jhu-clsp/mmBERT-small distilled from the teacher with zero gold labels; p(true); AUROC vs gold on 300 eval examples; AUROC 0.843 (teacher 0.837); gate: no tau reaches 90% precision(true); calibrated to gold (T=0.73).
- **banking77**: jhu-clsp/mmBERT-small distilled from the teacher with zero gold labels; argmax accuracy vs gold on 2000 eval examples; calibrated to gold (T=0.82).
- **georeview**: jhu-clsp/mmBERT-small distilled from the teacher with zero gold labels; expected level; MAE vs gold on 2000 eval examples; MAE 0.609 (teacher 0.619); calibrated to gold (T=1.09).
- **georeview-mmBERT-base**: jhu-clsp/mmBERT-base distilled from the teacher with zero gold labels; expected level; MAE vs gold on 2000 eval examples; MAE 0.594 (teacher 0.619); calibrated to gold (T=1.07).
- **headlines**: jhu-clsp/mmBERT-small distilled from the teacher with zero gold labels; argmax accuracy vs gold on 2000 eval examples; calibrated to gold (T=0.85).
- **headlines-8k**: jhu-clsp/mmBERT-small distilled from the teacher with zero gold labels; argmax accuracy vs gold on 2000 eval examples; calibrated to gold (T=0.75).
- **headlines-e12**: jhu-clsp/mmBERT-small distilled from the teacher with zero gold labels; argmax accuracy vs gold on 2000 eval examples; calibrated to gold (T=0.78).
- **headlines-nosynth**: jhu-clsp/mmBERT-small distilled from the teacher with zero gold labels; argmax accuracy vs gold on 2000 eval examples; calibrated to gold (T=0.80).
- **kinopoisk**: jhu-clsp/mmBERT-small distilled from the teacher with zero gold labels; argmax accuracy vs gold on 1500 eval examples; calibrated to gold (T=1.21).
- **kinopoisk-e12**: jhu-clsp/mmBERT-small distilled from the teacher with zero gold labels; argmax accuracy vs gold on 1500 eval examples; calibrated to gold (T=1.27).
- **kinopoisk-gold**: jhu-clsp/mmBERT-small distilled from the teacher with gold CE weight 1.0; argmax accuracy vs gold on 1500 eval examples; calibrated to gold (T=1.03).
- **kinopoisk-gold-n500**: jhu-clsp/mmBERT-small distilled from the teacher with gold CE weight 1.0; argmax accuracy vs gold on 1500 eval examples; calibrated to gold (T=0.61).
- **m2w-element**: jhu-clsp/mmBERT-small distilled from the teacher with zero gold labels; argmax accuracy vs gold on 900 eval examples; calibrated to gold (T=11318.59).
- **m2w-target**: jhu-clsp/mmBERT-small distilled from the teacher with no gold in the loss (but train rows picked 50/50 by gold); p(true); AUROC vs gold on 2000 eval examples; AUROC 0.904 (teacher 0.919); gate: 69% coverage at 90%+ precision(true); calibrated to gold (T=0.53).
- **swde-field**: jhu-clsp/mmBERT-small distilled from the teacher with zero gold labels; argmax accuracy vs gold on 2000 eval examples; calibrated to gold (T=0.59).
- **toxic**: jhu-clsp/mmBERT-small distilled from the teacher with no gold in the loss (but train rows picked 50/50 by gold); p(true); AUROC vs gold on 3000 eval examples; AUROC 0.856 (teacher 0.817); gate: 80% coverage at 90%+ precision(true); calibrated to gold (T=0.25).
<!-- results -->

Teacher for every row: **Qwen3.8-27B**, 3-bit GSQ quantisation (ISTA-DASLab) with MTP, served by vLLM
on the same box under the id `Qwen/Qwen3.8-27B-FP8`, called with `max_tokens: 1` and
`enable_thinking: false`. Every "teacher" number in the table is that model zero-shot; the students
cannot do better than it except by denoising it. Runs from before 2026-09-20 do not record the model
id in `runs/<task>/label.json` — newer ones do.

The table is regenerated by `openjev report` between its HTML comment markers — do not hand-edit
it. What each column means, and what it does not: [`docs/findings.md#what-the-numbers-mean`](docs/findings.md#what-the-numbers-mean).

## What we learned

Nine runs, one teacher, two nights. Every delta below is a paired bootstrap over shared eval rows
(`openjev compare`, 2000 resamples); a CI that includes 0 is written as *no measurable difference*,
never as a gain. The prose, with the numbers and the caveats behind each line, is in
[`docs/findings.md`](docs/findings.md); the tables and commands are in
[`docs/experiments.md`](docs/experiments.md).

- **500 gold labels buy more as a per-class calibration bias than as training signal.** `logits / T + b`
  fitted on the 500-row calib split: headlines 0.767 → **0.843**, kinopoisk 0.609 → **0.660**,
  georeview MAE 0.783 → **0.609**. CPU seconds, no retraining, no teacher calls. On agnews and
  banking77 it does nothing — the teacher's marginal there is already right.
- **With the bias, two students beat their own teacher and two match it.** The teacher's dominant
  error on the hard tasks is a shifted marginal, and a K-dim bias is exactly the shape that removes it.
- **The caveat: you have to know your prior.** The no-gold path (`prior:` declared) lands within half
  a point — but only because these benchmarks are balanced by construction.
- **Two knobs that turned out not to matter**, once measured with a control arm: 12 epochs vs 5, and
  the `openjev augment` (PGKD-lite) round. Both are *no measurable difference*.
- **The teacher is the budget.** 66,522 calls and 2.4 h of labeling across six tasks; the students
  train in 1.5–12 minutes and every post-hoc result is CPU seconds on logits already on disk.
- **The cascade is offline and often unnecessary.** Four of nine runs reach teacher parity at 0 %
  escalation. `serve` returns `"escalate": true/false` and never calls the teacher itself.

Where the method stops working — a wrong prior, per-example errors, drift, `K > 19` —
[`docs/findings.md#limitations-and-follow-ups`](docs/findings.md#limitations-and-follow-ups).

## Real tasks

Five more tasks, same pipeline, on agent-shaped inputs: judging a web-agent run from its action log,
picking the element to click, labelling a DOM node with a schema field. Four work, one fails
informatively — the numbers, the gate tables and the limits are in
[`docs/use-cases.md`](docs/use-cases.md).

- **A degenerate teacher can still be a usable scorer.** On `arb-success` the teacher answers *not
  successful* for 99.9 % of rows (gold: 26.6 % succeed) — macro-F1 0.426 — yet its probabilities
  rank runs at AUROC 0.837. The student lands at AUROC 0.843, macro-F1 0.695, ECE 0.214 → 0.064.
  But precision on *true* is 0.56 at the 80 %-coverage gate: a filter for a human, not an auto-accept.
- **On held-out websites the students beat their own teachers.** `swde-field` (33 fields, 3 teacher
  calls per row) 0.864 acc / 0.917 macro-F1 vs 0.803 / 0.876; `m2w-target` (act-or-ask on one
  element) 0.863 acc vs 0.783, AUROC 0.904, and the gate answers 87 % of elements at 0.870
  precision. 6 ms per decision, 714–1054 ex/s at batch 64.
- **`choice` is the wrong primitive for per-row candidates, and that is the finding.**
  `m2w-element` — "which of these 16 candidate elements" — collapses to 0.059 accuracy against a
  0.0625 chance floor: slot `G` means something different on every row, so there is nothing
  row-independent for a fixed head to learn. Ranking per candidate (`m2w-target`) is the same task
  and works.
- Weak result, reported as such: `arb-quality` (1–4 optimality) MAE 0.776 vs the teacher's 0.809 —
  the teacher is barely above chance, and the teacher is the ceiling.

Whole track: 35,900 teacher calls, 118 minutes of labeling, three seeds where it mattered
(`openjev compare` says *no measurable difference* between seeds on all three).

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

## Docs

- [`docs/pipeline.md`](docs/pipeline.md) — the five stages in detail, the teacher request, `openjev augment`.
- [`docs/task-spec.md`](docs/task-spec.md) — task YAML reference, command reference, worked examples, add your own task.
- [`docs/serving.md`](docs/serving.md) — batching, escalation, prediction sets, `openjev push`, comparing endpoints.
- [`docs/findings.md`](docs/findings.md) — what the numbers mean, the findings in full, limitations.
- [`docs/use-cases.md`](docs/use-cases.md) — the five agent-shaped tasks: gates, costs, the one that failed.
- [`docs/experiments.md`](docs/experiments.md) — every experiment with its commands, tables and CIs.
- [`docs/img/README.md`](docs/img/README.md) — the figures and how they are generated.

## Licence

MIT — see [LICENSE](LICENSE).
