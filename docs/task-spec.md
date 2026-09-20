# Task YAML reference

A task file is the only thing you write. It is parsed by `openjev/spec.py` (`load_task`) into a
`Task` object; everything downstream sees only the derived `labels` list (plus `values` for
`score`). Unknown keys are a hard error at load time, in every section — a typo fails immediately
instead of being silently ignored.

Run a task file with:

```bash
openjev check tasks/<name>.yaml   # dry run: prompt, splits, cost. No teacher calls.
openjev run   tasks/<name>.yaml   # label -> train -> calibrate -> eval
```

## The task files in this repo

Copy the one closest to your problem (`tasks/example.yaml` is the annotated blank template).

| file | type | K | lang | what it demonstrates |
|---|---|---|---|---|
| `agnews.yaml` | choice | 4 | en | the plain case: HF dataset, 4 options, gold used only to score |
| `agnews-nogold.yaml` | choice | 4 | en | the same task with `gold:` removed — distillation with zero labels, calibrated against the teacher |
| `banking77.yaml` | choice | 77 | en | K > 19: the chunked shortlist path (`teacher.max_options_per_call`), 77 options written out |
| `headlines.yaml` | choice | 6 | ru | non-English options — the prompt is Russian, the student is multilingual |
| `kinopoisk.yaml` | choice | 3 | ru | long texts: `max_chars` 1500 with `student.max_len: 512` so both see the same review |
| `georeview.yaml` | score | 5 | ru | `rubric` + `descriptions`; the served answer is an expectation over the levels, scored by MAE |
| `toxic.yaml` | noul | 2 | en | truth of a `statement`, a fractional `gold_prob` column, and `balance: true` on a 8 %-positive pool |

## Top level

| field | type | default | what it does |
|---|---|---|---|
| `name` | str | **required** | Run directory is `runs/<name>/`; also the model id used by `openjev serve` and the row name in the results table. |
| `type` | `choice` \| `score` \| `noul` | **required** | Picks the derived labels and the response view. Anything else is an error. |
| `question` | str | **required** for `choice`/`score` | First line of the teacher's system prompt. Optional for `noul`, where it only overrides the default head line `Is the following statement about the text true?` (the `statement` block follows it either way). |
| `lang` | str | `en` | The labeling prompt never sees it: it is the report column, and the language `openjev augment` asks the teacher to generate in. Write the prompt's language into `question` / `options` / `statement` yourself — a task in any other language works exactly the same way, as long as the teacher and the student checkpoint cover it. |
| `options` | list[str] | `null` | **`choice` only, required.** The label set, in a fixed order that is the class order end to end. |
| `rubric` | dict | `null` | **`score` only, required.** `{levels: [...], descriptions: [...]}` — see below. |
| `statement` | str | `null` | **`noul` only, required.** The proposition the teacher judges true/false. |
| `prior` | `uniform` \| {label: weight} | `null` | The class distribution you expect at deployment. It is used *only* by calibration, and only when the calib split has no gold (or `--ignore-gold` forces that): after fitting the temperature to the teacher's soft probabilities, a per-class bias is fitted so the mean calibrated probability over the calib rows matches this prior. That is a debiasing step with **zero gold labels**, which is the point — the teacher's dominant error on a hard task is a shifted marginal, and a K-dim bias is exactly the shape that fixes it. Weights are normalised, so `{Bad: 1, Neutral: 1, Good: 1}` and `uniform` are the same thing. Declare it only when you actually know it: a wrong prior moves the argmax the wrong way, and on a benchmark that is balanced by construction "knowing" it is a leak a real deployment does not get. `calib.json` records `target: prior-declared` when this path ran. |
| `data` | mapping | **required** | See [data](#data). |
| `teacher` | mapping | defaults | See [teacher](#teacher). |
| `student` | mapping | defaults | See [student](#student). |

Derived, do not set them yourself:

| derived | value |
|---|---|
| `labels` | `choice` → `options`; `score` → `str(level)` per level; `noul` → `["true", "false"]` (index 0 is `true`). |
| `values` | `score` only: `float(level)` per label, used for the expectation in the `score` view. `null` otherwise. |
| `k` | `len(labels)`. |
| `path` | The YAML path the task was loaded from. |

### `rubric` (score)

```yaml
rubric:
  levels: [1, 2, 3, 4, 5]                    # required; numeric, ordered low -> high
  descriptions: [very negative, negative, mixed or neutral, positive, very positive]  # optional
```

`levels` become the class labels *and* the numeric values the served `score` is an expectation
over. `descriptions` only affect the teacher prompt: each option line becomes `"<level>: <description>"`
(just `"<level>"` if `descriptions` is absent or shorter than `levels`).

## data

| field | type | default | what it does |
|---|---|---|---|
| `source` | mapping | **required** | Exactly one of `{hf: org/name}`, `{csv: path}`, `{jsonl: path}`. For `hf`, optional `config:` and `revision:` are passed to `datasets.load_dataset`. Any other key is an error, as everywhere else. |
| `text` | str | `"{text}"` | `str.format` template over the row's fields. Multi-field is fine: `"{title}\n{body}"`. A missing field raises. |
| `max_chars` | int | `2000` | Truncation applied *before* both teacher and student, so both see exactly the same input. |
| `gold` | str \| null | `null` | Column holding the ground-truth label. Optional: with no gold, everything still works — the temperature is fitted to the teacher's soft probabilities and the eval is reported against the teacher. |
| `gold_map` | list \| dict \| null | `null` | Applied to the raw gold value before interpretation: a list is indexed by an integer gold, a dict is keyed by the raw value. Use it when the dataset's label ids do not match your `options` order. |
| `gold_prob` | str \| null | `null` | `noul` only: column with a *fractional* gold probability (e.g. the share of annotators who said yes). Enables Brier score against that fraction in the eval. |
| `gold_threshold` | float | `0.5` | `noul` only: a gold value ≥ threshold becomes `true` (index 0), otherwise `false`. Applies to floats, ints and booleans alike. |
| `seed` | int | `0` | Sampling seed for all three splits. One seed, because the roles are drawn in order from the same pool — changing it reshuffles all of them together. |
| `train` | split | `{split: train, n: 1000, balance: false}` | Distillation set. |
| `calib` | split | `{split: train, n: 500, balance: false}` | Early stopping + temperature fitting. |
| `eval` | split | `{split: test, n: 2000, balance: false}` | The reported numbers. |

Gold interpretation, in order: `gold_map` if set → a number or boolean on a `noul` task becomes an
index via `gold_threshold` → a string is looked up in `labels` (exact match) → anything else is `int()`-ed
and used as an index.

### split blocks (`train` / `calib` / `eval`)

| field | type | default | what it does |
|---|---|---|---|
| `split` | str | `train` (`test` for `eval`) | The `datasets` split expression, slices included: `train[:60000]`. A plain CSV/JSONL file is always a single split named `train`. |
| `n` | int | `1000` (`500` calib, `2000` eval) | Number of examples to draw. |
| `balance` | bool | `false` | Draw `n // k` per gold class, then top up from the remaining rows. Requires `gold` (so a balanced split is not a zero-gold split), and is an error without it. Use it on train for a skewed task; leave `calib` and `eval` at the natural distribution, or the fitted temperature is calibrated to the wrong prior. A balanced role is drawn *after* the natural-rate ones (see below), so on a small pool the minority class can already be spent — the top-up then quietly makes the split less balanced than asked. |

Roles are drawn in the order train → calib → eval, each from the rows not yet taken from that
source split, except that any role with `balance: true` is drawn **last**: a balanced draw picks rows
by gold and depletes the minority class, which would otherwise shift the prior of the roles after it
(this is exactly what happened to `toxic`, see below). Two roles pointing at the same split are
guaranteed disjoint either way, and the draw is deterministic — re-running resumes rather than
re-labels. A role that cannot get its `n` rows out of what is left is an error, not a short split.

## teacher

| field | type | default | what it does |
|---|---|---|---|
| `concurrency` | int | `32` | In-flight requests to the teacher. |
| `max_options_per_call` | int | `19` | Options per teacher call. vLLM caps `top_logprobs` at 20 and `Z` is reserved for "none of the above", so 19 is the ceiling. Above `k`, the shortlist path kicks in (see below). |

The teacher endpoint is not in the YAML — it comes from `OPENJEV_TEACHER_URL`
(default `http://localhost:8000/v1`), and the model is whatever `/v1/models` reports first.

Prompt shape for `k <= max_options_per_call`, one call per example:

```
<question>            # or "Is the following statement about the text true?\nStatement: <statement>"
A: <label 0>          # letters A..S; for score, "<level>: <description>"
B: <label 1>
...
Answer with a single letter.
```

with the text as the user message. The soft label is the softmax over the returned option-letter
logprobs; letters that do not appear in `top_logprobs` get probability 0.

For `k > max_options_per_call` the options are split into `ceil(k / max_options_per_call)`
balanced chunks, each asked with an extra `Z: none of the above`; the per-chunk scores
`s_i = p(i | chunk)` (a chunk answering `none` puts little mass on its options) pick a shortlist of
`max_options_per_call` options, and one final call over the shortlist — lettered in the original
option order, not in score order — produces the distribution. Options outside the shortlist get
probability 0 (clamped to 1e-6 before the KL). Cost: `ceil(k/19) + 1` calls per example.

## student

| field | type | default | what it does |
|---|---|---|---|
| `model` | str | `jhu-clsp/mmBERT-small` | Any `AutoModelForSequenceClassification` checkpoint. `jhu-clsp/mmBERT-base` for a bigger run. Overridable per run with `--student`. |
| `max_len` | int | `256` | Tokenizer truncation length for training, eval and serving. 512 for long-text tasks — see below. |
| `epochs` | int | `5` | Overridable with `--epochs`. Upper bound, with early stopping (patience 2) on calib KL. 5 vs 12 epochs was measured on three tasks: **no measurable difference** — all three 12-epoch runs early-stopped (at 9, 7 and 5 epochs) and what `--epochs` mostly changes is the LR-decay horizon, not the training length ([findings.md](findings.md)). banking77 (77 classes) asks for 12 in its YAML. |
| `lr` | float | `5.0e-5` | AdamW learning rate, 6% linear warmup then linear decay. |
| `batch_size` | int | `32` | Training batch size. Halve it if you OOM next to a running vLLM. Overridable with `--batch-size`. |
| `gold_weight` | float | `0.0` | Weight of a CE term on gold added to the distillation KL, applied only to rows that have gold. `0.0` = pure distillation, which is the headline setting. Overridable with `--gold-weight`. |

The CLI overrides apply to every stage subcommand (`run`, `label`, `train`, `calibrate`, `eval`,
`augment`). `--student` and `--gold-weight` name a *variant* and add a `-<model>` / `-gold` suffix to
the run directory; `--batch-size` and `--epochs` are training-budget knobs only and leave the run
directory name alone, so they overwrite the run they are pointed at.

Write floats in YAML as `5.0e-5`, not `5e-5` — YAML 1.1 parses the latter as a string.

`max_chars` and `max_len` are two truncations in a row: `data.max_chars` cuts the raw text once, for
teacher and student alike, and `student.max_len` then cuts the student's *tokens*. If `max_len` binds
first the student never sees text the teacher was labelling (long Russian reviews: ~1500 chars is well
over 256 tokens), which caps agreement — raise `max_len` or lower `max_chars` so the two roughly meet.

## Command reference

Every stage subcommand (`run`, `label`, `train`, `calibrate`, `eval`, `augment`) takes a task YAML
and the overrides above. These are the rest:

| flag | on | what it does |
|---|---|---|
| `--seed N` | stage cmds | Training seed (head init + batch order). `N > 0` adds a `-sN` run-dir suffix; seed 0 *is* the baseline dir. Three seeds is the floor for any claim — CUDA is not bit-exact, so one seed is not a measurement. |
| `--tag X` | stage cmds | Free-form run-dir suffix (`runs/<task>-X`). Use it for a named variant; it replaces ad-hoc suffix rules. |
| `--no-synth` | stage cmds | Drop `source == "synth"` rows from training — the control arm for `openjev augment`. |
| `--gold-n N` | stage cmds | Put the CE term on the first N train rows in id order only. Needs `--gold-weight`; answers "what do 500 gold labels buy in the loss?". |
| `--ignore-gold` | `run`, `calibrate` | Calibrate as if the calib split had no gold. With `prior:` declared, this is what makes a `prior-declared` number a genuine no-gold number. |
| `--no-latency` | `run`, `eval` | Skip the latency/throughput measurement. Latency is only valid on an idle teacher (it shares the GPU), so every queued eval uses this and `openjev bench` measures later. |
| `--teacher vllm\|jev` | `run`, `label` | Teacher backend. `vllm` (default) is the local OpenAI-compatible server. `jev` calls TypeSafe Jev's decision API (`POST https://openrouter.ai/api/alpha/decisions`, key `OPENROUTER_API_KEY` from the env or `.env.local`), labels into `runs/<task>-jev/` and reuses `runs/<task>/teacher.jsonl` for the row list, so ids and splits match row for row. 8 requests in flight, 3 retries on 429/5xx, and a hard $1.00 spend stop per command (`OPENJEV_JEV_BUDGET`); `label.json` records the served model id and the summed `usage.cost`. |
| `--split ROLE` | `label` | Label only `train`/`calib`/`eval`. |
| `--force STAGE` | `run` | Re-run one stage and everything after it. Repeatable. |

| command | what it does |
|---|---|
| `openjev check TASK.yaml [--probe N]` | Dry-run a task before spending teacher time: splits, gold distribution, the prompt verbatim, three formatted examples, length stats, cost estimate. `--probe N` labels N calib rows into the real `teacher.jsonl` and reports the teacher's accuracy, marginal and under-predicted classes. See [Add your own task](#add-your-own-task). |
| `openjev compare A B` | Paired comparison of two run dirs over every seed they share: per-seed delta on the task's metric and one pooled 95 % bootstrap CI over eval rows, plus a verdict line. Nothing goes into the README without one. |
| `openjev report` | Rebuild the README results table from `results/*.json` (and print it). `-s<k>` files are grouped into one `mean ± sd (n)` row. |
| `openjev bench RUN_DIR` | Latency + throughput only, into the existing `results/<run>.json`. Waits for `vllm:num_requests_running == 0` first, so the number is not noise from a busy teacher. |
| `openjev serve RUN_DIR... [--port P] [--device cpu\|cuda]` | Serve one or more run dirs at `POST /v1/systemone`. A run dir may be `hf:user/name`, which is downloaded once into `runs/hf--user--name`. See [serving.md](serving.md). |
| `openjev push RUN_DIR --repo user/name [--dry-run] [--public]` | Upload `student/`, `openjev.json`, `conformal.json` and a generated model card to the Hugging Face Hub. Private by default. `--dry-run` prints the file list and the card and uploads nothing. |

## Worked examples

### choice — `tasks/agnews.yaml`

```yaml
name: agnews
type: choice
question: "What is the topic of this news article?"
options: [World, Sports, Business, Sci/Tech]
lang: en
data:
  source: {hf: fancyzhx/ag_news}
  text: "{text}"
  max_chars: 2000
  gold: label
  train: {split: train, n: 4000}
  calib: {split: train, n: 500}
  eval: {split: test, n: 2000}
```

`labels = ["World", "Sports", "Business", "Sci/Tech"]`, `k = 4`. Real response from
`openjev serve runs/agnews`:

```json
{"model": "agnews", "choice": "Business",
 "probabilities": {"World": 0.0058, "Sports": 0.0051, "Business": 0.9872, "Sci/Tech": 0.0018},
 "confidence": 0.9872}
```

### score — `tasks/georeview.yaml`

```yaml
name: georeview
type: score
question: "How positive is this review of an organization (in Russian)? Pick the star rating the author most likely gave."
rubric:
  levels: [1, 2, 3, 4, 5]
  descriptions: [very negative, negative, mixed or neutral, positive, very positive]
lang: ru
data:
  source: {hf: mteb/GeoreviewClassification}
  text: "{text}"
  max_chars: 2000
  gold: label          # 0-4 = index into levels 1-5 (stars - 1)
  train: {split: train, n: 4000}
  calib: {split: train, n: 500}
  eval: {split: test, n: 2000}
```

`labels = ["1", …, "5"]`, `values = [1.0, …, 5.0]`, `k = 5`. The served `score` is
`Σ p_k · value_k`, so it is continuous and lands between levels when the model is unsure.
Response shape (illustrative numbers):

```json
{"model": "georeview", "score": 4.31,
 "probabilities": {"1": 0.01, "2": 0.02, "3": 0.10, "4": 0.39, "5": 0.48},
 "confidence": 0.48}
```

Here gold is 0-based while levels are 1-based; that offset is handled because an integer gold is
used as an index into `labels`. If your column held the stars themselves you would add
`gold_map: [null, 0, 1, 2, 3, 4]` or `gold_map: {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}`.

### noul — `tasks/toxic.yaml`

```yaml
name: toxic
type: noul
statement: "This comment is toxic (rude, disrespectful, or likely to make someone leave a discussion)."
lang: en
data:
  source: {hf: google/civil_comments}
  text: "{text}"
  max_chars: 1500
  gold: toxicity       # fraction of raters; >= 0.5 -> true
  gold_prob: toxicity
  train: {split: "train[:60000]", n: 4000, balance: true}
  calib: {split: "train[:60000]", n: 500}
  eval: {split: test, n: 3000}
```

`labels = ["true", "false"]`, `k = 2`. The gold column is a fraction, so `gold_threshold` (0.5)
turns it into a class and `gold_prob` keeps the fraction for a Brier score against it. Train is
balanced because positives are ~8% of the data. Note that `balance: true` selects rows by gold, so a
balanced split does spend gold labels — and that in the run recorded in the README it was drawn
*first*, out of the same `train[:60000]` pool as calib, which left calib at **11/500 = 2.2%** positive
against **8.1%** on eval: T = 0.25 was fitted on 11 positives. It still improved eval ECE
(0.116 → 0.037), but fitting a temperature on a depleted pool is a weakness, which is why balanced
roles are now drawn after the others.
Response shape (illustrative numbers):

```json
{"model": "toxic", "probability": 0.04, "confidence": 0.96}
```

`probability` is p(statement is true); `confidence` is `max(p, 1-p)`, i.e. how far from the fence.

## Add your own task

1. **Write the YAML.** Copy `tasks/example.yaml` (or the closest row of the table above), set `name`,
   `type`, the question (plus `options` / `rubric` / `statement`), and point `data.source` at your HF
   dataset, CSV or JSONL. Set `data.text` to a template over your columns. Gold is optional — leave
   `gold` out and the run calibrates and reports against the teacher instead.
2. **`openjev check tasks/mytask.yaml`** — free, no teacher calls. It loads the spec, pings the
   teacher, draws the splits and prints the gold distribution per split, the system prompt verbatim,
   three examples exactly as the teacher will see them (after `data.text` and `max_chars`), the text
   length in characters *and* in student tokens, and an estimated teacher cost (±2×). It warns when
   `max_chars` truncates more than a quarter of the rows, and when more than a quarter of them are
   longer than `student.max_len` — i.e. when the student reads less than the teacher did. Spec and
   data errors come back as one line and exit 1, which is the fastest way to debug a `data.text`
   template. That `max_len` warning is the trap kinopoisk fell into: the teacher reads
   1500 characters, and a student left at the default `max_len: 256` would read about half of that.
3. **`openjev check tasks/mytask.yaml --probe 100`** — the only step that costs teacher time (a
   minute at most). It labels the first 100 calib rows *into the real* `runs/mytask/teacher.jsonl`,
   so the full run reuses them, and prints what the teacher actually answers: accuracy against gold,
   the predicted marginal next to the gold marginal, mean max-p, and a warning for any class the
   teacher under-predicts by more than 10 points. A skewed marginal is the teacher's dominant error
   on hard tasks and the student copies it faithfully, so this is the moment to reword an option and
   probe again — each iteration costs 100 calls, not 6000. On kinopoisk that warning is the whole story of
   the task:

   ```
   probe: 100 calib rows, teacher Qwen/Qwen3.8-27B-FP8   # the id is read back from label.json
     teacher accuracy vs gold: 0.710 on 100 rows
     predicted marginal: Bad 30.0%  Neutral 9.0%  Good 61.0%
     gold marginal:      Bad 34.0%  Neutral 24.0%  Good 42.0%
     mean max-p: 0.812
     WARNING the teacher under-predicts 'Neutral' (9% vs 24%): reword that option, or declare `prior:` ...
   ```
4. **Run it, then serve it.**

   ```bash
   openjev run tasks/mytask.yaml      # label -> train -> calibrate -> eval
   openjev report                     # refresh the README table from results/
   openjev serve runs/mytask
   ```

Stages skip themselves when their artifact exists; `--force <stage>` re-runs one (and everything
after it). `run` always walks the label stage first — it is a no-op when every row is cached, but it
does re-read the dataset and would call the teacher for any missing row. To re-run one stage and
nothing else, call it directly: `openjev eval tasks/<name>.yaml`.
