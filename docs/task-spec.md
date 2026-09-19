# Task YAML reference

A task file is the only thing you write. It is parsed by `openjev/spec.py` (`load_task`) into a
`Task` object; everything downstream sees only the derived `labels` list (plus `values` for
`score`). Unknown keys are a hard error at load time, in every section — a typo fails immediately
instead of being silently ignored.

Run a task file with:

```bash
openjev run tasks/<name>.yaml
```

## Top level

| field | type | default | what it does |
|---|---|---|---|
| `name` | str | **required** | Run directory is `runs/<name>/`; also the model id used by `openjev serve` and the row name in the results table. |
| `type` | `choice` \| `score` \| `noul` | **required** | Picks the derived labels and the response view. Anything else is an error. |
| `question` | str | **required** | First line of the teacher's system prompt. For `noul` the loader still requires it, but the prompt uses `statement` instead, so it is only documentation there. |
| `lang` | str | `en` | Not used for anything functional — it goes into the teacher prompt context you write yourself and into the report column. |
| `options` | list[str] | `null` | **`choice` only, required.** The label set, in a fixed order that is the class order end to end. |
| `rubric` | dict | `null` | **`score` only, required.** `{levels: [...], descriptions: [...]}` — see below. |
| `statement` | str | `null` | **`noul` only, required.** The proposition the teacher judges true/false. |
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
| `source` | mapping | **required** | Exactly one of `{hf: org/name}`, `{csv: path}`, `{jsonl: path}`. For `hf`, optional `config:` and `revision:` are passed to `datasets.load_dataset`. |
| `text` | str | `"{text}"` | `str.format` template over the row's fields. Multi-field is fine: `"{title}\n{body}"`. A missing field raises. |
| `max_chars` | int | `2000` | Truncation applied *before* both teacher and student, so both see exactly the same input. |
| `gold` | str \| null | `null` | Column holding the ground-truth label. Optional: with no gold, everything still works — calibration falls back to the teacher's argmax and the eval is reported against the teacher. |
| `gold_map` | list \| dict \| null | `null` | Applied to the raw gold value before interpretation: a list is indexed by an integer gold, a dict is keyed by the raw value. Use it when the dataset's label ids do not match your `options` order. |
| `gold_prob` | str \| null | `null` | `noul` only: column with a *fractional* gold probability (e.g. the share of annotators who said yes). Enables Brier score against that fraction in the eval. |
| `gold_threshold` | float | `0.5` | `noul` only: a float gold ≥ threshold becomes `true` (index 0), otherwise `false`. |
| `train` | split | `{split: train, n: 1000, balance: false, seed: 0}` | Distillation set. |
| `calib` | split | `{split: train, n: 500, balance: false, seed: 0}` | Early stopping + temperature fitting. |
| `eval` | split | `{split: test, n: 2000, balance: false, seed: 0}` | The reported numbers. |

Gold interpretation, in order: `gold_map` if set → a float on a `noul` task becomes an index via
`gold_threshold` → a string is looked up in `labels` (exact match) → anything else is `int()`-ed
and used as an index.

### split blocks (`train` / `calib` / `eval`)

| field | type | default | what it does |
|---|---|---|---|
| `split` | str | `train` (`test` for `eval`) | The `datasets` split expression, slices included: `train[:60000]`. A plain CSV/JSONL file is always a single split named `train`. |
| `n` | int | `1000` (`500` calib, `2000` eval) | Number of examples to draw. |
| `balance` | bool | `false` | Draw `n // k` per gold class, then top up from the remaining rows. Requires `gold`. Use it on train/calib for a skewed task; leave eval at the natural distribution. |
| `seed` | int | `0` | Sampling seed. The actual RNG seed is `"<seed>:<role>"`, so the three roles shuffle differently. |

Roles are drawn in the fixed order train → calib → eval, each from the rows not yet taken from
that source split. Two roles pointing at the same split are therefore guaranteed disjoint, and the
draw is deterministic — re-running resumes rather than re-labels.

## teacher

| field | type | default | what it does |
|---|---|---|---|
| `system_prompt` | str \| null | `null` | Replaces the generated head line (the `question`, or the statement block for `noul`). The option lines and `Answer with a single letter.` are still appended. |
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
balanced chunks, each asked with an extra `Z: none of the above`; per-chunk scores
`s_i = p(i | chunk) * (1 - p(none | chunk))` pick a shortlist of `max_options_per_call` options,
and one final call over the shortlist produces the distribution. Options outside the shortlist get
probability 0 (clamped to 1e-6 before the KL). Cost: `ceil(k/19) + 1` calls per example.

## student

| field | type | default | what it does |
|---|---|---|---|
| `model` | str | `jhu-clsp/mmBERT-small` | Any `AutoModelForSequenceClassification` checkpoint. `jhu-clsp/mmBERT-base` for a bigger run. Overridable per run with `--student`. |
| `max_len` | int | `256` | Tokenizer truncation length for training, eval and serving. |
| `epochs` | int | `5` | Upper bound; early stopping on calib KL with patience 2 usually stops sooner. |
| `lr` | float | `5.0e-5` | AdamW learning rate, 6% linear warmup then linear decay. |
| `batch_size` | int | `32` | Training batch size. Halve it if you OOM next to a running vLLM. |
| `gold_weight` | float | `0.0` | Weight of a CE term on gold added to the distillation KL, applied only to rows that have gold. `0.0` = pure distillation, which is the headline setting. Overridable with `--gold-weight`. |

Write floats in YAML as `5.0e-5`, not `5e-5` — YAML 1.1 parses the latter as a string.

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
question: "Is the following statement about the text true?"
statement: "This comment is toxic (rude, disrespectful, or likely to make someone leave a discussion)."
lang: en
data:
  source: {hf: google/civil_comments}
  text: "{text}"
  max_chars: 1500
  gold: toxicity       # fraction of raters; >= 0.5 -> true
  gold_prob: toxicity
  train: {split: "train[:60000]", n: 4000, balance: true}
  calib: {split: "train[:60000]", n: 500, balance: true}
  eval: {split: test, n: 3000}
```

`labels = ["true", "false"]`, `k = 2`. The gold column is a fraction, so `gold_threshold` (0.5)
turns it into a class and `gold_prob` keeps the fraction for a Brier score against it. Train and
calib are balanced because positives are ~8% of the data; eval stays at the natural rate.
Response shape (illustrative numbers):

```json
{"model": "toxic", "probability": 0.04, "confidence": 0.96}
```

`probability` is p(statement is true); `confidence` is `max(p, 1-p)`, i.e. how far from the fence.

## Add your own task in 3 steps

1. **Write the YAML.** Copy `tasks/example.yaml`, set `name`, `type`, the question (plus `options` /
   `rubric` / `statement`), and point `data.source` at your HF dataset, CSV or JSONL. Set
   `data.text` to a template over your columns. Gold is optional — leave `gold` out and the run
   calibrates and reports against the teacher instead.
2. **Check the draw before spending teacher time.**

   ```bash
   openjev label tasks/mytask.yaml --limit 20
   head -c 400 runs/mytask/teacher.jsonl
   ```

   Look at the `text` field: if the template or truncation is wrong, you see it here for the price
   of 20 calls. Labeling is append-only and resumable, so these 20 rows are reused by the full run.
3. **Run it, then serve it.**

   ```bash
   openjev run tasks/mytask.yaml      # label -> train -> calibrate -> eval
   openjev report                     # refresh the README table from results/
   openjev serve runs/mytask
   ```

Stages skip themselves when their artifact exists; `--force <stage>` re-runs one (and everything
after it).
