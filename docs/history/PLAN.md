# Open-Jev — implementation plan (overnight v0)

> Historical: this is the overnight plan as written before the build. The live spec is `docs/task-spec.md`.

Goal: `openjev run task.yaml` → teacher-labeled data → small calibrated encoder → eval report → servable
`/v1/systemone`. Three primitives (choice / score / noul) are **one mechanism**: a K-way classifier over
ordered labels plus a "view" that post-processes probabilities. Do not build three code paths.

Planner verified on 2026-09-20: teacher request format works (letters come back as bare tokens `"C"`,
not `" C"`); vLLM `--max-logprobs` is **20** (30 → HTTP 400); `.venv` in this repo already has
torch 2.13+cu130 (CUDA ok), transformers 5.16.1, datasets 5.0.1, fastapi, uvicorn, httpx, pyyaml,
pydantic, scikit-learn, numpy, pandas; HF cache has only empty stubs for mmBERT (must download);
`PolyAI/banking77` and `ai-forever/georeview-classification` are script datasets (unloadable with
datasets 5) → use the parquet mirrors `mteb/banking77`, `mteb/GeoreviewClassification`.

## 1. Package, layout, deps

Package `openjev`, flat layout, argparse CLI, no src/ dir, no plugin system.

```
pyproject.toml            # name openjev, deps below, [project.scripts] openjev = "openjev.cli:main"
openjev/
  spec.py       # Task dataclass/pydantic model, YAML load + validation, derived `labels` list
  data.py       # load HF/CSV/JSONL, apply text template, gold mapping, deterministic split/sample; JSONL append/read helpers
  teacher.py    # async httpx client to vLLM; letters prompt; chunked shortlist for K>19; append-only cache; resume
  train.py      # AutoModelForSequenceClassification(K); KL(+CE) loss; early stop on calib split; save
  calibrate.py  # temperature scaling (LBFGS on NLL), ECE/Brier helpers
  evaluate.py   # metrics vs gold and vs teacher, latency/throughput; writes eval.json + results/<task>.json
  serve.py      # FastAPI /v1/systemone over one or more run dirs
  views.py      # choice/score/noul post-processing of a prob vector -> response dict (~40 lines)
  cli.py        # openjev run|label|train|calibrate|eval|serve|report|augment
tasks/*.yaml    # the 5-6 tasks below
results/*.json  # committed; README table generated from them
scripts/queue.sh  # sequential overnight queue (label+train+eval per task)
tests/test_smoke.py  # tiny: spec parse, views math, calibrate on synthetic logits, teacher parsing on a saved response
```

Deps (all already in `.venv`): torch, transformers, datasets, httpx, pyyaml, pydantic, numpy,
scikit-learn (f1/metrics only), fastapi, uvicorn. Nothing else. Reuse `.venv` (`uv pip install -e .`).
Env: **unset `HTTP_PROXY/HTTPS_PROXY/ALL_PROXY` for localhost calls** (`teacher.py` uses
`httpx.AsyncClient(trust_env=False)`); download HF models with proxy vars unset (faster).

## 2. Task spec (YAML)

```yaml
name: agnews                  # run dir = runs/<name>/
type: choice                  # choice | score | noul
question: "What is the topic of this news article?"
options: [World, Sports, Business, Sci/Tech]      # choice only; order is fixed = label order
# score only:
# rubric: {levels: [1,2,3,4,5], descriptions: ["very negative", ..., "very positive"]}
# noul only:
# statement: "This comment is toxic (rude, disrespectful, or likely to make someone leave a discussion)."
lang: en                      # only used for the teacher prompt and the report
data:
  source: {hf: fancyzhx/ag_news}         # or {csv: path} | {jsonl: path}; hf may add `config:`, `revision:`
  text: "{text}"                          # str.format template over row fields; multi-field ok ("{title}\n{body}")
  max_chars: 2000                         # truncate before teacher and student (same text for both)
  gold: label                             # optional column; int -> index into options/levels, str -> matched by name
  gold_map: null                          # optional list/dict remapping raw gold values to option index
  train: {split: train, n: 4000, balance: false, seed: 0}
  calib: {split: train, n: 500}           # disjoint from train, same sampling
  eval:  {split: test,  n: 2000}
teacher:
  system_prompt: null                     # override; default built from question/options/statement (§3)
  concurrency: 32
  max_options_per_call: 19                # shortlist chunking above this (vLLM top_logprobs cap 20 incl. "none")
student:
  model: jhu-clsp/mmBERT-small            # default; jhu-clsp/mmBERT-base for "final" runs
  max_len: 256
  epochs: 5
  lr: 5.0e-5
  batch_size: 32
  gold_weight: 0.0                        # CE on gold added to KL; 0 = pure distillation (headline setting)
```

`spec.py` derives `labels`: choice → options; score → `str(level)` per level (+ `values` for expectation);
noul → `["true", "false"]`. Everything downstream only sees `labels` (+ `values` for score).

## 3. Pipeline stages and artifacts

`openjev run task.yaml` runs stages in order, each skipped if its artifact is complete (`--force STAGE`).

**label** → `runs/<name>/teacher.jsonl`, append-only, one line per example:
`{"id","split":"train|calib|eval","text","gold":int|null,"probs":[K floats],"raw":{letter:logprob},"source":"data|synth"}`.
Resume = read existing ids, skip. Flush every 64 completed. Request exactly as verified:
`max_tokens:1, temperature:0, logprobs:true, top_logprobs:min(K,20), structured_outputs:{choice:[letters]},
chat_template_kwargs:{enable_thinking:false}`. Default system prompt: `question` + one line per option
`"A: <label>"` + "Answer with a single letter."; user message = text. For noul the "question" is
`Is the following statement about the text true?` with `A: true`, `B: false`. For score, options are
`"<level>: <description>"`. `probs = softmax over returned option-letter logprobs`, missing letters → 0.
Retry with backoff (3x) on 5xx/timeouts; a failed example is logged and skipped, not fatal.

**K > 19 options (banking77)** — shortlist: split options into chunks of ≤19, each call adds
`"Z: none of the above"`; per-chunk score `s_i = p(i|chunk) * (1 - p(none|chunk))`; take the top-19 options
by `s_i` and make one final call with only those → `probs` over the shortlist, other labels 0 (clip to 1e-6
before KL). Cost = ceil(K/19)+1 calls per example (77 → 5). Tradeoff: exact softness only over the
shortlist; if the true class is missed by every chunk the label is wrong (we measure this: eval agreement).
Alternatives rejected: multi-token codes (no clean distribution), semantic grouping (extra teacher setup step).

**train** → `runs/<name>/student/` (save_pretrained + tokenizer) and `train.json` (loss curve, best epoch).
`AutoModelForSequenceClassification(num_labels=K)`, bf16 autocast, sdpa, AdamW, linear warmup 6% + linear
decay, loss = `KL(teacher_probs || softmax(logits)) + gold_weight * CE(gold)` (CE only for rows with gold).
Early stopping on calib KL. Memory: mmBERT-small bs32 len256 ≈ 3 GB; base ≈ 8 GB; both under 20 GB. Set
`TOKENIZERS_PARALLELISM=false`. `--student` CLI override, run dir suffix `-base` for the base variant.

**calibrate** → `runs/<name>/calib.json` `{"temperature": T, "target": "gold|teacher", "ece_before", "ece_after"}`.
Fit T on calib split by LBFGS on NLL. Target = gold when the calib split has gold, else teacher argmax
(this calibrates to the teacher's opinion; the report states which). ECE = 15 equal-width bins on max-prob.

**eval** → `runs/<name>/eval.json` and `results/<name>.json` (§6). On the eval split: student (calibrated)
vs gold, teacher vs gold, student vs teacher (agreement = argmax match, mean KL). Score view also reports
MAE of expectation vs gold level and exact-level accuracy; noul reports AUROC + Brier vs gold probability
when the gold is a fraction (civil_comments `toxicity`). Latency: batch-1 CUDA, 200 requests after 20 warmup,
p50/p99 ms; throughput at batch 64 on the eval texts (examples/s). Also CPU batch-1 p50 (one number; the
edge story).

**export** = the `student/` dir + `openjev.json` `{task spec, labels, values, temperature, metrics}`.
No ONNX tonight.

**serve**: `openjev serve runs/agnews runs/georeview ... --port 8080`. `POST /v1/systemone`
`{"model": "agnews", "input": "text or JSON object (formatted with the task's text template)"}` →
choice: `{"choice": "Sports", "probabilities": {label: p}, "confidence": max p}`;
score: `{"score": Σ p_k·value_k, "probabilities": {level: p}, "confidence": max p}`;
noul: `{"probability": p_true, "confidence": max(p_true, 1-p_true)}`. Probabilities are temperature-scaled.
`GET /v1/models` lists loaded runs. Type/options are fixed per model (we serve trained students, not a
general model — say so in README).

**views.py** is the single place where a prob vector becomes a response; eval and serve both call it.

## 4. Optional: PGKD-lite active loop (`openjev augment task.yaml --rounds 1 --per-class 100`)

1. From `eval.json` on the **calib** split (never eval), take the 3 classes with the lowest recall and the
   3 most frequent confusion pairs (teacher labels as reference, gold if present).
2. For each weak class, ask the teacher (normal generation, `temperature 0.9`, `max_tokens 2000`, JSON list
   output, `enable_thinking:false`) for `per-class` new texts in the task language: "Write N realistic
   examples of `<class>`, in the style of these 3 examples; also N that look like `<confused class>` but are
   really `<class>`". Parse leniently; drop duplicates and texts > max_chars.
3. Label them with the normal **label** stage (source `"synth"`, split `train`) — the teacher may disagree
   with the intended class; that is fine, it is only more targeted data. Retrain, re-calibrate, re-eval;
   results file gets `"augment_round": 1`.
Ceiling: no hard-negative mining beyond confusion pairs; upgrade path is the full PGKD selection loop.

## 5. Tonight's tasks (all ids verified 2026-09-20)

| task | type | K | lang | source (split sizes) | text | gold | notes |
|---|---|---|---|---|---|---|---|
| agnews | choice | 4 | en | `fancyzhx/ag_news` (120k/7.6k), label names World/Sports/Business/Sci-Tech | `{text}` | `label` int | short, ~8 min labeling; pipeline dev target |
| kinopoisk | choice | 3 | ru | `ai-forever/kinopoisk-sentiment-classification` train/validation/test (10.5k/1.5k/1.5k) | `{text}`, `max_chars 1500` | `label` 0=Bad,1=Neutral,2=Good | long reviews → slower (~3-4 req/s); eval n = all 1500 |
| georeview | score | 5 | ru | `mteb/GeoreviewClassification` (50k/5k/2048) | `{text}` | `label` 0-4 → stars 1-5 (`gold_map`) | rubric levels 1..5, descriptions "very negative".."very positive"; report MAE |
| toxic | noul | 2 | en | `google/civil_comments` (1.8M/97k/97k; load `train[:60000]`) | `{text}`, `max_chars 1500` | `toxicity ≥ 0.5` → true; keep raw fraction as `gold_prob` | statement in §2; train/calib `balance: true`; eval natural distribution, n=3000 |
| banking77 | choice | 77 | en | `mteb/banking77` (9993/3076) | `{text}` | `label` int, names in `label_text` | shortlist path, 5 calls/example; n_train 3000; run last |
| headlines (optional 6th) | choice | 6 | ru | `ai-forever/headline-classification` (36k/12k/12k) | `{text}` | `label` 0-5 (спорт, происшествия, политика, наука, культура, экономика) | cheapest ru task; add if the queue is idle |

Splits: `train n=4000`, `calib n=500` from the train split (disjoint, seeded), `eval n=2000` from test.
Rough teacher time (≈14 req/s short, proportionally slower for long): agnews 8 min, headlines 8 min,
georeview 15 min, toxic 15 min, kinopoisk ~40 min, banking77 ~90 min → ≈3 h sequential; runs in the
background while code is written. Teacher zero-shot is the baseline in every table row.

Student default: **`jhu-clsp/mmBERT-small`** (≈140M with the multilingual vocab, ~32M non-embedding,
ModernBERT arch, en+ru, fast to train; 1833 languages so a task YAML in any language works). Final rows use
`jhu-clsp/mmBERT-base` (307M) where time allows. Not `answerdotai/ModernBERT-base` (English only) and not
`deepvk/RuModernBERT-*` (Russian only) — one default must cover both languages. Download both mmBERT models at
t=0 with proxy vars unset (`huggingface-cli download`, ~0.6 GB + 1.2 GB).

## 6. Results format

`results/<task>[-base][-gold].json`:
```json
{"task":"agnews","type":"choice","k":4,"lang":"en","student":"jhu-clsp/mmBERT-small","n_train":4000,"n_synth":0,
 "gold_weight":0.0,"eval_n":2000,
 "student":{"acc":0.0,"macro_f1":0.0,"ece_raw":0.0,"ece_cal":0.0,"brier":0.0,"nll":0.0},
 "teacher":{"acc":0.0,"macro_f1":0.0,"ece":0.0,"brier":0.0},
 "agreement":{"argmax":0.0,"mean_kl":0.0},
 "score":{"mae":0.0}, "noul":{"auroc":0.0,"brier_vs_gold_prob":0.0},
 "latency_ms":{"gpu_b1_p50":0.0,"gpu_b1_p99":0.0,"cpu_b1_p50":0.0},"throughput_gpu_b64":0.0,
 "teacher_calls":0,"teacher_minutes":0.0,"train_minutes":0.0,"temperature":1.0,"calib_target":"gold","git":"sha"}
```
`openjev report` rewrites the README table between `<!-- results -->` markers:
`task | type | K | lang | n_train | student acc / F1 | teacher acc / F1 | agree | ECE raw→cal | Brier | GPU p50 ms | ex/s`.
Plus one line per task on what the number means (e.g. "student distilled with zero gold labels").

## 7. Milestones (t = hours after start; ~10 h total, 2 h buffer)

- **M0 (0:00-0:40) scaffold.** pyproject, package skeleton, `spec.py`, `data.py`, `teacher.py`, `tasks/agnews.yaml`.
  Start mmBERT downloads in the background immediately. *Accept:* `openjev label tasks/agnews.yaml --limit 50`
  writes 50 lines to `runs/agnews/teacher.jsonl`; probs sum to 1; rerun adds 0 lines; `pytest -q` passes.
- **M1 (0:40-2:00) end-to-end on agnews.** train/calibrate/eval/report/serve. *Accept:* full agnews run
  (4000/500/2000) finishes; student acc vs gold ≥ 0.85 and within 3 pts of the teacher; ECE after calibration
  < ECE before; `results/agnews.json` exists; `curl POST /v1/systemone` returns a choice dict; README table row.
- **M2 (2:00-4:30) all primitives.** Write the other YAMLs first, then start `scripts/queue.sh`
  (label → train → calibrate → eval per task, sequential: kinopoisk, georeview, toxic, headlines, banking77);
  training of task N overlaps labeling of task N+1 (student training uses ~3 GB; vLLM is unaffected).
  *Accept:* score view returns expectation in [1,5] and MAE is reported; noul reports AUROC; 4 result files.
- **M3 (4:30-6:30) hard/optional.** banking77 via shortlist (accept: student acc vs gold ≥ 0.75 and
  agreement with teacher ≥ 0.85, else document the gap); mmBERT-base on agnews+georeview; `gold_weight 1.0`
  variant on one task to show the gold-boost.
- **M4 (6:30-8:00) augment.** PGKD-lite one round on the task with the weakest per-class recall.
  *Accept:* results file with `augment_round: 1`, macro-F1 delta reported even if negative.
- **M5 (8:00-9:00) wrap.** README (what it is, 5-line quickstart, results table, honest caveats), tests
  green, `results/` committed, a `docs/task-spec.md` derived from §2.

Parallelization: the teacher is the only serial bottleneck → keep it busy from M0 (queue script);
code writing, model downloads and student training all overlap with labeling. Two coder agents max:
one on `teacher.py`+`data.py`+queue, one on `train.py`+`calibrate.py`+`evaluate.py`+`serve.py`; the spec
and the `teacher.jsonl` row format above are the contract between them.

Cut order if behind: M4 augment → mmBERT-base variants → banking77 (keep the shortlist code path but skip the
run) → headlines. Never cut: calibration, eval JSON, README table.

## 8. Risks and gotchas

- Proxy env vars break localhost and slow HF downloads: `trust_env=False` in httpx; run downloads with
  `env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY`. Check `~/.cache/huggingface/hub/models--jhu-clsp--*`
  actually contains weights (today they are 40 KB stubs).
- vLLM caps `top_logprobs` at 20; `structured_outputs` is a vLLM-specific field; response letters are bare
  tokens. Keep `max_tokens:1`, `finish_reason: "length"` is expected. Store `raw` logprobs for re-softmaxing.
- Script-based HF datasets fail on datasets 5 → use the mirrors listed; `ag_news` is `fancyzhx/ag_news`.
- Teacher is near one-hot on easy tasks → distillation ≈ hard labels; that is fine, but do not expect the
  student to be better-calibrated than the teacher without temperature scaling. Report ECE of both.
- Letter position bias in the teacher: fixed option order tonight; `shuffle_options: true` (average over 2
  permutations) is a documented follow-up, not tonight.
- Georeview/kinopoisk teacher accuracy will be modest (neutral/3-star ambiguity); the honest metric for
  score is MAE + agreement, not accuracy. Toxicity gold is itself a fraction: use it for Brier.
- Long Russian reviews: truncate to `max_chars` for both teacher and student so they see the same input.
- GPU is shared with vLLM: keep student batch ≤ 32×256 (base) and check `nvidia-smi` free memory before the
  queue starts; OOM → halve batch size, never touch vLLM.
- `AutoModelForSequenceClassification` for ModernBERT/mmBERT pools CLS by default; fine. Do not use
  flash-attn; sdpa is enough on GB10.
- Class imbalance (civil_comments 8% positives): `balance: true` for train/calib only; eval stays natural.
- Every stage writes to the run dir only; `runs/`, `cache/`, `models/` are gitignored; `results/` is not.
