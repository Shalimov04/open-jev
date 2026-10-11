# Preregistration — `decision-index-v1`: `open-jev-base` on the Decision Index 0.3 public suite

Written 2026-10-11, **before** `open-jev-base` has been run on any Decision Index row. Fixed
here: the suite, the rendering, the refusal rule, the disclosures and what may be claimed. Nothing
below changes after a number exists except by a dated amendment in §7.

Unlocks: `scripts/bench_decision_index.py` (written with this file, tested in
`tests/test_bench_decision_index.py`) and the one run in §5. No training, no teacher calls.

## 0. What had been read before the freeze

- The board ([`multimodalart/jev-decision-index`](https://huggingface.co/spaces/multimodalart/jev-decision-index),
  v0.3.1, updated 2026-10-10): its Full-score ranking of 117 entrants and its methodology page.
  The Full score is 20 % public suite + 50 % private same-skill tests + 30 % private new-domain
  tasks; only the maintainers can run the private 80 %.
- The kit, [`apolinario/decision-index`](https://github.com/apolinario/decision-index) at its
  `main` of 2026-10-11: README, `docs/engines.md`, `docs/format.md`, `engines/base.py`,
  `data/benchmarks.json`, `data/index-0.3.json`. Not a single suite row: the suite is not
  redistributed and has not been built here.
- Nothing from `open-jev-base` on any Decision Index row.

## 1. The decision

Whether the README and model card may carry one line, "Decision Index 0.3 public index: *X*
(coverage *c*), BANKING77 disclosed as trained-on", and whether `results/bench/decision-index/`
holds a complete run that the repo owner can choose to submit (§6). No ranking claim is made in
this round: the board ranks by the Full score, which we cannot compute.

## 2. The rendering, fixed for every benchmark

The kit forbids per-benchmark prompts, truncation and dropped options; so does this file.

- **choice** → `openjev.pairs.render` of a `choice` Task: `Q:` = the question's `instructions`
  (JSON if not a string), option line = `key: description` (the key alone when the description is
  empty or equals it — the same line the kit's `transformers` engine shows), contrast set cut at
  `pairs.OPTIONS_TOKENS` = 96 tokens with `…(+N more)` exactly as in training. Segment B = the
  `state` (JSON if not a string). One pair per option, so every option is scored; the
  probabilities are the per-option sigmoids renormalised over the K options offered, the choice
  is their argmax — `openjev.base.predict_logits`, unchanged.
- **noul** (RAGTruth, PhishNChips booleans) → a `noul` Task with the default head and
  `statement` = `instructions`; the answer is the sigmoid of its one pair, p(true).
- **Refusal.** Any pair of a request longer than `max_len` = 512 tokens (the length the model was
  trained at) raises `Unsupported` for the whole request; it counts as wrong in every index. Any
  other question type raises `Unsupported`.
- Precision: float32 weights, bf16 autocast on CUDA, batch 256, the shipped weights
  (`runs/base-none-v2/student`; the Hub snapshot's sha256 is recorded in `environment.json` and checked by STOP rule 2).

The 512-token refusal will cost coverage on long-context benchmarks (ContractNLI, RAGTruth,
HoVer, MuSR, BRIGHT, SGD at least). That is the model, not the harness: it is reported, not
worked around. A sliding-window variant would be a different model and a different prereg.

## 3. Overlap with our training mixture

Checked against every `data.source` in `tasks/`, `tasks/base/`, `tasks/unseen/`:

| Decision Index benchmark | our source | status |
|---|---|---|
| BANKING77 (in the 0.3 index, gold ★, Retrieval & Classification) | `banking77` (mteb/banking77 train); BTZSC's audit found 64.9 % of banking77 test texts verbatim in `data/pairs.jsonl` | **trained-on, disclosed**; the index is reported both as computed and with BANKING77's skill set to its chance level (0) |
| ANLI, ContractNLI, NLI4CT | `xnli-en`, `rte` are NLI, different sources | clean, task-adjacent |
| everything else | — | clean |

`yahoo-topics`, `sst5`, `ru-inappropriate` are calibration-only and do not appear in the suite.

## 4. What is reported

From the kit's own `score --edition 0.3`, untouched: `scores.json` (public index, raw index,
five areas), `benchmark-summary.json` (per benchmark: native metric, answered / unsupported /
error counts, median ms). Added by us: the BANKING77-at-chance index (§3), and the share of
requests refused by the 512-token rule per benchmark.

## 5. The run

```sh
# once, on a machine with Hub access (≈7 GB downloads, HLE terms accepted, `hf auth login`)
pip install -e ".[transformers,rebuild]"   # in a checkout of apolinario/decision-index
python -m decision_index suite rebuild --work work
python -m decision_index suite import \
    --rows work/artifacts/benchmark-suite/release-v2-rebuilt/selected-rows.jsonl.gz \
    --added-rows work/artifacts/benchmark-suite/release-v2-rebuilt/added-rows.jsonl.gz \
    --gsm8k-rows work/artifacts/benchmark-suite/release-v3-rebuilt/gsm8k-rows.jsonl.gz

# smoke: 100 sampled requests, must validate with zero `error` rows
python -m decision_index suite sample --n 100 --out sample-100.jsonl.gz
PYTHONPATH=<open-jev>/scripts python -m decision_index run \
    --engine bench_decision_index:OpenJevEngine --option model=sshalimov04/open-jev-base \
    --rows sample-100.jsonl.gz --out runs/di-open-jev-base-sample

# the run: 109,759 + 30,419 scoreable requests, resumable
PYTHONPATH=<open-jev>/scripts python -m decision_index pipeline \
    --engine bench_decision_index:OpenJevEngine --option model=sshalimov04/open-jev-base \
    --out runs/di-open-jev-base
```

Copied into `results/bench/decision-index/`: `scores.json`, `index.json`,
`benchmark-summary.json`, `environment.json` (not `results.jsonl`: the sources' terms forbid
redistributing their text).

**STOP rules.** (1) Any `error` row in the smoke run: fix the engine, rerun the smoke, record the
fix in §7 — never patch result files. (2) `environment.json` weights sha256 ≠
`runs/base-none-v2/student/model.safetensors`: stop. (3) `scores.json` not `"complete": true`
after resume: report it incomplete, no index line in the README.

## 6. Submission

Not part of this round. The kit's submission is a PR to the Space with the results dataset on
the Hub; the maintainers then run the private 80 %. It is prepared only after the repo owner has
read §4's numbers and says so, as for BTZSC (`docs/bench/btzsc-submission.md`).

## 7. Amendments

- **2026-10-11, before any run.** The repo owner decided to skip our own run (§5). Instead, the
  maintainers are asked to evaluate the model on the lab side; the request text is in
  `docs/bench/decision-index-request.md`. §1's README line and §4's report wait for numbers that the
  board publishes. §2's rendering, §3's disclosure and the engine are unchanged and are what the
  request hands over.
