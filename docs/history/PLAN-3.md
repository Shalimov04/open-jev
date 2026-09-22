# Open-Jev — track 3: real tasks (web parsing, browser actions, probability)

> Written 2026-09-20 after reading README, docs/task-spec.md, docs/experiments.md, tasks/*.yaml,
> openjev/{data,spec,teacher,check,evaluate}.py, and verifying every dataset below on the hub
> today (file lists, schemas, sample rows, sizes). Implementation by Opus agents. No code here.

## 0. What this track is for

The seven existing tasks measure the method. This track ships five tasks that look like the work
an agent stack actually does, all through the unchanged pipeline (`tasks/*.yaml` → `check` →
`check --probe 100` → `run`), each fed by one small preprocessing script that writes a JSONL the
task loads with `{jsonl: data/<task>.jsonl}`:

| task | primitive | K | family | data | teacher cost |
|---|---|---|---|---|---|
| `arb-success` | noul | 2 | **probability is the product**: p(agent completed the task) | AgentRewardBench | ~6 min |
| `arb-quality` | score | 4 | how well the agent did (expected optimality level) | same rows | ~6 min |
| `m2w-element` | choice | 16 | browser action: which element to act on | Multimodal-Mind2Web | ~21 min |
| `m2w-target` | noul | 2 | act-or-ask gate on a single (task, element) pair | same pages | ~13 min |
| `swde-field` | choice | 33 | web parsing: which schema field is this DOM node (K>19 shortlist, for real) | SWDE | ~30 min |

Total teacher time ≈ 76 min + 5 probes (≈ 4 min) = **~80 min** (rates from `check.py` RATES:
12 ex/s ≤ 300 chars, 5 ex/s ≤ 1000, 3.5 above, divided by calls per example). Concurrency stays 32;
vLLM on :8000 is never touched.

**The crux — how a page becomes ≤ 512 student tokens.** Never the page. Each row is one *decision*
rendered as a short structured text: the goal, a one-line history, and either (a) a numbered list of
candidate elements, one line each, or (b) a single node with its DOM path and neighbours. Element
lines are `tag + up to 2 salient attrs + inner text[:40]`; DOM path is the last 6 ancestors with
`#id`/`.class` hints. Every rendering below stays under `max_chars` with `student.max_len` chosen so
`check` does not print the "student reads less than the teacher" warning. Teacher and student see the
identical string (pipeline rule).

## 1. Datasets — verified today, and how they load

- **AgentRewardBench** `McGill-NLP/agent-reward-bench` (terms-of-use, research). `data/annotations.csv`
  (1408 rows = 1302 unique trajectories, 106 double-annotated, 13 disagreements): columns
  `benchmark, task_id, model_name, trajectory_success {Successful 395 / Unsuccessful 1012 / Unsure 1},
  trajectory_optimality {1..4}, trajectory_side_effect, trajectory_looping`. Do **not** download
  `cleaned/` (5–18 MB per trajectory, ~7–24 GB); instead read the text-only judge inputs
  `judgments/<bench>/<agent>/gpt-4o-mini-noscreen-noaxtree/<bench>.<id>.json` (13–20 KB each, 1304
  files ≈ 20 MB, all 16 bench×agent dirs exist except llama on visualwebarena, which annotations do not
  contain either): `goal` + `chat_messages.regular[1].content[1].text` = `Step/URL/Action/Reasoning`
  blocks. Loads with `huggingface_hub.hf_hub_download` per file (no `datasets` script needed).
- **Multimodal-Mind2Web** `osunlp/Multimodal-Mind2Web` (openrail; the text-only `osunlp/Mind2Web` is
  cc-by-4.0 but its test set is a password zip, `mind2web`, and 5.9 GB of JSON). Parquet, one row per
  action, splits train 7775 / test_task 1339 / test_website 1019 / test_domain 4060. Columns
  `confirmed_task, action_reprs, target_action_index, operation (json: op CLICK/TYPE/SELECT, value),
  pos_candidates, neg_candidates (json strings: tag, backend_node_id, attributes json,
  is_top_level_target), cleaned_html (median 90 k chars, every element carries backend_node_id=)`.
  Screenshots are 94 % of each shard: **read only the text columns with
  `pyarrow.parquet.ParquetFile(HfFileSystem().open(path)).read(columns=[...])`** — verified, 14 MB of
  a 238 MB shard — so the whole thing is ~0.8 GB of HTTP range reads, not 13 GB. huggingface_hub 1.32
  and pyarrow 25 are already in the venv. ~6 % of rows have empty `pos_candidates`: drop them.
- **SWDE** `abdo-Mansour/SWDE` (parquet + `files/<vertical>.7z`, 205 MB total, the original pages:
  `movie/movie-imdb(2000)/0000.htm`, verified on nbaplayer). Parquet rows are pages: `website_id`
  (`boxofficemojo_0001`), `schema` (json, field descriptions), `gt` (python-literal dict field →
  [values], `&nbsp;` entities inside). 8 verticals, 32 fields total, 10 sites per vertical, 124 k
  pages. Needs a 7z reader: no `7z`/`bsdtar` on the box; `uv run --with py7zr scripts/prep_swde.py`
  works (verified) and keeps py7zr out of `pyproject.toml`. lxml/bs4 are not installed — parse with
  stdlib `html.parser` (a tag stack is all the renderer needs).
- Rejected: Klarna (Zenodo, multi-GB MHTML, NC licence — future work); WebSRC (QA, no element
  labels); WebLINX (NC licence); `hazyresearch/based-swde` (1111 pages of page-text QA, not nodes).
- `datasets` 5.0.1 loads all of the above only via JSONL we write; split slices on a single JSONL
  (`train[:N]`, `train[N:]`) work (verified) — the scripts write source-train rows first and
  source-test rows after, and the YAML slices, so eval stays on **unseen websites / unseen agents'
  runs** with zero loader changes. `data/` is gitignored (Mind2Web asks not to redistribute).

## 2. The tasks

**T1 `arb-success` (noul).** `statement: "The agent completed the user's task: its final state or
final message satisfies the goal."`, `question` head: "Judge this web-agent run from its action log
only." Text (`{text}` written by `scripts/prep_arb.py`): `goal:` line; `steps: N`; last ≤ 12 steps as
`k. <action[:120]>` (drop Reasoning, keep URL host only when it changes); `final: <send_msg_to_user
text[:400]>` or `final: (none)`; `error:` if the last `last_action_error` is non-empty. ~900–1500
chars, `max_chars: 1800`, `max_len: 512`. Gold `success` = 1 if Successful (majority over duplicate
annotators, Unsure dropped) — 28 % positive. Also write `optimality` (1–4) and `benchmark`. Splits:
train 800 / calib 200 / eval 300, natural rate, `seed: 0`; rows in random order (one file). Cost
1302 × 1 call ≈ 6 min. **Why it is the calibration task:** the gate "accept the run if p ≥ τ, else
re-run or send to a human" is decided by the probability, not the argmax; the paper's headline is
that LLM judges over-estimate success (precision ~70 %), so the calibrated *number* is the product.
Publish if: student AUROC within 0.03 of the teacher, Brier vs gold below the constant-rate baseline
(0.28·0.72 = 0.20), and the risk-coverage table (§3) shows a τ with ≥ 90 % precision at ≥ 30 %
coverage. State the CI: 300 eval rows → AUROC ± ~0.06, three seeds mandatory, "preliminary" wording.

**T2 `arb-quality` (score).** Same JSONL, `gold: optimality`, `gold_map: {1: 0, 2: 1, 3: 2, 4: 3}`,
rubric levels [1,2,3,4] = complete failure / suboptimal / somewhat optimal / completely optimal.
MAE of the expectation, as georeview. Cost ≈ 6 min. First thing cut (§5) — it is the same rows.

**T3 `m2w-element` (choice, K = 16).** `question: "Which candidate element should the agent act on
for its next step?"`, `options: [candidate A, …, candidate P]`. Text: `task: <confirmed_task>`;
`done: <action_reprs[:target_idx][-2:] joined by " ; ">` (or `done: (nothing yet)`); `candidates:`
then 16 lines `A) <tag> <attrs> "<text[:40]>"` where attrs are at most two of
`type, role, aria_label, placeholder, title, alt, name, value` and text is the node's descendant text
from `cleaned_html` (stdlib parser, `backend_node_id` lookup; empty text → `""`). One positive
(`is_top_level_target` preferred) + 15 negatives sampled uniformly from `neg_candidates` (say so:
random negatives are easier than a ranker's top-k; the harder variant — prefer negatives with
non-empty text — is a `--hard` flag, seed-1 only if time), shuffled with a per-row seed; gold = slot
index (int). ~1100–1400 chars, `max_chars: 1600`, `max_len: 512`. Rows: train-source actions first
(7300 usable), test_website after (~950). YAML: train `train[:7300]` n 5000, calib `train[:7300]`
n 500, eval `train[7300:]` n 900. Cost 6400 × 1 call at ~5 ex/s ≈ 21 min. Publish if: student
accuracy ≥ teacher − 2 pts on unseen websites (chance 6.25 %), plus the cascade: accuracy at 0 / 10 /
25 % of steps escalated to the teacher — that is the "small model drives, big model on the hard
steps" number an agent stack wants. Note honestly that this is element selection with the positive
guaranteed among 16, the MindAct "oracle candidates" setting, not end-to-end task success.

**T4 `m2w-target` (noul).** `statement: "This element is the one the agent should act on next."`
Text: `task:`, `done:` as T3, then one element rendered with more context: the T3 line plus
`path: <last 4 ancestors with #id/.class>` and `parent text: "<parent descendant text[:100]>"`.
Rows: per action, the positive plus 3 sampled negatives → 25 % positives at the natural rate; train
1250 actions → 5000 rows, calib 500, eval from test_website 500 actions → 2000 rows; `balance: true`
on train only (as toxic; note that the draw spends gold). ~350–500 chars, `max_chars: 700`,
`max_len: 256`. Cost 7500 at ~10 ex/s ≈ 13 min. This is the pairwise ranker/verifier stage: score
every element independently, act when max p ≥ τ, ask otherwise. Publish: AUROC, Brier, and the
risk-coverage table; compare its gate against T3's confidence gate on the same actions in
`docs/use-cases.md`.

**T5 `swde-field` (choice, K = 33).** `question: "Which schema field does this text node hold?"`,
options = 32 `vertical.field` names (`movie.title`, `auto.price`, … from the parquet `schema`) plus
`none`. Text: `title: <page <title>[:80]>`; `path: <last 6 ancestors with #id/.class>`;
`before: "<previous text node[:60]>"`; `node: "<text[:120]>"`; `after: "<next text node[:60]>"`.
Positives: text nodes whose normalised text (collapse whitespace, unescape entities; keep label
prefixes like `Height:`) equals a `gt` value, or contains it with len(node) ≤ 2·len(value); 2 random
other text nodes per page → `none`, so `none` is ~30 %.
Eval = 2 held-out sites per vertical (unseen-site protocol), written after the train-site rows.
YAML: train n 4000, calib 500 (`train[:N]`), eval 2000 (`train[N:]`). ~250–400 chars, `max_chars:
600`, `max_len: 256`,
`teacher.max_options_per_call: 19` (default) → 2 chunks + final = **3 calls/row**, 6500 rows ≈ 30
min. This is the >19-option path on a label set with real semantics (banking77 is the only other
one). Publish: accuracy / macro-F1 on unseen sites vs teacher, per-vertical recall, and whether the
2-fold guard accepts a 33-dim bias. Honest ceiling: gold is string-matched, so a value that appears
twice on the page (title in header and body) is two positives, and `none` contains near-misses.

## 3. One small framework addition: the risk-coverage ("ask a human") table

`cascade` routes unsure rows to the *teacher*. The gates in T1/T4 route them to a *human*, i.e. to
gold. Add to `evaluate.py` next to `cascade`: `selective = [{tau, coverage, metric_covered}]` for
τ in 0.50…0.95 step 0.05 over the calibrated eval rows (`confidence ≥ τ` → answered; metric on the
answered subset: acc for choice, MAE for score, and for noul both acc and *precision of "true"*),
stored in `results/<run>.json` and printed by `report` for noul tasks as one extra line. ~20 lines,
CPU, no re-inference; tested in `tests/test_posthoc.py` on a synthetic 3-row case. Nothing else in
the pipeline changes. (`serve` already exposes `escalate` and conformal `set`; no new endpoint.)

## 4. Milestones (wall clock; one agent, GPU queue in the background)

**M0 — prep scripts (0:00–1:15).** `scripts/prep_arb.py`, `scripts/prep_m2w.py` (writes both
`data/m2w-element.jsonl` and `data/m2w-target.jsonl` from one pass over the parquet columns),
`scripts/prep_swde.py` (run via `uv run --with py7zr`). Each: idempotent, writes `data/<task>.jsonl`
with the fields named in §2 plus `source_id`, prints row counts and the N for the YAML slices. Stdlib
`html.parser` only. Each script ends with a 5-line `__main__` self-check (a hand-written HTML
snippet → expected rendering). Five YAMLs in `tasks/`.
Accept: `openjev check tasks/<t>.yaml` on all five — exit 0, no `max_len` warning, printed cost
within ±2× of §0, and the three printed examples read like §2 to a human.

**M1 — ARB first (1:15–2:15).** `check --probe 100` on `arb-success` (watch the marginal: the teacher
will over-predict *true*; that is the paper's finding and the calibration bias is the answer — do not
reword to hide it). `openjev run` both tasks, then seeds 1–2 through `scripts/gpu_queue.sh`
(`--no-latency`). Teacher ≈ 12 min, GPU ≈ 6 min. Accept: `results/arb-success.json` has `noul.auroc`,
`brier`, `cascade`; `openjev compare runs/arb-success runs/arb-success-s1` runs.

**M2 — Mind2Web (2:15–4:00).** Prep (~0.8 GB of range reads; cache the projected columns as
`cache/m2w/*.parquet` so a re-run is free), probe, run `m2w-element` then `m2w-target`, seeds 1–2.
Teacher ≈ 34 min, GPU ≈ 30 min (len 512, 5000 rows ≈ 8 min/seed). Accept: every eval `id` in
`runs/m2w-element/teacher.jsonl` is ≥ the slice boundary (test_website rows), cascade curve present.

**M3 — SWDE (4:00–5:30).** Prep (205 MB, py7zr), probe (report the per-chunk `none` rate — if the
final call gets the true field in the shortlist < 90 % of the time on the probe, lower
`max_options_per_call` to 17 and re-probe before spending 30 min), run, seeds 1–2. Teacher ≈ 30 min,
GPU ≈ 10 min.

**M4 — `selective` table + tests (can overlap M2/M3 teacher time; 1:00 of agent time).** §3.
Re-run `openjev eval --no-latency` on the five new run dirs only. Accept: `pytest -q tests` green;
`python -c "import json;print(json.load(open('results/arb-success.json'))['selective'][:3])"`.

**M5 — write-up and review (5:30–7:00).** `docs/use-cases.md`: per task — what an agent does with
the output (the two-line client pattern from the README), the numbers with `compare` CIs over three
seeds, the gate tables, and the "cannot show" list from §6. README: a short "Real tasks" section
linking it, and `openjev report` for the table (new rows only; existing rows untouched).
`openjev bench` on the five seed-0 dirs with an idle teacher. Honesty review by `oj-planner` with the
REVIEW-final brief; 30 min to fix. Commit per milestone; `data/` never committed.

Budget: 7 h wall clock; teacher lane ≈ 80 min serial; GPU lane ≈ 50 min serial.

## 5. Cut order, never-cut, and the 4 am fallback

Cut first: (1) `arb-quality`; (2) T3 `--hard` negatives variant; (3) SWDE seeds 1–2, then SWDE
train n 4000 → 2500; (4) `m2w-target` (T3's confidence gate already gives an act/ask curve);
(5) `openjev bench`. **Never cut:** `arb-success` with three seeds, `m2w-element` with three seeds,
the `selective` table, `docs/use-cases.md` with CIs, the honesty review. If the teacher is busy
(production traffic), the lane order is ARB → m2w-element → m2w-target → SWDE and whatever is
unlabelled at hour 5 is reported as "not run". If a prep script cannot get its data (hub outage), skip
that task; nothing here depends on another task's data.

## 6. What these tasks can and cannot show

- **They can show** that the same YAML-to-server loop works on agent-shaped inputs: a decision rendered
  from a DOM or an action log, a calibrated probability an agent thresholds on, a small model that
  handles the easy steps and knows when it is unsure. The numbers that matter: student vs teacher on
  held-out *websites* (T3/T4/T5) and held-out *runs* (T1); the risk-coverage gate; the cascade.
- **The teacher is the ceiling, and here it is a weaker teacher than on ag_news.** It reads a 40-char
  element line, not the page, and an action log, not a screenshot; its zero-shot accuracy on T3 and
  T5 will be well below supervised SOTA (MindAct ~50–60 % element accuracy with a trained ranker;
  SWDE supervised ~95 F1). A student that matches it is the claim; beating it needs the bias trick to
  find a marginal shift, which T1 (over-predicted *true*) is the likeliest to have.
- **Gold is noisier than the benchmarks.** T5 gold is string-matched; T1 has 13/106 annotator
  disagreements; T3 has candidate lists we sampled. Every accuracy carries the ±1.8 pt (n = 2000) or
  ±5 pt (n = 300) CI from PLAN-2 §5; one seed is not a measurement.
- **Not end-to-end.** T3 selects among oracle candidates; T1 judges from actions only; nothing here
  drives a browser. Do not write "agent accuracy" anywhere; write "element selection given 16
  candidates" and "run-success judgment from the action log".
- **Cascade caveat carries over:** escalation goes to the same zero-shot teacher that made the labels.
  The `selective` table is the honest one for the human-in-the-loop story, and it assumes the human
  is right.
