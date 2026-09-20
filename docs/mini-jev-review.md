# mini-jev review: what to borrow, what to skip, what to send back

Reviewed: [r-ms/mini-jev](https://github.com/r-ms/mini-jev) at commit `ca61219` (2026-09-18; 6 commits,
one author, MIT). Read in full: `README.md`, `PREREG.md` (v1 to v1.3), `DEFERRED.md`, `docs/ARTICLE.md`,
`analysis/ANALYSIS_C.md`, all of `minijev/`, `scripts/run.py`, `scripts/run_cost.py`, `scripts/smoke_*.py`,
`demo/server.py`, `tests/test_harness.py`, and the file list of their Hub dataset
`Mikhail/mini-jev-runs`. Every "theirs" pointer below is `file:line` in that checkout. Nothing in our repo
was changed, no teacher was called; the two measurements on our own data below are read-only passes over
`runs/*/teacher.jsonl`.

## One paragraph

They and we read the same number: the next-token logit of an option letter at the answer position of a
lettered multiple-choice prompt, on a frozen Qwen. They serve that read directly from a 4B model and measure
it against grammar-constrained JSON on CLINC150; we take the same read from a 27B model as a *training
label* and ship a 140M student. Our teacher request (`max_tokens: 1` + `structured_outputs: {choice:
letters}`) is literally their arm `A1letter`, which they show is the same decision as their logit read
`B1` on 13 595 / 13 600 questions (ARTICLE §2). So their mechanism results transfer to our labels one to
one; their cost results (frozen-model latency) do not transfer at all; their calibration story is empty by
their own declaration (`DEFERRED.md:14-21`); their evaluation discipline is stronger than ours on paper
(a frozen preregistration with metric definitions, non-inferiority margin, STOP rules, disclosed peeks)
and weaker in the record (see Q3). The things worth taking are small: a mechanism probe, a resume guard,
a shortlist miss rate, discordant counts, a prereg template. Nothing here changes our prompt.

## The five questions

### Q1. Prompt and logit handling

**Theirs.** System prompt "You are a classifier ... reply with exactly one capital letter" (`prompts.py:8-11`);
user turn is `TEXT:\n{text}\n\nQUESTION: {q}\nA = opt\nB = opt\n...\n\nANSWER:` (`prompts.py:32-36`), the
question deliberately *after* the text so the text's KV cache can be shared across fields
(`engine.py:210-246`, `broadcast_last_hidden`). Letter ids: bare `A`..`Z` and `" A"`..`" Z"`, both asserted
single-token (`letters.py:21-32`); candidate logits recomputed in fp32 from the last hidden state so a bf16
tie does not become a letter-A bias (`letters.py:1-8, 35-39`); the read is `softmax` over the k bare letters
(`p_cand`) plus a merged bare+space variant (`letters.py:52-68`). They also record `candidate_mass`, the share
of the *unconstrained* next-token distribution the k letters hold (`scripts/run.py:181-207`), and `argmax_class`
(is the unconstrained argmax a candidate letter at all, `letters.py:42-49`). k > 26 is refused
(`letters.py:15-18`). Position bias: measured only, not corrected: a rotation arm `B1perm` over 50 texts at
k = 4, 8 (`runner.py:39-50`) reports the chosen-position histogram (flat on the 4B: 0.275/0.245/0.240/0.240
at k = 4), accuracy by gold position and per-row agreement across shifts (3 / 50 texts flip,
`ANALYSIS_C.md` Q1). Their answer to option order is "one letter, fixed order" and a diagnostic, exactly ours.

**Ours.** Question + options in the system prompt, text as the user message (`teacher.system_prompt`,
`teacher.Teacher.ask`); vLLM prefix caching then shares the constant system prompt across every example of a
task, which is the right prefix to share for labeling thousands of rows with one question (theirs shares the
text across fields of one request, the right prefix for their multi-field serving; both are correct for their
own job). Bare and spaced letters are stripped and log-summed (`teacher.parse_logprobs`), equivalent to their
`p_merged`. K > 19 goes through the chunked shortlist with `Z: none of the above`, which they do not have.

**One thing we cannot see and they can.** In every one of our `teacher.jsonl` rows the letter logprobs
sum to exactly 1.0 (checked on agnews, kinopoisk, headlines, georeview, toxic, m2w-element, swde-field,
3000 rows each, `min = max = 1.0000`): vLLM returns *processed* logprobs, i.e. post-mask and renormalised.
Two consequences. (a) "letters missing from `top_logprobs` get 0" (`task-spec.md`) has never fired on any
task: all K letters were always present. (b) `candidate_mass` and letter emission are unobservable through
our constrained call. Their smoke model (Qwen3-0.6B) had a near-total letter-A prior and candidate mass
min 0.0158 (`PREREG.md:130-133, 289`); under our recipe that teacher would produce confident-looking
renormalised garbage and nothing in `check --probe` would say so. This is the one prompt/logit item worth
adopting (A1 below). Everything else in their prompt handling is either already ours or is for their job.

### Q2. Calibration and confidence

They do not calibrate and say so: the shares are "normalized candidate scores, never the probability that
the answer is right" (`README.md:31, 106`; `DEFERRED.md:14-21`); a reliability diagram is listed as "an hour
of offline work" not done. Their confidence is TypeSafe's own `(max p − 1/k) / (1 − 1/k)` copied verbatim
(`letters.py:71-82`), used only to score the `A1prob` arm; abstention is "use the gap" with no threshold
measured. Ours is strictly ahead: temperature or `logits / T + b` with the 2-fold held-out NLL guard
(`calibrate.select_method`), the declared-`prior:` no-gold path (`calibrate.fit_prior_bias`), ECE/Brier
reported, cascade parity and selective tables (`evaluate.cascade`, `evaluate.selective`), split-conformal
sets (`serve.pred_set`). Nothing to import. Their one relevant remark, "report calibration per
`candidate_mass` stratum, low mass is not high confidence" (`DEFERRED.md:19-21`), is moot for us because mass
is 1 by construction of the constrained call (Q1). Their confidence formula is worth one documentation
line (S10) because our `/v1/systemone` mirrors Jev's shape and a reader comparing `confidence` across the
two APIs should know ours is raw max-p.

### Q3. Evaluation discipline

**What they preregistered** (`PREREG.md`): the decision (B1 non-inferior to A1joint on `intent`, margin
−3.0 pp on the lower 95 % bound, §4 P1); sixteen metrics each with numerator, denominator, unit and matching
rule (§3); a cost criterion (P2) and mechanism criteria (P3: letter emission ≥ 0.95, zero enum violations,
free-text control mismatch ≤ 2 %); ten STOP rules that void the study (§5); an explicit "what was read before
the freeze" table that discloses a 64-row accuracy peek (§0); the corpus, seed and revisions; and four dated
amendments, each stating what had been read when it was written and what changed (v1.1: one rung was an
identity, not a measurement; v1.2: candidate mass added after external review, corpus enlarged, "generate vs
read" renamed because it was causally wrong; v1.3: a harness defect found by the demo, with decision rules for
which claims survive *written before the re-run was read*). Uncertainty: cluster bootstrap over `text_id`
(`scoring.py:78-102`, 2000 resamples) plus the McNemar minimal-detectable half-width from the discordant
counts (`scoring.py:105-113`), so "the design cannot resolve the bound" prints as INCONCLUSIVE rather than as
"no difference" (`report.py:115-120`); two frozen batch orders as the noise band (§4); total-variation
distance between distributions, not only argmax flips (v1.2 §E). Every test names the placebo that would
turn it red (`tests/test_harness.py:1-2`), and oracle engines travel the real run path (`oracles.py:1-11`).

**Where the record is weaker than the protocol.** The README's headline table (`README.md:15-21`) cites
runs `b1`, `b2`, `b3` for the JSON-parity, letter-vs-name (+10.0 pp) and write-probabilities (0.346)
results. Amendment v1.3 (`PREREG.md:353-406`) states that every multi-token generation arm in those runs
ran with the position-ids defect and is "under review, b4", and `ARTICLE.md:5, 51, 75, 104` marks those
numbers *provisional*. The README, committed after v1.3, carries no such flag, and neither the repo nor the
Hub dataset (files: `read_letters*`, `free_text_control`, `cost_by_length_b2/b3`; no `A1joint`, `A1label`,
`A1prob`, no `b4`) contains a corrected generation-arm record. The reading-arm results (mechanism fires,
candidate mass, position prior, noise bands, shared-prefix agreement) are unaffected and solid. The comparison
against JSON, which is the README's first line, is by their own protocol not yet claimed. `ANALYSIS_C.md`
also documents an earlier published control number (189 / 200) that was a wrong-prompt join, found and
retracted by a same-`prompt_sha` re-analysis, which is to their credit and a good pattern.

**Against ours.** We have the claim rule (CI excludes 0, `experiments.md` header), three seeds where a
training factor varies, paired row bootstrap with the same resampled rows in both arms (`evaluate.paired_ci`),
and at least one decision rule written before the numbers (the permutation-averaging rule, `experiments.md`
"decision rule"). Our eval rows are one per text, so a cluster bootstrap would collapse to ours; nothing to
change there. What we lack: a single frozen file per experiment stating metric definitions, the margin that
would count, the STOP conditions and what was peeked; and the discordant counts / resolvable half-width next
to a "no measurable difference" verdict, so an underpowered comparison is not read as a null result. Both
are cheap (A6, A7).

### Q4. Engineering to lift

- **Mechanism probe before accuracy** (`scripts/smoke_letters.py`, `report.py:27-57` "read BEFORE any
  accuracy"): the cheapest thing a newcomer can learn about a new teacher is whether it wants to answer with
  a letter at all. A1.
- **Resume refusal on a changed certified profile** (`manifest.py:26-40`; per-template hashes so adding an
  unused template does not invalidate records, `prompts.py:69-75`, `tests/test_harness.py:209-220`). We
  resume `teacher.jsonl` by id only: change `question:` or the served model and `openjev run` silently reuses
  labels made under the old prompt. A2.
- **Truncated-tail repair** (`records.py:31-50`). Our `read_jsonl` drops undecodable lines, but the
  partial line stays on disk and the next append glues a new record onto it, losing both. A5.
- **Discordant counts and minimal detectable delta** (`scoring.py:105-113`). A6.
- **Run-mode string on every record** (`config.py:88-96`), versions in the manifest. A8.
- **Shared-prefix broadcast** (`engine.py:210-246`) and the **teaching bench** (`demo/server.py`): a page that
  shows the exact prompt, the top-10 next tokens at the answer position, the letter scores, and what the
  grammar allowed per written token. Good pedagogy for their question. For ours, `openjev check --probe` is
  the bench, and A1 gives it the one panel it lacks (the raw top tokens). No HTML.
- **Pinned everything** (`pyproject.toml`: torch/transformers/xgrammar exact; model and dataset revisions
  and sha256 of the domain map, `config.py:7-16`, `data.py:19-41`). Right for a one-shot study, wrong for our
  install story; record versions instead of pinning (A8).
- **Tests**: 29 guards, each with a named placebo; two oracle engines; a two-sided smoke for the generation
  defect (`scripts/smoke_positions.py`, refuses to pass when the placebo does not loop). We have 24 tests
  and `test_end_to_end_tiny` is already an oracle-gold run; adopt the placebo docstring convention and add
  the oracle-next arm (A4).

### Q5. What they measured that confirms or contradicts us

| their measurement | ours | verdict |
|---|---|---|
| `A1letter` (one letter under a grammar) and `B1` (read the logit) agree 13 595 / 13 600; the 5 differ at fp32 gaps ≤ 0.15 (ARTICLE §2) | our teacher call *is* `A1letter` | confirms: our soft label is their `p_cand`; nothing between "generate one constrained letter" and "read the logit" |
| letter beats writing the option's name: +10.0 pp intent, +13.2 domain, nil on `true/false` (README:18; provisional per v1.3) | we use letters, and `noul` options are `true/false` | supports our recipe; the boolean row says the letter buys nothing there, consistent with our `noul` working fine |
| scoring option names by sequence likelihood (`B1score`, LMQL-style) is worse than both, 0.443 on intent, and two length normalisations disagree by 12 points (ARTICLE §3.3) | never done | confirms not to |
| the model *writing* probabilities (`A1prob`, TypeSafe's adapter form) scores 0.346 vs 0.896; 62 % pick the first option (§3.6, provisional) | we read logits; we also have the real Jev API: kinopoisk 0.694 vs Qwen 0.652, georeview MAE 0.502 vs 0.619, probabilities quantised to 0.01, 60 % of kinopoisk rows > 0.999 (`findings.md`) | confirms reading over writing; their "TypeSafe form" is the published adapter shape, not the product, and should not be read as a Jev number |
| position prior flat at k = 4, 8 on real texts; 3 / 50 rows flip under rotation | reversing order moves kinopoisk teacher acc 0.646 → 0.693 and the *Good* rate 0.564 → 0.419; *Neutral* stays starved in both orders | not a contradiction, different regime (English 15-token utterances vs Russian 1500-char reviews); more important, a flat chosen-position histogram cannot see a semantic marginal shift, which is our dominant error. Their diagnostic would have passed our kinopoisk teacher |
| letters pick "none of the above" less than JSON on out-of-scope texts (41 vs 47 / 50; "none loses its long name and becomes a one-token candidate") | our chunked shortlist rests on a `Z: none` letter per chunk | measured on our data: on banking77 the gold option is outside the final shortlist in **272 / 5500 rows (4.9 %)**, capping the teacher at 0.951 before it answers; `Z` wins the chunk 77 % of the time when gold is absent and 14 % when it is present. swde-field: 0 / 6500 misses. Real, bounded, and now a number (A3) |
| a boolean that depends on an earlier field loses 5.3 pp asked alone; the apparent JSON edge on `domain` was a prompt leak (`ANALYSIS_C` Q2) | single-field tasks only | no action; if a multi-question task type ever exists, feed the earlier answer into the later question |
| 0.6B smoke model: near-total letter-A prior, candidate mass min 0.016 | invisible under our constrained call | motivates A1 |
| cost: 0.24× JSON at 32 tokens; loses at 2048 tokens without the shared prefix; bit-identical across two 4090s; 0.42 % flips Mac vs CUDA | our cost is teacher-once vs student at 6 ms; our noise band is seeds (0.6–1.2 pts) | not comparable; nothing to import |

## Ranked integration plan

Effort: S = under an hour, M = half a day, L = more. Every item is post-hoc or probe-time; none changes a
shipped label or the prompt.

### Adopt

**A1. Mechanism probe: letter emission and candidate mass in `openjev check --probe`.**
Theirs: `letters.py:42-49` (`classify_argmax`), `scripts/run.py:181-207` (mass from the model's own head),
`scripts/smoke_letters.py`, PREREG S3 gate (letter emission < 0.90 on the first 64 rows → stop).
Ours: `openjev/check.py:probe()` and `openjev/teacher.py:Teacher.ask()`. Add an unconstrained variant of the
same request (no `structured_outputs`, `top_logprobs: 20`) on the probe rows and print, *before* the accuracy
line: share of rows whose top token is one of the letters; median and min of `sum(exp(logprob))` over the
letters; the top-5 tokens for the three printed examples. Warn below 0.90 emission or 0.5 mass ("this teacher
does not want to answer with a letter; the constrained labels will look confident anyway"). Because vLLM
returns processed logprobs, this must be a separate unconstrained call; cost is the same 100 calls the probe
already makes. Effort S. Accept: `openjev check tasks/agnews.yaml --probe 100` prints `letter emission
100/100, candidate mass median 1.000 min 0.99x` on the production teacher, and a monkeypatched response with
top token `\n` triggers the warning in `tests/test_check.py`.

**A2. Refuse to resume `teacher.jsonl` under a changed prompt or model.**
Theirs: `manifest.py:26-40` (`refuse_resume_on_mismatch`), `prompts.py:69-75` (per-template hashes),
`tests/test_harness.py:209-228`. Ours: `openjev/teacher.py:run()` writes `label.json`; add `prompt_sha`
(sha256 of `system_prompt(task, range(K))`, the `with_none` variant, and `max_options_per_call`) and keep
`model`. On a resume with rows present, exit with the two values when either differs, unless `--force label`
(which today only re-walks the stage; make it also mean "relabel into a fresh file after backing the old one
up"). `check` prints the stored hash next to the live one. Effort S. Accept: edit `question:` in a copied
task → `openjev label` refuses with the diff; revert → resumes with "N cached".

**A3. Report the shortlist miss rate for K > `max_options_per_call`.**
Theirs: the out-of-scope finding (README:19, ARTICLE §3.5). Ours: `openjev/evaluate.py:run()` teacher block
and `check.py:probe()`: from `raw.final` and `gold`, `teacher.shortlist_miss = mean(gold not in final)`, plus
the two `p(none)` medians (chunk with gold / chunk without). Measured today: banking77 0.049, swde-field 0.000.
Effort S. Accept: `results/banking77.json` carries `teacher.shortlist_miss = 0.049`; the README note line for
a K > 19 task mentions it; `findings.md` "K > 19 is approximate" gets the number.

**A4. Placebo-named tests and the oracle-next arm.**
Theirs: `tests/test_harness.py:1-2` (docstring rule), `oracles.py` (gold-piked and next-piked fake engines
through the real path). Ours: `tests/test_core.py:test_end_to_end_tiny` is an oracle-gold run already; add
the oracle-next placebo (teacher rows whose probs point at `(gold+1) % K` → eval accuracy ≈ 0, agreement ≈ 1),
and one test through `parse_logprobs → softmax_letters → probs` with a saved response whose top letter is `B`
asserting `argmax == 1`. Adopt the convention: every new guard's docstring names the mutation that makes it
red. Effort S. Accept: two new tests; `pytest -q tests` stays under a minute on CPU.

**A5. Do not glue a new record onto a truncated last line.**
Theirs: `records.py:31-50`. Ours: `openjev/teacher.py:run()` opens `teacher.jsonl` in append mode; before
appending, if the file is non-empty and does not end with `\n`, write `\n` (the partial line is then dropped
by `read_jsonl` as today and the new rows survive). Effort S, three lines. Accept: a test truncates the last
line mid-JSON, appends one row, reads back `n-1 + 1` rows.

**A6. Discordant counts and resolvable half-width in `openjev compare`.**
Theirs: `scoring.py:105-113` (`minimal_detectable_delta`), `report.py:115-120`. Ours:
`openjev/evaluate.py:compare()`: for `acc` tasks print `discordant: A-only-right b, B-only-right c;
resolvable ±1.96·sqrt(b+c)/n` per seed, and let the verdict say `no measurable difference (resolvable to
±x pts)` so a null with wide resolution reads as inconclusive. Effort S. Accept: `openjev compare
runs/headlines runs/headlines-nosynth` prints the b/c counts and the half-width.

**A7. A preregistration file for the next experiment.**
Theirs: `PREREG.md` §0 (what was read before the freeze), §3 (metrics with numerator/denominator/unit),
§4 (positive result with the margin), §5 (STOP), amendments dated with "what had been read". Ours:
`docs/prereg/TEMPLATE.md` (~25 lines) and one rule added to the header of `experiments.md`: a comparison
enters `findings.md` only if its prereg file's commit predates the arm's `label.json` / `train.json`
timestamps, and every amendment states what had been read. Not retrofitted to past runs. Effort S to write;
the cost is the discipline. Accept: the next `experiments.md` section links its prereg and the timestamps
check out.

**A8. Record the environment in `results/*.json`.**
Theirs: `config.run_mode():88-96`, `manifest.CERTIFIED`. Ours: `evaluate.run()` already writes `git`; add
`env: {torch, transformers, student_model, student_revision (from the HF cache commit if available),
teacher_model (from label.json), label_prompt_sha (from A2)}`. Effort S. Accept: the key appears in the next
`results/<run>.json`; `openjev report` ignores it.

### Adapt

**B1. Per-row rotation stability from the runs we already have.**
Theirs: `report.py:218-231` (position prior) and PREREG v1.2 §G (three separate readouts: chosen-position
histogram, accuracy by gold position, agreement across shifts). Ours: `scripts/perm_check.py` reports
accuracy, marginal and max-p for original / reversed / averaged; add the per-row flip rate between the two
orders and accuracy by gold position, computed from `runs/kinopoisk-rev` and `runs/headlines-rev` with zero
new teacher calls. It answers what fraction of the marginal move is near-tie rows (their 3 / 50) versus a
whole-class shift. Effort S. Accept: `python scripts/perm_check.py kinopoisk` prints `rows flipped under
reversal: x/3500` and a 3-row accuracy-by-gold-position table without contacting the teacher.

**B2. Text-first prompt layout, chunked path only, measured before merged.**
Theirs: question after the text (`prompts.py:32-36`) so the text prefix is shared across the fields of one
request (`engine.py:210-246`; README "many fields on one long text"). Ours: for K > 19 the three calls per
row share the text but not the system prompt, so vLLM's prefix cache reuses nothing across them; putting the
text first would share it and cost the cross-example sharing of the system prompt. Only the chunked path
could gain, only on long texts, and it changes the prompt (new `prompt_sha`, labels not comparable). Do it
only as an A/B: label 200 swde-field rows both ways, compare ex/s and argmax agreement. Effort M. Accept:
≥ 1.5× ex/s on the chunked path with agreement ≥ 0.95, else drop it and record the number.

### Skip

- **S1. Frozen-model serving, KV broadcast, the cost sweep** (`engine.py`, `scripts/run_cost.py`): we ship
  students; vLLM's automatic prefix caching does the sharing on the teacher side.
- **S2. xgrammar and every generation arm** (`grammar.py`, `A1*`, `A0`): our pipeline generates nothing.
- **S3. fp32 recompute of candidate logits** (`letters.py:1-8`): server-side in vLLM, not ours to change, and
  their measured effect is 5 / 13 600 argmax differences at gaps ≤ 0.15.
- **S4. `p_merged` over bare and spaced letters** (`letters.py:61`): `teacher.parse_logprobs` already strips
  and log-sums duplicates.
- **S5. Cluster bootstrap** (`scoring.py:78-102`): their unit is a field-question nested in a text; ours is
  one row per text, so a cluster bootstrap is our row bootstrap. `compare` already pairs rows across arms and
  seeds.
- **S6. Their calibration and confidence story**: there is none (`DEFERRED.md`), and the gap-based abstention
  is unmeasured. Ours has T + per-class bias with a held-out guard, a no-gold prior path, ECE/Brier, cascade,
  selective and conformal sets.
- **S7. k > 26 handling**: they refuse it; our chunked shortlist exists and its cost is now measured (A3).
- **S8. The teaching-bench web page** (`demo/index.html`, `demo/server.py`): the right artefact for their
  "look at the mechanism" question; for a newcomer to *our* repo the bench is `check --probe`, and A1 adds
  the one panel it lacked. No HTML, no second server.
- **S9. Exact dependency pins**: hurts `uv pip install -e .` on a fresh box; A8 records versions instead.
- **S10. TypeSafe's rescaled confidence** (`letters.py:71-82`): changing our `confidence` would move every
  `escalate_below` and `selective` threshold for a cosmetic gain. Add one line to `serving.md`: ours is raw
  max-p; Jev's is `(max p − 1/k)/(1 − 1/k)`; convert before comparing.
- **S11. Sequence likelihood of option names, model-written probabilities**: measured worse by them;
  we never did either.
- **S12. Multi-field dependent questions**: not our task shape; note kept in Q5.

## What we should send them back

1. **The marginal-shift diagnostic and the per-class bias result.** A flat chosen-position histogram is not
   an unbiased teacher: on kinopoisk the 27B starves *Neutral* in both option orders (12.9 % / 12.0 % vs
   gold 33.7 %) and the histogram cannot see it; a predicted-vs-gold marginal per field can. And their
   "normalized candidate scores" become usable probabilities with 500 labels and CPU seconds:
   `logits / T + b` moved our students +5.1 (kinopoisk) and +7.5 pts (headlines), CI excluding 0, and the
   bias is exactly the shape of the error. Their run records (`p_cand`, `gold_pos`) are enough to fit it.
2. **vLLM returns processed logprobs under structured outputs.** Their `candidate_mass` is invisible through
   the OpenAI-compatible API unless the server is started in a raw-logprobs mode; measured on 21 000 of our
   rows, the letter logprobs sum to 1.0 exactly. Relevant to their stated follow-up on production latency.
3. **Per-row candidate letters do not distill.** Reading letters over "which of these 16 elements" works on
   a frozen model (their setting, our teacher: 0.649) and collapses to chance in a fixed-head student
   (0.059 against 0.0625): slot `G` means something different on every row. Their "extraction-as-choice"
   recommendation (README:35) is a frozen-model pattern and should say so for anyone distilling.
4. **Real Jev numbers.** Their `A1prob` (0.346) is TypeSafe's published adapter *shape* on a 4B; the product
   scores 0.694 on kinopoisk against our Qwen 27B's 0.652 and MAE 0.502 vs 0.619 on georeview, with
   probabilities quantised to 0.01 and 60 % of rows above 0.999. Worth a footnote so the 0.35 is not read
   as a Jev result.
5. **The "none" letter, quantified for a shortlist.** Their OOS finding predicts that a one-token "none"
   under-fires; on banking77 our chunked shortlist loses the gold option in 4.9 % of rows.
6. **A question, not a finding:** the README cites runs b1/b2/b3 for the generation-arm results that
   PREREG v1.3 places under review pending b4; nothing named b4 is in the repo or the Hub dataset. Did the
   re-run land, and if so, where?

## Licence and attribution

mini-jev is MIT, `Copyright (c) 2026 Mikhail Rakutko`. Reimplementing an idea (A1, A2, A3, A6, A7, A8, B1)
carries no obligation; a courtesy line in `docs/findings.md` or the changelog ("probe and resume guard after
r-ms/mini-jev") is the right citizenship. Copying code, verbatim or lightly edited (candidates:
`scoring.minimal_detectable_delta`, `records.Writer._load_existing`, `letters.classify_argmax`), requires
keeping their copyright and permission notice with the copied portion: a header comment naming the source
file and commit `ca61219`, and a `THIRD_PARTY_NOTICES.md` with the MIT text, since our own `LICENSE` covers
only our copyright. Do not copy `letters.choice_confidence`: by their own note it is itself copied from
TypeSafe's `system-one-adapter`, whose licence would have to be checked first, and S10 says we do not want it
anyway.
