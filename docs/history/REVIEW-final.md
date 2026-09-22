# Final review (2026-09-20, HEAD f02c4a0, clean tree)

Checked: README, docs/task-spec.md, PLAN.md, REVIEW.md, openjev/*.py, tasks/*.yaml, results/*.json,
git log, `runs/` artifacts read-only. `pytest -q tests` (CPU-forced): 14 passed in 21 s. `openjev --help`
and every subcommand's help work. README results table is byte-identical to a fresh `report()`. The
README curl example reproduces from `runs/agnews` (CPU gives 0.9863 vs 0.9872 printed — bf16/GPU vs fp32,
same choice). Nothing under runs/ was touched; the teacher was not contacted.

**Bottom line:** the work is sound and the numbers are real. Every number I could recompute from
`runs/` and git history matches the README (kinopoisk teacher 56.5 % Good / 12.7 % Neutral, 66,522 calls /
144 min / 42 % banking77, headlines 0.762→0.767 / 0.755→0.760, 5-epoch banking77 calib KL 0.407, the
idle-GPU re-measure at 03:02–03:11 after labeling ended 02:59). Two claims are overstated, one doc claim
is false, one documented behaviour crashes. Fix list below; the must-fix items are all one-liners.

## Must fix before the user reads this

1. **README "Findings" → toxic: "The student beat its teacher on toxic — accuracy 0.917 vs 0.893".**
   The eval split is 8.1 % positive (244/3000), so majority-class accuracy is **0.919**; the student's
   0.917 is the base rate, not a win, and macro-F1 is a tie (0.626 vs 0.624). The real win is AUROC
   0.856 vs 0.817 and Brier-vs-gold-fraction 0.035 vs 0.048. Also "The mechanism is ordinary — ...
   denoises a teacher that is itself poorly thresholded" is speculation stated as fact, and thresholding
   cannot move AUROC. Fix: lead with AUROC/Brier, say accuracy on an 8 % task is uninformative (majority
   0.919), and change "The mechanism is" to "One plausible mechanism is".

2. **README auto-notes contradict the prose on toxic.** The generated bullet says
   "**toxic**: ... distilled from the teacher with zero gold labels" and three paragraphs later the README
   says toxic "is not a zero-gold row". `openjev report` will regenerate the bullet, so hand-editing it
   is pointless. Fix in `openjev/evaluate.py`: in `run()` add `"balanced_train": task.data.train.balance`
   to `res`; in `report()` replace the `gold = "zero gold labels" ...` phrase with
   `gold = "no gold in training" + (" (train rows balanced by gold)" if r.get("balanced_train") else "")`,
   add `"balanced_train": true` to `results/toxic.json` (and `false` to the others, or leave absent), and
   re-run `openjev report`. This is also REVIEW.md C12, the one review item never applied.

3. **The teacher is never named.** README says "a local Qwen teacher" / "a local Qwen3 served by vLLM";
   results JSON and label.json record no teacher model at all. The teacher is the ceiling for every row,
   so "which teacher" is the single most load-bearing fact in the repo. Per the serving note the box runs
   Qwen3.8-27B 3-bit GSQ (ISTA-DASLab) + MTP, served under the id `Qwen/Qwen3.8-27B-FP8`. Fix: one line
   under Results ("Teacher for every row: Qwen3.8-27B, 3-bit GSQ quantisation, vLLM, `max_tokens 1`,
   `enable_thinking false`"), and in `teacher._run` write `"model": model` into `label.json` so future
   runs carry it.

4. **docs/task-spec.md `epochs` row: "early stopping on calib KL with patience 2 usually stops sooner".**
   False for every run in the repo: in all 7 `runs/*/train.json` the best epoch is the last epoch and
   calib KL was still falling (agnews 0.051→0.050, headlines 0.176→0.169, kinopoisk 0.123→0.113,
   georeview 0.145→0.131, banking77 0.351→0.347 at epoch 12). Fix: "Early stopping (patience 2) never
   triggered in the seven runs here; calib KL was still falling at the last epoch in every one, so
   `epochs` is the binding knob — raise it." Correspondingly, README Findings "The default epoch count is
   too low for large label sets" → "too low, full stop; it was still falling at epoch 5 on every task".

5. **docs/task-spec.md (noul example) + tasks/toxic.yaml comment: "calib and eval stay at the natural
   rate, so the temperature is fitted under the prior it is evaluated on".** Not true: the balanced train
   draw takes 2000 positives out of the same 60k pool *first*, so the calib split has **11/500 = 2.2 %**
   positives vs **8.1 %** on eval; T = 0.25 was fitted on 11 positives. It happened to work (eval ECE
   0.116→0.037) but the sentence is wrong. Fix: say "calib is drawn after the balanced train split from
   the same pool, so its positive rate (2.2 %) is *below* the eval rate (8.1 %); T = 0.25 was fitted on 11
   positives and still improved eval ECE, but this is a weakness of drawing calib from a depleted pool".
   (Code fix, later: draw calib from a different slice, e.g. `train[60000:70000]`.)

6. **Documented behaviour that crashes: `rubric.descriptions` shorter than `levels`.** Spec says "(just
   `<level>` if `descriptions` is absent or shorter than `levels`)"; `teacher.option_lines` zips and the
   prompt then does `lines[i]` → `IndexError` at label time (verified). Fix, `openjev/teacher.py:27`:
   `descs = list(task.rubric.get("descriptions") or []); descs += [""] * (task.k - len(descs))`.

7. **REVIEW.md has no status banner.** It opens with "Confirmed bugs A1–A4" against code at 025fa84;
   a reader cannot tell they were fixed in fa67ccf. Fix: one line at the top like PLAN.md's:
   "> Historical: review of 025fa84. A1–A4, B1–B4, B7–B9, C1–C4, C7, C8 and D were fixed in fa67ccf and
   later; still open: B5 (label.json `n_failed` overwritten, stats only on clean exit), B6 (row order
   depends on teacher completion order — seeded, not bit-exact), C11, C12."

8. **No LICENSE, and `pyproject.toml` has no `license`/`description`/`readme`.** If this is going
   anywhere public, pick one (MIT/Apache-2.0) and add the file; the README says "open reimplementation".

## Nice to have

9. **banking77 shortlist caveat, quantified.** README says "if every chunk misses the true class the
   label is simply wrong" but never says how often. Measured from `runs/banking77/teacher.jsonl`: gold has
   probability 0 (outside the shortlist) on **4.55 % of eval rows** (91/2000), 4.6 % calib, 5.3 % train.
   So the teacher's 0.764 already includes a 4.6 % hard floor and the "teacher" row for banking77 is the
   shortlist pipeline, not the LLM's 77-way ability. One sentence in Findings and in Limitations.

10. **README Findings, banking77: "12 epochs moved accuracy 0.7425 → 0.7485".** The 5-epoch results
    file was overwritten before it was ever committed; 0.7425 is not reproducible from the repo (the KL
    0.407 is, from run.log). Either say "(5-epoch result not kept)" or drop the number.

11. **README Requirements memory numbers are stale.** "mmBERT-base is roughly 8 GB" — measured 9.8 GB
    (agnews). "mmBERT-small ... peaks around 5.5 GB" — true at len 256; at `max_len: 512`
    (georeview/kinopoisk) it is 8.9 GB. The georeview mmBERT-base run in `scripts/variants.sh` died after
    loading (`runs/georeview-mmBERT-base.log`), and kinopoisk-base / kinopoisk-gold never ran; not claimed
    anywhere, but the script suggests they did.

12. **`tasks/example.yaml`: `lang: en  # only affects the teacher prompt wording`** is wrong — the
    labeling prompt never sees `lang` (`teacher.system_prompt` ignores it); only `augment` uses it
    (`language "{task.lang}"`) plus the report column. Same softening needed in task-spec's `lang` row.
    For a third language nothing breaks (question/options are free text, mmBERT is multilingual), but the
    README should say so in one line: "any language the teacher and mmBERT cover; write `question` and
    `options` in whatever language you want the prompt in."

13. **Newcomer error messages** (all verified): missing `data:` → bare `KeyError: 'data'`; a gold string
    not in `options` → `ValueError: 'A' is not in list` (no hint it is the gold column); `balance: true`
    without `gold` is accepted and silently does nothing; `source: {hf: x, bogus: 1}` is accepted although
    the docs promise "unknown keys are a hard error in every section"; a CSV shorter than
    train+calib+eval silently yields short splits and an empty eval crashes in `torch.cat`. Each is one
    `raise ValueError` in `spec.load_task` / `data.examples`.

14. **README never says how to run the tests.** Add: "`pytest -q tests` — 14 tests, ~20 s on CPU; two of
    them need `jhu-clsp/mmBERT-small` and `fancyzhx/ag_news` in the HF cache."

15. **Provenance nit.** Every `results/*.json` says `"git": "52a3730-dirty"`; the code that produced them
    is what f02c4a0 committed. One sentence under Results, or accept.

16. **Path inconsistency.** `--runs` is cwd-relative; `results/` and `README.md` are repo-relative
    (`evaluate.ROOT`). Fine for an editable install from the repo root; say "run from the repo root".

17. **`.claude/agents/*.md` are committed.** Harmless build tooling (planner/coder agent definitions);
    keep if you want the workflow visible, otherwise `git rm -r .claude`.

18. **`evaluate.py:75/78`** uses `task.labels.index("true")` and then hardcodes `cal[:, 0]`; and
    `teacher._run` still overwrites `n_failed` instead of summing (REVIEW B5). Cosmetic.

## Verified fine (do not re-check)

- Table numbers, "One night, seven runs", 66,522 calls / 144 min / 42 %, 1.5–12 min training, 5.3 ms /
  869 ex/s, +0.4 pts for half throughput and 9.8 vs 5.5 GB, kinopoisk/georeview vs teacher, kinopoisk
  teacher prior (848/191/461 of 1500, gold 500/500/500), ECE deltas, headlines "kept 1.0", banking77
  ECE 0.017→0.028 while calib improved 0.065→0.052, augment 0.762→0.767 (argmax is temperature-invariant,
  so the T change between the two headlines runs does not confound it).
- The idle-GPU re-measure: labeling ended 02:59:40, augment training 03:01, evals 03:02–03:11 sequential
  ~40 s apart; latencies fell from 18–22 ms (all measured while vLLM was labeling) to 5.2–7.0 ms.
- Quickstart runs against the real CLI; `serve`/`report`/`label --limit` exist as documented; the
  augment write-up matches `augment.py` (targets, dedup, `synth:<round>:<i>`, normal label stage).
- Tree is clean, `.gitignore` covers `.venv`, `runs/`, `*.egg-info`, caches; nothing large or secret is
  tracked; git log is 13 legible, well-scoped commits.
