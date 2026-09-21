# README figures

Illustrations for the top-level README (`pipeline`, `response-card`, `student-vs-teacher`, `terminal`),
for `README.ru.md` (the same set), for `docs/findings.md` (`calibration`, `cascade`, `cost`)
and for `docs/use-cases.md` (`use-cases`).
The `srcset`/`src` paths below are written for a file at the repo root; from inside `docs/` drop the
`docs/` prefix. All hand-written SVG — no matplotlib, no
browser, no fonts to download. Regenerate with:

```bash
python docs/img/make_figures.py
```

Every number is read out of `results/*.json` and `runs/agnews/*.json` at generation time, so
the figures follow the repo instead of drifting from it. Re-run the script after any `openjev
eval`. The only hand-written parts are the labels, the prose captions and the log-line
*formatting* in `terminal.svg` (the values in it are real; the `calib:`/`eval:` prefixes are
illustrative — `calibrate` prints nothing today and `evaluate` prints a markdown table).

Each figure exists as a `-light` / `-dark` pair. They use presentation attributes only
(`fill=`, `stroke=`), no `<style>`, no external fonts, and a generic font stack, so they
survive GitHub's inline-SVG sanitizer and render at 900px wide.

## Embedding

GitHub picks the right one with `<picture>` (works in README.md; the `srcset` paths are
relative to the file doing the embedding):

```markdown
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/img/pipeline-dark.svg">
  <img alt="open-jev pipeline" src="docs/img/pipeline-light.svg" width="900">
</picture>
```

`![...](docs/img/pipeline-light.svg)` alone works too — it just shows a white card on dark theme.
`terminal.svg` is one dark file, embedded as a plain image.

## The figures

| file | what it shows | data |
|---|---|---|
| `pipeline-{light,dark}.svg` | The whole project in five boxes: YAML task → Qwen teacher (one constrained letter, softmax over `top_logprobs`) → mmBERT-small student trained on KL → `logits / T + b` calibration → typed JSON at `/v1/systemone`. Carries the "paid once vs paid per call" split as a caption. | `teacher_calls`, `teacher_minutes`, `latency_ms`, `throughput_gpu_b64` summed/ranged over the six base runs |
| `student-vs-teacher-{light,dark}.svg` | Grouped bars, student accuracy against teacher accuracy on seven tasks. The zero-gold `agnews-nogold` run is highlighted and scored honestly (offline, against real ag_news labels), and `toxic` carries the "accuracy is the wrong metric on an 8%-positive split" footnote. | `student.acc` / `teacher.acc`, plus `gold_acc_offline` / `teacher_gold_acc_offline` from `results/variants/agnews-nogold-goldacc.json` for agnews-nogold; headlines is the mean of its three seeds |
| `cascade-{light,dark}.svg` | Escalation rate vs accuracy. One curve per task that has one, with each teacher's accuracy as a dashed line and the best point annotated. The story: on agnews, kinopoisk and headlines-8k escalating a tenth of the rows buys +0.7 to +1.3 points and then turns back down; on headlines the student is 6 points above its teacher, so escalation only costs; banking77 is the one task still behind its teacher, and escalating a third of it gains +2.8. | `cascade[]` and `cascade_parity` in `results/*.json` — accuracy tasks only (georeview is MAE, toxic AUROC, agnews-nogold is scored against the teacher), one series per task, seed 0 |
| `calibration-{light,dark}.svg` | Dumbbell chart of ECE before → after calibration (`logits / T + b`), every run, sorted by raw ECE. Green = ECE improved, red = ECE got worse — the fit is selected on held-out NLL and accuracy, not on ECE, so headlines gains 7.5 accuracy points while its ECE goes 0.024 → 0.036. Each row carries its fitted `T`. | `student.ece_raw`, `student.ece_cal`, `temperature`, `calib_target` |
| `cost-{light,dark}.svg` | Two panels: teacher minutes per task (paid once, 66,522 calls / 2.4 h, banking77 is 42% of it) against student batch-1 p50 latency per task (paid per call). Ends on the amortised 130 ms/label vs 5.2 ms/call comparison. | `teacher_calls`, `teacher_minutes`, `latency_ms.gpu_b1_p50`, `throughput_gpu_b64` |
| `use-cases-{light,dark}.svg` | Small multiples for the five agent-shaped tasks: one panel per task, each with its own metric named in it (`arb-success` AUROC, `m2w-target` AUROC, `swde-field` macro-F1, `m2w-element` accuracy, `arb-quality` MAE) and its own axis, student vs teacher inside the panel only. The five metrics are not comparable, so they never share a bar scale. `m2w-element` is marked as the failed task and carries its `1/16 = 0.0625` chance floor as a dashed line; `arb-quality`'s MAE bars hang from the top with a "better ↓" arrow. Panels say `n=3 seeds (mean)` or `n=1 seed`. | `noul.auroc` / `noul.teacher_auroc`, `student.macro_f1` / `teacher.macro_f1`, `student.acc` / `teacher.acc`, `score.mae` / `score.teacher_mae`, `k` and `eval_n` from `results/{arb-success,arb-quality,swde-field,m2w-target,m2w-element}.json`; the three-seed panels average the `-s1`/`-s2` files |
| `response-card-{light,dark}.svg` | The actual request and typed JSON response, with callouts on `choice`, `probabilities` and `confidence`, and the `score`/`noul` shapes from `openjev/views.py`. | the README quickstart's verbatim request/response; probabilities abridged to 4 dp |
| `terminal.svg` | Animated (SMIL) terminal: `openjev run tasks/agnews.yaml` types itself out, the stage logs appear, then a `curl` and the served JSON. Dark only — it's a terminal. | `runs/agnews/{label,train,calib}.json` and `results/agnews.json` |

## On the animated one

`terminal.svg` animates with SMIL (`<animate>`, `<set>`) — no JS, no CSS, no `<style>`.

- Referenced as an image (`![](docs/img/terminal.svg)` or `<img src=...>`), GitHub serves the
  file as its own document through camo and does **not** sanitize it, so the animation plays.
  This is the same mechanism every `readme-typing-svg` badge uses.
- Pasted *inline* as `<svg>…</svg>` in Markdown, GitHub's sanitizer runs and `<animate>` /
  `<set>` are very likely stripped. That case is handled: every animated element's **base
  attributes are its final state**, so a stripped file degrades to a correct static screenshot
  of the finished run rather than to a blank box. Embed it as an image and you get the
  animation; embed it inline and you get the screenshot. Either way it is not broken.
- It runs once (~7 s) and freezes, apart from a blinking cursor at the end. 4.7 KB.
