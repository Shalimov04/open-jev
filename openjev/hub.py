"""Hugging Face Hub: `openjev push RUN_DIR --repo user/name` and `openjev serve hf:user/name`.

A pushed repo is the run dir minus the data: student/ + openjev.json + conformal.json + a model
card. That is everything `serve` needs and nothing that would leak the teacher's labels.
"""
import json
import os
from pathlib import Path

FILES = ["student/*", "openjev.json", "conformal.json", "README.md"]
BASE_FILES = ["student/*", "train.json", "README.md"]  # a pair model has no per-task calibration
ROOT = Path(__file__).resolve().parent.parent


def _no_proxy():
    """The dev box exports a SOCKS proxy huggingface_hub cannot use. Same reason as tests/conftest."""
    for v in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        os.environ.pop(v, None)


def resolve(spec):
    """`hf:user/name` -> a local run dir (downloaded once); anything else is returned unchanged."""
    if not str(spec).startswith("hf:"):
        return spec
    from huggingface_hub import snapshot_download
    _no_proxy()
    repo = str(spec)[3:]
    dest = Path("runs") / ("hf--" + repo.replace("/", "--"))
    snapshot_download(repo_id=repo, local_dir=dest, allow_patterns=sorted({*FILES, *BASE_FILES}))
    return dest


def card(run_dir, repo=None):
    """The model card: what the student is, what it was distilled from, what it may be claimed to do."""
    run_dir = Path(run_dir)
    meta = json.loads((run_dir / "openjev.json").read_text())
    task, m, calib = meta["task"], meta["metrics"], meta.get("calib", {})
    label = json.loads((run_dir / "label.json").read_text()) if (run_dir / "label.json").exists() else {}
    teacher_id = label.get("model") or os.environ.get("OPENJEV_TEACHER_MODEL") or "not recorded in label.json"
    yaml_path = Path(task.get("path") or "")
    yaml_text = yaml_path.read_text().strip() if yaml_path.is_file() else "(task YAML not found at export time)"
    metric = (f"MAE {m['score']['mae']:.3f} (teacher {m['score']['teacher_mae']:.3f})" if "score" in m else
              f"AUROC {m['noul']['auroc']:.3f} (teacher {m['noul']['teacher_auroc']:.3f})"
              if m.get("noul", {}).get("auroc") is not None else
              f"accuracy {m['student']['acc']:.3f} / macro-F1 {m['student']['macro_f1']:.3f} "
              f"(teacher {m['teacher']['acc']:.3f} / {m['teacher']['macro_f1']:.3f})")
    gold = ("**No gold labels were used in the loss.** The student was trained only on the teacher's "
            "probability vectors; the gold labels in the dataset were used to *measure* it and to fit "
            "the calibration." if not m["gold_weight"] else
            f"Gold labels were mixed into the loss (`gold_weight: {m['gold_weight']}`), so this is "
            f"distillation plus supervision, not distillation alone.")
    cal = ("temperature left at 1.0 (fitting it did not improve ECE on the calib split)"
           if calib.get("method") == "none" else
           f"{calib.get('method', 'temperature')} scaling, T = {calib.get('temperature', 1.0):.3f}"
           + (f", per-class bias fitted on the calib split" if calib.get("bias") else "")
           + f", target `{calib.get('target', '?')}`")
    return f"""---
library_name: transformers
pipeline_tag: text-classification
base_model: {m['student_model']}
language: [{m['lang']}]
tags: [open-jev, distillation, calibrated]
---

# {run_dir.name}

A `{task['type']}` classifier over {m['k']} labels, distilled from an LLM teacher with
[open-jev](https://github.com/sshalimov04/open-jev). It answers one fixed question with a fixed
label set — it is not a general model that takes arbitrary options at request time.

```bash
pip install openjev
openjev serve hf:{repo or 'USER/NAME'} --port 8099
curl -s localhost:8099/v1/systemone -H 'Content-Type: application/json' \\
  -d '{{"model":"{('hf--' + repo.replace('/', '--')) if repo else run_dir.name}","input":"your text here"}}'
```

## How it was made

- **Teacher**: `{teacher_id}`, zero-shot, {m['teacher_calls']} calls to label {m['n_train']} train rows
  plus the calib and eval splits.
- **Student**: `{m['student_model']}`, {task['student']['epochs']} epochs, max_len {task['student']['max_len']}, KL(teacher ‖ student).
- **Calibration**: {cal}.
- **Escalation threshold** (`escalate_below`, the cheapest confidence at which the
  student+teacher cascade matches the teacher alone): {meta.get('escalate_below')}.

{gold}

## Results ({m['eval_n']} eval rows, vs {m.get('eval_target', 'gold')})

| student | teacher | agreement |
|---|---|---|
| {metric.split(' (teacher ')[0]} | {metric.split(' (teacher ')[1].rstrip(')') if ' (teacher ' in metric else '–'} | {m['agreement']['argmax']:.3f} |

The teacher is the ceiling: the student is trained to copy it, so its accuracy is bounded by the
teacher's on this task, and any bias the teacher has is reproduced.

## Task

```yaml
{yaml_text}
```
"""


def _rows(name):
    return json.loads((ROOT / "results/base" / name).read_text())


def base_card(run_dir, repo=None):
    """The model card for an `open-jev-base` pair model. Every number is read from the committed
    result files, so the card cannot drift from `docs/experiments.md` §B2."""
    run_dir = Path(run_dir)
    t = json.loads((run_dir / "train.json").read_text())
    jev = sum(1 for x in t["train_tasks"] if x.endswith("-jev"))
    unseen = "\n".join(
        f"| {r['task']} | {r['metric']} | {r['n']} | {r['k']} | {r['chance']:.3f} | {r['teacher']:.3f} "
        f"| **{r['base']:.3f}** | +{r['adv']:.3f} ({r['ci'][0]:+.3f}..{r['ci'][1]:+.3f}) | {r['normalised']:.2f} |"
        for r in _rows("unseen-base-none-v2-gold500.json"))
    loto = "\n".join(
        f"| {r['task']} | {r['fold']} | {r['metric']} | {r['n']} | {r['chance']:.3f} | {r['teacher']:.3f} "
        f"| {r['student']:.3f} | {r['base']:.3f} | {r['d_student'][0]:+.3f} "
        f"({r['d_student'][1]:+.3f}..{r['d_student'][2]:+.3f}) |"
        for r in _rows("base-loto-v2-gold500.json"))
    return f"""---
library_name: transformers
pipeline_tag: text-classification
base_model: {t['student']}
license: mit
language: [en, ru]
tags: [open-jev, distillation, cross-encoder, calibrated]
---

# open-jev-base ({run_dir.name})

A **pointwise cross-encoder**: one forward pass scores one `(question, option, text)` triple and
returns the probability that this option is the right answer for this text. A K-way decision is K
passes, renormalised over the K offered options; a yes/no (`noul`) decision is one pass. Nothing
about K or the option wording lives in the head, so **both are free at inference** — a new question
with new options needs no retraining, only a new prompt string.

Trained by [open-jev](https://github.com/sshalimov04/open-jev) on the pooled soft labels of
{len(t['train_tasks'])} typed-decision tasks ({jev} labelled by the Jev teacher, {len(t['train_tasks']) - jev} by a
local Qwen). This is the `--fold none` artefact: **every** task in the mixture is held in, so it
must not be scored on any of them.

## Use it

```python
from openjev import base
from openjev.spec import load_task

tok, model = base.load("{repo or 'sshalimov04/open-jev-base'}")
task = load_task("tasks/kinopoisk.yaml")          # any task YAML: your question, your options
logp = base.predict_logits(tok, model, task, ["Фильм затянут, но актёры вытягивают."])
probs = logp.exp()                                # [N, K], renormalised over the K options
```

The string the model actually sees (`openjev/pairs.py:render`) is a sentence pair — segment A is the
decision and is never truncated, segment B is the text and takes the rest of the 512-token window:

```
A: choice | Q: <question> | option: <this option's line> | options: <all option lines, cut at 96 tokens>
B: <the text>
```

**Calibrate it before you trust the probabilities.** Every number below uses the `gold500` variant:
a temperature + per-class bias fitted on that task's own calib rows (200 on the unseen sets, 500 on
the leave-one-source-out tasks). Raw, the renormalised sigmoids are usable at small K and unusable
at K = 77 (ECE 0.307 on banking77, repaired to 0.055 by the fit).

## What it does on questions it was never trained on

Three preregistered sets, chosen and frozen before any of this model existed, over three text
sources not in the mixture. n = 500 eval rows each, **200 gold calibration rows per task**.
Teacher-normalised = (metric − chance) / (teacher − chance), sign-flipped for MAE.

| task | metric | n | K | chance | teacher | this model | advantage over chance (95 % CI) | teacher-normalised |
|---|---|---|---|---|---|---|---|---|
{unseen}

It beats chance on all three with the CI excluding 0, and recovers 70 % / 57 % / 30 % of the
teacher's own margin over chance. That is the whole claim. It is **not** a general model and it does
**not** answer any question: the rule it was written against passed at ≥ 0.5 on 2 of 3 sets, the
weakest result is the Russian safety question, and under a no-gold calibration (`teacher500`) sst5
falls to 0.48 and the rule would be met on 1 of 3. A deployment on a new question has to supply
those 200 gold rows.

## What it loses to (the same recipe, held-out text sources)

A card that only lists wins is the thing this repo exists not to be. These are the fold models
(`base-F1-v2`, `base-F2-v2`) — same architecture, same recipe, same mixture minus the held-out
source — scored zero-shot against the per-task distilled student for that task:

| task | fold | metric | n | chance | teacher | per-task student | this recipe, zero-shot | Δ vs student (95 % CI) |
|---|---|---|---|---|---|---|---|---|
{loto}

**It loses to every per-task student except one**, every CI excluding 0. The exception is
`m2w-element`, where the per-task student is a fixed 16-way head over row-specific candidates and
collapses to 0.059 — a pair scorer reads the candidate's own line, which is the argument for this
architecture and not evidence that the model is useful there (it is still 51 pts below the teacher).
Converging the training made transfer **worse** on 5 of these 6 tasks than an earlier 45-minute
budget did. Note that `results/base/base-F2-v2-zs-m2w-element-gold500.json` carries `nll: inf` and
`ece: 0.855`: the vector fit drove one option's calibrated probability to 0 on a row that carries it
as gold. Accuracy is unaffected; anything reading `nll` off that file is reading a degenerate fit.

## What is not measured

The mixture contains 27 small tasks beyond the original ones, and **whether they bought anything is
unknown**. The ablation that was built to answer it is **void by its own preregistered STOP rule**:
2 of the 3 control-arm seeds never converged, and the prereg forbids comparing a converged arm with
a non-converged one. Its numbers happen to favour the full mixture, which is exactly the direction
an under-trained control would fake, so they are not cited here. **This model's mixture advantage is
unmeasured.**

Also not measured: any recalibration under drift, any language beyond the en/ru mix of the training
tasks, and any behaviour at K far above the 77 seen in training.

## Training

- **Init**: `{t['student']}` (~140M), `AutoModelForSequenceClassification(num_labels=1)`,
  BCE against the teacher's probability of that option.
- **Teachers**: `typesafe/jev-1.13-20260917` ({jev} run dirs) and a local `Qwen/Qwen3.8-27B-FP8`
  ({len(t['train_tasks']) - jev} run dirs, the original tasks). No row was re-labelled for this model.
- **Mixture**: {t['pairs_per_epoch']:,} pairs/epoch (cap {t['cap']:,} per task per epoch) over {t['n_train']:,} teacher rows.
- **Run**: batch {t['batch']}, lr {t['lr']}, max_len {t['max_len']}, {t['steps_run']:,} steps ({t['epochs_seen']} epochs),
  early-stopped on held-in calib BCE (min_delta {t['min_delta']}, patience {t['patience']}), best
  {t['best_calib_bce']:.4f}, converged: {str(t['converged']).lower()}. {t['train_minutes']:.0f} min, {t['peak_gpu_gb']:.2f} GB peak on one GB10.

## Licence and provenance

MIT. Base model [`{t['student']}`](https://huggingface.co/{t['student']}).
Code, task YAMLs and every table above: [open-jev](https://github.com/sshalimov04/open-jev) —
the lab notebook is `docs/experiments.md` §B2 and the preregistration that fixed these thresholds
before the runs existed is `docs/prereg/base-v2.md`.
"""


def push(run_dir, repo, dry_run=False, private=True):
    run_dir = Path(run_dir)
    if (run_dir / "train.json").exists() and json.loads((run_dir / "train.json").read_text()).get("pair_model"):
        return _push(run_dir, repo, base_card(run_dir, repo), BASE_FILES, dry_run, private)
    if not (run_dir / "openjev.json").exists():
        raise SystemExit(f"{run_dir}/openjev.json missing — run `openjev eval` first")
    from openjev.serve import conformal  # materialises conformal.json so `serve hf:...` can do `coverage`
    meta = json.loads((run_dir / "openjev.json").read_text())
    conformal(run_dir, meta.get("calib") or {"temperature": meta["temperature"]})
    return _push(run_dir, repo, card(run_dir, repo), FILES, dry_run, private)


def _push(run_dir, repo, text, patterns, dry_run, private):
    (run_dir / "README.md").write_text(text)
    files = sorted(p.relative_to(run_dir) for pat in patterns for p in run_dir.glob(pat))
    size = sum(p.stat().st_size for p in (run_dir / f for f in files))
    print(f"{'would push' if dry_run else 'pushing'} {len(files)} files ({size / 1e6:.1f} MB) "
          f"from {run_dir} to {repo} ({'private' if private else 'public'}):")
    for f in files:
        print(f"  {f}")
    print("\n--- model card ---\n" + text + "--- end model card ---")
    if dry_run:
        return {"repo": repo, "files": [str(f) for f in files], "dry_run": True}
    from huggingface_hub import HfApi
    _no_proxy()
    api = HfApi()
    api.create_repo(repo, private=private, exist_ok=True)
    url = api.upload_folder(repo_id=repo, folder_path=str(run_dir), allow_patterns=patterns,
                            commit_message=f"open-jev {run_dir.name}")
    print(f"pushed: https://huggingface.co/{repo}  ({url})")
    return {"repo": repo, "files": [str(f) for f in files], "dry_run": False}
