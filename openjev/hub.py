"""Hugging Face Hub: `openjev push RUN_DIR --repo user/name` and `openjev serve hf:user/name`.

A pushed repo is the run dir minus the data: student/ + openjev.json + conformal.json + a model
card. That is everything `serve` needs and nothing that would leak the teacher's labels.
"""
import json
import os
from pathlib import Path

FILES = ["student/*", "openjev.json", "conformal.json", "README.md"]


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
    snapshot_download(repo_id=repo, local_dir=dest, allow_patterns=FILES)
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


def push(run_dir, repo, dry_run=False, private=True):
    run_dir = Path(run_dir)
    if not (run_dir / "openjev.json").exists():
        raise SystemExit(f"{run_dir}/openjev.json missing — run `openjev eval` first")
    from openjev.serve import conformal  # materialises conformal.json so `serve hf:...` can do `coverage`
    meta = json.loads((run_dir / "openjev.json").read_text())
    conformal(run_dir, meta.get("calib") or {"temperature": meta["temperature"]})
    text = card(run_dir, repo)
    (run_dir / "README.md").write_text(text)
    files = sorted(p.relative_to(run_dir) for pat in FILES for p in run_dir.glob(pat))
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
    url = api.upload_folder(repo_id=repo, folder_path=str(run_dir), allow_patterns=FILES,
                            commit_message=f"open-jev {run_dir.name}")
    print(f"pushed: https://huggingface.co/{repo}  ({url})")
    return {"repo": repo, "files": [str(f) for f in files], "dry_run": False}
