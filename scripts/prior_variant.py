#!/usr/bin/env python
"""Calibrate an existing run with the gold labels deliberately ignored, against a declared prior.

    python scripts/prior_variant.py tasks/kinopoisk.yaml [uniform]

Makes `runs/<task>-prior` (student and labels symlinked to the base run -- same weights, same
teacher rows, so the *only* thing that varies is the calibration target), runs calibrate with
`ignore_gold=True`, then eval. Gold is used to *score* the result and never to fit it, which is what
makes `calib_target: prior-declared` a genuine no-gold number.

Results go to results/variants/ so they cannot leak into the README table (PLAN-2 §5).
Until `prior:` is a spec field, the prior is passed here instead of read from the YAML.
"""
import sys
from pathlib import Path

from openjev import calibrate, evaluate
from openjev.spec import load_task

ROOT = Path(__file__).resolve().parent.parent


def main(yaml_path, prior="uniform", tag="prior"):
    task = load_task(yaml_path)
    base, run_dir = ROOT / "runs" / task.name, ROOT / "runs" / f"{task.name}-{tag}"
    run_dir.mkdir(exist_ok=True)
    for f in ("teacher.jsonl", "label.json", "student", "train.json"):
        link = run_dir / f
        if not link.is_symlink() and (base / f).exists():
            link.symlink_to(Path("..") / task.name / f)
    cal = calibrate.run(task, run_dir, ignore_gold=True, prior=prior)
    print(f"{run_dir.name}: {cal['target']} T={cal['temperature']:.3f} "
          f"bias={None if cal['bias'] is None else [round(b, 3) for b in cal['bias']]}")
    res = evaluate.run(task, run_dir, results_dir=ROOT / "results" / "variants", latency=False)
    m = res["score"]["mae"] if task.type == "score" else res["student"]["acc"]
    print(f"{run_dir.name}: {'mae' if task.type == 'score' else 'acc'}={m:.4f} "
          f"(teacher {res['teacher']['acc']:.4f} acc), ece_cal={res['student']['ece_cal']:.4f}")
    return res


if __name__ == "__main__":
    main(*sys.argv[1:])
