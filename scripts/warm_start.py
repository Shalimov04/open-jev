"""PLAN-4 M4: how many labelled rows does a new task need when it starts from a fold model?

Both arms are the *same* pair architecture on the *same* N teacher-labelled rows of the task; the
only factor is the init (a fold model that never saw the task, vs mmBERT-small). Calibration and
eval ids are the ones the per-task student already used.

    python3 scripts/warm_start.py --task kinopoisk --fold F1 --n 100 500 --seeds 0 1 2
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from openjev import base  # noqa: E402
from openjev.spec import load_task  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--task", default="kinopoisk")
ap.add_argument("--fold", default="F1", help="the fold model whose training excluded the task")
ap.add_argument("--n", type=int, nargs="+", default=[100, 500])
ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
ap.add_argument("--steps-per-n", type=float, default=4.0, help="steps per labelled row")
a = ap.parse_args()

task = load_task(ROOT / f"tasks/{a.task}.yaml")
src = ROOT / "runs" / a.task
fold_model = ROOT / "runs" / f"base-{a.fold}"
assert a.task in json.loads((fold_model / "train.json").read_text())["excluded"], \
    f"{a.task} was in base-{a.fold}'s mixture: warm-starting from it would be a leak"

for n in a.n:
    for seed in a.seeds:
        for arm, init in (("warm", str(fold_model / "student")), ("scratch", "jhu-clsp/mmBERT-small")):
            name = f"{a.task}-{arm}-n{n}" + (f"-s{seed}" if seed else "")
            d = ROOT / "runs" / f"{name}-tmp"
            if (ROOT / "runs" / name / "eval_rows.jsonl").exists():
                print(f"skip {name}", flush=True)
                continue
            t0 = time.time()
            base.train("none", d, steps=max(200, int(a.steps_per_n * n)), eval_every=50,
                       patience=4, seed=seed, init=init, only=[a.task], limit=n)
            base.zs_eval(d, task, src, variants=("gold500",), name=name)
            print(f"{name} done in {(time.time() - t0) / 60:.1f} min", flush=True)
