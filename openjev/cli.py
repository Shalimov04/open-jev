"""openjev run|label|train|calibrate|eval|augment|serve|report"""
import argparse
import os
from pathlib import Path

from openjev.spec import load_task

STAGES = ["label", "train", "calibrate", "eval"]
DONE = {"train": "train.json", "calibrate": "calib.json", "eval": "eval.json"}  # label always resumes


def setup(args):
    task = load_task(args.task)
    suffix = []
    if getattr(args, "student", None):
        task.student.model = args.student
        suffix.append(args.student.split("/")[-1])  # full model name: mmBERT-base != ModernBERT-base
    if getattr(args, "gold_weight", None) is not None:
        task.student.gold_weight = args.gold_weight
        if args.gold_weight:  # --gold-weight 0 is the default setting, not a variant
            suffix.append("gold")
    run_dir = Path(args.runs) / "-".join([task.name, *suffix])
    run_dir.mkdir(parents=True, exist_ok=True)
    if suffix:  # variants share the base run's teacher labels
        for f in ("teacher.jsonl", "label.json"):
            if not (run_dir / f).is_symlink():
                (run_dir / f).symlink_to(Path("..") / task.name / f)
    return task, run_dir


def stage(name, task, run_dir, args):
    if name == "label":
        from openjev import teacher
        if run_dir.name != task.name:
            run_dir = run_dir.parent / task.name  # variants never label into their own dir
        return teacher.run(task, run_dir, limit=getattr(args, "limit", None))
    if name == "train":
        from openjev import train as mod
    elif name == "calibrate":
        from openjev import calibrate as mod
    else:
        from openjev import evaluate as mod
    return mod.run(task, run_dir)


def main(argv=None):
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    p = argparse.ArgumentParser(prog="openjev")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ["run", "augment", *STAGES]:
        s = sub.add_parser(name)
        s.add_argument("task")
        s.add_argument("--runs", default="runs")
        s.add_argument("--student", help="override student model; run dir gets a -<model name> suffix")
        s.add_argument("--gold-weight", type=float, help="override gold_weight; run dir gets a -gold suffix")
        if name == "label":  # a limit on `run` would label train rows only and leave calib/eval empty
            s.add_argument("--limit", type=int, help="label at most N examples")
        if name == "run":
            s.add_argument("--force", action="append", default=[], choices=STAGES, help="re-run STAGE")
        if name == "augment":
            s.add_argument("--rounds", type=int, default=1)
            s.add_argument("--per-class", type=int, default=100, help="texts asked for per prompt")
            s.add_argument("--targets", type=int, default=6, help="prompts per round (weak classes + pairs)")
    s = sub.add_parser("serve")
    s.add_argument("run_dirs", nargs="+")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8080)
    sub.add_parser("report")
    args = p.parse_args(argv)

    if args.cmd == "serve":
        from openjev import serve
        serve.main(args.run_dirs, args.host, args.port)
        return
    if args.cmd == "report":
        from openjev import evaluate
        evaluate.report()
        return
    task, run_dir = setup(args)
    if args.cmd == "augment":
        from openjev import augment
        for _ in range(args.rounds):
            augment.round_once(task, run_dir, args.per_class, args.targets)
            for name in STAGES[1:]:  # label already happened inside the round
                stage(name, task, run_dir, args)
        return
    if args.cmd != "run":
        stage(args.cmd, task, run_dir, args)
        return
    rerun = False  # once a stage re-runs, everything downstream re-runs too
    for name in STAGES:
        if name in DONE and (run_dir / DONE[name]).exists() and not (rerun or name in args.force):
            print(f"skip {name}: {run_dir / DONE[name]} exists")
            continue
        stage(name, task, run_dir, args)
        rerun = name != "label"


if __name__ == "__main__":
    main()
