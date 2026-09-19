"""openjev run|label|train|calibrate|eval|serve|report"""
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
        suffix.append("base" if "base" in args.student.lower() else args.student.split("/")[-1])
    if getattr(args, "gold_weight", None) is not None:
        task.student.gold_weight = args.gold_weight
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
        return teacher.run(task, run_dir, limit=args.limit)
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
    for name in ["run", *STAGES]:
        s = sub.add_parser(name)
        s.add_argument("task")
        s.add_argument("--runs", default="runs")
        s.add_argument("--student", help="override student model; run dir gets a -base/-<name> suffix")
        s.add_argument("--gold-weight", type=float, help="override gold_weight; run dir gets a -gold suffix")
        s.add_argument("--limit", type=int, help="label at most N examples (label stage)")
        if name == "run":
            s.add_argument("--force", action="append", default=[], choices=STAGES, help="re-run STAGE")
    s = sub.add_parser("serve")
    s.add_argument("run_dirs", nargs="+")
    s.add_argument("--host", default="0.0.0.0")
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
