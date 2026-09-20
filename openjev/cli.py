"""openjev run|label|train|calibrate|eval|augment|serve|report|bench"""
import argparse
import os
from pathlib import Path

from openjev.spec import load_task

STAGES = ["label", "train", "calibrate", "eval"]
DONE = {"train": "train.json", "calibrate": "calib.json", "eval": "eval.json"}  # label always resumes


def setup(args):
    task = load_task(args.task)
    jev = getattr(args, "teacher", "vllm") == "jev"
    suffix = ["jev"] if jev else []
    if getattr(args, "student", None):
        task.student.model = args.student
        suffix.append(args.student.split("/")[-1])  # full model name: mmBERT-base != ModernBERT-base
    if getattr(args, "gold_weight", None) is not None:
        task.student.gold_weight = args.gold_weight
        if args.gold_weight:  # --gold-weight 0 is the default setting, not a variant
            suffix.append("gold")
    if getattr(args, "gold_n", None) is not None:
        suffix.append(f"n{args.gold_n}")
    for knob in ("batch_size", "epochs"):  # training-budget knobs: never part of the run-dir name
        if getattr(args, knob, None) is not None:
            setattr(task.student, knob, getattr(args, knob))
    if getattr(args, "tag", None):
        suffix.append(args.tag)
    if getattr(args, "seed", 0):  # seed 0 is the baseline run dir, not a variant
        suffix.append(f"s{args.seed}")
    run_dir = Path(args.runs) / "-".join([task.name, *suffix])
    run_dir.mkdir(parents=True, exist_ok=True)
    if suffix and not jev:  # variants share the base run's teacher labels
        for f in ("teacher.jsonl", "label.json"):
            if not (run_dir / f).is_symlink():
                (run_dir / f).symlink_to(Path("..") / task.name / f)
    return task, run_dir


def stage(name, task, run_dir, args):
    if name == "label":
        from openjev import teacher
        backend = getattr(args, "teacher", "vllm")
        if run_dir.name != task.name and backend != "jev":
            run_dir = run_dir.parent / task.name  # variants never label into their own dir
        return teacher.run(task, run_dir, limit=getattr(args, "limit", None),
                           split=getattr(args, "split", None), backend=backend)
    if name == "train":
        from openjev import train as mod
        return mod.run(task, run_dir, seed=getattr(args, "seed", 0) or 0,
                       no_synth=getattr(args, "no_synth", False), gold_n=getattr(args, "gold_n", None))
    if name == "eval":
        from openjev import evaluate as mod
        return mod.run(task, run_dir, latency=not getattr(args, "no_latency", False))
    from openjev import calibrate as mod
    return mod.run(task, run_dir, ignore_gold=getattr(args, "ignore_gold", False))


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
        s.add_argument("--batch-size", type=int, help="override student.batch_size (run dir unchanged)")
        s.add_argument("--epochs", type=int, help="override student.epochs (run dir unchanged)")
        s.add_argument("--seed", type=int, default=0, help="training seed; run dir gets a -sN suffix for N > 0")
        s.add_argument("--tag", help="free-form run-dir suffix")
        s.add_argument("--no-synth", action="store_true", help="drop synthetic (augmented) train rows")
        if name in ("run", "calibrate"):
            s.add_argument("--ignore-gold", action="store_true",
                           help="calibrate as if the calib split had no gold labels; with `prior:` in the "
                                "YAML this gives a genuine no-gold calibration")
        s.add_argument("--gold-n", type=int,
                       help="CE term on the first N train rows in id order only (needs --gold-weight)")
        if name in ("run", "eval"):
            s.add_argument("--no-latency", action="store_true",
                           help="skip the latency/throughput measurement (it needs an idle teacher; "
                                "use `openjev bench RUN_DIR` later)")
        if name in ("run", "label"):
            s.add_argument("--teacher", choices=["vllm", "jev"], default="vllm",
                           help="teacher backend; `jev` labels into runs/<task>-jev/ from the "
                                "existing runs/<task>/ rows (same ids and splits)")
        if name == "label":  # a limit on `run` would label train rows only and leave calib/eval empty
            s.add_argument("--split", choices=["train", "calib", "eval"], help="label only this split")
            s.add_argument("--limit", type=int, help="label at most N examples")
        if name == "run":
            s.add_argument("--force", action="append", default=[], choices=STAGES, help="re-run STAGE")
        if name == "augment":
            s.add_argument("--rounds", type=int, default=1)
            s.add_argument("--per-class", type=int, default=100, help="texts asked for per prompt")
            s.add_argument("--targets", type=int, default=6, help="prompts per round (weak classes + pairs)")
    c = sub.add_parser("check", help="dry-run a task YAML: prompt, examples, splits, cost (no teacher calls)")
    c.add_argument("task")
    c.add_argument("--runs", default="runs")
    c.add_argument("--probe", type=int, metavar="N", default=0,
                   help="label N calib rows and report the teacher's accuracy and marginal "
                        "(cached into runs/<task>/teacher.jsonl, so `run` reuses them)")
    s = sub.add_parser("serve")
    s.add_argument("run_dirs", nargs="+", help="run dirs, or hf:user/name to pull one from the Hub")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8080)
    s.add_argument("--device", choices=["cpu", "cuda"], help="default: cuda when available")
    h = sub.add_parser("push", help="upload a run dir (student/ + openjev.json + model card) to the HF Hub")
    h.add_argument("run_dir")
    h.add_argument("--repo", required=True, metavar="USER/NAME")
    h.add_argument("--dry-run", action="store_true", help="print the file list and the card, upload nothing")
    h.add_argument("--public", action="store_true", help="create the repo public (default: private)")
    sub.add_parser("report")
    c = sub.add_parser("compare", help="paired-bootstrap comparison of two run dirs, over all seeds")
    c.add_argument("a")
    c.add_argument("b")
    c.add_argument("--boot", type=int, default=2000, help="bootstrap resamples")
    b = sub.add_parser("bench")
    b.add_argument("run_dir")
    b.add_argument("--timeout", type=int, default=1800, help="give up waiting for an idle teacher after N s")
    args = p.parse_args(argv)

    if args.cmd == "check":
        from openjev import check
        check.main(args.task, args.probe, args.runs)
        return
    if args.cmd == "serve":
        from openjev import serve
        serve.main(args.run_dirs, args.host, args.port, args.device)
        return
    if args.cmd == "push":
        from openjev import hub
        hub.push(args.run_dir, args.repo, dry_run=args.dry_run, private=not args.public)
        return
    if args.cmd == "report":
        from openjev import evaluate
        evaluate.report()
        return
    if args.cmd == "compare":
        from openjev import evaluate
        evaluate.compare(args.a, args.b, n_boot=args.boot)
        return
    if args.cmd == "bench":
        from openjev import evaluate
        evaluate.bench(args.run_dir, timeout=args.timeout)
        return
    try:  # spec errors are the newcomer's first wall: one line, and point at `check`
        task, run_dir = setup(args)
    except (ValueError, FileNotFoundError, KeyError) as e:
        raise SystemExit(f"error: {e}\nhint: `openjev check {args.task}` prints the spec, the prompt "
                         f"and the first examples before anything is spent")
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
