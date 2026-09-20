"""Eval on the eval split: student (calibrated) vs gold, teacher vs gold, student vs teacher, latency.
Writes runs/<name>/eval.json, runs/<name>/openjev.json (export) and results/<run>.json; `report()` rebuilds the README table."""
import dataclasses
import json
import os
import re
import statistics
import subprocess
import time
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score

from openjev.calibrate import apply, brier, ece, targets_for
from openjev.data import read_rows
from openjev.train import load_student, predict_logits
from openjev.views import view

ROOT = Path(__file__).resolve().parent.parent
MARK = "<!-- results -->"


def _cls(probs, y):
    pred = probs.argmax(-1)
    return {"acc": float((pred == y).float().mean()), "macro_f1": float(f1_score(y, pred, average="macro")),
            "ece": ece(probs, y), "brier": brier(probs, y)}


def metric_fn(task, probs, y):
    """The task's headline metric as (name, higher_is_better, f(row_idx=None) -> float).
    Everything post-hoc (cascade, compare) scores through this one definition."""
    if task.type == "score":
        v = torch.tensor(task.values)
        err = ((probs * v).sum(-1) - v[y]).abs().numpy()
        return "mae", False, lambda i=None: float(err.mean() if i is None else err[i].mean())
    if task.type == "noul":
        ti = task.labels.index("true")
        t, s = (y == ti).numpy(), probs[:, ti].numpy()

        def auroc(i=None):
            tt, ss = (t, s) if i is None else (t[i], s[i])
            return float(roc_auc_score(tt, ss)) if 0 < tt.sum() < len(tt) else float("nan")

        return "auroc", True, auroc
    ok = (probs.argmax(-1) == y).numpy().astype(float)
    return "acc", True, lambda i=None: float(ok.mean() if i is None else ok[i].mean())


TAUS = [round(0.30 + 0.05 * i, 2) for i in range(15)]


def cascade(task, student, teacher, y, taus=TAUS, slack=0.005):
    """Student answers when its confidence >= tau, else the row is escalated to the teacher. Purely
    offline: both probability matrices are already on disk, so the sweep costs no teacher call.
    An escalated row takes the teacher's probabilities (= the teacher's argmax, for accuracy).
    -> (curve, parity), parity = the cheapest tau whose metric is within `slack` of the teacher's."""
    conf = student.max(-1).values
    name, higher, tm = metric_fn(task, teacher, y)
    t_val, curve = tm(), []
    for tau in taus:
        esc = conf < tau
        p = torch.where(esc[:, None], teacher, student)
        curve.append({"tau": tau, "escalation": float(esc.float().mean()),
                      name: metric_fn(task, p, y)[2]()})
    within = [c for c in curve if ((c[name] - t_val) if higher else (t_val - c[name])) >= -slack]
    return curve, {"metric": name, "teacher": t_val, "slack": slack,
                   "parity": min(within, key=lambda c: c["escalation"]) if within else None}


SEL_TAUS = [round(0.50 + 0.05 * i, 2) for i in range(10)]


def selective(task, student, y, taus=SEL_TAUS):
    """Risk-coverage: the student answers when its confidence >= tau and abstains otherwise -- the
    abstained row goes to a *human* (gold), not to the teacher as in `cascade`. -> one row per tau
    with the coverage and the metric over the answered rows only (acc / mae, plus, for noul, the
    precision of the "true" answers: the number an "accept the run automatically" gate rides on)."""
    conf, pred = student.max(-1).values.numpy(), student.argmax(-1)
    ok = (pred == y).numpy()
    name, _, mf = metric_fn(task, student, y)
    if task.type != "score":  # acc on the covered rows, also for noul (metric_fn's auroc is global)
        name, mf = "acc", lambda i: float(ok[i].mean())
    ti = task.labels.index("true") if task.type == "noul" else None
    curve = []
    for tau in taus:
        i = conf >= tau
        r = {"tau": tau, "coverage": float(i.mean()), name: mf(i) if i.any() else None}
        if ti is not None:
            p = i & (pred.numpy() == ti)
            r["precision_true"] = float(ok[p].mean()) if p.any() else None
        curve.append(r)
    return curve


def _latency(tok, model, texts, max_len, n=200, warmup=20):
    ts = []
    for i in range(warmup + n):
        t0 = time.perf_counter()
        predict_logits(tok, model, [texts[i % len(texts)]], max_len)  # .cpu() inside syncs CUDA
        ts.append((time.perf_counter() - t0) * 1000)
    ts = sorted(ts[warmup:])
    return ts[len(ts) // 2], ts[int(len(ts) * 0.99) - 1]


def _git():
    def git(*a):
        return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()

    sha = git("rev-parse", "--short", "HEAD")
    return (sha + "-dirty" if git("status", "--porcelain") else sha) if sha else "none"


def run(task, run_dir, results_dir=None, latency=True):
    run_dir = Path(run_dir)
    rows = read_rows(run_dir, "eval")
    texts, max_len = [r["text"] for r in rows], task.student.max_len
    calib = json.loads((run_dir / "calib.json").read_text())
    train = json.loads((run_dir / "train.json").read_text())
    T = calib["temperature"]
    tok, model = load_student(run_dir / "student")
    logits = predict_logits(tok, model, texts, max_len)
    raw, cal = logits.softmax(-1), apply(logits, calib).softmax(-1)
    teacher = torch.tensor([r["probs"] for r in rows]).clamp_min(1e-6)
    teacher = teacher / teacher.sum(-1, keepdim=True)
    y, eval_target = targets_for(rows)  # gold, or teacher argmax when gold is missing
    s = _cls(cal, y)
    student = {"acc": s["acc"], "macro_f1": s["macro_f1"], "ece_raw": ece(raw, y), "ece_cal": s["ece"],
               "brier": s["brier"], "nll": float(F.nll_loss(cal.log(), y))}
    res = {"task": task.name, "type": task.type, "k": task.k, "lang": task.lang, "student_model": train["student"],
           "n_train": train["n_train"], "n_synth": train["n_synth"],
           "augment_round": train.get("augment_round", 0), "gold_weight": train["gold_weight"],
           "balanced_train": task.data.train.balance,
           "eval_n": len(rows), "eval_target": eval_target,
           "student": student, "teacher": _cls(teacher, y),
           "agreement": {"argmax": float((cal.argmax(-1) == teacher.argmax(-1)).float().mean()),
                         "mean_kl": float(F.kl_div(cal.log(), teacher, reduction="batchmean"))}}
    res["cascade"], res["cascade_parity"] = cascade(task, cal, teacher, y)
    res["selective"] = selective(task, cal, y)
    if task.type == "score":
        v = torch.tensor(task.values)
        gold_v = v[y]
        res["score"] = {"mae": float(((cal * v).sum(-1) - gold_v).abs().mean()),
                        "teacher_mae": float(((teacher * v).sum(-1) - gold_v).abs().mean())}
    if task.type == "noul":
        ti = task.labels.index("true")  # C11: p(true) is a named column, not column 0 by luck
        is_true = (y == ti).numpy()
        noul = {}
        if 0 < is_true.sum() < len(is_true):
            noul["auroc"] = float(roc_auc_score(is_true, cal[:, ti]))
            noul["teacher_auroc"] = float(roc_auc_score(is_true, teacher[:, ti]))
        if all(r.get("gold_prob") is not None for r in rows):
            gp = torch.tensor([float(r["gold_prob"]) for r in rows])
            noul["brier_vs_gold_prob"] = float(((cal[:, ti] - gp) ** 2).mean())
            noul["teacher_brier_vs_gold_prob"] = float(((teacher[:, ti] - gp) ** 2).mean())
        res["noul"] = noul
    if latency:
        p50, p99 = _latency(tok, model, texts, max_len)
        t0 = time.perf_counter()
        predict_logits(tok, model, texts, max_len, batch_size=64)
        res["throughput_gpu_b64"] = len(texts) / (time.perf_counter() - t0)
        _, cpu_model = load_student(run_dir / "student", "cpu")
        cpu_p50, _ = _latency(tok, cpu_model, texts, max_len, n=50, warmup=5)
        res["latency_ms"] = {"gpu_b1_p50": p50, "gpu_b1_p99": p99, "cpu_b1_p50": cpu_p50}
    label = json.loads((run_dir / "label.json").read_text()) if (run_dir / "label.json").exists() else {}
    calls = 1 if task.k <= task.teacher.max_options_per_call else -(-task.k // task.teacher.max_options_per_call) + 1
    res.update({"teacher_calls": label.get("calls", calls * sum(1 for _ in open(run_dir / "teacher.jsonl"))),
                "teacher_minutes": label.get("minutes"), "train_minutes": train["train_minutes"],
                "peak_gpu_gb": train.get("peak_gpu_gb"), "temperature": T, "calib_target": calib["target"],
                "ece_calib_split": [calib["ece_before"], calib["ece_after"]], "git": _git()})
    prev = Path(results_dir or ROOT / "results") / f"{run_dir.name}.json"
    if not latency and prev.exists():  # a re-eval must not drop what `bench` measured on an idle teacher
        res.update({k: v for k, v in json.loads(prev.read_text()).items()
                    if k in ("latency_ms", "throughput_gpu_b64")})
    # per-class recall + confusions on the calib split (never eval) for augment
    crows = read_rows(run_dir, "calib")
    cy, ctarget = targets_for(crows)
    cpred = apply(predict_logits(tok, model, [r["text"] for r in crows], max_len), calib).argmax(-1)
    cm = confusion_matrix(cy, cpred, labels=list(range(task.k)))
    ev = dict(res, calib_split={"target": ctarget, "confusion": cm.tolist(),
                                "recall": dict(zip(task.labels, (cm.diagonal() / cm.sum(1).clip(min=1)).tolist()))},
              examples=[dict(view(task.type, task.labels, cal[i].tolist(), task.values), text=texts[i][:200])
                        for i in range(min(5, len(rows)))])
    (run_dir / "eval.json").write_text(json.dumps(ev, indent=2, ensure_ascii=False))
    # one line per eval row, so every post-hoc analysis (cascade, vector scaling, compare) is a
    # file read instead of a re-inference.
    with open(run_dir / "eval_rows.jsonl", "w") as f:
        for i, r in enumerate(rows):
            line = {"id": r["id"], "gold": r.get("gold"),
                    "student_probs": cal[i].tolist(), "student_logits": logits[i].tolist(),
                    "teacher_probs": teacher[i].tolist()}
            if r.get("gold_prob") is not None:
                line["gold_prob"] = float(r["gold_prob"])
            f.write(json.dumps(line) + "\n")
    parity = res["cascade_parity"]["parity"]
    export = {"task": dataclasses.asdict(task), "labels": task.labels, "values": task.values,
              "temperature": T, "calib": calib, "metrics": res,
              # the cheapest tau that buys the teacher's metric; serve's default escalate_below
              "escalate_below": parity["tau"] if parity else None}
    (run_dir / "openjev.json").write_text(json.dumps(export, indent=2, ensure_ascii=False))
    prev.parent.mkdir(parents=True, exist_ok=True)
    prev.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    return res


SEEDED = re.compile(r"-s(\d+)$")


def paired_ci(f_a, f_b, n, n_boot=2000, seed=0):
    """Bootstrap the paired delta f_a - f_b over the *same* resampled rows -> (delta, lo, hi).
    `f(idx=None)` scores an arm on those rows. Paired is the whole point: the rows are identical in
    both arms, so the row-to-row noise cancels and only the difference is resampled (PLAN-2 §5)."""
    draws = np.random.default_rng(seed).integers(0, n, (n_boot, n))
    boot = sorted(f_a(i) - f_b(i) for i in draws)
    return f_a() - f_b(), boot[int(0.025 * n_boot)], boot[int(0.975 * n_boot) - 1]


def _seed_dirs(run_dir):
    """{seed: dir} for a run dir and its -sN siblings, keeping only the ones that have been eval'd."""
    run_dir = Path(run_dir)
    out = {0: run_dir} if (run_dir / "eval_rows.jsonl").exists() else {}
    for p in sorted(run_dir.parent.glob(run_dir.name + "-s*")):
        m = SEEDED.fullmatch(p.name[len(run_dir.name):])
        if m and (p / "eval_rows.jsonl").exists():
            out[int(m.group(1))] = p
    return out


def _eval_rows(d):
    return {r["id"]: r for r in (json.loads(l) for l in open(Path(d) / "eval_rows.jsonl") if l.strip())}


def _task_view(d):
    """Enough of the task to score with, straight out of the run dir's export."""
    m = json.loads((Path(d) / "openjev.json").read_text())
    return SimpleNamespace(type=m["task"]["type"], labels=m["labels"], values=m["values"])


def compare(a, b, n_boot=2000, seed=0, out=print):
    """Paired comparison of two arms, seed by seed, on the eval rows they share.

    Per-seed delta on the task's metric, then one pooled 95 % bootstrap CI over *rows* (the same
    resampled rows for both arms and every seed -- that is what makes it paired). The verdict line
    is the only thing allowed into the README (PLAN-2 §5)."""
    A, B = _seed_dirs(a), _seed_dirs(b)
    seeds = sorted(set(A) & set(B))
    if not seeds:
        raise SystemExit(f"no seed is eval'd in both arms: {sorted(A)} vs {sorted(B)} "
                         f"(need eval_rows.jsonl -- run `openjev eval`)")
    task = _task_view(A[seeds[0]])
    rows = [_eval_rows(A[s]) for s in seeds] + [_eval_rows(B[s]) for s in seeds]
    ids = sorted(set.intersection(*(set(r) for r in rows)))
    if not ids:
        raise SystemExit("the two arms share no eval ids")
    gold = [rows[0][i]["gold"] for i in ids]
    if any(g is None for g in gold):  # no gold: score both arms against the teacher, as eval does
        y = torch.tensor([max(range(len(rows[0][i]["teacher_probs"])),
                              key=rows[0][i]["teacher_probs"].__getitem__) for i in ids])
    else:
        y = torch.tensor(gold)
    fns = {}
    for k, s in enumerate(seeds):
        for off, arm in ((0, "A"), (len(seeds), "B")):
            p = torch.tensor([rows[k + off][i]["student_probs"] for i in ids])
            name, higher, f = metric_fn(task, p, y)
            fns[arm, s] = f
    per_seed = [{"seed": s, "a": fns["A", s](), "b": fns["B", s]()} for s in seeds]
    for r in per_seed:
        r["delta"] = r["a"] - r["b"]
    n = len(ids)
    arm = lambda a: (lambda i=None: sum(fns[a, s](i) for s in seeds) / len(seeds))  # noqa: E731
    delta, lo, hi = paired_ci(arm("A"), arm("B"), n, n_boot, seed)
    scale, unit = (1.0, f" {name}") if name == "mae" else (100.0, " pts")
    fmt = (lambda v: f"{v * scale:+.3f}") if name == "mae" else (lambda v: f"{v * scale:+.1f}")
    d = f"Δ = {fmt(delta)} ± {fmt((hi - lo) / 2).lstrip('+')}{unit} (95 % CI {fmt(lo)}..{fmt(hi)})"
    verdict = (f"no measurable difference ({d})" if lo <= 0 <= hi else
               f"{'A' if (delta > 0) == higher else 'B'} is better: {d}")
    out(f"A = {a}   B = {b}   metric: {name} ({'higher' if higher else 'lower'} is better), "
        f"{n} shared eval rows, {len(seeds)} seed(s), {n_boot} bootstrap resamples")
    out("| seed | A | B | Δ |\n|---|---|---|---|")
    for r in per_seed:
        out(f"| {r['seed']} | {r['a']:.4f} | {r['b']:.4f} | {r['delta']:+.4f} |")
    out(f"verdict: {verdict}")
    return {"a": str(a), "b": str(b), "metric": name, "higher_is_better": higher, "seeds": seeds,
            "n_rows": n, "per_seed": per_seed, "delta": delta, "ci": [lo, hi],
            "n_boot": n_boot, "verdict": verdict}


def teacher_running():
    """vllm:num_requests_running from the teacher's /metrics. Proxy-free: the teacher is localhost
    and the dev box exports a SOCKS proxy that would swallow the call."""
    url = os.environ.get("OPENJEV_TEACHER_URL", "http://localhost:8000/v1").rsplit("/v1", 1)[0] + "/metrics"
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    body = opener.open(url, timeout=10).read().decode()
    n = [float(l.rsplit(" ", 1)[1]) for l in body.splitlines() if l.startswith("vllm:num_requests_running")]
    if not n:
        raise RuntimeError(f"no vllm:num_requests_running in {url}")
    return sum(n)


def bench(run_dir, results_dir=None, timeout=1800, poll=15):
    """Latency + throughput only, on an idle teacher (they share the GPU), into the existing results
    JSON. Everything else in that file stays as eval wrote it."""
    run_dir = Path(run_dir)
    meta = json.loads((run_dir / "openjev.json").read_text())
    max_len = meta["task"]["student"]["max_len"]
    texts = [r["text"] for r in read_rows(run_dir, "eval")]
    t0 = time.time()
    while teacher_running():
        if time.time() - t0 > timeout:
            raise SystemExit(f"teacher still busy after {timeout}s; not benching (numbers would be noise)")
        print(f"teacher busy, waiting {poll}s", flush=True)
        time.sleep(poll)
    tok, model = load_student(run_dir / "student")
    p50, p99 = _latency(tok, model, texts, max_len)
    t1 = time.perf_counter()
    predict_logits(tok, model, texts, max_len, batch_size=64)
    thr = len(texts) / (time.perf_counter() - t1)
    _, cpu_model = load_student(run_dir / "student", "cpu")
    cpu_p50, _ = _latency(tok, cpu_model, texts, max_len, n=50, warmup=5)
    lat = {"gpu_b1_p50": p50, "gpu_b1_p99": p99, "cpu_b1_p50": cpu_p50}
    path = Path(results_dir or ROOT / "results") / f"{run_dir.name}.json"
    res = json.loads(path.read_text())
    res.update({"latency_ms": lat, "throughput_gpu_b64": thr})
    path.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(f"{run_dir.name}: gpu b1 p50 {p50:.1f} ms p99 {p99:.1f} ms, cpu b1 p50 {cpu_p50:.1f} ms, "
          f"{thr:.0f} ex/s -> {path}")
    return res


NOTES = {"choice": "argmax accuracy", "score": "expected level; MAE", "noul": "p(true); AUROC"}


SEED = re.compile(r"-s\d+$")


def _agg(vals, fmt="{:.3f}"):
    """mean over the seeds of a group, ± sd when the seeds actually differ, '–' when nothing ran."""
    vals = [v for v in vals if v is not None]
    if not vals:
        return "–"
    m = fmt.format(sum(vals) / len(vals))
    sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return m if sd == 0 else f"{m} ± {fmt.format(sd)}"


def report(results_dir=None, readme=None):
    results_dir = Path(results_dir or ROOT / "results")
    readme = Path(readme or ROOT / "README.md")
    head = ("| task | type | K | lang | n_train | student acc / F1 | teacher acc / F1 | agree | ECE raw→cal | Brier "
            "| GPU p50 ms | ex/s |\n|" + "---|" * 12)
    lines, notes = [head], []
    groups = {}  # results/<name>-s<k>.json is a seed of results/<name>.json, not a separate run
    for p in sorted(results_dir.glob("*.json")):
        # seed 0 first: its file has no -sN suffix, and the note line quotes g[0]
        groups.setdefault(SEED.sub("", p.stem), []).append((p.stem, json.loads(p.read_text())))
    groups = {k: [r for _, r in sorted(v, key=lambda t: len(t[0]))] for k, v in groups.items()}
    for stem, g in sorted(groups.items()):
        r = g[0]
        st = lambda k: _agg([x["student"][k] for x in g])          # noqa: E731
        te = lambda k: _agg([x["teacher"][k] for x in g])          # noqa: E731
        name = stem + (f" (n={len(g)})" if len(g) > 1 else "")
        lines.append(f"| {name} | {r['type']} | {r['k']} | {r['lang']} | {r['n_train']} "
                     f"| {st('acc')} / {st('macro_f1')} | {te('acc')} / {te('macro_f1')} "
                     f"| {_agg([x['agreement']['argmax'] for x in g])} "
                     f"| {st('ece_raw')}→{st('ece_cal')} | {st('brier')} "
                     f"| {_agg([x.get('latency_ms', {}).get('gpu_b1_p50') for x in g], '{:.1f}')} "
                     f"| {_agg([x.get('throughput_gpu_b64') for x in g], '{:.0f}')} |")
        extra = ""
        if "score" in r:
            extra = f"; MAE {r['score']['mae']:.3f} (teacher {r['score']['teacher_mae']:.3f})"
        if r.get("noul", {}).get("auroc") is not None:
            extra = f"; AUROC {r['noul']['auroc']:.3f} (teacher {r['noul']['teacher_auroc']:.3f})"
        if r["type"] == "noul" and r.get("selective"):  # the "ask a human" gate, one line
            g = [c for c in r["selective"] if (c.get("precision_true") or 0) >= 0.9]
            extra += ("; gate: " + (f"{max(g, key=lambda c: c['coverage'])['coverage']:.0%} coverage at "
                                    f"90%+ precision(true)" if g else "no tau reaches 90% precision(true)"))
        gold = (f"gold CE weight {r['gold_weight']}" if r["gold_weight"] else
                "no gold in the loss (but train rows picked 50/50 by gold)" if r.get("balanced_train") else
                "zero gold labels")
        target = r["calib_target"]
        cal = (f"temperature kept at 1.0 (fitting it did not improve ECE on the calib split)"
               if target.endswith("-kept-1.0") else
               f"calibrated to {target} (T={r['temperature']:.2f})")
        notes.append(f"- **{stem}**: {r['student_model']} distilled from the teacher with {gold}; "
                     f"{NOTES[r['type']]} vs {r.get('eval_target', 'gold')} on {r['eval_n']} eval examples"
                     f"{extra}; {cal}.")
    block = f"{MARK}\n" + "\n".join(lines) + "\n\n" + "\n".join(notes) + f"\n{MARK}"
    text = readme.read_text() if readme.exists() else f"# Open-Jev\n\n## Results\n\n{MARK}\n{MARK}\n"
    if text.count(MARK) < 2:
        text += f"\n## Results\n\n{MARK}\n{MARK}\n"
    a, b = text.index(MARK), text.index(MARK, text.index(MARK) + 1) + len(MARK)
    readme.write_text(text[:a] + block + text[b:])
    print(block)
    return block
