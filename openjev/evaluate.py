"""Eval on the eval split: student (calibrated) vs gold, teacher vs gold, student vs teacher, latency.
Writes runs/<name>/eval.json, runs/<name>/openjev.json (export) and results/<run>.json; `report()` rebuilds the README table."""
import dataclasses
import json
import subprocess
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score

from openjev.calibrate import brier, ece, targets_for
from openjev.data import read_rows
from openjev.train import load_student, predict_logits
from openjev.views import view

ROOT = Path(__file__).resolve().parent.parent
MARK = "<!-- results -->"


def _cls(probs, y):
    pred = probs.argmax(-1)
    return {"acc": float((pred == y).float().mean()), "macro_f1": float(f1_score(y, pred, average="macro")),
            "ece": ece(probs, y), "brier": brier(probs, y)}


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
    raw, cal = logits.softmax(-1), (logits / T).softmax(-1)
    teacher = torch.tensor([r["probs"] for r in rows]).clamp_min(1e-6)
    teacher = teacher / teacher.sum(-1, keepdim=True)
    y, eval_target = targets_for(rows)  # gold, or teacher argmax when gold is missing
    s = _cls(cal, y)
    student = {"acc": s["acc"], "macro_f1": s["macro_f1"], "ece_raw": ece(raw, y), "ece_cal": s["ece"],
               "brier": s["brier"], "nll": float(F.nll_loss(cal.log(), y))}
    res = {"task": task.name, "type": task.type, "k": task.k, "lang": task.lang, "student_model": train["student"],
           "n_train": train["n_train"], "n_synth": train["n_synth"],
           "augment_round": train.get("augment_round", 0), "gold_weight": train["gold_weight"],
           "eval_n": len(rows), "eval_target": eval_target,
           "student": student, "teacher": _cls(teacher, y),
           "agreement": {"argmax": float((cal.argmax(-1) == teacher.argmax(-1)).float().mean()),
                         "mean_kl": float(F.kl_div(cal.log(), teacher, reduction="batchmean"))}}
    if task.type == "score":
        v = torch.tensor(task.values)
        gold_v = v[y]
        res["score"] = {"mae": float(((cal * v).sum(-1) - gold_v).abs().mean()),
                        "teacher_mae": float(((teacher * v).sum(-1) - gold_v).abs().mean())}
    if task.type == "noul":
        is_true = (y == task.labels.index("true")).numpy()
        noul = {}
        if 0 < is_true.sum() < len(is_true):
            noul["auroc"] = float(roc_auc_score(is_true, cal[:, 0]))
            noul["teacher_auroc"] = float(roc_auc_score(is_true, teacher[:, 0]))
        if all(r.get("gold_prob") is not None for r in rows):
            gp = torch.tensor([float(r["gold_prob"]) for r in rows])
            noul["brier_vs_gold_prob"] = float(((cal[:, 0] - gp) ** 2).mean())
            noul["teacher_brier_vs_gold_prob"] = float(((teacher[:, 0] - gp) ** 2).mean())
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
    # per-class recall + confusions on the calib split (never eval) for augment
    crows = read_rows(run_dir, "calib")
    cy, ctarget = targets_for(crows)
    cpred = predict_logits(tok, model, [r["text"] for r in crows], max_len).argmax(-1)
    cm = confusion_matrix(cy, cpred, labels=list(range(task.k)))
    ev = dict(res, calib_split={"target": ctarget, "confusion": cm.tolist(),
                                "recall": dict(zip(task.labels, (cm.diagonal() / cm.sum(1).clip(min=1)).tolist()))},
              examples=[dict(view(task.type, task.labels, cal[i].tolist(), task.values), text=texts[i][:200])
                        for i in range(min(5, len(rows)))])
    (run_dir / "eval.json").write_text(json.dumps(ev, indent=2, ensure_ascii=False))
    export = {"task": dataclasses.asdict(task), "labels": task.labels, "values": task.values,
              "temperature": T, "metrics": res}
    (run_dir / "openjev.json").write_text(json.dumps(export, indent=2, ensure_ascii=False))
    results_dir = Path(results_dir or ROOT / "results")
    results_dir.mkdir(exist_ok=True)
    (results_dir / f"{run_dir.name}.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
    return res


NOTES = {"choice": "argmax accuracy", "score": "expected level; MAE vs gold stars", "noul": "p(true); AUROC vs gold"}


def report(results_dir=None, readme=None):
    results_dir = Path(results_dir or ROOT / "results")
    readme = Path(readme or ROOT / "README.md")
    head = ("| task | type | K | lang | n_train | student acc / F1 | teacher acc / F1 | agree | ECE raw→cal | Brier "
            "| GPU p50 ms | ex/s |\n|" + "---|" * 12)
    lines, notes = [head], []
    for p in sorted(results_dir.glob("*.json")):
        r = json.loads(p.read_text())
        s, t, lat = r["student"], r["teacher"], r.get("latency_ms", {})
        lines.append(f"| {p.stem} | {r['type']} | {r['k']} | {r['lang']} | {r['n_train']} "
                     f"| {s['acc']:.3f} / {s['macro_f1']:.3f} | {t['acc']:.3f} / {t['macro_f1']:.3f} "
                     f"| {r['agreement']['argmax']:.3f} | {s['ece_raw']:.3f}→{s['ece_cal']:.3f} | {s['brier']:.3f} "
                     f"| {lat.get('gpu_b1_p50', 0):.1f} | {r.get('throughput_gpu_b64', 0):.0f} |")
        extra = ""
        if "score" in r:
            extra = f"; MAE {r['score']['mae']:.3f} (teacher {r['score']['teacher_mae']:.3f})"
        if r.get("noul", {}).get("auroc") is not None:
            extra = f"; AUROC {r['noul']['auroc']:.3f} (teacher {r['noul']['teacher_auroc']:.3f})"
        gold = "zero gold labels" if not r["gold_weight"] else f"gold CE weight {r['gold_weight']}"
        notes.append(f"- **{p.stem}**: {r['student_model']} distilled from the teacher with {gold}; "
                     f"{NOTES[r['type']]} vs {r.get('eval_target', 'gold')} on {r['eval_n']} eval examples"
                     f"{extra}; calibrated to {r['calib_target']} (T={r['temperature']:.2f}).")
    block = f"{MARK}\n" + "\n".join(lines) + "\n\n" + "\n".join(notes) + f"\n{MARK}"
    text = readme.read_text() if readme.exists() else f"# Open-Jev\n\n## Results\n\n{MARK}\n{MARK}\n"
    if text.count(MARK) < 2:
        text += f"\n## Results\n\n{MARK}\n{MARK}\n"
    a, b = text.index(MARK), text.index(MARK, text.index(MARK) + 1) + len(MARK)
    readme.write_text(text[:a] + block + text[b:])
    return block
