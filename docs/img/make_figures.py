#!/usr/bin/env python3
"""Regenerate the README figures in docs/img/ as standalone SVGs.

    python docs/img/make_figures.py

No dependencies beyond the standard library, no browser, no GPU. Every number is read
out of results/*.json and runs/*/ at generation time -- nothing is typed in here except
labels and prose. Each figure is emitted twice, `-light` and `-dark`, so a README can
pick one with <picture>; see docs/img/README.md.

SVGs are written with presentation attributes only (no <style>, no external fonts) so
they survive GitHub's sanitizer when inlined and render identically when served as an
<img>.
"""
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = pathlib.Path(__file__).resolve().parent

FONT = "-apple-system,BlinkMacSystemFont,Segoe UI,Helvetica,Arial,sans-serif"
MONO = "ui-monospace,SFMono-Regular,Menlo,Consolas,monospace"

# GitHub Primer colors: picked per theme so contrast holds on both backgrounds.
THEMES = {
    "light": dict(bg="#ffffff", fg="#1f2328", muted="#59636e", grid="#d1d9e0",
                  panel="#f6f8fa", edge="#afb8c1", student="#0969da", teacher="#bf8700",
                  good="#1a7f37", bad="#cf222e", flat="#8b949e", accent="#8250df"),
    "dark": dict(bg="#0d1117", fg="#e6edf3", muted="#9198a1", grid="#30363d",
                 panel="#161b22", edge="#484f58", student="#58a6ff", teacher="#d29922",
                 good="#3fb950", bad="#f85149", flat="#6e7681", accent="#a371f7"),
}

# ---------------------------------------------------------------- data


ALL = {p.stem: json.loads(p.read_text()) for p in sorted((ROOT / "results").glob("*.json"))}
# `openjev eval` rewrites results/agnews-nogold.json, so the offline-against-real-labels score
# is kept out of its way (scripts/nogold_gold_acc.py).
NOGOLD = json.loads((ROOT / "results/variants/agnews-nogold-goldacc.json").read_text())
# the six distinct labeling jobs (one teacher pass each); variants reuse teacher.jsonl
BASE = ["agnews", "banking77", "georeview", "headlines", "kinopoisk", "toxic"]
TEACHER_CALLS = sum(ALL[t]["teacher_calls"] for t in BASE)
TEACHER_MIN = sum(ALL[t]["teacher_minutes"] for t in BASE)
LAT = [r["latency_ms"]["gpu_b1_p50"] for r in ALL.values() if "latency_ms" in r]
THR = [r["throughput_gpu_b64"] for r in ALL.values() if "throughput_gpu_b64" in r]

# ---------------------------------------------------------------- svg helpers


def esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def t(x, y, s, fill, size=13, weight=None, anchor=None, font=FONT, opacity=None):
    a = f'<text x="{x:.1f}" y="{y:.1f}" fill="{fill}" font-family="{font}" font-size="{size}"'
    if weight:
        a += f' font-weight="{weight}"'
    if anchor:
        a += f' text-anchor="{anchor}"'
    if opacity is not None:
        a += f' opacity="{opacity}"'
    return a + f">{esc(s)}</text>"


def rect(x, y, w, h, fill, stroke=None, rx=0, sw=1, opacity=None):
    a = f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(w, 0):.1f}" height="{max(h, 0):.1f}" fill="{fill}"'
    if stroke:
        a += f' stroke="{stroke}" stroke-width="{sw}"'
    if rx:
        a += f' rx="{rx}"'
    if opacity is not None:
        a += f' opacity="{opacity}"'
    return a + "/>"


def ln(x1, y1, x2, y2, stroke, sw=1, dash=None, cap=None):
    a = (f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
         f'stroke="{stroke}" stroke-width="{sw}"')
    if dash:
        a += f' stroke-dasharray="{dash}"'
    if cap:
        a += f' stroke-linecap="{cap}"'
    return a + "/>"


def poly(pts, fill):
    return '<polygon points="%s" fill="%s"/>' % (
        " ".join(f"{x:.1f},{y:.1f}" for x, y in pts), fill)


def circ(x, y, r, fill, stroke=None, sw=1.5):
    a = f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{fill}"'
    if stroke:
        a += f' stroke="{stroke}" stroke-width="{sw}"'
    return a + "/>"


def polyline(pts, stroke, sw=2):
    return ('<polyline points="%s" fill="none" stroke="%s" stroke-width="%s" '
            'stroke-linejoin="round" stroke-linecap="round"/>' % (
                " ".join(f"{x:.1f},{y:.1f}" for x, y in pts), stroke, sw))


def arrow(x1, y, x2, color, sw=2):
    """Left-to-right arrow ending at x2."""
    return ln(x1, y, x2 - 6, y, color, sw) + poly(
        [(x2, y), (x2 - 7, y - 4.5), (x2 - 7, y + 4.5)], color)


def wrap(text, width):
    """Greedy wrap by character count (no font metrics available)."""
    out, line = [], ""
    for w in text.split():
        if line and len(line) + 1 + len(w) > width:
            out.append(line)
            line = w
        else:
            line = f"{line} {w}".strip()
    if line:
        out.append(line)
    return out


def doc(w, h, th, body, label):
    return ('<?xml version="1.0" encoding="UTF-8"?>\n'
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
            f'viewBox="0 0 {w} {h}" role="img" aria-label="{esc(label)}">\n'
            f'<title>{esc(label)}</title>\n'
            + rect(0, 0, w, h, th["bg"]) + "\n" + "\n".join(body) + "\n</svg>\n")


def head(th, title, sub, w=900):
    o = [t(20, 30, title, th["fg"], 17, "700")]
    if sub:
        o.append(t(20, 51, sub, th["muted"], 12.5))
    return o


# ---------------------------------------------------------------- 1. pipeline


def fig_pipeline(th):
    W, H = 900, 312
    b = head(th, "open-jev: a prompt becomes a calibrated 140M classifier",
             "one YAML task → soft labels from a local LLM → distilled encoder "
             "→ temperature + per-class bias → typed JSON")
    steps = [
        ("1", "Task", th["muted"], ["tasks/agnews.yaml", "question + options", "4,000 rows",
                                    "zero gold labels"]),
        ("2", "Teacher", th["teacher"], ["Qwen3.8-27B · vLLM", "max_tokens: 1,",
                                         "constrained letter", "softmax(top_logprobs)",
                                         "= soft label"]),
        ("3", "Student", th["student"], ["mmBERT-small, 140M", "KL(teacher ‖ student)",
                                         "no gold in the loss", "early stop on calib KL"]),
        ("4", "Calibrate", th["accent"], ["logits / T + b,", "LBFGS on 500", "held-out gold rows",
                                          "kept on a 2-fold", "held-out NLL test"]),
        ("5", "Serve", th["good"], ["POST /v1/systemone", "typed JSON decision",
                                    "+ probabilities", "you can threshold"]),
    ]
    bw, gap, x0, y0, bh = 156, 20, 20, 72, 136
    for i, (num, name, col, lines) in enumerate(steps):
        x = x0 + i * (bw + gap)
        b.append(rect(x, y0, bw, bh, th["panel"], th["grid"], rx=8))
        b.append(rect(x, y0, bw, 4, col, rx=2))
        b.append(circ(x + 16, y0 + 24, 9, col))
        b.append(t(x + 16, y0 + 28, num, th["bg"], 11, "700", "middle"))
        b.append(t(x + 32, y0 + 28, name, th["fg"], 14, "700"))
        for j, s in enumerate(lines):
            b.append(t(x + 12, y0 + 52 + j * 16, s, th["muted"], 11, font=MONO if j == 0 else FONT))
        if i < len(steps) - 1:
            b.append(arrow(x + bw + 4, y0 + bh / 2, x + bw + gap - 4, th["edge"]))
    # cost captions
    cy = y0 + bh + 26
    b.append(rect(x0 + 1 * (bw + gap), cy - 14, bw * 2 + gap, 22, th["teacher"], rx=6, opacity=0.14))
    b.append(t(x0 + 1 * (bw + gap) + 10, cy + 2,
               f"paid once: {TEACHER_CALLS:,} calls · {TEACHER_MIN / 60:.1f} h",
               th["teacher"], 12, "700"))
    b.append(rect(x0 + 4 * (bw + gap), cy - 14, bw, 22, th["good"], rx=6, opacity=0.14))
    b.append(t(x0 + 4 * (bw + gap) + 10, cy + 2, "paid per call: ~5 ms", th["good"], 12, "700"))
    b.append(t(20, H - 14,
               f"6 tasks: {TEACHER_CALLS:,} teacher calls in {TEACHER_MIN:.0f} min · "
               f"training 1.5–12 min/task · serving {min(LAT):.1f}–{max(LAT):.1f} ms "
               f"p50 batch-1, up to {max(THR):,.0f} ex/s batch-64",
               th["muted"], 11))
    return doc(W, H, th, b, "open-jev pipeline: unlabeled text, LLM teacher soft labels, "
                            "distilled encoder student, temperature calibration, typed JSON response")


# ---------------------------------------------------------------- 2. student vs teacher


def fig_bars(th):
    W, H = 900, 482
    hl = [ALL[k]["student"]["acc"] for k in ("headlines", "headlines-s1", "headlines-s2")]
    ng = NOGOLD
    rows = [
        ("agnews", ALL["agnews"]["student"]["acc"], ALL["agnews"]["teacher"]["acc"], ""),
        ("agnews-nogold", ng["gold_acc_offline"], ng["teacher_gold_acc_offline"], "*"),
        ("banking77", ALL["banking77"]["student"]["acc"], ALL["banking77"]["teacher"]["acc"], ""),
        ("georeview", ALL["georeview"]["student"]["acc"], ALL["georeview"]["teacher"]["acc"], ""),
        ("headlines", sum(hl) / len(hl), ALL["headlines"]["teacher"]["acc"], "†"),
        ("kinopoisk", ALL["kinopoisk"]["student"]["acc"], ALL["kinopoisk"]["teacher"]["acc"], ""),
        ("toxic", ALL["toxic"]["student"]["acc"], ALL["toxic"]["teacher"]["acc"], "‡"),
    ]
    b = head(th, "The student tracks its teacher — and stops there",
             "argmax accuracy vs gold on the eval split; the teacher is the ceiling, not the "
             "state of the art")
    px, py, pw, ph = 62, 76, 818, 290

    def yv(v):
        return py + ph - v * ph

    for g in range(0, 11, 2):
        v = g / 10
        b.append(ln(px, yv(v), px + pw, yv(v), th["grid"], 1))
        b.append(t(px - 8, yv(v) + 4, f"{v:.1f}", th["muted"], 11, anchor="end"))
    gw = pw / len(rows)
    for i, (name, s, tc, mark) in enumerate(rows):
        cx = px + gw * (i + 0.5)
        bw = 40
        if mark == "*":
            b.append(rect(px + gw * i + 4, py - 6, gw - 8, ph + 6, th["student"], rx=6, opacity=0.09))
        for k, (v, col, lab) in enumerate([(s, th["student"], "student"), (tc, th["teacher"], "teacher")]):
            x = cx - bw - 3 + k * (bw + 6)
            b.append(rect(x, yv(v), bw, ph - (yv(v) - py), col, rx=3))
            b.append(t(x + bw / 2, yv(v) - 6, f"{v:.3f}", col, 11, "700", "middle"))
        b.append(t(cx, py + ph + 18, name + mark, th["fg"], 11.5, "600", "middle"))
    b.append(ln(px, py + ph, px + pw, py + ph, th["edge"], 1.5))
    # legend
    b.append(rect(px, 58, 11, 11, th["student"], rx=2))
    b.append(t(px + 16, 68, "student (mmBERT-small, 140M)", th["fg"], 11.5))
    b.append(rect(px + 210, 58, 11, 11, th["teacher"], rx=2))
    b.append(t(px + 226, 68, "teacher (Qwen3.8-27B, zero-shot)", th["fg"], 11.5))
    notes = [
        ("*", "agnews-nogold: no gold label is used anywhere in the pipeline — not in "
              "training, not for the temperature. Both bars are scored offline against the real "
              "ag_news test labels."),
        ("†", "headlines: mean of 3 seeds (" + ", ".join(f"{v:.3f}" for v in hl) + ")."),
        ("‡", "toxic: accuracy is misleading on an 8.1%-positive split — always "
                   "answering “not toxic” scores 0.919. The student wins on AUROC "
                   "(0.856 vs 0.817) and Brier (0.035 vs 0.048)."),
    ]
    y = py + ph + 44
    for mark, text in notes:
        for j, lineb in enumerate(wrap(text, 145)):
            b.append(t(20 if j == 0 else 32, y,
                       f"{mark} {lineb}" if j == 0 else lineb, th["muted"], 10.5))
            y += 13
        y += 3
    return doc(W, H, th, b, "Grouped bars of student vs teacher accuracy on seven tasks")


# ---------------------------------------------------------------- 3. cascade


def fig_cascade(th):
    W, H = 900, 456
    cols = [th["student"], th["accent"], th["good"], th["teacher"]]
    # one series per task: several seeds of the same task carry the same curve shape
    by_task = {}
    for k, v in sorted(ALL.items()):
        # accuracy tasks only: the georeview (mae) and toxic (auroc) curves are not on this axis,
        # and agnews-nogold is scored against the teacher, so its curve walks to 1.000 by definition
        if "cascade" in v and "acc" in v["cascade"][0] and v["task"] != "agnews-nogold":
            by_task.setdefault(v["task"], v)
    series = [(name, v, cols[i % len(cols)]) for i, (name, v) in enumerate(by_task.items())]
    b = head(th, "Escalating the uncertain cases to the teacher buys a point or two, at most",
             "send a prediction to the teacher when the student's max-p < τ — "
             "x = fraction escalated, y = accuracy of the pair (baseline runs, seed 0)")
    px, py, pw, ph = 70, 78, 700, 250
    accs = [c["acc"] for _, r, _ in series for c in r["cascade"]]
    lo = (min(accs) // 0.03) * 0.03
    hi = (max(accs) // 0.03 + 1) * 0.03

    def yv(v):
        return py + ph - (v - lo) / (hi - lo) * ph

    def xv(v):
        return px + v * pw

    v = lo
    while v <= hi + 1e-9:
        b.append(ln(px, yv(v), px + pw, yv(v), th["grid"], 1))
        b.append(t(px - 8, yv(v) + 4, f"{v:.2f}", th["muted"], 11, anchor="end"))
        v += 0.03
    for f in range(0, 101, 20):
        b.append(ln(xv(f / 100), py, xv(f / 100), py + ph, th["grid"], 1, dash="2 4"))
        b.append(t(xv(f / 100), py + ph + 18, f"{f}%", th["muted"], 11, anchor="middle"))
    b.append(ln(px, py + ph, px + pw, py + ph, th["edge"], 1.5))
    b.append(t(px + pw / 2, py + ph + 38, "share of eval examples escalated to the teacher",
               th["fg"], 12, "600", "middle"))
    b.append(t(16, py + ph / 2, "accuracy", th["fg"], 12, "600", "middle")
             .replace("<text ", f'<text transform="rotate(-90 16 {py + ph / 2:.1f})" '))
    for name, r, col in series:
        tac = r["teacher"]["acc"]
        b.append(ln(px, yv(tac), px + pw, yv(tac), col, 1.2, dash="6 4"))
        b.append(t(px + pw + 8, yv(tac) + 4, f"teacher {tac:.3f}", col, 10.5, opacity=0.9))
        pts = [(xv(c["escalation"]), yv(c["acc"])) for c in r["cascade"]]
        b.append(polyline(pts, col, 2.2))
        for x, y in pts:
            b.append(circ(x, y, 2.6, col))
        best = max(r["cascade"], key=lambda c: c["acc"])
        b.append(t(px + 6, yv(r["cascade"][0]["acc"]) - 9, name, col, 12, "700"))
        if best["escalation"] > 0.01:
            bx, by = xv(best["escalation"]), yv(best["acc"])
            b.append(ln(bx, by - 8, bx, by - 26, col, 1))
            b.append(t(bx, by - 31,
                       f"best {best['acc']:.3f} at {best['escalation'] * 100:.0f}%",
                       col, 10.5, "700", "middle"))
    for j, (name, r, col) in enumerate(series):
        best = max(r["cascade"], key=lambda c: c["acc"])
        base, tac = r["student"]["acc"], r["teacher"]["acc"]
        gain = (best["acc"] - base) * 100
        note = (f"{name}: the student alone scores {base:.3f} against a teacher at {tac:.3f}; "
                + (f"escalating {best['escalation'] * 100:.0f}% of examples adds "
                   f"{gain:+.1f} points." if gain > 0.05
                   else "every escalated call only drags it back down."))
        b.append(t(20, py + ph + 62 + j * 14, note, col, 10.5))
    b.append(t(20, H - 14,
               "τ = 1.0 escalates everything and collapses onto the teacher. "
               "Source: cascade[] in results/*.json.", th["muted"], 10.5))
    return doc(W, H, th, b, "Cascade curves: escalation rate versus accuracy for agnews and headlines")


# ---------------------------------------------------------------- 4. calibration


def fig_calibration(th):
    W = 900
    rows = sorted(
        [(k, v["student"]["ece_raw"], v["student"]["ece_cal"], v["temperature"], v["calib_target"])
         for k, v in ALL.items() if k not in ("headlines-s1",)],
        key=lambda r: -r[1])
    px, py, pw = 196, 104, 420
    H = py + len(rows) * 30 + 90
    b = head(th, "Calibration is fitted for accuracy, and ECE does not always follow",
             "expected calibration error on the eval split, before → after the fit "
             "(logits / T + b) on 500 held-out rows")
    mx = 0.28

    def xv(v):
        return px + v / mx * pw

    for g in range(0, 8):
        v = g * 0.04
        b.append(ln(xv(v), py - 14, xv(v), py + len(rows) * 30 - 8, th["grid"], 1))
        b.append(t(xv(v), py - 20, f"{v:.2f}", th["muted"], 10.5, anchor="middle"))
    b.append(t(px + pw / 2, py - 38, "ECE (lower is better)", th["fg"], 11.5, "600", "middle"))
    for i, (name, raw, cal, temp, target) in enumerate(rows):
        y = py + i * 30 + 6
        d = raw - cal
        col = th["good"] if d > 0.002 else (th["bad"] if d < -0.002 else th["flat"])
        b.append(t(px - 12, y + 4, name, th["fg"], 12, "600", "end"))
        b.append(ln(xv(min(raw, cal)), y, xv(max(raw, cal)), y, col, 3, cap="round"))
        b.append(circ(xv(raw), y, 5.5, th["bg"], th["muted"], 2))
        b.append(circ(xv(cal), y, 5.5, col))
        lo, hi = (raw, cal) if raw < cal else (cal, raw)
        b.append(t(xv(hi) + 12, y + 4, f"{raw:.3f} → {cal:.3f}", col, 11, "600"))
        note = f"T = {temp:.2f}" if abs(temp - 1.0) > 1e-6 else "T = 1.00 (fit rejected)"
        if target.startswith("teacher"):
            note += " · teacher-soft"
        b.append(t(722, y + 4, note, th["muted"], 10.5))
    ly = py + len(rows) * 30 + 14
    b.append(circ(px + 6, ly, 5.5, th["bg"], th["muted"], 2))
    b.append(t(px + 18, ly + 4, "raw", th["muted"], 11))
    b.append(circ(px + 62, ly, 5.5, th["good"]))
    b.append(t(px + 74, ly + 4, "improved", th["good"], 11))
    b.append(circ(px + 148, ly, 5.5, th["bad"]))
    b.append(t(px + 160, ly + 4, "made worse", th["bad"], 11))
    b.append(circ(px + 244, ly, 5.5, th["flat"]))
    b.append(t(px + 256, ly + 4, "no-op", th["flat"], 11))
    foot = ("Red rows are not failures: the per-class bias is selected on held-out NLL and "
            "accuracy, not on ECE, so a run can gain accuracy and lose ECE — headlines goes "
            "0.024 -> 0.036 ECE while gaining 7.5 accuracy points. agnews-nogold has no gold "
            "at all, so its fit is made against the teacher's soft probabilities instead.")
    for j, w in enumerate(wrap(foot, 148)):
        b.append(t(20, ly + 32 + j * 14, w, th["muted"], 10.5))
    return doc(W, H, th, b, "ECE before and after temperature scaling for every run")


# ---------------------------------------------------------------- 5. cost


def fig_cost(th):
    W, H = 900, 438
    b = head(th, "What you pay once, and what you pay per call",
             "the teacher is the whole budget; the student is rounding")
    lx, rx0 = 20, 470
    b.append(rect(lx, 64, 410, 300, th["panel"], th["grid"], rx=8))
    b.append(rect(rx0, 64, 410, 300, th["panel"], th["grid"], rx=8))
    b.append(t(lx + 16, 88, "PAID ONCE — teacher labeling", th["teacher"], 12, "700"))
    b.append(t(rx0 + 16, 88, "PAID PER CALL — student inference", th["good"], 12, "700"))

    mins = sorted(((k, ALL[k]["teacher_minutes"], ALL[k]["teacher_calls"]) for k in BASE),
                  key=lambda r: -r[1])
    bx, bw = lx + 108, 214
    mmax = mins[0][1]
    for i, (name, m, calls) in enumerate(mins):
        y = 108 + i * 30
        b.append(t(bx - 10, y + 15, name, th["fg"], 11.5, "600", "end"))
        b.append(rect(bx, y + 4, bw * m / mmax, 15, th["teacher"], rx=3))
        b.append(t(bx + bw + 10, y + 15, f"{m:.0f} min", th["muted"], 11))
    b.append(ln(lx + 16, 296, lx + 394, 296, th["grid"], 1))
    b.append(t(lx + 16, 320, f"{TEACHER_CALLS:,} calls", th["fg"], 20, "700"))
    b.append(t(lx + 16, 340, f"{TEACHER_MIN:.0f} minutes ({TEACHER_MIN / 60:.1f} h) of wall-clock, "
                             f"once, ever", th["muted"], 11))
    b.append(t(lx + 16, 356, f"banking77 alone is {mins[0][1] / TEACHER_MIN * 100:.0f}% of it "
                             f"(77 classes → 6 calls/example)", th["muted"], 11))

    lat = sorted(((k, ALL[k]["latency_ms"]["gpu_b1_p50"], ALL[k]["throughput_gpu_b64"])
                  for k in BASE), key=lambda r: r[1])
    bx2, bw2 = rx0 + 168, 150
    lmax = 8.0
    for i, (name, ms, thr) in enumerate(lat):
        y = 108 + i * 30
        b.append(t(bx2 - 10, y + 15, name, th["fg"], 11.5, "600", "end"))
        b.append(rect(bx2, y + 4, bw2 * ms / lmax, 15, th["good"], rx=3))
        b.append(t(bx2 + bw2 * ms / lmax + 8, y + 15, f"{ms:.1f} ms", th["muted"], 11))
    b.append(ln(rx0 + 16, 296, rx0 + 394, 296, th["grid"], 1))
    per_call_ms = TEACHER_MIN * 60_000 / TEACHER_CALLS
    b.append(t(rx0 + 16, 320, f"{min(LAT):.1f} ms p50", th["fg"], 20, "700"))
    b.append(t(rx0 + 16, 340, f"batch-1 on one GB10 GPU · "
                              f"{min(THR):,.0f}–{max(THR):,.0f} ex/s at batch 64",
               th["muted"], 11))
    b.append(t(rx0 + 16, 356, f"the teacher averaged {per_call_ms:.0f} ms/label "
                              f"— ~{per_call_ms / min(LAT):.0f}× slower", th["muted"], 11))
    b.append(t(20, H - 28,
               f"Teacher wall-clock was measured under concurrency, so {per_call_ms:.0f} ms/label "
               "is labeling throughput, not single-call latency.", th["muted"], 10.5))
    b.append(t(20, H - 14,
               "Student latency is batch-1 p50 of 200 requests on one GB10 with vLLM resident "
               "but idle. Source: results/*.json.", th["muted"], 10.5))
    return doc(W, H, th, b, "Teacher labeling cost paid once versus student inference cost per call")


# ---------------------------------------------------------------- 6. response card


def fig_response(th):
    W, H = 900, 404
    b = head(th, "The output is a typed decision, not a string to parse",
             "POST /v1/systemone against runs/agnews")
    px, pw = 20, 530
    b.append(rect(px, 66, pw, 70, th["panel"], th["grid"], rx=8))
    b.append(t(px + 14, 88, "curl -s localhost:8099/v1/systemone \\", th["muted"], 11.5, font=MONO))
    b.append(t(px + 14, 106, "  -d '{\"model\":\"agnews\",\"input\":", th["muted"], 11.5, font=MONO))
    b.append(t(px + 14, 124, "  \"Shares of the airline fell 8% after it cut its forecast.\"}'",
               th["muted"], 11.5, font=MONO))
    b.append(ln(px + pw / 2, 140, px + pw / 2, 152, th["edge"], 2))
    b.append(poly([(px + pw / 2, 158), (px + pw / 2 - 5, 150), (px + pw / 2 + 5, 150)], th["edge"]))

    by = 166
    b.append(rect(px, by, pw, 180, th["panel"], th["grid"], rx=8))
    lines = [
        [("{", th["muted"])],
        [('  "model"', th["student"]), (": ", th["muted"]), ('"agnews"', th["good"]), (",", th["muted"])],
        [('  "choice"', th["student"]), (": ", th["muted"]), ('"Business"', th["good"]), (",", th["muted"])],
        [('  "probabilities"', th["student"]), (": {", th["muted"])],
        [('    "World"', th["student"]), (":    ", th["muted"]), ("0.0058", th["accent"]), (",", th["muted"])],
        [('    "Sports"', th["student"]), (":   ", th["muted"]), ("0.0051", th["accent"]), (",", th["muted"])],
        [('    "Business"', th["student"]), (": ", th["muted"]), ("0.9872", th["accent"]), (",", th["muted"])],
        [('    "Sci/Tech"', th["student"]), (": ", th["muted"]), ("0.0018", th["accent"])],
        [("  },", th["muted"])],
        [('  "confidence"', th["student"]), (": ", th["muted"]), ("0.9872", th["accent"])],
        [("}", th["muted"])],
    ]
    for i, parts in enumerate(lines):
        x = px + 16
        for s, col in parts:
            b.append(t(x, by + 24 + i * 15, s, col, 11.5, font=MONO))
            x += len(s) * 6.95
    cx = px + pw + 26
    notes = [
        (by + 24 + 2 * 15, 192, "choice",
         "the argmax, already decoded to your option label — no parsing, no regex, "
         "no retry loop"),
        (by + 24 + 3 * 15, 252, "probabilities",
         "calibrated by the fitted temperature; these are the numbers you threshold on "
         "or feed to a cascade"),
        (by + 24 + 9 * 15, 318, "confidence",
         "max(p). Below your threshold → escalate to the teacher, ask a human, "
         "or abstain"),
    ]
    for row_y, y, title, text in notes:
        b.append(ln(px + pw + 4, row_y - 4, cx - 8, y, th["edge"], 1, dash="3 3"))
        b.append(t(cx, y + 4, title, th["fg"], 12, "700", font=MONO))
        for j, w in enumerate(wrap(text, 44)):
            b.append(t(cx, y + 20 + j * 13, w, th["muted"], 10.5))
    b.append(t(20, H - 28,
               "Other task types return the same shape from openjev/views.py: "
               "score → {\"score\", \"probabilities\", \"confidence\"}, "
               "noul → {\"probability\", \"confidence\"}.", th["muted"], 10.5))
    b.append(t(20, H - 14,
               "Request and response are verbatim from the README quickstart; probabilities "
               "abridged to 4 dp.", th["muted"], 10.5))
    return doc(W, H, th, b, "The typed JSON decision returned by POST /v1/systemone")


# ---------------------------------------------------------------- 7. animated terminal


def fig_terminal():
    """SMIL-animated terminal. Base attributes are the FINAL state, so a sanitizer that
    strips <animate>/<set> leaves a correct static screenshot."""
    W, H = 900, 280
    bg, chrome, fg, dim = "#0d1117", "#161b22", "#e6edf3", "#8b949e"
    grn, blu, ylw, mag = "#3fb950", "#58a6ff", "#d29922", "#a371f7"
    a = ALL["agnews"]
    calib = json.loads((ROOT / "runs" / "agnews" / "calib.json").read_text())
    tr = json.loads((ROOT / "runs" / "agnews" / "train.json").read_text())
    ep = tr["history"][tr["best_epoch"]]
    lab = json.loads((ROOT / "runs" / "agnews" / "label.json").read_text())
    secs = lab["minutes"] * 60
    cmd = "openjev run tasks/agnews.yaml"
    log = [
        (f"agnews: {lab['calls']} examples, 0 cached, {lab['calls']} to label", dim),
        (f"agnews: labeled {lab['n_labeled']} in {secs:.0f}s "
         f"({lab['n_labeled'] / secs:.1f} ex/s), failed {lab['n_failed']}", ylw),
        (f"epoch {ep['epoch']} train_loss {ep['train_loss']:.4f} "
         f"calib_kl {ep['calib_kl']:.4f}", blu),
        (f"calib: T={calib['temperature']:.3f}  ECE {calib['ece_before']:.4f} -> "
         f"{calib['ece_after']:.4f}  (target: {calib['target']})", mag),
        (f"agnews: gpu b1 p50 {a['latency_ms']['gpu_b1_p50']:.1f} ms "
         f"p99 {a['latency_ms']['gpu_b1_p99']:.1f} ms, "
         f"{a['throughput_gpu_b64']:.0f} ex/s -> results/agnews.json", grn),
        (f"eval: student {a['student']['acc']:.3f}  teacher {a['teacher']['acc']:.3f}  "
         f"agree {a['agreement']['argmax']:.3f}", fg),
    ]
    b = [rect(0, 0, W, H, bg, rx=8)]
    b.append(rect(0, 0, W, 30, chrome, rx=8))
    b.append(rect(0, 22, W, 8, chrome))
    for i, c in enumerate(["#ff5f57", "#febc2e", "#28c840"]):
        b.append(circ(20 + i * 18, 15, 5.5, c))
    b.append(t(W / 2, 19, "openjev — agnews", dim, 11.5, anchor="middle"))
    ch, y0, lh = 6.95, 56, 19
    typed_w = len(cmd) * ch
    # line 1: prompt + typed command revealed by an animated clip rect
    b.append(t(20, y0, "$", grn, 12.5, "700", font=MONO))
    b.append('<clipPath id="typ"><rect x="34" y="42" height="18" width="%0.1f">'
             '<animate attributeName="width" begin="0.3s" dur="1.6s" '
             'calcMode="discrete" values="%s" fill="freeze"/></rect></clipPath>' % (
                 typed_w,
                 ";".join(f"{typed_w * i / len(cmd):.1f}" for i in range(len(cmd) + 1))))
    b.append('<g clip-path="url(#typ)">' + t(34, y0, cmd, fg, 12.5, font=MONO) + "</g>")
    # caret: rides the typing, then parks under the last output line
    b.append('<rect x="34" y="45" width="7" height="14" fill="%s" opacity="0">'
             '<animate attributeName="x" from="34" to="%0.1f" begin="0.3s" dur="1.6s" '
             'fill="freeze"/>'
             '<set attributeName="opacity" to="1" begin="0.3s" dur="1.6s"/>'
             '</rect>' % (grn, 34 + typed_w))
    for i, (line, col) in enumerate(log):
        y = y0 + 30 + i * lh
        begin = 2.1 + i * 0.55
        b.append('<g opacity="1">'
                 f'<set attributeName="opacity" to="0" begin="0s" dur="{begin:.2f}s"/>'
                 f'<animate attributeName="opacity" values="0;1" begin="{begin:.2f}s" '
                 f'dur="0.22s" fill="freeze"/>'
                 + t(34, y, line, col, 12, font=MONO) + "</g>")
    cy = y0 + 30 + len(log) * lh + 14
    curl = ("$ curl -s :8099/v1/systemone -d '{\"model\":\"agnews\",\"input\":"
            "\"Shares of the airline fell 8% ...\"}'")
    resp = ('{"model":"agnews","choice":"Business","probabilities":{...},'
            '"confidence":0.9872}')
    for j, (line, col, size) in enumerate([(curl, dim, 11.5), (resp, grn, 12)]):
        begin = 2.1 + len(log) * 0.55 + j * 0.7
        b.append('<g opacity="1">'
                 f'<set attributeName="opacity" to="0" begin="0s" dur="{begin:.2f}s"/>'
                 f'<animate attributeName="opacity" values="0;1" begin="{begin:.2f}s" '
                 f'dur="0.25s" fill="freeze"/>'
                 + t(20, cy + j * 22, line, col, size, font=MONO) + "</g>")
    # blinking caret at the end
    end = 2.1 + len(log) * 0.55 + 2 * 0.7
    b.append('<rect x="20" y="%0.1f" width="7" height="14" fill="%s" opacity="0">'
             '<animate attributeName="opacity" calcMode="discrete" values="1;0" '
             'begin="%0.2fs" dur="1.1s" repeatCount="indefinite"/></rect>'
             % (cy + 22 + 8, grn, end))
    return ('<?xml version="1.0" encoding="UTF-8"?>\n'
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
            f'viewBox="0 0 {W} {H}" role="img" '
            'aria-label="Terminal running openjev run tasks/agnews.yaml and curling the served '
            'endpoint">\n<title>openjev run tasks/agnews.yaml</title>\n'
            + "\n".join(b) + "\n</svg>\n")


# ---------------------------------------------------------------- main

FIGURES = {
    "pipeline": fig_pipeline,
    "student-vs-teacher": fig_bars,
    "cascade": fig_cascade,
    "calibration": fig_calibration,
    "cost": fig_cost,
    "response-card": fig_response,
}


def main():
    written = []
    for name, fn in FIGURES.items():
        for theme, th in THEMES.items():
            p = OUT / f"{name}-{theme}.svg"
            p.write_text(fn(th), encoding="utf-8")
            written.append(p)
    p = OUT / "terminal.svg"
    p.write_text(fig_terminal(), encoding="utf-8")
    written.append(p)
    # self-check: every file is well-formed XML and not empty
    import xml.dom.minidom
    for p in written:
        xml.dom.minidom.parse(str(p))
        assert p.stat().st_size > 800, p
        print(f"{p.relative_to(ROOT)}  {p.stat().st_size / 1024:.1f} KB")
    assert abs(TEACHER_CALLS - 66522) < 1, TEACHER_CALLS  # README's Findings number


if __name__ == "__main__":
    main()
