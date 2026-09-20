"""FastAPI /v1/systemone over one or more exported run dirs (student/ + openjev.json).

The request shape mirrors TypeSafe's Jev endpoint closely enough that one harness
(`scripts/compare_api.py`) can drive both. Extras over the plain view:
  input: str | {template fields} | list of either   -> one forward pass, list response
  escalate_below: tau                               -> "escalate": true/false (we never call the teacher)
  coverage: 0.9                                     -> "set": [...] (split-conformal, LAC)
"""
import json
import math
from pathlib import Path

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from openjev.calibrate import apply, targets_for
from openjev.data import read_rows
from openjev.train import load_student, predict_logits
from openjev.views import view


class Req(BaseModel):
    model: str
    input: str | dict | list[str | dict]
    escalate_below: float | None = None   # per-request override of openjev.json's parity tau
    coverage: float | None = None         # e.g. 0.9 -> a prediction set that covers gold 90 % of the time


def conformal(run_dir, calib):
    """Sorted LAC nonconformity scores (1 − p_true) on the calib split, or None when the run dir
    has no calib logits. Cached in conformal.json because `push` ships that but not teacher.jsonl."""
    run_dir = Path(run_dir)
    cached = run_dir / "conformal.json"
    if cached.exists():
        return json.loads(cached.read_text())
    if not (run_dir / "calib_logits.pt").exists():
        return None
    d = torch.load(run_dir / "calib_logits.pt", weights_only=False)
    rows = {r["id"]: r for r in read_rows(run_dir, "calib")}
    if not all(i in rows for i in d["ids"]):
        return None
    y, target = targets_for([rows[i] for i in d["ids"]])
    p = apply(d["logits"], calib).softmax(-1)
    out = {"target": target,  # "gold", or "teacher" = pseudo-gold: the set covers the *teacher*, not truth
           "scores": sorted((1 - p[torch.arange(len(y)), y]).tolist())}
    cached.write_text(json.dumps(out))
    return out


def pred_set(labels, probs, conf, coverage):
    """Split conformal at level `coverage`: keep every label whose prob is at least 1 − q, where q is
    the ceil((n+1)·coverage)/n empirical quantile of the calib scores. Asking for more coverage than
    n+1 rows can certify returns every label, which is the honest answer."""
    scores, n = conf["scores"], len(conf["scores"])
    k = math.ceil((n + 1) * coverage)
    q = 1.0 if k > n else scores[k - 1]
    keep = [l for l, p in zip(labels, probs) if p >= 1 - q]
    return keep or [labels[max(range(len(probs)), key=probs.__getitem__)]]  # never empty


def load(run_dir, device=None):
    run_dir = Path(run_dir)
    meta = json.loads((run_dir / "openjev.json").read_text())
    tok, model = load_student(run_dir / "student", device)
    calib = meta.get("calib") or {"temperature": meta["temperature"]}
    return {"meta": meta, "tok": tok, "model": model, "calib": calib,
            "conformal": conformal(run_dir, calib)}


def app_for(run_dirs, device=None):
    models = {Path(d).name: load(d, device) for d in run_dirs}
    app = FastAPI(title="open-jev")

    @app.get("/v1/models")
    def list_models():
        return {"data": [{"id": n, "type": m["meta"]["task"]["type"], "labels": m["meta"]["labels"],
                          "escalate_below": m["meta"].get("escalate_below"),
                          "conformal": m["conformal"]["target"] if m["conformal"] else None}
                         for n, m in models.items()]}

    @app.post("/v1/systemone")
    def systemone(req: Req):
        m = models.get(req.model)
        if m is None:
            raise HTTPException(404, f"unknown model {req.model!r}; loaded: {sorted(models)}")
        task, meta = m["meta"]["task"], m["meta"]
        batch = req.input if isinstance(req.input, list) else [req.input]
        texts = []
        for one in batch:
            if isinstance(one, dict):
                try:
                    one = task["data"]["text"].format(**one)
                except KeyError as e:
                    raise HTTPException(422, f"input object missing field {e}")
            texts.append(one[:task["data"]["max_chars"]])
        if not texts:
            raise HTTPException(422, "input list is empty")
        logits = predict_logits(m["tok"], m["model"], texts, task["student"]["max_len"])
        probs = apply(logits, m["calib"]).softmax(-1)
        tau = req.escalate_below if req.escalate_below is not None else meta.get("escalate_below")
        if req.coverage is not None and m["conformal"] is None:
            raise HTTPException(422, "no conformal calibration in this run dir "
                                     "(needs calib_logits.pt + teacher.jsonl, or a conformal.json)")
        out = []
        for row in probs.tolist():
            r = {"model": req.model, **view(task["type"], meta["labels"], row, meta["values"])}
            if tau is not None:
                r["escalate"] = r["confidence"] < tau
            if req.coverage is not None:
                r["set"] = pred_set(meta["labels"], row, m["conformal"], req.coverage)
                r["set_target"] = m["conformal"]["target"]
            out.append(r)
        return out if isinstance(req.input, list) else out[0]

    return app


def main(run_dirs: list[str], host: str = "127.0.0.1", port: int = 8080, device: str | None = None):
    from openjev.hub import resolve
    uvicorn.run(app_for([resolve(d) for d in run_dirs], device), host=host, port=port)
