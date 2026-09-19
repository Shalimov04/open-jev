"""FastAPI /v1/systemone over one or more exported run dirs (student/ + openjev.json)."""
import json
from pathlib import Path

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from openjev.train import load_student, predict_logits
from openjev.views import view


class Req(BaseModel):
    model: str
    input: str | dict


def load(run_dir):
    run_dir = Path(run_dir)
    meta = json.loads((run_dir / "openjev.json").read_text())
    tok, model = load_student(run_dir / "student")
    return {"meta": meta, "tok": tok, "model": model}


def app_for(run_dirs):
    models = {Path(d).name: load(d) for d in run_dirs}
    app = FastAPI(title="open-jev")

    @app.get("/v1/models")
    def list_models():
        return {"data": [{"id": n, "type": m["meta"]["task"]["type"], "labels": m["meta"]["labels"]}
                         for n, m in models.items()]}

    @app.post("/v1/systemone")
    def systemone(req: Req):
        m = models.get(req.model)
        if m is None:
            raise HTTPException(404, f"unknown model {req.model!r}; loaded: {sorted(models)}")
        task, meta = m["meta"]["task"], m["meta"]
        text = req.input
        if isinstance(text, dict):
            try:
                text = task["data"]["text"].format(**text)
            except KeyError as e:
                raise HTTPException(422, f"input object missing field {e}")
        text = text[:task["data"]["max_chars"]]
        logits = predict_logits(m["tok"], m["model"], [text], task["student"]["max_len"])[0]
        probs = torch.softmax(logits / meta["temperature"], -1).tolist()
        return {"model": req.model, **view(task["type"], meta["labels"], probs, meta["values"])}

    return app


def main(run_dirs: list[str], host: str = "127.0.0.1", port: int = 8080):
    uvicorn.run(app_for(run_dirs), host=host, port=port)
