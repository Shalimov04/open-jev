#!/usr/bin/env python
"""Decision Index 0.3 (apolinario/decision-index) for `open-jev-base` -- docs/prereg/decision-index-v1.md.

An `Engine` for the kit's runner: `engine(state, questions) -> (response, raw)`. Every option of
every question is one (segment A, segment B) pair rendered by `openjev.pairs.render`, exactly the
training format; all pairs of a request go through the model in one batched pass. Nothing in
`openjev/` changes.

  PYTHONPATH=scripts python -m decision_index pipeline \
      --engine bench_decision_index:OpenJevEngine \
      --option model=sshalimov04/open-jev-base --out runs/di-open-jev-base

The kit's rule is "do not truncate, do not drop options, refuse instead": a pair longer than
`max_len` tokens raises `Unsupported` for the whole request (it counts as wrong), it is never cut.
The contrast set inside segment A is cut at `pairs.OPTIONS_TOKENS` as in training; every option is
still scored on its own pair.
"""
import hashlib
import json
from pathlib import Path

try:
    from decision_index.engines import Engine, Unsupported
except ModuleNotFoundError:      # the tests need no harness install
    class Unsupported(ValueError):
        pass

    class Engine:
        name, provenance = "engine", {}

        def __init__(self, **options):
            self.options = options

        def runtime(self):
            return {}

        def synchronize(self):
            pass

        def close(self):
            pass

DEFAULT_MODEL = "sshalimov04/open-jev-base"


def as_text(x):
    """The kit's own `engines.base.text`: a string as is, anything else as compact JSON."""
    if x is None:
        return ""
    return x if isinstance(x, str) else json.dumps(x, ensure_ascii=False, separators=(",", ":"))


def option_line(key, desc):
    """`key: description`, the line the kit's transformers engine shows; the key alone when the
    description is empty or repeats it."""
    desc = as_text(desc).strip()
    return key if not desc or desc == key else f"{key}: {desc}"


def to_task(q):
    """A kit question -> an `openjev.spec.Task` that `pairs.render` reads. Fixed for every
    benchmark (prereg §2): no per-benchmark wording."""
    from openjev.spec import DataSpec, Task
    data = DataSpec(source={"jsonl": "not-read"})
    instructions = as_text(q.get("instructions")).strip()
    if q["type"] == "noul":
        return Task(name="di", type="noul", data=data, statement=instructions,
                    labels=["true", "false"])
    if q["type"] != "choice":
        raise Unsupported(f"question type {q['type']!r}")
    keys = list(q["criteria"])
    return Task(name="di", type="choice", data=data, question=instructions,
                options=[option_line(k, q["criteria"][k]) for k in keys],
                labels=[option_line(k, q["criteria"][k]) for k in keys])


def resolve(path):
    """(local dir, revision). The Hub repo is a whole run dir; the weights sit in `student/`."""
    p = Path(path)
    if not p.exists():
        from huggingface_hub import snapshot_download
        p = Path(snapshot_download(str(path)))
    rev = p.name if len(p.name) == 40 else "local"
    return (p / "student" if (p / "student" / "config.json").exists() else p), rev


class OpenJevEngine(Engine):
    name = "open-jev-base"
    latency = ("In-process request wall time: rendering, tokenisation and one batched forward pass "
               "over every (question, option) pair of the request; excludes model loading.")

    def __init__(self, model=DEFAULT_MODEL, batch_size=256, max_len=512, device=None, **options):
        super().__init__(**options)
        from openjev import base, pairs
        self.pairs = pairs
        model_dir, rev = resolve(model)
        self.tok, self.model = base.load(str(model_dir), device)
        self.batch_size, self.max_len = int(batch_size), int(max_len)
        weights = Path(model_dir) / "model.safetensors"
        self.provenance = {
            "kind": "openjev-pair-scorer", "repo": str(model), "revision": rev,
            "weights_sha256": (hashlib.sha256(weights.read_bytes()).hexdigest()
                               if weights.exists() else None),
            "params": sum(p.numel() for p in self.model.parameters()),
            "device": str(next(self.model.parameters()).device),
            "precision": "float32 weights, bf16 autocast on CUDA",
            "max_len": self.max_len, "options_tokens": pairs.OPTIONS_TOKENS,
            "rendering": "openjev.pairs.render; option line `key: description`; noul statement = instructions",
        }

    def synchronize(self):
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def _pairs(self, state, questions):
        """[(question key, Task, [(seg_a, seg_b), ...])]; raises Unsupported on any over-long pair."""
        text = as_text(state)
        out = []
        for key, q in questions.items():
            task = to_task(q)
            k = 1 if task.type == "noul" else task.k
            flat = [self.pairs.render(task, i, text) for i in range(k)]
            lens = self.tok([a for a, _ in flat], [b for _, b in flat], truncation=False)["input_ids"]
            longest = max(len(x) for x in lens)
            if longest > self.max_len:
                raise Unsupported(f"maximum context length {self.max_len} tokens: question "
                                  f"{key!r} needs {longest}")
            out.append((key, task, flat))
        self.pairs._options_cache.clear()    # keyed by the label set; would grow over 110k requests
        return out

    def _sigmoids(self, flat):
        import torch
        from openjev.base import _encode
        dev = next(self.model.parameters()).device
        out = []
        with torch.no_grad():
            for i in range(0, len(flat), self.batch_size):
                b = [{"seg_a": a, "seg_b": s} for a, s in flat[i:i + self.batch_size]]
                with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                    out.append(self.model(**_encode(self.tok, b, dev)).logits.float().squeeze(-1).cpu())
        return torch.cat(out).sigmoid().clamp(1e-6, 1 - 1e-6).tolist()

    def __call__(self, state, questions):
        """Same math as `openjev.base.predict_logits`: per-option sigmoids renormalised over the
        options offered; `noul` is p(true) of its one pair."""
        plan = self._pairs(state, questions)
        p = self._sigmoids([pair for _, _, flat in plan for pair in flat])
        answers, raw, at = {}, {}, 0
        for key, task, flat in plan:
            s, at = p[at:at + len(flat)], at + len(flat)
            raw[key] = s
            if task.type == "noul":
                answers[key] = {"type": "noul", "noul": s[0]}
                continue
            z = sum(s)
            probs = {k: v / z for k, v in zip(questions[key]["criteria"], s)}
            answers[key] = {"type": "choice", "choice": max(probs, key=probs.get),
                            "probabilities": probs}
        return {"model": self.name, "answers": answers}, {"sigmoid": raw}
