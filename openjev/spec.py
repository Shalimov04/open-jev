"""Task spec: YAML -> Task dataclass. Downstream code only needs `labels` (+ `values` for score).

Task fields (all read-only after load):
  name, type ("choice"|"score"|"noul"), question (choice/score only), lang, path (yaml path)
  options: list[str]            # choice only
  rubric: {levels: [...], descriptions: [...]}  # score only
  statement: str                # noul only
  labels: list[str]             # derived: choice -> options; score -> str(level); noul -> ["true","false"]
  values: list[float] | None    # score only: numeric level per label (for expectation)
  data: DataSpec, teacher: TeacherSpec, student: StudentSpec   (see dataclasses below)
  k -> len(labels)
"""
from dataclasses import dataclass, field, fields
from pathlib import Path

import yaml


@dataclass
class SplitSpec:
    split: str = "train"
    n: int = 1000
    balance: bool = False


@dataclass
class DataSpec:
    source: dict  # {hf: id, config?, revision?, slice?} | {csv: path} | {jsonl: path}
    text: str = "{text}"
    max_chars: int = 2000
    seed: int = 0     # one sampling seed for all three splits (they are drawn in order)
    gold: str | None = None
    gold_map: list | dict | None = None
    gold_prob: str | None = None  # optional column with a fractional gold probability (noul)
    gold_threshold: float = 0.5   # noul with float gold: gold = 0 (true) if value >= threshold else 1
    train: SplitSpec = field(default_factory=SplitSpec)
    calib: SplitSpec = field(default_factory=lambda: SplitSpec(n=500))
    eval: SplitSpec = field(default_factory=lambda: SplitSpec(split="test", n=2000))


@dataclass
class TeacherSpec:
    concurrency: int = 32
    max_options_per_call: int = 19


@dataclass
class StudentSpec:
    model: str = "jhu-clsp/mmBERT-small"
    max_len: int = 256
    epochs: int = 5
    lr: float = 5e-5
    batch_size: int = 32
    gold_weight: float = 0.0


@dataclass
class Task:
    name: str
    type: str
    data: DataSpec
    question: str | None = None  # required for choice/score; optional head override for noul
    lang: str = "en"
    options: list[str] | None = None
    rubric: dict | None = None
    statement: str | None = None
    teacher: TeacherSpec = field(default_factory=TeacherSpec)
    student: StudentSpec = field(default_factory=StudentSpec)
    labels: list[str] = field(default_factory=list)
    values: list[float] | None = None
    path: str | None = None

    @property
    def k(self) -> int:
        return len(self.labels)


def _build(cls, d):
    d = d or {}
    known = {f.name for f in fields(cls)}
    unknown = set(d) - known
    if unknown:
        raise ValueError(f"{cls.__name__}: unknown keys {sorted(unknown)}")
    return cls(**d)


def load_task(path) -> Task:
    raw = yaml.safe_load(Path(path).read_text())
    derived = sorted({"labels", "values", "path"} & set(raw))
    if derived:
        raise ValueError(f"Task: {derived} are derived from type/options/rubric, not settable")
    data = dict(raw.pop("data"))
    for s in ("train", "calib", "eval"):
        if s in data:
            data[s] = _build(SplitSpec, data[s])
    t = _build(Task, {**raw, "data": _build(DataSpec, data), "path": str(path),
                      "teacher": _build(TeacherSpec, raw.get("teacher")),
                      "student": _build(StudentSpec, raw.get("student"))})
    if t.type in ("choice", "score") and not t.question:
        raise ValueError(f"{t.type} task needs a question")
    if t.type == "choice":
        if not t.options:
            raise ValueError("choice task needs options")
        t.labels = list(t.options)
    elif t.type == "score":
        levels = (t.rubric or {}).get("levels")
        if not levels:
            raise ValueError("score task needs rubric.levels")
        t.labels = [str(v) for v in levels]
        t.values = [float(v) for v in levels]
    elif t.type == "noul":
        if not t.statement:
            raise ValueError("noul task needs statement")
        t.labels = ["true", "false"]
    else:
        raise ValueError(f"unknown type {t.type!r}")
    return t
