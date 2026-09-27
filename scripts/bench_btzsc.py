#!/usr/bin/env python
"""BTZSC (arXiv 2603.11991) for `open-jev-base` -- the run preregistered in docs/prereg/bench-v1.md.

One file, in the prereg's order:

  --overlap-audit                          # data only, no model; exits 1 on a text leak
  --model <hub id> --type reranker --datasets ...   # environment sanity on a published model
  --model <path> --render choice|noul      # the run; score matrices cached under scores/
  --table                                  # bootstrap CI + the §4 decision, from the cache
  --t0 copa hellaswag                      # the per-row-option secondary

Nothing in `openjev/` changes: the wrapper is a `btzsc.models.base.BaseModel` whose
`predict_scores` renders the §2 Task and calls `openjev.base.predict_logits`.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results/bench/btzsc"
PUBLISHED = Path("/tmp/claude-1000/-home-spark-4551/1f465fa6-ebda-44e8-bb8f-c19302f68665/"
                 "scratchpad/btzsc-results/results")

QUESTION = "Which of these statements about the text is true?"   # prereg §2, fixed
CLEAN = ["yahootopics", "wikitoxic_insult", "wikitoxic_obscene", "wikitoxic_threat",
         "wikitoxic_toxicaggregated", "amazonpolarity", "imdb", "appreviews", "yelpreviews",
         "rottentomatoes", "empathetic", "biasframes_intent", "biasframes_offensive",
         "biasframes_sex", "trueteacher", "manifesto", "capsotu"]
# prereg §2: the text source was labelled by a teacher and trained on -> a labelled example
MIXTURE = {"agnews": ["agnews", "agnews-jev"], "banking77": ["banking77"],
           "emotiondair": ["emotion-dair-jev"], "massive": ["massive-en-jev"],
           "financialphrasebank": ["fin-sentiment-jev"]}
# prereg §2's K column, from Hub metadata; the harness derives K from the data itself and wins
# (amendment 2026-09-27: banking77 is K=72/3080 texts in the data, not the table's 77/2880).
K_PREREG = {"agnews": 4, "banking77": 77, "emotiondair": 6, "massive": 59,
              "financialphrasebank": 3, "yahootopics": 10, "wikitoxic_insult": 2,
              "wikitoxic_obscene": 2, "wikitoxic_threat": 2, "wikitoxic_toxicaggregated": 2,
              "amazonpolarity": 2, "imdb": 2, "appreviews": 2, "yelpreviews": 2,
              "rottentomatoes": 2, "empathetic": 32, "biasframes_intent": 2,
              "biasframes_offensive": 2, "biasframes_sex": 2, "trueteacher": 2,
              "manifesto": 56, "capsotu": 21}
LEAK_FRAC = 0.005


# ---------------------------------------------------------------- metrics / decision

def macro_f1(refs, preds, k):
    """sklearn f1_score(average="macro", zero_division=0) from a bincount confusion matrix."""
    cm = np.bincount(refs * k + preds, minlength=k * k).reshape(k, k)
    tp = np.diag(cm).astype(float)
    den = 2 * tp + (cm.sum(0) - tp) + (cm.sum(1) - tp)
    return float(np.where(den > 0, 2 * tp / np.maximum(den, 1), 0.0).mean())


def verdict(mean, lo):
    """prereg §4, verbatim. Thresholds are fixed and not touched after a number exists."""
    if lo >= 0.611:
        return "Best published model under 200M on BTZSC-17"
    if lo >= 0.575 and mean >= 0.581:
        return "Competitive with the published field at 140M"
    return "Below the published median at 140M"


def bootstrap(per_dataset, draws=2000, seed=0):
    """Paired bootstrap over texts, per dataset and (same draws) jointly over the mean.

    per_dataset: {name: (refs, preds, k)}. Returns {name: (lo, hi)} plus ("__mean__", (lo, hi))."""
    rng = np.random.default_rng(seed)
    names = list(per_dataset)
    samples = {n: np.empty(draws) for n in names}
    means = np.empty(draws)
    for b in range(draws):
        acc = []
        for n in names:
            refs, preds, k = per_dataset[n]
            idx = rng.integers(0, len(refs), len(refs))
            samples[n][b] = macro_f1(refs[idx], preds[idx], k)
            acc.append(samples[n][b])
        means[b] = float(np.mean(acc))
    out = {n: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
           for n, v in samples.items()}
    out["__mean__"] = (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))
    return out


# ---------------------------------------------------------------- data

def load_checked(name, cache_dir=None):
    """The harness's own loader plus the prereg's STOP assertions (§5)."""
    from btzsc.data import load_btzsc_dataset
    ds = load_btzsc_dataset(name, max_samples=None, cache_dir=cache_dir)
    assert ds.n_samples > 0 and len(ds.texts) == len(ds.references) == ds.n_samples
    assert len(ds.labels) == ds.n_classes > 1, f"{name}: K={ds.n_classes}"
    return ds


def assert_fixed_hypotheses(name, ds, cache_dir=None):
    """STOP: the K hypotheses must be identical across every row (the harness assumes it)."""
    from datasets import load_dataset
    raw = load_dataset("btzsc/btzsc", name=name, split="test", cache_dir=cache_dir)
    hyp, k = raw["hypothesis"], ds.n_classes
    assert len(raw) == len(hyp)
    assert len(hyp) // k == ds.n_samples, f"{name}: {len(hyp)} rows, {ds.n_samples} texts at K={k}"
    for i in range(0, (len(hyp) // k) * k, k):
        if hyp[i:i + k] != ds.labels:
            raise AssertionError(f"{name}: hypotheses differ at row {i}")
    return len(raw)


def is_leak(hit, n_texts):
    """prereg §5: > 0.5 % of a clean dataset's texts present in data/pairs.jsonl disqualifies it."""
    return hit / n_texts > LEAK_FRAC


# ---------------------------------------------------------------- the model wrapper

try:
    from btzsc.models.base import BaseModel
except ModuleNotFoundError:      # the audit/table/tests paths need no harness
    BaseModel = object


class OpenJevModel(BaseModel):
    """`open-jev-base` as a BTZSC adapter. A pointwise cross-encoder over (text, label) -> reranker."""

    model_type = "reranker"

    def __init__(self, path, render="choice", batch_size=256, scores_dir=None, max_len=512):
        from openjev import base
        self._base = base
        self.model_name = str(path)
        self.render = render
        self.batch_size = batch_size
        self.max_len = max_len
        self.scores_dir = Path(scores_dir) if scores_dir else None
        self.tokenizer, self.model = base.load(path)
        self.device = str(next(self.model.parameters()).device)
        self.n_params = sum(p.numel() for p in self.model.parameters())
        self.revision = getattr(self.model.config, "_commit_hash", None) or "unknown"
        self.precision_note = "float32 weights, bf16 autocast"
        self.dataset = "adhoc"      # set by evaluate(): names the cached score matrix

    def _task(self, labels, idx=None):
        from openjev.spec import DataSpec, Task
        data = DataSpec(source={"jsonl": "not-read"})
        if self.render == "noul":
            return Task(name=f"btzsc-{self.dataset}-{idx}", type="noul", data=data,
                        statement=labels[idx], labels=["true", "false"])
        return Task(name=f"btzsc-{self.dataset}", type="choice", data=data, question=QUESTION,
                    options=list(labels), labels=list(labels))

    def predict_scores(self, texts, labels, batch_size=None):
        """[N, K]. `choice`: renormalised log-probabilities over the K hypotheses (argmax is the
        harness's argmax). `noul`: log p(true) of one statement-per-label task."""
        cache = self.scores_dir / f"{self.dataset}.npy" if self.scores_dir else None
        if cache is not None and cache.exists():
            m = np.load(cache)
            assert m.shape == (len(texts), len(labels)), f"{cache}: {m.shape}"
            return m
        bs = batch_size or self.batch_size
        if self.render == "noul":
            cols = [self._base.predict_logits(self.tokenizer, self.model, self._task(labels, i),
                                              texts, bs, self.max_len)[:, 0]
                    for i in range(len(labels))]
            import torch
            m = torch.stack(cols, -1).numpy()
        else:
            m = self._base.predict_logits(self.tokenizer, self.model, self._task(labels), texts,
                                          bs, self.max_len).numpy()
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache, m.astype(np.float32))
        return m

    def predict(self, texts, labels, batch_size=None):
        return self.predict_scores(texts, labels, batch_size).argmax(axis=1)


def fix_qwen_reranker(m, pad_side=None, dtype=None):
    """STOP-rule-1 environment fix (amendment 2026-09-27): `btzsc.models.reranker` reads
    `logits[:, -1, :]` at the "yes"/"no" vocab ids -- CausalLM logits -- but loads the model with
    `AutoModelForSequenceClassification`, whose logits are `[B, 2]` over a randomly initialised
    head, and which refuses a batch > 1 without `config.pad_token_id`. Load the head the adapter
    actually indexes; nothing else about the adapter or our own scoring path changes."""
    from transformers import AutoModelForCausalLM
    m.model = AutoModelForCausalLM.from_pretrained(m.model_name, dtype=dtype).to(m.device).eval()
    if pad_side:
        m.tokenizer.padding_side = pad_side
    return m


def weights_sha(path=ROOT / "runs/base-none-v2/student/model.safetensors"):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resolve_student(path):
    """(local dir to load, revision). `openjev.hub` publishes the whole run dir, so the shipped
    weights sit in `student/` on the Hub -- `from_pretrained` needs the subdirectory."""
    p = Path(path)
    if not p.exists():
        from huggingface_hub import snapshot_download
        p = Path(snapshot_download(str(path)))
    rev = p.name if len(p.name) == 40 else "local"
    return (p / "student" if (p / "student" / "config.json").exists() else p), rev


def assert_shipped(model_dir):
    """STOP: the weights must be the shipped artefact (runs/base-none-v2/student)."""
    local = weights_sha()
    got = hashlib.sha256((Path(model_dir) / "model.safetensors").read_bytes()).hexdigest()
    assert got == local, f"{model_dir}/model.safetensors {got} != runs/base-none-v2 {local}"
    return local


# ---------------------------------------------------------------- runs

def evaluate(model, names, batch_size, cache_dir=None, ece_bins=10):
    """The harness's metrics over `names`; keeps the score matrix so ECE and the CI are free."""
    from btzsc.data import TASK_GROUPS
    from btzsc.metrics import compute_metrics, compute_task_summary
    per, extra = {}, {}
    for name in names:
        ds = load_checked(name, cache_dir)
        n_rows = assert_fixed_hypotheses(name, ds, cache_dir)
        if hasattr(model, "dataset"):
            model.dataset = name
        t0 = time.time()
        scores = model.predict_scores(ds.texts, ds.labels, batch_size)
        dt = time.time() - t0
        preds = scores.argmax(axis=1)
        per[name] = compute_metrics(predictions=preds, references=ds.references)
        extra[name] = {"n_texts": ds.n_samples, "n_pair_rows": n_rows, "k": ds.n_classes,
                       "k_prereg": K_PREREG[name],
                       "seconds": dt, "pairs_per_s": ds.n_samples * ds.n_classes / max(dt, 1e-9)}
        if getattr(model, "render", None) == "choice":
            import torch
            p = torch.from_numpy(np.asarray(scores)).float().exp()
            from openjev import calibrate
            extra[name]["ece"] = calibrate.ece(p, torch.from_numpy(ds.references), ece_bins)
        if getattr(model, "scores_dir", None):
            np.save(model.scores_dir / f"{name}.refs.npy", ds.references)
        print(f"{name}: macro_f1 {per[name]['macro_f1']:.4f} acc {per[name]['accuracy']:.4f} "
              f"({ds.n_samples} texts x K={ds.n_classes}, {extra[name]['pairs_per_s']:.0f} pairs/s)",
              flush=True)
    return per, compute_task_summary(per, TASK_GROUPS), extra


def write_result(model, per, summary, extra, out, model_name=None, params=None, revision=None):
    from btzsc.benchmark import BTZSCResults
    res = BTZSCResults(per_dataset_results=per, task_summary=summary,
                       model_name=model_name or model.model_name,
                       model_type=model.model_type, model_params=params,
                       model_revision=revision or "unknown",
                       precision=getattr(model, "precision_note", None) or str(
                           next(model.model.parameters()).dtype).replace("torch.", ""),
                       batch_size=getattr(model, "batch_size", 32), max_samples=None,
                       device=getattr(model, "device", "cpu"))
    out = Path(out)
    payload = res.to_json()
    # `_get_btzsc_commit` git-rev-parses upwards from site-packages and picks up *our* repo's HEAD.
    payload["evaluation"]["btzsc_commit"] = "n/a (btzsc==0.1.2 wheel from PyPI, no git checkout)"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    total_pairs = sum(v["n_pair_rows"] for v in extra.values())
    side = {"per_dataset": extra, "render": getattr(model, "render", None),
            "question": QUESTION if getattr(model, "render", None) == "choice" else None,
            "max_len": getattr(model, "max_len", None),
            "weights_sha256": getattr(model, "weights_sha256", None),
            "achieved_pairs_per_s": (sum(v["n_texts"] * v["k"] for v in extra.values())
                                     / sum(v["seconds"] for v in extra.values())),
            "total_pair_rows": total_pairs,
            "clean": CLEAN, "trained_on_this_source": sorted(MIXTURE)}
    out.with_name(out.stem + "-extras.json").write_text(json.dumps(side, indent=2))
    return payload


# ---------------------------------------------------------------- overlap audit

def our_texts(path=ROOT / "data/pairs.jsonl"):
    """(every exact `text` we trained on, every task name in the file)."""
    seen, tasks = set(), set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            seen.add(r["text"])
            tasks.add(r["task"])
    return seen, tasks


def overlap_audit(out=OUT / "overlap.json", cache_dir=None, pairs=ROOT / "data/pairs.jsonl"):
    from btzsc.data import BTZSC_DATASETS
    mine, trained = our_texts(pairs)
    report, stops = {}, []
    for name in BTZSC_DATASETS:
        ds = load_checked(name, cache_dir)
        hit = sum(1 for t in ds.texts if t in mine)
        frac = hit / ds.n_samples
        in_mixture = name in MIXTURE
        leak = is_leak(hit, ds.n_samples)
        report[name] = {"n_texts": ds.n_samples, "k": ds.n_classes, "exact_text_overlap": hit,
                        "overlap_frac": frac, "text_leak": leak,
                        "in_mixture": in_mixture, "mixture_tasks": MIXTURE.get(name, []),
                        "clean": name in CLEAN}
        if leak and not in_mixture:
            stops.append(f"{name}: {frac:.4%} of its texts are in data/pairs.jsonl")
        print(f"{name:28s} texts {ds.n_samples:6d} overlap {hit:5d} ({frac:.4%})"
              f"{'  MIXTURE' if in_mixture else ''}{'  LEAK' if leak else ''}", flush=True)
    missing = {d: [t for t in ts if t not in trained] for d, ts in MIXTURE.items()}
    payload = {"n_our_texts": len(mine), "leak_threshold": LEAK_FRAC, "datasets": report,
               "mixture_tasks_absent_from_pairs_jsonl": {k: v for k, v in missing.items() if v},
               "primary_set": [n for n in CLEAN if not report[n]["text_leak"]], "stops": stops}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(payload, indent=2))
    return payload


# ---------------------------------------------------------------- table

def published_17(published=PUBLISHED):
    """Every published model's mean macro-F1 over the 17 clean datasets."""
    out = {}
    for p in sorted(Path(published).glob("*/*.json")):
        d = json.loads(p.read_text())
        bd = d["results"]["by_dataset"]
        out[d["model"]["name"]] = {
            "params": d["model"]["params"],
            "mean17": float(np.mean([bd[c]["macro_f1"] for c in CLEAN])),
            "mean22": float(np.mean([v["macro_f1"] for v in bd.values()])),
        }
    return out


def table(result=OUT / "open-jev-base.json", scores=OUT / "scores", published=PUBLISHED,
          out=OUT / "table.md", draws=2000):
    data = json.loads(Path(result).read_text())
    bd = data["results"]["by_dataset"]
    per = {}
    for name in CLEAN:
        m = np.load(Path(scores) / f"{name}.npy")
        refs = np.load(Path(scores) / f"{name}.refs.npy")
        preds = m.argmax(1)
        point = macro_f1(refs, preds, m.shape[1])
        assert abs(point - bd[name]["macro_f1"]) < 1e-6, f"{name}: cache {point} vs json"
        per[name] = (refs, preds, m.shape[1])
    ci = bootstrap(per, draws=draws)
    mean = float(np.mean([bd[c]["macro_f1"] for c in CLEAN]))
    lo, hi = ci["__mean__"]
    field = published_17(published)
    med = float(np.median([v["mean17"] for v in field.values()]))
    (Path(out).parent / "field-17.json").write_text(json.dumps(
        {"source": "btzsc/btzsc-results, 35 model JSONs", "median17": med, "models": field}, indent=2))
    extras_path = Path(result).with_name(Path(result).stem + "-extras.json")
    extras = json.loads(extras_path.read_text()) if extras_path.exists() else {"per_dataset": {}}
    med_ds = {}
    for name in bd:
        vals = [json.loads(p.read_text())["results"]["by_dataset"][name]["macro_f1"]
                for p in sorted(Path(published).glob("*/*.json"))]
        med_ds[name] = float(np.median(vals))
    lines = [f"# BTZSC — `open-jev-base` (prereg docs/prereg/bench-v1.md)", "",
             f"17-clean mean macro-F1 **{mean:.4f}**, 95 % CI [{lo:.4f}, {hi:.4f}] "
             f"({draws} paired bootstrap draws over texts, seed 0)", "",
             f"**Verdict (§4): {verdict(mean, lo)}**", "",
             f"field median (35 models, 17 datasets) {med:.4f}; "
             f"22-dataset mean (all, incl. the 5 trained-on sources) "
             f"{np.mean([v['macro_f1'] for v in bd.values()]):.4f}", "",
             "| dataset | K | texts | macro-F1 | 95 % CI | acc | field median | ECE | status |",
             "|---|---|---|---|---|---|---|---|---|"]
    for name in list(CLEAN) + sorted(MIXTURE):
        e = extras["per_dataset"].get(name, {})
        c = ci.get(name)
        lines.append(
            f"| {name} | {e.get('k', K_PREREG[name])} | {e.get('n_texts', '')} | "
            f"{bd[name]['macro_f1']:.4f} | "
            f"{('[%.4f, %.4f]' % c) if c else '—'} | {bd[name]['accuracy']:.4f} | "
            f"{med_ds[name]:.3f} | {e.get('ece', float('nan')):.3f} | "
            f"{'clean' if name in CLEAN else 'trained on this source'} |")
    lines += ["", "## Published field, recomputed on the same 17 datasets", "",
              "| model | params | 17-mean | 22-mean |", "|---|---|---|---|"]
    for n, v in sorted(field.items(), key=lambda kv: -kv[1]["mean17"]):
        lines.append(f"| {n} | {v['params']} | {v['mean17']:.4f} | {v['mean22']:.4f} |")
    Path(out).write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:12]))
    return {"mean17": mean, "ci": [lo, hi], "verdict": verdict(mean, lo), "per_dataset_ci": ci,
            "field_median17": med}


# ---------------------------------------------------------------- T0 secondary

def t0(model_path, sets=("copa", "hellaswag"), out=ROOT / "results/bench/t0", batch_size=256,
       draws=2000, seed=0):
    from datasets import load_dataset
    from openjev import base
    from openjev.spec import DataSpec, Task
    local, revision = resolve_student(model_path)
    tok, model = base.load(local)
    data = DataSpec(source={"jsonl": "not-read"})
    res = {}
    for name in sets:
        if name == "copa":
            ds = load_dataset("aps/super_glue", "copa", split="validation")
            rows = [(r["premise"],
                     f"Which alternative is the more plausible {r['question']} of the premise?",
                     [r["choice1"], r["choice2"]], r["label"]) for r in ds]
        elif name == "hellaswag":
            ds = load_dataset("Rowan/hellaswag", split="validation")
            rows = [(r["ctx"], "Which ending most plausibly continues the text?",
                     list(r["endings"]), int(r["label"])) for r in ds]
        else:
            raise ValueError(name)
        t0_ = time.time()
        preds, k = [], len(rows[0][2])
        for text, q, options, _ in rows:
            task = Task(name=f"{name}-row", type="choice", data=data, question=q,
                        options=options, labels=list(options))
            preds.append(int(base.predict_logits(tok, model, task, [text], batch_size)[0].argmax()))
        dt = time.time() - t0_
        preds = np.array(preds)
        gold = np.array([r[3] for r in rows])
        acc = float((preds == gold).mean())
        rng = np.random.default_rng(seed)
        boot = np.array([float((preds[i] == gold[i]).mean())
                         for i in (rng.integers(0, len(gold), len(gold)) for _ in range(draws))])
        chance = 1.0 / k
        res[name] = {"n": len(gold), "k": k, "accuracy": acc, "chance": chance,
                     "advantage": acc - chance,
                     "advantage_ci": [float(np.percentile(boot, 2.5) - chance),
                                      float(np.percentile(boot, 97.5) - chance)],
                     "above_chance": bool(np.percentile(boot, 2.5) - chance > 0),
                     "seconds": dt, "pairs_per_s": len(gold) * k / dt}
        print(name, json.dumps(res[name]), flush=True)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "open-jev-base.json").write_text(json.dumps(
        {"model": str(model_path), "revision": revision, "weights_sha256": weights_sha(),
         "question_copa": "Which alternative is the more plausible "
         "{cause/effect} of the premise?", "results": res}, indent=2))
    return res


# ---------------------------------------------------------------- cli

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--overlap-audit", action="store_true")
    ap.add_argument("--model")
    ap.add_argument("--type", dest="model_type")
    ap.add_argument("--render", choices=("choice", "noul"), default="choice")
    ap.add_argument("--datasets", nargs="*")
    ap.add_argument("--out")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--table", action="store_true")
    ap.add_argument("--t0", nargs="*")
    ap.add_argument("--published", default=str(PUBLISHED))
    ap.add_argument("--no-cache-scores", action="store_true")
    ap.add_argument("--pad-side", choices=("left", "right"))
    ap.add_argument("--dtype", choices=("float32", "bfloat16", "float16"))
    ap.add_argument("--via-harness", action="store_true",
                    help="score the given datasets through BTZSCBenchmark.evaluate instead of our "
                         "own loop, as a check that the two agree (diagnostic)")
    a = ap.parse_args(argv)

    if a.overlap_audit:
        p = overlap_audit()
        print(json.dumps({"stops": p["stops"], "primary_set": len(p["primary_set"])}, indent=2))
        return 1 if p["stops"] else 0
    if a.table:
        table(published=a.published)
        return 0
    if a.t0 is not None:
        t0(a.model or "sshalimov04/open-jev-base", tuple(a.t0) or ("copa", "hellaswag"),
           batch_size=a.batch_size)
        return 0
    if not a.model:
        ap.error("need --model, --overlap-audit, --table or --t0")

    from btzsc.data import BTZSC_DATASETS
    names = a.datasets or BTZSC_DATASETS
    out = Path(a.out or OUT / f"{a.model.split('/')[-1]}.json")
    if a.model_type:      # a published baseline, through the harness's own adapter
        from btzsc.benchmark import BTZSCBenchmark
        model = BTZSCBenchmark._resolve_model(a.model, a.model_type)
        if getattr(model, "is_qwen_reranker", False):
            fix_qwen_reranker(model, a.pad_side, a.dtype)
        model.scores_dir = None
        model.render = None
        model.batch_size = a.batch_size
        params, revision = None, "published-baseline"
    else:
        scores = None if a.no_cache_scores else (
            OUT / ("scores" if a.render == "choice" else "scores-noul"))
        local, revision = resolve_student(a.model)
        model = OpenJevModel(local, render=a.render, batch_size=a.batch_size, scores_dir=scores)
        model.model_name = a.model          # the shipped id goes in the JSON, not the cache path
        model.weights_sha256 = assert_shipped(local)
        params = model.n_params
    if a.via_harness:      # diagnostic: the harness's own orchestrator over the same adapter
        from btzsc.benchmark import BTZSCBenchmark
        res = BTZSCBenchmark(tasks=list(names)).evaluate(model, batch_size=a.batch_size)
        print(json.dumps({k: round(v["macro_f1"], 6) for k, v in res.per_dataset_results.items()},
                         indent=2))
        return 0
    per, summary, extra = evaluate(model, names, a.batch_size)
    payload = write_result(model, per, summary, extra, out, params=params, revision=revision)
    print(json.dumps(payload["results"]["overall"], indent=2))
    print(f"17-clean mean: {np.mean([per[c]['macro_f1'] for c in CLEAN if c in per]):.4f}"
          if all(c in per for c in CLEAN) else "(subset run: no 17-mean)")
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
