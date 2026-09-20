# Serving

`openjev serve runs/agnews runs/kinopoisk --port 8099` loads every run dir it is given and routes on
the `model` field. `--device cpu` forces CPU (the default is CUDA when it is there). The forward
pass is 5 ms on GPU and 30 ms on CPU for mmBERT-small (the [results table](../README.md#results)); measured end to end through
HTTP on a busy box, one request is ~21 ms on GPU and ~160 ms on CPU, so most of the CPU number is
tokenisation and framing, not the model. Either way CPU-only serving is a real option, not a
fallback.

**Batch.** `input` may be a list. One forward pass, one list back, same order:

```bash
curl -s localhost:8099/v1/systemone -H 'Content-Type: application/json' \
  -d '{"model":"agnews","input":["Shares of the airline fell 8% after it cut its forecast.",
                                 "Manchester United beat Arsenal 2-1."]}'
```

**Escalation.** `eval` writes the cascade *parity point* into `openjev.json` as `escalate_below`: the
lowest confidence threshold at which "student answers when confident, teacher answers the rest"
matches the teacher's own metric. Serve returns `"escalate": true/false` against it, and
`escalate_below` in the request overrides it. **The server never calls the teacher** — that is your
decision to make, and it keeps the teacher's URL, key and concurrency out of this process:

```python
r = httpx.post(f"{OPENJEV}/v1/systemone", json={"model": "kinopoisk", "input": text}).json()
answer = ask_the_llm(text) if r["escalate"] else r["choice"]
```

**Prediction sets.** `"coverage": 0.9` adds `"set": [...]`, a split-conformal (LAC) set built from
the calib split's `1 − p_true` quantile — it contains the right label ~90 % of the time, and it is
wider exactly where the model is unsure. `"set_target"` says what "right" was measured against:
`gold`, or `teacher` when the calib split had no gold labels, in which case the set covers *the
teacher's* answer and not the truth.

```bash
curl -s localhost:8099/v1/systemone -H 'Content-Type: application/json' \
  -d '{"model":"kinopoisk","input":"Ну такое. Актёры старались, но сценарий провальный.","coverage":0.9}'
# {"model":"kinopoisk","choice":"Bad",
#  "probabilities":{"Bad":0.6697,"Neutral":0.2994,"Good":0.0309},"confidence":0.6697,
#  "escalate":false,"set":["Bad","Neutral"],"set_target":"gold"}
```

`GET /v1/models` lists what is loaded, with each model's `escalate_below` and whether it has a
conformal calibration.

### Share a trained student

```bash
openjev push runs/agnews --repo you/openjev-agnews --dry-run   # prints the file list and the card
openjev push runs/agnews --repo you/openjev-agnews             # private; --public to publish
openjev serve hf:you/openjev-agnews --port 8099                # snapshot_download, then as usual
```

`push` uploads `student/`, `openjev.json`, `conformal.json` and a generated model card — the task
YAML, the teacher id, the results row, the calibration method, and the "no gold in the loss"
sentence when that is what the run actually did. It does **not** upload `teacher.jsonl`, so the
teacher's labels stay yours. Needs a Hugging Face token with write access (`hf auth login`).

### Comparing against another endpoint

`scripts/compare_api.py` scores any `/v1/systemone`-shaped endpoint on the same eval rows this
repo's numbers come from, and writes `results/api/api-<name>-<task>.json` with accuracy, argmax
agreement with our student, and p50 latency:

```bash
python scripts/compare_api.py --adapter openjev --url http://localhost:8099 --task agnews --limit 200
JEV_API_KEY=... python scripts/compare_api.py --adapter jev --task agnews --limit 200
```

The `jev` adapter targets [TypeSafe's Jev](https://docs.typesafe.ai/api.md), whose request shape
(`state` + a `questions` map of Choice/Score/Noul) this project's own endpoint deliberately mirrors.
It is written from the published docs and has not been run against the live API — there is no key
here. Latency across two providers is two different machines and is not a comparison; accuracy and
agreement on identical rows are.
