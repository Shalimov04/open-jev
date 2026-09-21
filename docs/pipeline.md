# The pipeline, stage by stage

What `openjev run` does between a task YAML and a served endpoint. Field reference for the YAML itself: [`task-spec.md`](task-spec.md). Results and what they mean: [`findings.md`](findings.md).


1. **Task** — one YAML: type (`choice` / `score` / `noul`), the question, the option set, where the
   text comes from, split sizes.
2. **Teacher** — a local vLLM endpoint (`OPENJEV_TEACHER_URL`, default `http://localhost:8000/v1`)
   is asked one constrained question per example; the request is exactly:

   ```json
   {"model": "<first id from /v1/models>", "max_tokens": 1, "temperature": 0,
    "logprobs": true, "top_logprobs": 4,
    "messages": [{"role": "system", "content": "<question>\nA: World\nB: Sports\nC: Business\nD: Sci/Tech\nAnswer with a single letter."},
                 {"role": "user", "content": "<the text>"}],
    "structured_outputs": {"choice": ["A", "B", "C", "D"]},
    "chat_template_kwargs": {"enable_thinking": false}}
   ```

   Letters come back as bare tokens; `softmax` over their logprobs is the soft label. `finish_reason:
   "length"` is expected. More than 19 options → chunked shortlist (`top_logprobs` caps at 20).
3. **Student** — `AutoModelForSequenceClassification` (default `jhu-clsp/mmBERT-small`) trained on
   `KL(teacher ‖ student)`, optionally plus a gold CE term; early stop on calibration-split KL.
4. **Calibrate** — `logits / T` fitted by LBFGS on the calib split's NLL, or `logits / T + b` with a
   per-class bias `b ∈ R^K` when that wins a 2-fold held-out NLL test on the same rows. The bias is
   what removes the teacher's shifted marginal ([findings.md](findings.md#the-calibration-bias)).
   With no gold it targets a declared `prior:`.
5. **Serve** — `openjev serve runs/<task>`: the calibrated probability vector goes through one view
   per type (`openjev/views.py`) and comes back as a typed JSON decision.

Everything lands in `runs/<task>/`: append-only `teacher.jsonl`, `student/`, `calib.json`,
`eval.json`, and `openjev.json` (spec + labels + temperature + metrics — the exportable bundle).

## Targeted synthetic data (`openjev augment`)

`openjev augment tasks/<task>.yaml --rounds 1 --per-class 100` is a PGKD-lite active loop. It reads
the per-class recall and the confusion matrix that `eval` measured **on the calib split** (never on
eval), picks the weakest classes and the most frequent confusion pairs, and asks the teacher — normal
generation, `temperature 0.9`, `enable_thinking: false`, JSON-list output — for new texts in the
task's language, few-shot-prompted with three real train examples of that class. Confusion pairs get
the "make half of them look like *B* but really be *A*" variant. The parser is lenient (JSON array if
one can be found, else one text per line); duplicates (against the whole train split and within the
batch) and texts over `max_chars` are dropped.

The survivors are appended to the same append-only `teacher.jsonl` as `{"source": "synth", "split":
"train"}` rows with ids `synth:<round>:<i>`, which the dataset can never produce (dataset ids are
`<source_split>:<row_idx>`), and then labeled by the **normal** label stage. The teacher often
disagrees with the class the text was generated for; that label is kept as-is — the point is targeted
data near the decision boundary, not more hard labels. Train, calibrate and eval then re-run, and
`results/<task>.json` carries `augment_round` and `n_synth`.

Ceiling: no hard-negative mining beyond the confusion pairs, and no filter on whether a synthetic
text is actually useful; the upgrade path is the full PGKD selection loop.
