# Preregistration — `<experiment id>`

Copy to `docs/prereg/<id>.md`, fill it in, commit it **before** the arms are labeled or trained, and
link it from the `experiments.md` section. The point is not ceremony: it is that the decision rule
exists before the number does. (Shape borrowed from r-ms/mini-jev's `PREREG.md`.)

## 0. What had been read before the freeze

Every number from this data that the author had already seen, and how it was obtained. "Nothing"
is a valid answer and the only one that needs no detail. A peek is not a sin; an undisclosed one is.

## 1. The decision

One sentence: what changes if this comes out one way rather than the other (ship / do not ship,
keep the knob / drop it, which default).

## 2. Arms

| arm | run dir | what differs |
|---|---|---|
| A | `runs/<x>` | baseline |
| B | `runs/<x>-<variant>` | the one factor that varies |

Everything else held fixed (teacher labels and their `prompt_sha`, split ids, student checkpoint,
`max_len`, `lr`, `batch_size`, calibration rule). Seeds: `<0,1,2>`.

## 3. The metric

| | |
|---|---|
| name | `acc` / `mae` / `auroc` (the task's `metric_fn`) |
| numerator | e.g. eval rows whose calibrated argmax equals gold |
| denominator | e.g. all eval rows with gold |
| unit | pts / MAE units |
| matching rule | rows are paired by `id` across arms and seeds |
| read from | `results/<run>.json`, via `openjev compare A B` |

Secondary readouts (diagnostic, never a claim): ECE, Brier, agreement, cascade parity, marginal.

## 4. What counts as a result

- **Improvement**: paired bootstrap 95 % CI on Δ excludes 0 (`experiments.md` claim rule).
- **Non-inferiority**: the lower bound of the 95 % CI is above `−<margin> pts` — state the margin
  here, before the run, and why that much is acceptable.
- **Inconclusive**: CI contains 0 *and* `openjev compare` reports a resolvable half-width larger
  than the margin. That is not a null result and must not be written as one.

## 5. STOP rules (any one voids the comparison)

- `openjev check --probe` shows letter emission < 0.90 or candidate mass median < 0.5.
- `label.json`'s `prompt_sha` or `model` differs between the arms.
- A run fails or is resumed under a changed prompt (`openjev label` refuses; do not `--force`).
- The arms share fewer than `<n>` eval rows.
- Any eval-fitted quantity leaks into the arm (fit on calib only).
- `<anything else that would make you distrust the number — name it now>`

## 6. Amendments

Append only, each dated, each stating **what had been read when it was written** and what changed.

| date | what had been read | change | why |
|---|---|---|---|
