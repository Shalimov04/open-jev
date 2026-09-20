# Findings v2 — draft bullets (scratch, agent R / M2)

Not the README. M4 merges these. Every number traces to `results/*.json` or a table in
`docs/experiments.md`; every Δ carries a paired-bootstrap 95 % CI from `openjev compare`.
One-seed deltas are marked **(1 seed, docs-only)** and must not reach README Findings.

## R1 — 500 gold labels are worth more as a calibration bias than as a CE term

- **A K-dim bias on the student's logits, fitted on the 500 gold calib rows, is the single largest
  post-hoc win in the repo**: headlines +7.5 pts (95 % CI +6.0..+9.1), kinopoisk +5.1 pts
  (95 % CI +2.9..+7.5), georeview MAE −0.17 (95 % CI −0.19..−0.16). No teacher calls, no
  retraining, CPU, seconds. (1 seed per run, docs-only until the 3-seed arms are folded in.)
- **With no gold at all it is almost as good**, if you can declare the class prior: headlines
  +7.5 pts, kinopoisk +4.7 pts (95 % CI +2.3..+7.1) against a declared `prior: uniform`. The caveat
  ships with it: these benchmarks are balanced by construction, so `uniform` is a *leak* a real
  deployment does not get for free.
- **Where it does nothing**: banking77 (+0.2 pts, CI spans 0) and toxic (AUROC cannot move under a
  monotone per-class shift). The bias removes a marginal shift; where there is none it does nothing.
- <!-- GOLD-SPEND BULLET: filled when runs/kinopoisk-gold{,-n500} land -->

## R2 — the cascade curve

- Four of nine runs reach teacher parity at **0 % escalation** — after the bias the student matches
  or beats its own teacher on the eval split. On headlines escalating *costs* accuracy (0.845 → 0.783).
- Where escalation buys something: banking77 0.758 → 0.776 at 25 %, kinopoisk 0.660 → 0.673 at 25 %,
  georeview MAE 0.606 → 0.583 at 25 %, agnews 0.889 → 0.897 at 10 %.
- May not claim: any saving in teacher *quality*. Escalation routes to the same zero-shot teacher
  that produced the training labels, scored on the rows it labelled.

## R3 — honesty: the knobs that turned out not to matter

- **Augmentation (PGKD-lite): no measurable difference.** 3 seeds per arm, only the 522 synthetic
  train rows varying: Δ = +0.0 ± 0.5 pts (95 % CI −0.5..+0.6). Seed 1 has the opposite sign to
  seeds 0 and 2 — the earlier "+0.5 pt gain" was one seed.
- **Epochs: the default stays at 5.** 12 epochs vs 5, one factor, seed 0: agnews +0.0 pts
  (95 % CI −0.6..+0.6), headlines +0.7 (95 % CI −0.2..+1.6), kinopoisk +0.0 (95 % CI −1.5..+1.6) —
  Δ is 5-epoch minus 12-epoch, so neither arm wins anywhere. **(1 seed, docs-only.)** No 12-epoch
  run ever reached epoch 12 (patience 2 stopped them at 9, 7 and 5); `--epochs` sets the LR-decay
  horizon, and moving that horizon 2.4× changed nothing measurable.
- **Seed noise is the floor everything else is measured against**: 3 seeds move accuracy 0.6–1.2 pts
  peak-to-peak on identical data (agnews 0.888 ± 0.006, headlines 0.838 ± 0.005,
  kinopoisk 0.657 ± 0.003). Every one-seed delta above sits inside that band.
- **The teacher's option-order bias is semantic, not positional** — the diagnostic ran, the
  averaging feature was not built (limitation sentence below).
- <!-- 8K BULLET: filled when runs/headlines-8k{,-s1,-s2} land -->

## Limitations sentences (verbatim, for the README)

> Reversing the option order moves the teacher's *Good* rate on kinopoisk from 56.4 % to 41.9 %
> (gold 33.4 %) and its accuracy from 0.646 to 0.693 — but the *Neutral* collapse survives both
> orders (12.9 % and 12.0 % against a gold 33.7 %), and averaging the two orders makes it worse
> (9.0 %). The bias is semantic, not positional, so openjev labels each row once and removes the
> marginal shift post-hoc with the calibration bias instead — on kinopoisk that is
> `[+0.80 Bad, +0.59 Neutral, −1.39 Good]`, fitted on 500 calib rows, for zero extra teacher calls.
