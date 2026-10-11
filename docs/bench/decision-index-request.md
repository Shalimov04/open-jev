# Decision Index evaluation request: prepared, **not** posted

Nothing has been posted anywhere yet. This is the text of an evaluation request for the Decision Index
maintainers. Post it as an issue on [`apolinario/decision-index`](https://github.com/apolinario/decision-index/issues)
or as a discussion on the Space's Community tab
([`multimodalart/jev-decision-index`](https://huggingface.co/spaces/multimodalart/jev-decision-index/discussions)).

We have **not** run the public suite ourselves (prereg `docs/prereg/decision-index-v1.md`, amendment
2026-10-11). The kit's documented route is a self-run public suite plus a PR to `submissions/README.md`.
This request asks instead for a lab-run entry, which the board already shows for some entrants
("Public board run by the lab"). The maintainers may decline it and ask for a self-run submission.

---

## Title

    Evaluation request: open-jev-base (141M pair scorer), run with the kit's Engine interface

## Body

Could you add **open-jev-base** to the Decision Index? We have not run the public suite ourselves.
We're asking whether you would run it on the lab side, public and private parts, the same way as the
entrants marked "Public board run by the lab". If you only take self-run submissions, please tell us
and we will do that instead.

**Model.** [`sshalimov04/open-jev-base`](https://huggingface.co/sshalimov04/open-jev-base) is a 141M
pointwise cross-encoder (`jhu-clsp/mmBERT-small`, `num_labels=1`). It was distilled from teacher soft
labels on 40 typed-decision tasks. It scores one (question, option, state) pair per forward pass and
renormalises the sigmoids over the options offered, so option count and wording are free at inference.
The weights are in `student/` of the Hub repo (the repo holds the whole run directory). License: MIT.

**How to run it.** The kit's `Engine` interface, no server:

```sh
git clone https://github.com/Shalimov04/open-jev && pip install -e open-jev
PYTHONPATH=open-jev/scripts python -m decision_index pipeline \
    --engine bench_decision_index:OpenJevEngine \
    --option model=sshalimov04/open-jev-base --out runs/open-jev-base
```

Settings: float32 weights with bf16 autocast on CUDA, batch 256, `max_len` 512. The engine is
`scripts/bench_decision_index.py`, and the rendering is fixed for every benchmark:

- **choice:** option line `key: description`, `instructions` as the question, `state` as the second
  segment. Every option is scored on its own pair.
- **noul:** p(true) of one pair, with `instructions` as the statement.

**Please read before running: two disclosures.**

1. **BANKING77 is in our training mixture** (mteb/banking77 train). In a BTZSC overlap audit, 64.9 % of
   banking77 test texts appeared verbatim in our training file. Please treat our BANKING77 number, and
   its private counterpart if it is drawn from the same source, as trained-on.
2. **The model does not truncate, it refuses.** It was trained at 512 tokens. Any request with a pair
   longer than that raises `Unsupported`, which is a declared capacity limit. Expect low coverage on the
   long-context benchmarks (ContractNLI, RAGTruth, HoVer, MuSR, BRIGHT and others). Nothing is cut.

Latency should be far below the 1,000 ms limit: one batched encoder pass over K pairs per question.

Model: https://huggingface.co/sshalimov04/open-jev-base ·
Code: https://github.com/Shalimov04/open-jev
