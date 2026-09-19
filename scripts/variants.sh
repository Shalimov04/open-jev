#!/usr/bin/env bash
# GPU-only variants, run while the teacher is busy labeling: a bigger student,
# and one task trained with gold labels on top of distillation.
cd "$(dirname "$0")/.." && export PATH=$PWD/.venv/bin:$PATH
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
for t in agnews georeview kinopoisk; do
  echo "$(date +%T) $t base"
  openjev run tasks/$t.yaml --student jhu-clsp/mmBERT-base >> runs/$t-mmBERT-base.log 2>&1 || echo "FAILED $t base"
done
echo "$(date +%T) kinopoisk gold"
openjev run tasks/kinopoisk.yaml --gold-weight 1.0 >> runs/kinopoisk-gold.log 2>&1 || echo "FAILED kinopoisk gold"
echo "$(date +%T) done"
