#!/usr/bin/env bash
# Teacher lane for the real-use-case track (PLAN-3 §4): serial, ~81 min total.
cd "$(dirname "$0")/.." && export PATH=$PWD/.venv/bin:$PATH
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
for t in arb-success arb-quality m2w-element m2w-target swde-field; do
  echo "$(date +%T) label $t"
  openjev label tasks/$t.yaml >> runs/$t/label.log 2>&1 || echo "$(date +%T) FAILED $t"
done
echo "$(date +%T) teacher lane done"
