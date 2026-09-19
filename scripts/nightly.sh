#!/usr/bin/env bash
# Tail of the overnight run: one augment round on headlines once banking77 is done,
# then an idle-GPU re-measurement of agnews latency and a fresh README table.
cd "$(dirname "$0")/.." && export PATH=$PWD/.venv/bin:$PATH
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
until [ -f results/banking77.json ]; do sleep 120; done
echo "$(date +%T) augment headlines"
openjev augment tasks/headlines.yaml --rounds 1 --per-class 100 >> runs/headlines/augment.log 2>&1 || echo "FAILED augment"
echo "$(date +%T) re-eval agnews on an idle GPU"
openjev run tasks/agnews.yaml --force eval >> runs/agnews/run.log 2>&1 || echo "FAILED agnews eval"
openjev report
echo "$(date +%T) done"
