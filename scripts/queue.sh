#!/usr/bin/env bash
# Overnight queue: labeling runs sequentially in one lane; train -> calibrate -> eval of task N
# runs in a second lane as soon as its labels are done, overlapping labeling of task N+1.
# Usage: scripts/queue.sh [task ...]   (default: all).  Extra args for `openjev run`: OPENJEV_RUN_ARGS.
cd "$(dirname "$0")/.." && export PATH=$PWD/.venv/bin:$PATH
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
tasks=("$@")
[ $# -eq 0 ] && tasks=(agnews kinopoisk georeview toxic headlines banking77)

for t in "${tasks[@]}"; do  # label lane (a concurrent labeler of the same task waits on runs/<t>/label.lock)
  mkdir -p runs/$t
  openjev label tasks/$t.yaml >> runs/$t/label.log 2>&1 && touch runs/$t/.labeled
done &

for t in "${tasks[@]}"; do  # train lane
  while [ ! -e runs/$t/.labeled ]; do sleep 30; done
  echo "$(date +%T) run $t"
  openjev run tasks/$t.yaml $OPENJEV_RUN_ARGS >> runs/$t/run.log 2>&1 || echo "$(date +%T) FAILED $t (see runs/$t/run.log)"
done
wait
