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
  rm -f runs/$t/.labelfail
  openjev label tasks/$t.yaml >> runs/$t/label.log 2>&1 && touch runs/$t/.labeled || touch runs/$t/.labelfail
done &
label_lane=$!

for t in "${tasks[@]}"; do  # train lane
  # wait for this task's labels; stop waiting if its labeling failed or the whole label lane died
  while [ ! -e runs/$t/.labeled ] && [ ! -e runs/$t/.labelfail ] && kill -0 $label_lane 2>/dev/null; do
    sleep 30
  done
  if [ ! -e runs/$t/.labeled ]; then  # skip this task only; the queue moves on to the next one
    echo "$(date +%T) SKIP $t (labeling did not finish, see runs/$t/label.log)"
    continue
  fi
  echo "$(date +%T) run $t"
  openjev run tasks/$t.yaml $OPENJEV_RUN_ARGS >> runs/$t/run.log 2>&1 || echo "$(date +%T) FAILED $t (see runs/$t/run.log)"
done
wait
