#!/usr/bin/env bash
# Label the diversity corpus with Jev. Serial, resumable, budget-capped per task.
cd "$(dirname "$0")/.." && export PATH=$PWD/.venv/bin:$PATH
unset HTTP_PROXY ALL_PROXY http_proxy all_proxy
export OPENJEV_JEV_BUDGET=${OPENJEV_JEV_BUDGET:-0.40}
for f in tasks/base/*.yaml; do
  [ "$(basename "$f")" = folds.yaml ] && continue
  t=$(basename "$f" .yaml)
  [ -s "runs/$t-jev/teacher.jsonl" ] && continue
  echo "$(date +%T) label $t"
  openjev label "$f" --teacher jev >> "runs/${t}-jev.log" 2>&1 || echo "$(date +%T) FAILED $t"
done
echo "$(date +%T) base labelling done"
