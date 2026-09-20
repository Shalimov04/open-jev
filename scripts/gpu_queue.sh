#!/usr/bin/env bash
# Serial GPU job runner. One job at a time (flock), each under choom so the host OOM killer picks
# the trainer and not the 84 GB vLLM on :8000 -- which is production and is never touched here.
#
#   scripts/gpu_queue.sh runs/queue2.jobs        # one `openjev ...` command per line, # = comment
#
# Before each job: wait until CUDA free >= 12 GB AND host available >= 10 GB. The queue waits; it
# never shrinks the batch size (a different batch size is a different experiment). Log: runs/queue2.log
set -uo pipefail
cd "$(dirname "$0")/.." && export PATH=$PWD/.venv/bin:$PATH
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy   # the teacher is localhost

JOBS=${1:-runs/queue2.jobs}
LOG=runs/queue2.log
CUDA_GB=${CUDA_GB:-12}
HOST_GB=${HOST_GB:-10}
mkdir -p runs
log() { echo "$(date +%F' '%T) $*" | tee -a "$LOG"; }

gate() {  # CUDA free GB (via torch: this box reports N/A to nvidia-smi) and host available GB
  local cuda host
  cuda=$(python -c 'import torch;print(int(torch.cuda.mem_get_info()[0]/2**30))' 2>/dev/null || echo 0)
  host=$(free -g | awk '/^Mem:/{print $7}')
  [ "$cuda" -ge "$CUDA_GB" ] && [ "$host" -ge "$HOST_GB" ] && return 0
  echo "$cuda $host"
  return 1
}

run_one() {
  local cmd=$1 waited=0 free
  until free=$(gate); do
    [ $((waited % 300)) -eq 0 ] && log "WAIT (cuda/host free: $free GB, need $CUDA_GB/$HOST_GB) -- $cmd"
    sleep 30; waited=$((waited + 30))
  done
  log "START $cmd"
  # choom -n 1000: if the host runs out of memory, this process dies, vLLM lives
  if choom -n 1000 -- bash -c "$cmd" >> "$LOG" 2>&1; then log "OK    $cmd"; else log "FAIL  $cmd (rc=$?)"; fi
}

[ -f "$JOBS" ] || { echo "no job file: $JOBS" >&2; exit 1; }
log "queue start: $JOBS"
exec 9>runs/.gpu.lock
flock 9                              # one training at a time, across queues and hand-started runs
while IFS= read -r line; do
  line=${line%%#*}; line=$(echo "$line" | xargs)   # strip comment + surrounding whitespace
  [ -z "$line" ] && continue
  run_one "$line"
done < "$JOBS"
log "queue done: $JOBS"
