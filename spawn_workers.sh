#!/usr/bin/env bash
# Usage: spawn_workers.sh NUM_WORKERS
#
# Launch NUM_WORKERS Slurm jobs, each running a FAISS worker that
# (a) binds to PORT=50051+ID
# (b) appends "<hostname>:<PORT>" to addresses.txt exactly once.

set -euo pipefail

NUM_WORKERS=${1:-}
if [[ -z "$NUM_WORKERS" ]]; then
  echo "Usage: $0 NUM_WORKERS"
  exit 1
fi

# Shared state
SHARED_FILE="$(pwd)/addresses.txt"
LOCK_FILE="${SHARED_FILE}.lock"

# Clean slate
: >"$SHARED_FILE"
: >"$LOCK_FILE"

BASE_PORT=50051   # every worker gets BASE_PORT+ID

for ID in $(seq 0 $((NUM_WORKERS - 1))); do
  PORT=$((BASE_PORT + ID))
  sbatch --job-name=faissW$ID \
         --output=logs/worker_%j.out \
         --export=ALL,ID=${ID},PORT=${PORT},SHARED_FILE=${SHARED_FILE},LOCK_FILE=${LOCK_FILE} \
         worker.sh
  echo "Submitted worker $ID  →  will listen on *:${PORT}"
done
