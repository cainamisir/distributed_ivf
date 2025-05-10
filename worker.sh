#!/usr/bin/env bash
#SBATCH -c 32
#SBATCH --mem=64G
#SBATCH -t 0-02:00


module load cuda/12.0.1-fasrc01 cudnn cmake

# init & activate the env
source /n/home13/asaluja/miniconda3/etc/profile.d/conda.sh
conda activate faiss_env

set -euo pipefail

PORT=${PORT:?}
SHARED_FILE=${SHARED_FILE:?}
LOCK_FILE=${LOCK_FILE:?}

echo "[worker $$] launching on $(hostname -f):${PORT}"
python new_server_loadbalancing.py --port "${PORT}" &
SERVER_PID=$!

sleep 2

(
  flock -x 200
  echo "$(hostname -f):${PORT}" >> "${SHARED_FILE}"
  echo "[worker $$] registered in ${SHARED_FILE}"
) 200>"${LOCK_FILE}"

wait "${SERVER_PID}"
