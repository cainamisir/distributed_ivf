#!/bin/bash
#SBATCH -c 12                                        # Number of cores (-c)
#SBATCH --gres=gpu:1                                 # GPU
#SBATCH -t 0-12:00                                   # Runtime in D-HH:MM
#SBATCH --mem=32000                                  # Memory pool for all cores
#SBATCH -o ./gen_data_%j.out                         # STDOUT → gen_data_<jobid>.out
#SBATCH -e ./gen_data_%j.err                         # STDERR → gen_data_<jobid>.err
#SBATCH -p gpu_test                                  # Partition
# SBATCH --constraint=a100|h100                      

############################################
# Usage:
#   sbatch generate.sh [N] [k] [distribution] [seed] [m] [run_test]
#
#   N            = # base vectors    (default: 100)
#   k            = vector dimension  (default: 1000)
#   distribution = uniform|normal    (default: uniform)
#   seed         = RNG seed          (default: 42)
#   m            = # query vectors   (default: 10)
#   run_test     = 1=verify, 0=skip   (default: 1)
############################################

N=${1:-4000000}
k=${2:-1000}
distribution=${3:-uniform}
seed=${4:-44}
m=${5:-100000}
run_test=${6:-1}

DATA_ROOT="/n/idreos_lab/users/1/aadit_tori/DistributedIVF/data"
SUBDIR="${DATA_ROOT}/${m}x10_${N}_${distribution}_s${seed}"
mkdir -p "$SUBDIR"

BASE_F="${SUBDIR}/base_vectors.npy"
QUERY_F="${SUBDIR}/queries.npy"
GT_F="${SUBDIR}/ground_truth.npy"

echo ">>> Params: N=$N, k=$k, dist=$distribution, seed=$seed, m=$m, run_test=$run_test"
echo ">>> Writing into: $SUBDIR"
echo "    base_vectors → $BASE_F"
echo "    queries      → $QUERY_F"
echo "    ground_truth → $GT_F"

echo ">>> Generating base vectors"
python generate_data.py \
    -N $N -k $k -d $distribution -s $seed \
    -o "$BASE_F"

echo ">>> Generating queries & ground truth"
python generate_gnd_truth.py \
    --base       "$BASE_F" \
    --num_queries $m \
    --distribution $distribution \
    --seed        $seed \
    --top_k       10 \
    --query_output "$QUERY_F" \
    --gt_output    "$GT_F" \
    --batch_size   512

if [[ "$run_test" -eq 1 ]]; then
  echo ">>> Verifying (top-5 on 3 random queries)"
  python test_generated_data.py \
      --base    "$BASE_F" \
      --queries "$QUERY_F" \
      --gt      "$GT_F" \
      --top_k   5 \
      --checks  3
else
  echo ">>> Skipping verification step"
fi

echo ">>> All done."
