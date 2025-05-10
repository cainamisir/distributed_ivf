python new_client_loadbalancing.py \
  --data /n/idreos_lab/users/1/aadit_tori/DistributedIVF/data/100000x10_4000000_uniform_s44/base_vectors.npy \
  --queries /n/idreos_lab/users/1/aadit_tori/DistributedIVF/data/100000x10_4000000_uniform_s44/queries.npy \
  --groundtruth /n/idreos_lab/users/1/aadit_tori/DistributedIVF/data/100000x10_4000000_uniform_s44/ground_truth.npy \
  --addresses addresses.txt \
  --num-clusters 64 \
  --fanout 64 \
  --nprobe 1 \
  --k 5

