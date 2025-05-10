#!/usr/bin/env python3
import argparse
import numpy as np
import random
import sys

def main():
    parser = argparse.ArgumentParser(
        description="Verify shapes and a few top-5 matches between base, queries, and ground_truth arrays."
    )
    parser.add_argument('--base',    default='base_vectors.npy',
                        help="Path to base_vectors.npy")
    parser.add_argument('--queries', default='queries.npy',
                        help="Path to queries.npy")
    parser.add_argument('--gt',      default='ground_truth.npy',
                        help="Path to ground_truth.npy")
    parser.add_argument('--top_k',   type=int, default=5,
                        help="Number of neighbors to verify")
    parser.add_argument('--checks',  type=int, default=3,
                        help="How many random queries to check")
    args = parser.parse_args()

    # Load
    try:
        base    = np.load(args.base)
        queries = np.load(args.queries)
        gt      = np.load(args.gt)
    except Exception as e:
        print(f"Failed to load files: {e}")
        sys.exit(1)

    # Shape checks
    if base.ndim != 2:
        print(f"base_vectors must be 2D, got shape {base.shape}")
        sys.exit(1)
    N, k = base.shape

    if queries.ndim != 2:
        print(f"queries must be 2D, got shape {queries.shape}")
        sys.exit(1)
    m, k2 = queries.shape

    if k2 != k:
        print(f"Dim mismatch: base has k={k}, queries have k={k2}")
        sys.exit(1)

    if gt.ndim != 2:
        print(f"ground_truth must be 2D, got shape {gt.shape}")
        sys.exit(1)
    m2, K = gt.shape

    if m2 != m:
        print(f"Row count mismatch: queries have m={m}, ground_truth have m={m2}")
        sys.exit(1)
    if K < args.top_k:
        print(f"ground_truth lists only K={K} neighbors, but top_k={args.top_k}")
        sys.exit(1)

    print(f"✅ Shapes OK: base={base.shape}, queries={queries.shape}, ground_truth={gt.shape}")

    # Randomly verify a few
    idxs = random.sample(range(m), min(args.checks, m))
    for qi in idxs:
        q = queries[qi]
        # squared L2 distances
        d2 = np.sum((base - q)**2, axis=1)
        # compute true top_k
        gt_idxs = gt[qi, :args.top_k]
        # brute-force top_k
        brute_idxs = np.argsort(d2)[:args.top_k]

        match = np.array_equal(gt_idxs, brute_idxs)
        status = "✔️" if match else "MISMATCH"
        print(f"\nQuery #{qi}:")
        print(f"  stored GT top-{args.top_k}: {gt_idxs.tolist()}")
        print(f"  brute-force   : {brute_idxs.tolist()}")
        print(f"  {status}")

    print("\nDone.")

if __name__ == '__main__':
    main()
