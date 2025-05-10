import argparse
import time

import numpy as np
import faiss

def main():
    p = argparse.ArgumentParser(
        description="Sequential baseline: load vectors, build IVF, then run batched queries and time throughput"
    )
    p.add_argument(
        "--data", required=True,
        help="Path to .npy file containing your (4e6, D) float32 vectors"
    )
    p.add_argument(
        "--nlist", type=int, default=4096,
        help="Number of IVF clusters (default: 4096)"
    )
    p.add_argument(
        "--queries", required=True,
        help="Path to .npy file containing your (Q, D) float32 query vectors"
    )
    p.add_argument(
        "--batch_size", type=int, default=128,
        help="Number of queries to search per batch (default: 128)"
    )
    p.add_argument(
        "--nprobe", type=int, default=10,
        help="IVF nprobe parameter for search (default: 10)"
    )
    p.add_argument(
        "--k", type=int, default=10,
        help="Number of nearest neighbours to return (default: 10)"
    )
    args = p.parse_args()

    # 1) load training vectors
    print(f"[1/4] Loading database vectors from {args.data} …")
    X = np.load(args.data).astype("float32")
    N, d = X.shape
    print(f" Loaded {N:,} vectors of dimension {d}")

    # 2) train IVF
    quantizer = faiss.IndexFlatL2(d)
    ivf = faiss.IndexIVFFlat(quantizer, d, args.nlist, faiss.METRIC_L2)

    print(f"[2/4] Training IVF (nlist={args.nlist}) …")
    t0 = time.time()
    ivf.train(X)
    train_time = time.time() - t0
    print(f" Training done in {train_time:.2f} s")

    # 3) add all database vectors
    print(f"[3/4] Adding {N:,} vectors to the index …")
    t0 = time.time()
    ivf.add(X)
    add_time = time.time() - t0
    print(f" Add done in {add_time:.2f} s (ntotal={ivf.ntotal:,})")

    # 4) load queries and run batched search
    print(f"[4/4] Loading {args.queries} and running batched search…")
    Q = np.load(args.queries).astype("float32")
    Qn, dq = Q.shape
    if dq != d:
        raise ValueError(f"Query dim {dq} != database dim {d}")

    ivf.nprobe = args.nprobe
    print(f" {Qn:,} queries, batch_size={args.batch_size}, nprobe={args.nprobe}, k={args.k}")

    total_search_time = 0.0
    for i in range(0, Qn, args.batch_size):
        batch = Q[i : i + args.batch_size]
        t1 = time.time()
        D, I = ivf.search(batch, args.k)
        total_search_time += (time.time() - t1)

    throughput = Qn / total_search_time
    print(f" Total search time: {total_search_time:.2f} s")
    print(f" Throughput: {throughput:,.1f} queries/s")

    # summary
    print("\nSummary:")
    print(f"  Train time: {train_time:.2f} s")
    print(f"  Add time:   {add_time:.2f} s")
    print(f"  Search time:{total_search_time:.2f} s")
    print(f"  QPS:        {throughput:,.1f}")

if __name__ == "__main__":
    main()
