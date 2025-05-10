import argparse
import math
import time
from concurrent.futures import ThreadPoolExecutor

import grpc
import numpy as np
from tqdm import tqdm

import faiss_service_pb2
import faiss_service_pb2_grpc

CHANNEL_OPTS = [
    ('grpc.max_send_message_length',    200 * 1024 * 1024),
    ('grpc.max_receive_message_length', 200 * 1024 * 1024),
]

CHUNK_SIZE = 10**6

BATCH_SIZE = 1024


def load_workers(path):
    addrs, idxs = [], []
    with open(path) as f:
        for line in f:
            host, port, index_path = line.strip().split(":", 2)
            addrs.append(f"{host}:{port}")
            idxs.append(index_path)
    return addrs, idxs


def send_and_train(shard, dim, addr, nlist, cid):
    stub = faiss_service_pb2_grpc.FaissServiceStub(
        grpc.insecure_channel(addr, options=CHANNEL_OPTS)
    )
    flat = shard.astype("float32").ravel()
    total = flat.size
    num_chunks = math.ceil(total / CHUNK_SIZE)

    def gen():
        for i in tqdm(range(num_chunks), desc=f"[C{cid}] send", leave=False):
            start = i * CHUNK_SIZE
            end = min(total, start + CHUNK_SIZE)
            yield faiss_service_pb2.ShardRequest(
                data=flat[start:end].tolist(), dim=dim
            )

    resp = stub.SendShard(gen())
    if not resp.ok:
        raise RuntimeError(f"[C{cid}] SendShard failed @ {addr}")

    tr = stub.TrainIndex(faiss_service_pb2.TrainRequest(nlist=nlist))
    if not tr.ok:
        raise RuntimeError(f"[C{cid}] TrainIndex failed: {tr.error}")

    tqdm.write(f"[C{cid}] training done")


def search_phase(queries, stubs, centroids, k, nprobe):
    total = 0
    t0 = time.time()
    NQ = queries.shape[0]

    for bs in tqdm(range(0, NQ, BATCH_SIZE), desc="query batches"):
        be = min(NQ, bs + BATCH_SIZE)
        batch_q = queries[bs:be]

        nearest = np.argmin(
            ((centroids[None, :, :] - batch_q[:, None, :])**2).sum(-1),
            axis=1
        )

        # group by stub
        worker_to_local = {w: [] for w in range(len(stubs))}
        for i, w in enumerate(nearest):
            worker_to_local[w].append(i)

        # launch all RPCs asynchronously
        futures = []
        for w, stub in enumerate(stubs):
            idxs = worker_to_local[w]
            if not idxs:
                continue
            req = faiss_service_pb2.SearchBatchRequest()
            for i in idxs:
                req.requests.append(
                    faiss_service_pb2.SearchRequest(
                        query_vector=batch_q[i].tolist(),
                        k=k,
                        nprobe=nprobe,
                    )
                )
            fut = stub.SearchBatch.future(req)
            futures.append(fut)

        # now we collect all results
        for fut in futures:
            resp = fut.result()        
            total += len(resp.responses)

    dt = time.time() - t0
    avg = dt / total if total else 0
    print(f"\n→ {total} total results in {dt:.2f}s "
          f"→ {total/dt:.1f} calls/sec "
          f"→ {avg:.4f}s/query")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data",        required=True, help="(N,D) base vectors .npy")
    p.add_argument("--assignments", required=True, help="(N,) cluster IDs .npy")
    p.add_argument("--centroids",   required=True, help="(K,D) centroids .npy")
    p.add_argument("--queries",     required=True, help="(Q,D) query vectors .npy")
    p.add_argument("--addresses",   required=True,
                   help="file of host:port:index_path")
    p.add_argument("--nlist",       type=int, default=1024)
    p.add_argument("--k",           type=int, default=10)
    p.add_argument("--nprobe",      type=int, default=5)
    p.add_argument("--no-send",     action="store_true",
                   help="skip shard send & train, assume indexes loaded")
    args = p.parse_args()

    X = np.load(args.data).astype("float32")
    A = np.load(args.assignments).astype("int64")
    C = np.load(args.centroids).astype("float32")
    Q = np.load(args.queries).astype("float32")
    addrs, index_paths = load_workers(args.addresses)

    if X.shape[0] != A.shape[0] or C.shape[0] != len(addrs):
        raise ValueError("Mismatch in data, assignments, centroids, or addresses")

    dim = X.shape[1]
    shards = [X[A == i] for i in range(len(addrs))]

    if not args.no_send:
        with ThreadPoolExecutor(max_workers=len(shards)) as exe:
            list(tqdm(
                exe.map(
                    lambda t: send_and_train(*t),
                    [(shards[i], dim, addrs[i], args.nlist, i)
                     for i in range(len(shards))]
                ),
                total=len(shards),
                desc="train clusters",
            ))
    else:
        print("Skipping shard send & train (--no-send enabled)")

    stubs = [
        faiss_service_pb2_grpc.FaissServiceStub(
            grpc.insecure_channel(addr, options=CHANNEL_OPTS)
        )
        for addr in addrs
    ]

    search_phase(Q, stubs, C, args.k, args.nprobe)

if __name__ == "__main__":
    main()
