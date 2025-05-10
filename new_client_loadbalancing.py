import argparse
import random
import time
from concurrent.futures import ThreadPoolExecutor

import grpc
import numpy as np
from tqdm import tqdm
import faiss

import faiss_service_pb2
import faiss_service_pb2_grpc

CHANNEL_OPTS = [
    ("grpc.max_send_message_length",    200 * 1024 * 1024),
    ("grpc.max_receive_message_length", 200 * 1024 * 1024),
]
BATCH_SIZE = 64


def kmeans_level1(x, k, seed):
    d = x.shape[1]
    km = faiss.Kmeans(
        d, k,
        niter=20, seed=seed, verbose=False, spherical=False,
        min_points_per_centroid=1024, max_points_per_centroid=1024,
    )
    km.train(x)
    _, I = km.index.search(x, 1)
    return km.centroids.copy(), I.ravel().astype("int64")


def load_workers(path):
    addrs, _ = [], []
    with open(path) as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            hostport, *_ = ln.split(",")
            addrs.append(hostport)
    return addrs


def send_cluster(stub, cid, shard, fanout):
    def it():
        yield faiss_service_pb2.ShardRequest(
            dim=shard.shape[1],
            cluster_id=cid,
            data=shard.astype("float32").ravel().tolist(),
        )
    if not stub.SendShard(it()).ok:
        raise RuntimeError(f"[C{cid}] SendShard failed")
    if not stub.TrainIndex(
        faiss_service_pb2.TrainRequest(cluster_id=cid, nlist=fanout)
    ).ok:
        raise RuntimeError(f"[C{cid}] TrainIndex failed")
    tqdm.write(f"[C{cid}] index ready")


def distribute_clusters(X, C, A, stubs, fanout, c2n):
    with ThreadPoolExecutor(max_workers=len(stubs)) as ex: # should experiment with the workers, no time left
        futs = []
        for cid in range(len(C)):
            wid = c2n[cid]
            shard = X[A == cid]
            n_pts = shard.shape[0]
            if n_pts == 0:
                tqdm.write(f"[main] cluster {cid} empty, skipping")
                continue
            fanout_i = min(fanout, n_pts)
            tqdm.write(f"[main] → cluster {cid:3d} ({shard.shape[0]} vecs) → worker {wid}, , fanout={fanout_i}")
            futs.append(ex.submit(send_cluster, stubs[wid], cid, shard, fanout_i))
        for _ in tqdm(futs, desc="building worker indexes", total=len(futs)):
            _.result()


def search(q, C, c2n, stubs, k, nprobe):
    nq = q.shape[0]
    preds = np.zeros((nq, k), dtype=np.int64)
    routed = np.zeros(nq, dtype=np.int64)
    t0 = time.time()

    for b0 in tqdm(range(0, nq, BATCH_SIZE), desc="query batches"):
        be = min(nq, b0 + BATCH_SIZE)
        batch = q[b0:be]

        nearest = np.argmin(((C[None] - batch[:, None])**2).sum(-1), axis=1)

        by_w = {}
        for li, cid in enumerate(nearest):
            wid = c2n[cid]
            by_w.setdefault(wid, []).append((li, cid))
            routed[b0+li] = cid

        calls = {}
        for wid, items in by_w.items():
            req = faiss_service_pb2.SearchBatchRequest()
            for li, cid in items:
                s = req.requests.add()
                s.cluster_id    = cid
                s.query_vector.extend(batch[li].tolist())
                s.k, s.nprobe   = k, nprobe
            calls[wid] = stubs[wid].SearchBatch.future(req)

        for wid, fut in calls.items():
            resp = fut.result()
            items = by_w[wid]
            for (li, cid), r in zip(items, resp.responses):
                preds[b0 + li] = np.array(r.indices, dtype=np.int64)

    dt = time.time() - t0
    print(f"\n→ {nq} queries in {dt:.2f}s ({nq/dt:.1f} q/s)")
    return preds, routed


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data",        required=True, help="(N,D) vectors .npy")
    p.add_argument("--queries",     required=True, help="(Q,D) queries .npy")
    p.add_argument("--addresses",   required=True, help="file of host:port[,path]")
    p.add_argument("--groundtruth", required=True, help="(Q,10) gt idxs .npy")
    p.add_argument("--num-clusters", type=int, default=128)
    p.add_argument("--fanout",       type=int, default=64)
    p.add_argument("--k",            type=int, default=10)
    p.add_argument("--nprobe",       type=int, default=10)
    p.add_argument("--seed",         type=int, default=1243)
    p.add_argument("--no-send", action="store_true")
    args = p.parse_args()

    print(f"BATCH SIZE: {BATCH_SIZE}")

    # load
    X  = np.load(args.data).astype("float32")
    Qv = np.load(args.queries).astype("float32")
    GT = np.load(args.groundtruth).astype("int64")  # shape (Q,10)

    # 1) level‑1 KMeans
    print(f"[main] KMeans K={args.num_clusters} …")
    C, A = kmeans_level1(X, args.num_clusters, args.seed)
    print("[main] clustering done")

    # build global‑id map per cluster
    cluster_to_global = {
        cid: np.where(A == cid)[0] for cid in range(args.num_clusters)
    }

    # 2) split among workers
    addrs = load_workers(args.addresses)
    rnd = random.Random(args.seed)
    order = list(range(args.num_clusters)); rnd.shuffle(order)
    c2n = {cid: order[cid] % len(addrs) for cid in order}
    print("[main] cluster→node map:")
    for cid, w in c2n.items():
        print(f"  • C{cid:3d} → W{w}")

    # grpc stubs
    stubs = [
        faiss_service_pb2_grpc.FaissServiceStub(
            grpc.insecure_channel(addr, options=CHANNEL_OPTS)
        )
        for addr in addrs
    ]

    # 3) build indexes
    if not args.no_send:
        distribute_clusters(X, C, A, stubs, args.fanout, c2n)
    else:
        print("[main] skipping distribution (--no-send)")

    # 4) search + debug
    preds_local, routed = search(Qv, C, c2n, stubs, args.k, args.nprobe)

    # 5) DEBUG: print first 5 queries, their local & global preds, and GT
    for i in range(min(100, Qv.shape[0])):
        cid = routed[i]
        local_ids = preds_local[i]
        global_ids = cluster_to_global[cid][local_ids]
        print(f"\n=== Query {i} (cluster={cid}) ===")
        # print("  vector:", Qv[i].tolist())
        print("  local preds:", local_ids.tolist())
        print("  global preds:", global_ids.tolist())
        print("  groundtruth:", GT[i].tolist())

    # 6) correct recall using global IDs

    n_queries = preds_local.shape[0]

    for k in (1, 5, 10):
        hit_count = 0

        for i in range(n_queries):
            cid        = routed[i]
            local_ids  = preds_local[i, :k]                  
            global_ids = cluster_to_global[cid][local_ids]   

            if any(g in GT[i] for g in global_ids):
                hit_count += 1

        recall_at_k = hit_count / n_queries
        print(f"Recall@{k}: {recall_at_k * 100:.2f}%")
    # for k in (1, 5, 10):
    #     recall_scores = []
    #     hit_rates    = []

    #     for i in range(preds_local.shape[0]):
    #         cid        = routed[i]
    #         local_ids  = preds_local[i, :k]                    
    #         global_ids = cluster_to_global[cid][local_ids]     

    #         true_ids   = np.asarray(GT[i])                     
    #         # how many of those top‑k were correct?
    #         n_hits     = np.intersect1d(global_ids, true_ids).size

    #         recall_scores.append(n_hits / len(true_ids))       
    #         hit_rates.append(1.0 if n_hits > 0 else 0.0)        

    #     print(f"Recall@{k}:   {np.mean(recall_scores) * 100:.2f}%")
    #     print(f"Hit rate@{k}: {np.mean(hit_rates) * 100:.2f}%")


if __name__ == "__main__":
    main()
