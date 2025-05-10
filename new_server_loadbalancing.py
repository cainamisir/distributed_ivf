import argparse, time, grpc, numpy as np, faiss
from array import array
from concurrent.futures import ThreadPoolExecutor

import faiss_service_pb2
import faiss_service_pb2_grpc


class FaissServicer(faiss_service_pb2_grpc.FaissServiceServicer):
    def __init__(self):
        faiss.omp_set_num_threads(32) # set number of threads to what our system has
        self._shards  = {}  
        self._indexes = {} 

    # ---------- shard streaming ----------
    def SendShard(self, req_iter, context):
        buf, dim, cid = array("f"), None, None
        for msg in req_iter:
            dim = dim or msg.dim
            cid = cid or msg.cluster_id
            buf.extend(msg.data)

        if cid is None:
            return faiss_service_pb2.ShardResponse(ok=False,
                                                   error="missing cluster_id")

        X = np.frombuffer(buf, dtype="float32").reshape(-1, dim)
        self._shards[cid] = X
        print(f"[worker] got shard for C{cid}: {X.shape}")
        return faiss_service_pb2.ShardResponse(ok=True)

    def TrainIndex(self, req, context):
        cid, nlist = req.cluster_id, req.nlist
        if cid in self._indexes:
            return faiss_service_pb2.TrainResponse(ok=True)

        X = self._shards.get(cid)
        if X is None:
            return faiss_service_pb2.TrainResponse(
                ok=False, error="no shard for this cid"
            )

        n, d = X.shape
        print(f"[worker] training IVF C{cid} (n={n}, d={d}, nlist={nlist})")
        ivf = faiss.IndexIVFFlat(faiss.IndexFlatL2(d), d, nlist, faiss.METRIC_L2)
        ivf.train(X)
        for i in range(0, n, 100_000):
            ivf.add(X[i:min(n, i + 100_000)])
        ivf.nprobe = 10
        self._indexes[cid] = ivf
        del self._shards[cid]
        print(f"[worker] index built for C{cid}")
        return faiss_service_pb2.TrainResponse(ok=True)


    # ---------- batched search ----------
    def SearchBatch(self, req, context):
        out = faiss_service_pb2.SearchBatchResponse()
        for sub in req.requests:
            idx = self._indexes.get(sub.cluster_id)
            if idx is None:
                out.responses.append(
                    faiss_service_pb2.SearchResponse(
                        error=f"no index for C{sub.cluster_id}"
                    )
                )
                continue
            q = np.array([sub.query_vector], dtype="float32")
            idx.nprobe = sub.nprobe
            D, I = idx.search(q, sub.k)
            out.responses.append(
                faiss_service_pb2.SearchResponse(
                    distances=D[0].tolist(),
                    indices=I[0].tolist()
                )
            )
        return out


def serve(port):
    server = grpc.server(
        ThreadPoolExecutor(max_workers=4),
        options=[
            ("grpc.max_send_message_length",    200 * 1024 * 1024),
            ("grpc.max_receive_message_length", 200 * 1024 * 1024),
        ],
    )
    faiss_service_pb2_grpc.add_FaissServiceServicer_to_server(
        FaissServicer(), server
    )
    server.add_insecure_port(f"0.0.0.0:{port}")
    server.start()
    print(f"[worker] listening on :{port}")
    try:
        while True:
            time.sleep(86400)
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=50051)
    args = ap.parse_args()
    serve(args.port)
