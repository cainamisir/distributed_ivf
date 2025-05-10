import argparse, time, grpc, numpy as np, faiss
from array import array
from concurrent.futures import ThreadPoolExecutor

import faiss_service_pb2
import faiss_service_pb2_grpc
import os


class FaissServicer(faiss_service_pb2_grpc.FaissServiceServicer):
    def __init__(self, index_path=None):
        self._shard = None
        self.index  = None
        self.index_path = index_path
        if index_path and os.path.exists(index_path):
            print(f"[worker] Loading index from {index_path}")
            self.index = faiss.read_index(index_path)

    def SendShard(self, request_iterator, context):
        if self.index is not None:
            return faiss_service_pb2.ShardResponse(ok=True)
        buf = array('f')
        dim = None
        for req in request_iterator:
            if dim is None:
                dim = req.dim
            buf.extend(req.data)
        X = np.frombuffer(buf, dtype='float32').reshape(-1, dim)
        self._shard = X
        print(f"[worker] Received shard of shape {X.shape}")
        return faiss_service_pb2.ShardResponse(ok=True)

    def TrainIndex(self, request, context):
        if self.index is not None:
            return faiss_service_pb2.TrainResponse(ok=True)
        X = self._shard
        n, d = X.shape
        print(f"[worker] Training IVF (nlist={request.nlist}) on {n}×{d}")
        quant = faiss.IndexFlatL2(d)
        ivf   = faiss.IndexIVFFlat(quant, d, request.nlist, faiss.METRIC_L2)
        ivf.train(X)
        block = 100_000
        for i in range(0, n, block):
            ivf.add(X[i : min(n, i + block)])
        ivf.nprobe = 10
        self.index = ivf
        print("[worker] Training complete")
        if self.index_path:
            faiss.write_index(self.index, self.index_path)
            print(f"[worker] Saved index to {self.index_path}")
        return faiss_service_pb2.TrainResponse(ok=True)

    def SearchBatch(self, request, context):
        out = faiss_service_pb2.SearchBatchResponse()
        for req in request.requests:
            q = np.array([req.query_vector], dtype='float32')
            self.index.nprobe = req.nprobe
            D, I = self.index.search(q, req.k)
            out.responses.append(
                faiss_service_pb2.SearchResponse(
                    distances=D[0].tolist(),
                    indices=I[0].tolist()
                )
            )
        return out

def serve(port, index_path):
    server = grpc.server(
        ThreadPoolExecutor(max_workers=4),
        options=[
            ('grpc.max_send_message_length',    200 * 1024 * 1024),
            ('grpc.max_receive_message_length', 200 * 1024 * 1024),
        ]
    )
    servicer = FaissServicer(index_path=index_path)
    faiss_service_pb2_grpc.add_FaissServiceServicer_to_server(servicer, server)
    server.add_insecure_port(f'0.0.0.0:{port}')
    server.start()
    print(f"[worker] serving on port {port}")
    try:
        while True:
            time.sleep(86400)
    except KeyboardInterrupt:
        server.stop(0)

if __name__ == "__main__":
    import os
    p = argparse.ArgumentParser()
    p.add_argument('--port',       type=int,   default=50051)
    p.add_argument('--index-path', type=str,   help='path to load/save index')
    args = p.parse_args()
    serve(args.port, args.index_path)
