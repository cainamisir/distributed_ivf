import argparse
import os
import numpy as np
import torch
from tqdm import tqdm

def get_random_vectors(m, k, distribution, seed, device):
    torch.manual_seed(seed)
    if distribution == 'uniform':
        return torch.rand(m, k, device=device)
    elif distribution == 'normal':
        return torch.randn(m, k, device=device)
    else:
        raise ValueError(f"Unknown distribution: {distribution}")

def compute_ground_truth(base, queries, top_k, batch_size=1024):
    m, k = queries.shape
    N, _ = base.shape

    base_norms = (base * base).sum(dim=1)  # on GPU

    gt = torch.empty((m, top_k), dtype=torch.long, device='cpu')

    num_batches = (m + batch_size - 1) // batch_size
    for b in tqdm(range(num_batches), desc="Computing ground truth"):
        start = b * batch_size
        end   = min(m, start + batch_size)
        qb    = queries[start:end]
        q_norms = (qb * qb).sum(dim=1)
        dots = qb @ base.T
        d2 = base_norms.unsqueeze(0) + q_norms.unsqueeze(1) - 2 * dots
        _, idx = torch.topk(d2, top_k, largest=False)
        gt[start:end] = idx.cpu()

    return gt.numpy()

def main():
    parser = argparse.ArgumentParser(
        description="Generate m queries and their top-k ground truth (GPU-optimized)."
    )
    parser.add_argument('-b', '--base',         type=str, default='base_vectors.npy')
    parser.add_argument('-m', '--num_queries',  type=int, default=100000)
    parser.add_argument('-d', '--distribution', type=str, default='uniform',
                        choices=['uniform', 'normal'])
    parser.add_argument('-s', '--seed',         type=int, default=123)
    parser.add_argument('--top_k',              type=int, default=10)
    parser.add_argument('-o', '--query_output', type=str, default='queries.npy')
    parser.add_argument('-g', '--gt_output',    type=str, default='ground_truth.npy')
    parser.add_argument('--batch_size',         type=int, default=1024,
                        help="Batch size for distance computation")

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_np = np.load(args.base)
    base = torch.from_numpy(base_np).to(device)
    N, k = base_np.shape
    print(f"Loaded base: N={N}, dim={k} on {device}")

    queries = get_random_vectors(
        args.num_queries, k, args.distribution, args.seed, device
    )
    os.makedirs(os.path.dirname(args.query_output) or '.', exist_ok=True)
    np.save(args.query_output, queries.cpu().numpy())
    print(f"Saved {args.num_queries} queries to '{args.query_output}'")

    gt_arr = compute_ground_truth(base, queries, args.top_k, args.batch_size)
    os.makedirs(os.path.dirname(args.gt_output) or '.', exist_ok=True)
    np.save(args.gt_output, gt_arr)
    print(f"Saved ground truth (top {args.top_k}) to '{args.gt_output}'")

if __name__ == '__main__':
    main()
