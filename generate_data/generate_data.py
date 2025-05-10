import argparse
import os
import numpy as np
import torch

def get_random_vectors(N, k, distribution, seed, device):
    torch.manual_seed(seed)
    if distribution == 'uniform':
        return torch.rand(N, k, device=device)
    elif distribution == 'normal':
        return torch.randn(N, k, device=device)
    else:
        raise ValueError(f"Unknown distribution: {distribution}")

def main():
    parser = argparse.ArgumentParser(
        description="Generate N base vectors of dimension k (using GPU if available)."
    )
    parser.add_argument('-N', '--num_vectors', type=int, required=True)
    parser.add_argument('-k', '--dim',         type=int, required=True)
    parser.add_argument('-d', '--distribution',type=str, default='uniform',
                        choices=['uniform', 'normal'])
    parser.add_argument('-s', '--seed',        type=int, default=42)
    parser.add_argument('-o', '--output',      type=str, default='base_vectors.npy')

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vectors = get_random_vectors(
        args.num_vectors, args.dim,
        args.distribution, args.seed,
        device
    )
    arr = vectors.cpu().numpy()

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    np.save(args.output, arr)

    print(f"Saved {args.num_vectors} vectors (dim={args.dim}) to '{args.output}' on {device}")

if __name__ == '__main__':
    main()
