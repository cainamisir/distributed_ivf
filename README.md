To run the codebase, you first must have `faiss`, `grpc`, `protobuf`, `numpy` and `tqdm` alongside a Python 3 implementation. With this setup, you can run this code on almost any CPU cluster.

We use the following setup:
`32 CPU Cores, 64GBs of Memory`

We tested our implementation with a dataset containing 4 million vectors of dimension 1000. You might have to toggle the hyperparameters if your dataset is vastly different.

First spawn the worker nodes by running `spawn_workers.sh N` where `N` represents number of worker nodes.

```
$ ./spawn_workers N
```

Then edit `run.sh` to link to your database, queries, and ground truth. We use npy files but with a minimal change to the dataloader, you can use any files. If you would like to not test against groundtruth, you can just comment out the last section in main.

With these configurations updates, you should run `./run.sh`

```
$ ./run.sh
```

This creates the index on the worker nodes and performs the queries, return their global index.
