import numpy as np
import faiss
import time
import os

def cluster_with_faiss_gpu(
    npy_file_path,
    n_clusters,
    random_seed=42, 
    n_init=30,     
    max_iter=300, 
    verbose=True
):
    if not os.path.exists(npy_file_path):
        raise FileNotFoundError(f"The file {npy_file_path} was not found.")

    print(f"Loading data from {npy_file_path}...")
    vectors = np.load(npy_file_path).astype(np.float32)

    num_vectors, num_dimensions = vectors.shape
    print(f"Data loaded: {num_vectors} vectors, {num_dimensions} dimensions.")
    print(f"Data type: {vectors.dtype}")

    if n_clusters <= 0:
        raise ValueError("Number of clusters (K) must be a positive integer.")
    if n_clusters > num_vectors:
        raise ValueError("Number of clusters cannot exceed the number of vectors.")

    print(f"Initializing FAISS K-Means for GPU with K={n_clusters} clusters...")

    kmeans = faiss.Kmeans(
        d=num_dimensions,
        k=n_clusters,
        niter=max_iter,    
        nredo=n_init,     
        verbose=verbose,
        gpu=True,        
        seed=random_seed
    )

    print("Starting FAISS K-Means training on GPU...")
    start_time = time.time()

    kmeans.train(vectors)

    end_time = time.time()
    duration = end_time - start_time
    print(f"FAISS K-Means training completed in {duration:.2f} seconds.")

    cluster_centers = kmeans.centroids

    print("Assigning all vectors to clusters...")

    try:
        gpu_resources = faiss.StandardGpuResources() 
        index_gpu = faiss.index_cpu_to_gpu(gpu_resources, 0, faiss.IndexFlatL2(num_dimensions)) # 0 is the GPU ID
        index_gpu.add(vectors) 
        index_for_assignment_cpu = faiss.IndexFlatL2(num_dimensions)
        index_for_assignment_gpu = faiss.index_cpu_to_gpu(gpu_resources, 0, index_for_assignment_cpu)
        index_for_assignment_gpu.add(kmeans.centroids) 
        D, cluster_assignments = index_for_assignment_gpu.search(vectors, 1)
        cluster_assignments = cluster_assignments.ravel() 

    except AttributeError as e:
        print(f"Note: GPU index for assignment had an issue ({e}), or for clarity, using CPU-based index with trained centroids.")
        index_cpu = faiss.IndexFlatL2(num_dimensions)
        index_cpu.add(kmeans.centroids) 
        D, cluster_assignments = index_cpu.search(vectors, 1) 
        cluster_assignments = cluster_assignments.ravel()


    print("Assignments complete.")

    return cluster_assignments, cluster_centers, duration

if __name__ == "__main__":
    file_path = "/n/idreos_lab/users/1/aadit_tori/DistributedIVF/data/100000x10_4000000_uniform_s44/base_vectors.npy"

    num_example_vectors = 4_000_000
    num_example_dimensions = 1000 

    if not os.path.exists(file_path):
        print(f"Warning: Your specified file_path '{file_path}' was not found.")
        file_path = "large_dataset_base_vectors.npy"
        if not os.path.exists(file_path):
            print(f"Creating a dummy dataset at {file_path} for demonstration...")
            dummy_data = np.random.rand(num_example_vectors, num_example_dimensions).astype(np.float32)
            np.save(file_path, dummy_data)
            print(f"Dummy dataset created with shape {dummy_data.shape} and dtype {dummy_data.dtype}.")
            del dummy_data
        else:
            print(f"Using existing local dummy dataset at {file_path}.")
    else:
        print(f"Using your specified dataset at {file_path}.")

    N = 4  

    try:
        print("\nAttempting GPU K-Means using FAISS...")
        assignments, centers, time_taken = cluster_with_faiss_gpu(
            npy_file_path=file_path,
            n_clusters=N,
            random_seed=123, 
            n_init=10,
            max_iter=50, 
            verbose=True
        )

        print(f"\n--- FAISS GPU Clustering Results ---")
        print(f"Successfully clustered {assignments.shape[0]} vectors into {N} clusters using FAISS on GPU.")
        print(f"Time taken for training: {time_taken:.2f} seconds.")
        print(f"Shape of assignment table (on CPU): {assignments.shape}")
        print(f"Example assignments (first 20): {assignments[:20]}")
        print(f"Shape of cluster centers (on CPU): {centers.shape}")

        np.save("cluster_assignments_faiss_gpu.npy", assignments)
        np.save("cluster_centers_faiss_gpu.npy", centers)

    except ImportError:
        print("Error: FAISS not found. Please ensure 'faiss-gpu' is installed correctly.")
    except RuntimeError as e:
        if " spécialistes " in str(e) or "no GPU support" in str(e).lower() or "failed to create clusterer_default" in str(e).lower():
            print(f"FAISS GPU Error: {e}. This might mean FAISS was not compiled with GPU support, or no GPU was found/usable.")
            print("Please ensure you have 'faiss-gpu' installed and a compatible NVIDIA GPU + drivers + CUDA.")
        elif "out of memory" in str(e).lower():
            print(f"GPU Out of Memory Error: {e}. Your GPU may not have enough memory. FAISS Kmeans can be memory intensive.")
        else:
            print(f"A runtime error occurred during FAISS GPU processing: {e}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")