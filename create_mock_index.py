import faiss
import numpy as np
import os

dimension = 128    
num_vectors = 10000 
index_filename = "mock_index.faiss"

print(f"Generating {num_vectors} random vectors of dimension {dimension}...")
vectors = np.random.random((num_vectors, dimension)).astype('float32')
print("Random vectors generated.")


print(f"Creating IndexFlatL2 with dimension {dimension}...")
index = faiss.IndexFlatL2(dimension)

print(f"Adding {vectors.shape[0]} vectors to the index...")
index.add(vectors)
print(f"Index populated. Total vectors in index: {index.ntotal}")

print(f"Saving index to file: {index_filename}")
faiss.write_index(index, index_filename)
print(f"Mock index saved successfully to {os.path.abspath(index_filename)}")