from pathlib import Path

import h5py
import numpy as np
from tqdm import trange

from scann_impl.apq import APQ
from scann_impl.pq import PQ


def compute_recall(neighbors: np.ndarray, true_neighbors: np.ndarray) -> float:
    total = 0
    for gt_row, row in zip(true_neighbors, neighbors):
        total += np.intersect1d(gt_row, row).shape[0]
    return total / true_neighbors.size


# with tempfile.TemporaryDirectory() as tmp:
#    response = requests.get("http://ann-benchmarks.com/glove-100-angular.hdf5")
#    loc = os.path.join(tmp, "glove.hdf5")
#    with open(loc, "wb") as f:
#        f.write(response.content)

glove_h5py = h5py.File(Path(__file__).parents[2] / "data" / "glove.hdf5", "r")

rng = np.random.default_rng(seed=1234)

dataset = glove_h5py["train"]
training_sample_size = 10000
training_dataset = dataset[()][rng.choice(len(dataset), training_sample_size, replace=False)]
training_dataset = training_dataset / np.linalg.norm(training_dataset, axis=1)[:, np.newaxis]

queries = glove_h5py["test"][()]

true_neighbors = glove_h5py["neighbors"][()]
# Instantiate with M=8 sub-spaces
for vq_class in [PQ,APQ]:
    pq = vq_class(M=25,Ks=16,metric="dot")

    # Train codewords
    pq.fit(training_dataset,iter=15)

    # Encode to PQ-codes
    X_code = pq.encode(dataset)  # (10000, 8) with dtype=np.uint8

    # Results: create a distance table online, and compute Asymmetric Distance to each PQ-code
    neighbors = np.empty((len(true_neighbors),100), dtype=np.int64)
    reordered_neighbors = np.empty((len(true_neighbors),100), dtype=np.int64)
    for i in trange(len(queries)):
        dists = pq.dtable(queries[i]).adist(X_code)  # (10000, )
        closest_indices = np.argsort(dists)[::-1][:2000]
        neighbors[i,:] = closest_indices[:100]
        sorted_indices = np.sort(closest_indices)
        # reordering
        reordered_neighbors[i,:] = sorted_indices[np.argsort((queries[i] * dataset[sorted_indices]).sum(axis=1))[::-1][:100]]

    # we are given top 100 neighbors in the ground truth, so select top 10
    print(f"Recall of {vq_class.__name__}(no reorder): {compute_recall(neighbors, true_neighbors[:, :10])}")
    print(f"Recall of {vq_class.__name__}(reorder): {compute_recall(reordered_neighbors, true_neighbors[:, :10])}")
