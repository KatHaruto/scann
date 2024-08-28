
import numpy as np
from tqdm import tqdm


def dist_l2(q, x):
    return np.linalg.norm(q - x, ord=2, axis=1) ** 2


def dist_ip(q, x):
    return q @ x.T


metric_function_map = {"l2": dist_l2, "dot": dist_ip}

def _initialize_kmeans_plusplus(X:np.ndarray, n_clusters:int, random_state:int):
    n_samples, n_features = X.shape
    centers = np.empty((n_clusters, n_features), dtype=X.dtype)
    centers[0] = X[np.random.RandomState(random_state).randint(n_samples)]
    for i in range(1, n_clusters):
        print(i)
        dist = np.linalg.norm(X[:,None] - centers[:i],axis=2)
        dist = np.min(dist, axis=1) / (dist ** 2).sum(axis=1) if i > 1 else np.min(dist, axis=1)

        indices  = np.random.choice(n_samples, p=dist / dist.sum())
        centers[i] = X[indices]
    return centers

class APQ:
    """Pure python implementation of Product Quantization (PQ) [Jegou11]_.

    For the indexing phase of database vectors,
    a `D`-dim input vector is divided into `M` `D`/`M`-dim sub-vectors.
    Each sub-vector is quantized into a small integer via `Ks` codewords.
    For the querying phase, given a new `D`-dim query vector, the distance beween the query
    and the database PQ-codes are efficiently approximated via Asymmetric Distance.

    All vectors must be np.ndarray with np.float32

    .. [Jegou11] H. Jegou et al., "Product Quantization for Nearest Neighbor Search", IEEE TPAMI 2011

    Args:
        M (int): The number of sub-space
        Ks (int): The number of codewords for each subspace
            (typically 256, so that each sub-vector is quantized
            into 8 bits = 1 byte = uint8)
        metric (str): Type of metric used among vectors (either 'l2' or 'dot')
            Note that even for 'dot', kmeans and encoding are performed in the Euclidean space.
        verbose (bool): Verbose flag

    Attributes:
        M (int): The number of sub-space
        Ks (int): The number of codewords for each subspace
        metric (str): Type of metric used among vectors
        verbose (bool): Verbose flag
        code_dtype (object): dtype of PQ-code. Either np.uint{8, 16, 32}
        codewords (np.ndarray): shape=(M, Ks, Ds) with dtype=np.float32.
            codewords[m][ks] means ks-th codeword (Ds-dim) for m-th subspace
        Ds (int): The dim of each sub-vector, i.e., Ds=D/M

    """

    def __init__(self, M: int, Ks:int =256, metric: str="l2", anisotropic_threshold: float=0.2, verbose: bool=True):
        assert 0 < Ks <= 2**32
        assert metric in ["l2", "dot"]
        self.M, self.Ks, self.metric, self.verbose = M, Ks, metric, verbose
        self.code_dtype = (
            np.uint8 if Ks <= 2**8 else (np.uint16 if Ks <= 2**16 else np.uint32)
        )
        self.anisotropic_threshold = anisotropic_threshold
        self.codewords = None
        self.Ds = None

        if verbose:
            print(
                f"M: {M}, Ks: {Ks}, metric : {self.code_dtype}, code_dtype: {metric}",
            )


    def fit(self, vecs, iter=20, seed=123, minit="points"):
        """Given training vectors, run k-means for each sub-space and create
        codewords for each sub-space.

        This function should be run once first of all.

        Args:
            vecs (np.ndarray): Training vectors with shape=(N, D) and dtype=np.float32.
            iter (int): The number of iteration for k-means
            seed (int): The seed for random process
            minit (str): The method for initialization of centroids for k-means (either 'random', '++', 'points', 'matrix')

        Returns:
            object: self

        """
        assert vecs.dtype == np.float32
        assert vecs.ndim == 2
        N, D = vecs.shape
        assert self.Ks < N, "the number of training vector should be more than Ks"
        assert D % self.M == 0, "input dimension must be dividable by M"
        assert minit in ["random", "++", "points", "matrix"]
        self.Ds = int(D / self.M)

        np.random.seed(seed)
        if self.verbose:
            print(f"iter: {iter}, seed: {seed}")

        # [m][ks][ds]: m-th subspace, ks-the codeword, ds-th dim
        init_code_data = vecs[np.random.choice(len(vecs),self.Ks, replace=False)]
        self.codewords = np.zeros(self.Ks *  D)
        for m in range(self.M):
            self.codewords[m*self.Ds*self.Ks:(m+1)*self.Ds*self.Ks] = init_code_data[:,m * self.Ds:(m + 1) * self.Ds].flatten()

        all_assign_label = np.zeros((N,self.M),dtype=np.int32)
        for i in range(iter):

            A = np.zeros((self.Ks * D,self.Ks * D), dtype=np.float32)
            C = np.zeros(self.Ks * D, dtype=np.float32)

            for j,data in enumerate(tqdm(vecs)):
                B = np.zeros((D, D * self.Ks), dtype=np.int32)
                norm_squared = np.linalg.norm(data) ** 2
                eta = self.eta(data)
                quantized = np.zeros(D)


                for m in range(self.M):
                    ind = np.random.randint(0,self.Ks) if i ==0 else all_assign_label[i][m]
                    quantized[m*self.Ds: (m+1)*self.Ds] = self.codewords[m * self.Ds*self.Ks + ind * self.Ds : m * self.Ds*self.Ks + (ind+1) * self.Ds]

                assign_indices = np.zeros(self.M, dtype=np.int32)

                for it in range(3):
                    for m in range(self.M):
                        anisotropic_losses = []
                        for k in range(self.Ks):
                            quantized[m*self.Ds: (m+1)*self.Ds] = self.codewords[m * self.Ds*self.Ks + k * self.Ds : m * self.Ds*self.Ks + (k+1) * self.Ds]

                            loss = self.anisiotropic_loss(data, quantized, eta)
                            anisotropic_losses.append(loss)
                        sub_cluster_index = np.argmin(anisotropic_losses)
                        assign_indices[m] = sub_cluster_index
                        quantized[m*self.Ds: (m+1)*self.Ds] = self.codewords[m * self.Ds*self.Ks + sub_cluster_index * self.Ds : m * self.Ds*self.Ks + (sub_cluster_index+1) * self.Ds]

                for m in range(self.M):
                    ind = assign_indices[m]
                    B[m*self.Ds:(m+1)*self.Ds,m*self.Ds*self.Ks + ind*self.Ds:m*self.Ds*self.Ks + (ind+1)*self.Ds] = np.eye(self.Ds)

                all_assign_label[j,:] = assign_indices
                x = data[:, np.newaxis]
                A +=  B.T @ ((eta - 1) / norm_squared * x @ x.T + np.eye(D)) @ B
                C += eta * B.T @ data



            try:
                from scipy.linalg import cho_factor, cho_solve
                c0, low = cho_factor(A)  # O(K**3 D**3)
            except:
                eig_set = np.linalg.eig(A)[0]
                min_eign = eig_set.min()
                max_eign = eig_set.max()
                reg = 1e-3
                print(f"{min_eign=},{max_eign=},{reg=}")
                dim = A.shape[0]
                c0, low = cho_factor(A + reg * np.eye(dim))

            old_codewords = self.codewords.copy()
            self.codewords = cho_solve((c0, low), C).flatten()#np.linalg.inv(A) @ C
            if np.sum(old_codewords - self.codewords) <= 1e-5:
                break
            print(f"iter:{i}, diff:{np.linalg.norm(old_codewords - self.codewords)}")
        np.save("data/codewords.npy",self.codewords)
        return self


    def eta(self, data:np.ndarray):
        T2 = self.anisotropic_threshold ** 2

        norm_squared = np.linalg.norm(data) ** 2
        dim = data.shape[-1]
        return (dim-1) * (T2 / norm_squared) / (1 - T2 / norm_squared)

    def anisiotropic_loss(self, data:np.ndarray, quantized: np.ndarray, eta=None):
        if eta is None:
            eta = self.eta(data)

        r_parallel = np.dot(data - quantized, data) * data / (np.linalg.norm(data) ** 2)
        r_parallel_norm = np.linalg.norm(r_parallel) ** 2
        r_perp = (data - quantized) - r_parallel
        r_perp_norm = np.linalg.norm(r_perp) ** 2
        return eta * r_parallel_norm + r_perp_norm

    def encode(self, vecs):
        """Encode input vectors into PQ-codes.

        Args:
            vecs (np.ndarray): Input vectors with shape=(N, D) and dtype=np.float32.

        Returns:
            np.ndarray: PQ codes with shape=(N, M) and dtype=self.code_dtype

        """
        assert vecs.dtype == np.float32
        assert vecs.ndim == 2
        N, D = vecs.shape
        assert self.Ds * self.M == D, "input dimension must be Ds * M"
        quantized = np.zeros(D)

        codes = np.empty((N, self.M), dtype=np.uint8)
        for i,data in enumerate(tqdm(vecs)):
            eta = self.eta(data)
            for m in range(self.M):
                ind = np.random.randint(0,self.Ks)
                quantized[m*self.Ds: (m+1)*self.Ds] = self.codewords[m * self.Ds*self.Ks + ind * self.Ds : m * self.Ds*self.Ks + (ind+1) * self.Ds]

            assign_indices = np.zeros(self.M, dtype=np.int32)
            for m in range(self.M):
                anisotropic_losses = []
                for k in range(self.Ks):
                    quantized[m*self.Ds: (m+1)*self.Ds] = self.codewords[m * self.Ds*self.Ks + k * self.Ds : m * self.Ds*self.Ks + (k+1) * self.Ds]

                    loss= self.anisiotropic_loss(data, quantized, eta)
                    anisotropic_losses.append(loss)
                sub_cluster_index = np.argmin(anisotropic_losses)
                assign_indices[m] = sub_cluster_index
            codes[i,:] = assign_indices
        return codes

    def decode(self, codes):
        """Given PQ-codes, reconstruct original D-dimensional vectors
        approximately by fetching the codewords.

        Args:
            codes (np.ndarray): PQ-cdoes with shape=(N, M) and dtype=self.code_dtype.
                Each row is a PQ-code

        Returns:
            np.ndarray: Reconstructed vectors with shape=(N, D) and dtype=np.float32

        """
        assert codes.ndim == 2
        N, M = codes.shape
        assert M == self.M
        assert codes.dtype == self.code_dtype

        vecs = np.empty((N, self.Ds * self.M), dtype=np.float32)
        for m in range(self.M):
            vecs[:, m * self.Ds : (m + 1) * self.Ds] = self.codewords[m][codes[:, m], :]

        return vecs

    def dtable(self, query):
        """Compute a distance table for a query vector.
        The distances are computed by comparing each sub-vector of the query
        to the codewords for each sub-subspace.
        `dtable[m][ks]` contains the squared Euclidean distance between
        the `m`-th sub-vector of the query and the `ks`-th codeword
        for the `m`-th sub-space (`self.codewords[m][ks]`).

        Args:
            query (np.ndarray): Input vector with shape=(D, ) and dtype=np.float32

        Returns:
            nanopq.DistanceTable:
                Distance table. which contains
                dtable with shape=(M, Ks) and dtype=np.float32

        """
        assert query.dtype == np.float32
        assert query.ndim == 1, "input must be a single vector"
        (D,) = query.shape
        assert self.Ds * self.M == D, "input dimension must be Ds * M"

        # dtable[m] : distance between m-th subvec and m-th codewords (m-th subspace)
        # dtable[m][ks] : distance between m-th subvec and ks-th codeword of m-th codewords
        dtable = np.empty((self.M, self.Ks), dtype=np.float32)
        for m in range(self.M):
            query_sub = query[m * self.Ds : (m + 1) * self.Ds]
            codewords = self.codewords[m * self.Ds * self.Ks : (m + 1) * self.Ds * self.Ks].reshape(self.Ks,self.Ds)
            dtable[m, :] = metric_function_map[self.metric](
                query_sub, codewords,
            )

            # In case of L2, the above line would be:
            # dtable[m, :] = np.linalg.norm(self.codewords[m] - query_sub, axis=1) ** 2

        return DistanceTable(dtable, metric=self.metric)


class DistanceTable:
    """Distance table from query to codewords.
    Given a query vector, a PQ/OPQ instance compute this DistanceTable class
    using :func:`PQ.dtable` or :func:`OPQ.dtable`.
    The Asymmetric Distance from query to each database codes can be computed
    by :func:`DistanceTable.adist`.

    Args:
        dtable (np.ndarray): Distance table with shape=(M, Ks) and dtype=np.float32
            computed by :func:`PQ.dtable` or :func:`OPQ.dtable`
        metric (str): metric type to calculate distance

    Attributes:
        dtable (np.ndarray): Distance table with shape=(M, Ks) and dtype=np.float32.
            Note that dtable[m][ks] contains the squared Euclidean distance between
            (1) m-th sub-vector of query and (2) ks-th codeword for m-th subspace.

    """

    def __init__(self, dtable, metric="l2"):
        assert dtable.ndim == 2
        assert dtable.dtype == np.float32
        assert metric in ["l2", "dot"]
        self.dtable = dtable
        self.metric = metric

    def adist(self, codes):
        """Given PQ-codes, compute Asymmetric Distances between the query (self.dtable)
        and the PQ-codes.

        Args:
            codes (np.ndarray): PQ codes with shape=(N, M) and
                dtype=pq.code_dtype where pq is a pq instance that creates the codes

        Returns:
            np.ndarray: Asymmetric Distances with shape=(N, ) and dtype=np.float32

        """

        assert codes.ndim == 2
        N, M = codes.shape
        assert self.dtable.shape[0] == M

        # Fetch distance values using codes.
        dists = np.sum(self.dtable[range(M), codes], axis=1)

        # The above line is equivalent to the followings:
        # dists = np.zeros((N, )).astype(np.float32)
        # for n in range(N):
        #     for m in range(M):
        #         dists[n] += self.dtable[m][codes[n][m]]

        return dists
