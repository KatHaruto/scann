from __future__ import annotations

import numpy as np
from tqdm import trange


class AnisotropicKmeans:
    def __init__(self, n_clusters: int, max_iter: int, random_state: int, anisotropic_quantization_threshold:float=0.0) -> None:
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.anisotropic_quantization_threshold = anisotropic_quantization_threshold

    def fit(self, data:np.ndarray) -> AnisotropicKmeans:

        self.pq_codebook = np.zeros((self.n_clusters, data.shape[1]))
        rng = np.random.default_rng(self.random_state)
        self.centroids = data[rng.choice(data.shape[0], self.n_clusters, replace=False), :]

        clusters = self.transform(data)
        # fig = go.Figure()
        for it in trange(self.max_iter):
            # 中心点の更新
            # 各クラスタのデータ点の平均をとる
            centroids = self.update_centroids(data, clusters)

            # 所属クラスタの更新
            # 一番近い中心点のクラスタを所属クラスタに更新する
            # np.linalg.normでノルムが計算できる
            # argminで最小値のインデックスを取得できる

            # TODO: scann
            new_clusters = np.array([self.anisotropic_loss(data, c ) for c in centroids]).argmin(axis = 0)

            # 空のクラスタがあった場合は中心点をランダムな点に割り当てなおす
            for n in range(self.n_clusters):
                if not np.any(new_clusters == n):
                    print(f"cluster {n} is empty")
                    centroids[n] = data[rng.choice(data.shape[0], 1), :]

            # 収束判定
            # fig.add_scatter(x=new_centers_compressed[:,0], y=new_centers_compressed[:,1],mode="markers",name=f"iter:{it}")
            if np.allclose(clusters, new_clusters):
                break

            clusters = new_clusters

        # fig.show()
        self.centroids = centroids
        return self

    def transform(self, data:np.ndarray) -> np.ndarray:

        return np.array([self.anisotropic_loss(data, c ) for c in self.centroids]).argmin(axis = 0)

    def fit_transform(self, data:np.ndarray)->np.ndarray:
        return self.fit(data).transform(data)


    def update_centroids(self, data:np.ndarray, clusters:np.ndarray) -> np.ndarray:

        new_centers = np.zeros((self.n_clusters, data.shape[1]))
        self_outer_func = lambda x:np.outer(x,x)
        for i in range(self.n_clusters):
            data_i = data[clusters == i]
            if len(data_i) == 0:
                print(f"cluster {i} is empty")
                continue
            eta = self.eta(data_i)
            squared_norm = np.linalg.norm(data_i, ord=2, axis=1) ** 2

            a = (((eta-1) / squared_norm).reshape(-1,1,1)  * np.apply_along_axis(self_outer_func,1,data_i) + np.eye(data.shape[1])).sum(axis=0)
            b = (eta.reshape(-1,1) * data_i).sum(axis=0)
            new_centers[i,:] = np.linalg.solve(a, b)

        return new_centers


    def eta(self, data:np.ndarray) -> np.ndarray:
        assert data.ndim == 2
        t2 = self.anisotropic_quantization_threshold ** 2
        norm_squared = np.linalg.norm(data, axis=1) ** 2
        dim = data.shape[-1]
        return (dim-1) * (t2 / norm_squared) / (1 - t2 / norm_squared)

    def anisotropic_loss(self, data:np.ndarray, quantized: np.ndarray, eta: np.ndarray | None=None) -> np.ndarray:
        assert data.ndim == 2 and quantized.ndim == 1

        if eta is None:
            eta = self.eta(data)

        r_parallel = ((data - quantized)* data).sum(axis=1)[:,None] * data  \
            / (np.linalg.norm(data,axis=1,keepdims=True) ** 2)
        r_parallel_norm = np.linalg.norm(r_parallel,axis=1) ** 2
        r_perp = (data - quantized) - r_parallel
        r_perp_norm = np.linalg.norm(r_perp,axis=1) ** 2
        return eta * r_parallel_norm + r_perp_norm


def akmeans2(x: np.ndarray,
             n_clusters: int,
               iter_num: int,
               random_state: int,
               anisotropic_quantization_threshold: float,
               ) -> tuple[np.ndarray, np.ndarray]:
    kmeans = AnisotropicKmeans(n_clusters=n_clusters, max_iter=iter_num,
                               random_state=random_state,
                               anisotropic_quantization_threshold=anisotropic_quantization_threshold)
    cluster = kmeans.fit_transform(x)
    return kmeans.centroids,cluster
