import numpy as np
import pandas as pd

from typing import Tuple

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples


class OptimalCluster:

    def __init__(self, n_init: int):
        """
        :param n_init: int, number of random initialization fed into KMeans
        """

        self._n_init = n_init

    @staticmethod
    def _concat_output(X: pd.DataFrame, cluster_keep: dict, cluster: dict) -> Tuple[pd.DataFrame, dict, pd.Series]:
        """
        Concatenate accepted base clustering result with rerun unqualified clustering result

        :param X: pd.DataFrame, observation matrix
        :param cluster_keep: dict, clustering info kept from base clustering
        :param cluster: dict, clustering info from rerun unqualified clusters
        """

        cluster_final = dict()

        for label in cluster_keep.keys():
            cluster_final[len(cluster_final.keys())] = list(cluster_keep[label])

        for label in cluster.keys():
            cluster_final[len(cluster_final.keys())] = list(cluster[label])

        new_idx = [idx for label in cluster_final.keys() for idx in cluster_final[label]]
        matrix = X.iloc[new_idx, :]

        cluster_labels = np.zeros(matrix.shape[0])
        for label in cluster_final.keys():
            idx = [matrix.index.get_loc(k) for k in cluster_final[label]]
            cluster_labels[idx] = label

        score = pd.Series(silhouette_samples(matrix, cluster_labels), index=matrix.index)

        return matrix, cluster_final, score

    def _base_clustering(self, X: pd.DataFrame, max_clusters: int) -> Tuple[pd.DataFrame, dict, pd.Series]:
        """
        Base-level clustering to find optimal number of clusters and initialization

        :param X: pd.DataFrame, N * F observation matrix with N observations and F features
        :param max_clusters: int, maximum acceptable number of clusters

        For i in n_init:
            create a random initialization
            For k in max_clusters:
                try KMeans with k clusters
                find optimal silhouette score

        Silhouette score S(i) = [b(i) - a(i)] / max{a(i), b(i)}
        a(i) is the average distance between i and all other elements in the same cluster
        b(i) is the average distance between i and all the elements in the nearest cluster
        S(i) \in [-1, 1], S(i) = -1 -> poorly clustered and S(i) = 1 -> well clustered
        For a given partition, clustering quality q = E[S] / STD(S)
        """

        optimal_score, optimal_model = pd.Series(), None

        for i in range(self._n_init):
            for j in range(2, max_clusters + 1):

                model = KMeans(n_clsters=i, n_jobs=1, n_init=1).fit(X)
                score = silhouette_samples(X, model.labels_)

                q = score.mean() / score.std()
                optimal_q = optimal_score.mean() / optimal_score.std()

                if np.isnan(optimal_q) or q > optimal_q:
                    optimal_score, optimal_model = score, model

        # Reordered observation matrix
        matrix = X.iloc[np.argsort(optimal_model.labels_)]

        # Cluster result mapping
        clusters = {label: X.index[np.where(optimal_model.labels_ == label)[0]].tolist()
                    for label in np.unique(optimal_model.labels_)}

        # Silhouette score
        optimal_score = pd.Series(optimal_score, index=X.index)

        return matrix, clusters, optimal_score

    def optimal_cluster(self, X: pd.DataFrame, max_clusters: int) -> Tuple[pd.DataFrame, dict, pd.Series]:
        """
        Optimal Clustering Algorithm:
            (1) Base Clustering on observation matrix
            (2) For clusters in D(0) = {k | q(k) < mean_k q(k)}, rerun Base Clustering on D(0) -> D(1)
            (3) While ||D(i)|| > 1: repeat step (2)

        :param X:  pd.DataFrame, N * F observation matrix with N observations and F features
        :param max_clusters: int, maximum acceptable number of clusters
        """

        matrix, clusters, score = self._base_clustering(X, min(max_clusters, X.shape[0] - 1))

        # Cluster-level clustering quality
        cluster_q = {label: np.mean(score[clusters[label]]) / np.std(score[clusters[label]])
                     for label in clusters.keys()}

        # Mean cluster-level clustering quality
        mean_q = np.mean(cluster_q.values())

        # Unqualified cluster
        redo_cluster = [label for label in clusters.keys() if cluster_q[label] < mean_q]

        if len(redo_cluster) <= 1:
            return matrix, clusters, score

        else:
            index = [idx for label in redo_cluster for idx in clusters[label]]
            redo_X = X.loc[index, :]
            redo_mean_q = np.mean([cluster_q[label] for label in redo_cluster])

            _, cluster_top, _ = self.optimal_cluster(redo_X, min(max_clusters, redo_X.shape[0] - 1))

            keep_cluster = {label: clusters[label] for label in clusters.keys() if label not in redo_cluster}
            matrix_final, cluster_final, score_final = self._concat_output(X, keep_cluster, cluster_top)

            final_cluster_q = {label: np.mean(score_final[cluster_final[label]])
                                    / np.std(score_final[cluster_final[label]]) for label in cluster_final.keys()}
            final_mean_q = np.mean(final_cluster_q.values())

            if final_mean_q <= redo_mean_q:
                return matrix, clusters, score
            else:
                return matrix_final, cluster_final, score_final
