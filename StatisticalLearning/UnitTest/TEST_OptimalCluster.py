import unittest
import warnings
import numpy as np
import pandas as pd

from sklearn.datasets import make_blobs

from StatisticalLearning.Clustering.OptimalCluster import OptimalCluster


class TEST_OptimalCluster(unittest.TestCase):

    def setUp(self):

        warnings.filterwarnings('ignore')

        X, _ = make_blobs(n_samples=400, cluster_std=5, centers=10, n_features=20, random_state=1)
        self._X = pd.DataFrame(X)

    def test_base_clustering(self):
        model = OptimalCluster(n_init=10)
        matrix, cluster, score = model._base_clustering(self._X, 10)
        self.assertTrue(len(cluster) == 10)

    def test_optimal_clustering(self):
        model = OptimalCluster(n_init=10)
        matrix, cluster, score = model.optimal_cluster(self._X, 10)
        self.assertTrue(len(cluster) == 10)
        self.assertTrue(np.all([abs(len(value) - 40) <= 1 for _, value in cluster.items()]))
