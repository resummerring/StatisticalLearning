import unittest
import numpy as np
import pandas as pd
from sklearn import datasets
from StatisticalLearning.ModelSelection.BestSubset import LinearRegressionBestSubset


class TEST_LinearRegressionBestSubset(unittest.TestCase):

    def setUp(self):
        X, y = datasets.load_diabetes(return_X_y=True)
        X, y = pd.DataFrame(X), pd.Series(y)
        self._corr = np.abs(X.corrwith(y, axis=0))
        self._lr = LinearRegressionBestSubset(X, y)

    def test_find_best_model_with_fixed_size(self):
        best_index = self._corr.argmax()
        self.assertEqual(self._lr.find_best_model_with_fixed_size(1), best_index)

    # TODO: add tests for finding optimal model
    def test_find_best_model(self):
        return


