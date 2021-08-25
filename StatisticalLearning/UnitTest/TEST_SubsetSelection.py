import unittest
import pandas as pd
from sklearn import datasets
from StatisticalLearning.ModelSelection.BestSubset import LinearRegressionBestSubset


class TEST_LinearRegressionBestSubset(unittest.TestCase):

    def setUp(self):
        X, y = datasets.load_diabetes(return_X_y=True)
        X, y = pd.DataFrame(X), pd.Series(y)
        self._lr = LinearRegressionBestSubset(X, y)

    def test_find_best_model_with_fixed_size(self):
        """
        Expected result from R - leap: regsubsets() function
        """

        self.assertEqual(self._lr.find_best_model_with_fixed_size(1), [2])
        self.assertEqual(self._lr.find_best_model_with_fixed_size(2), [2, 8])
        self.assertEqual(self._lr.find_best_model_with_fixed_size(3), [2, 3, 8])
        self.assertEqual(self._lr.find_best_model_with_fixed_size(4), [2, 3, 4, 8])
        self.assertEqual(self._lr.find_best_model_with_fixed_size(5), [1, 2, 3, 6, 8])

    # TODO: add tests for finding optimal model
    def test_find_best_model(self):
        self._lr.find_best_model()



