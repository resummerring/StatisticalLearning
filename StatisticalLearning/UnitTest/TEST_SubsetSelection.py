import unittest
import warnings
import pandas as pd
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from StatisticalLearning.ModelSelection.BestSubset import BestSubset
from StatisticalLearning.LinearModel.LinearRegression import LinearRegression as linear_regression


class TEST_LinearRegressionBestSubset(unittest.TestCase):

    def setUp(self):

        warnings.filterwarnings('ignore')

        X, y = datasets.load_diabetes(return_X_y=True)
        self._X, self._y = pd.DataFrame(X), pd.Series(y)
        lr = LinearRegression(fit_intercept=True, copy_X=True)
        self._selector = BestSubset(self._X, self._y, lr)

    def test_find_best_model_with_fixed_size(self):
        """
        Expected result from R - leap: regsubsets() function
        """

        self.assertEqual(self._selector.find_best_model_with_fixed_size(1), [2])
        self.assertEqual(self._selector.find_best_model_with_fixed_size(2), [2, 8])
        self.assertEqual(self._selector.find_best_model_with_fixed_size(3), [2, 3, 8])
        self.assertEqual(self._selector.find_best_model_with_fixed_size(4), [2, 3, 4, 8])
        self.assertEqual(self._selector.find_best_model_with_fixed_size(5), [1, 2, 3, 6, 8])
        self.assertEqual(self._selector.find_best_model_with_fixed_size(6), [1, 2, 3, 4, 5, 8])
        self.assertEqual(self._selector.find_best_model_with_fixed_size(7), [1, 2, 3, 4, 5, 7, 8])
        self.assertEqual(self._selector.find_best_model_with_fixed_size(8), [1, 2, 3, 4, 5, 7, 8, 9])
        self.assertEqual(self._selector.find_best_model_with_fixed_size(9), [1, 2, 3, 4, 5, 6, 7, 8, 9])
        self.assertEqual(self._selector.find_best_model_with_fixed_size(10), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    def test_find_best_model(self):
        self.assertEqual(self._selector.find_best_model(), [1, 2, 3, 6, 8])

    def test_parameter_pass(self):
        lr = linear_regression(fit_intercept=True)
        selector = BestSubset(self._X, self._y, lr)
        self.assertEqual(selector.find_best_model(solver='NormalEquation'), [1, 2, 3, 6, 8])
