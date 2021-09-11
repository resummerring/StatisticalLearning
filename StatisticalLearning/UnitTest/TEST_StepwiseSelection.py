import unittest
import warnings
import pandas as pd
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from StatisticalLearning.ModelSelection.Stepwise import StepwiseForward, StepwiseBackward
from StatisticalLearning.LinearModel.LinearRegression import LinearRegression as linear_regression


class TEST_LinearRegressionStepwiseForward(unittest.TestCase):

    def setUp(self):

        warnings.filterwarnings('ignore')

        X, y = datasets.load_diabetes(return_X_y=True)
        self._X, self._y = pd.DataFrame(X), pd.Series(y)
        lr = LinearRegression(fit_intercept=True, copy_X=True)
        self._selector = StepwiseForward(self._X, self._y, lr)

    def test_find_best_model_next_step(self):
        self.assertEqual(self._selector.find_best_model_next_step([]), [2])
        self.assertEqual(self._selector.find_best_model_next_step([2]), [2, 8])
        self.assertEqual(self._selector.find_best_model_next_step([2, 8]), [2, 3, 8])
        self.assertEqual(self._selector.find_best_model_next_step([2, 3, 8]), [2, 3, 4, 8])
        self.assertEqual(self._selector.find_best_model_next_step([2, 3, 4, 8]), [1, 2, 3, 4, 8])
        self.assertEqual(self._selector.find_best_model_next_step([1, 2, 3, 4, 8]), [1, 2, 3, 4, 5, 8])

    def test_find_best_model(self):
        self.assertEqual(self._selector.find_best_model(), [1, 2, 3, 4, 5, 8])

    def test_parameter_pass(self):
        lr = linear_regression(fit_intercept=True)
        selector = StepwiseForward(self._X, self._y, lr)
        self.assertEqual(selector.find_best_model(solver='NormalEquation'), [1, 2, 3, 4, 5, 8])


class TEST_LinearRegressionStepwiseBackward(unittest.TestCase):

    def setUp(self):

        warnings.filterwarnings('ignore')

        X, y = datasets.load_diabetes(return_X_y=True)
        self._X, self._y = pd.DataFrame(X), pd.Series(y)
        lr = LinearRegression(fit_intercept=True, copy_X=True)
        self._selector = StepwiseBackward(self._X, self._y, lr)

    def test_find_best_model_next_step(self):
        self.assertEqual(self._selector.find_best_model_next_step([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
                         [1, 2, 3, 4, 5, 6, 7, 8, 9])
        self.assertEqual(self._selector.find_best_model_next_step([1, 2, 3, 4, 5, 6, 7, 8, 9]),
                         [1, 2, 3, 4, 5, 7, 8, 9])
        self.assertEqual(self._selector.find_best_model_next_step([1, 2, 3, 4, 5, 7, 8, 9]), [1, 2, 3, 4, 5, 7, 8])
        self.assertEqual(self._selector.find_best_model_next_step([1, 2, 3, 4, 5, 7, 8]), [1, 2, 3, 4, 5, 8])

    def test_find_best_model(self):
        self.assertEqual(self._selector.find_best_model(), [1, 2, 3, 4, 5, 8])

    def test_parameter_pass(self):
        lr = linear_regression(fit_intercept=True)
        selector = StepwiseBackward(self._X, self._y, lr)
        self.assertEqual(selector.find_best_model(solver='NormalEquation'), [1, 2, 3, 4, 5, 8])
