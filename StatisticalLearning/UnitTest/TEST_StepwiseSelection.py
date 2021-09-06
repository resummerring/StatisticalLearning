import unittest
import warnings
import pandas as pd
from sklearn import datasets
from StatisticalLearning.ModelSelection.Stepwise import LinearRegressionStepwiseForward, \
                                                        LinearRegressionStepwiseBackward


class TEST_LinearRegressionStepwiseForward(unittest.TestCase):

    def setUp(self):

        warnings.filterwarnings('ignore', category=FutureWarning)

        X, y = datasets.load_diabetes(return_X_y=True)
        X, y = pd.DataFrame(X), pd.Series(y)
        self._lr = LinearRegressionStepwiseForward(X, y)

    def test_find_best_model_next_step(self):
        self.assertEqual(self._lr.find_best_model_next_step([]), [2])
        self.assertEqual(self._lr.find_best_model_next_step([2]), [2, 8])
        self.assertEqual(self._lr.find_best_model_next_step([2, 8]), [2, 3, 8])
        self.assertEqual(self._lr.find_best_model_next_step([2, 3, 8]), [2, 3, 4, 8])
        self.assertEqual(self._lr.find_best_model_next_step([2, 3, 4, 8]), [1, 2, 3, 4, 8])
        self.assertEqual(self._lr.find_best_model_next_step([1, 2, 3, 4, 8]), [1, 2, 3, 4, 5, 8])

    def test_find_best_model(self):
        self.assertEqual(self._lr.find_best_model(), [2, 3, 8])


class TEST_LinearRegressionStepwiseBackward(unittest.TestCase):

    def setUp(self):
        X, y = datasets.load_diabetes(return_X_y=True)
        X, y = pd.DataFrame(X), pd.Series(y)
        self._lr = LinearRegressionStepwiseBackward(X, y)

    def test_find_best_model_next_step(self):
        self.assertEqual(self._lr.find_best_model_next_step([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
                                                            [1, 2, 3, 4, 5, 6, 7, 8, 9])
        self.assertEqual(self._lr.find_best_model_next_step([1, 2, 3, 4, 5, 6, 7, 8, 9]), [1, 2, 3, 4, 5, 7, 8, 9])
        self.assertEqual(self._lr.find_best_model_next_step([1, 2, 3, 4, 5, 7, 8, 9]), [1, 2, 3, 4, 5, 7, 8])
        self.assertEqual(self._lr.find_best_model_next_step([1, 2, 3, 4, 5, 7, 8]), [1, 2, 3, 4, 5, 8])

    def test_find_best_model(self):
        self.assertEqual(self._lr.find_best_model(), [2, 3, 8])
