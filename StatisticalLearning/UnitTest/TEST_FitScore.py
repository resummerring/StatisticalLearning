import unittest
import numpy as np
import pandas as pd
import statsmodels.api as sm
from StatisticalLearning.ModelSelection.FitScore import FitScore
from sklearn.metrics import mean_squared_error, r2_score


class TEST_FitScore(unittest.TestCase):

    def setUp(self):
        self._X = sm.add_constant(pd.DataFrame([[1], [2], [3], [4], [5]]))
        self._y_true = pd.Series([2, 3.2, 4.1, 6.5, 5.5])
        self._result = sm.OLS(self._y_true, self._X).fit()
        self._y_pred = self._result.predict(self._X)

    def test_sum_square_error(self):
        x = FitScore.sum_square_error(self._y_true, self._y_pred)
        self.assertAlmostEqual(FitScore.sum_square_error(self._y_true, self._y_pred),
                               mean_squared_error(self._y_true, self._y_pred) * len(self._y_true))
        self.assertAlmostEqual(FitScore.sum_square_error(self._y_true, 0),
                               mean_squared_error(self._y_true, np.zeros(len(self._y_true))) * len(self._y_true))

    def test_mean_square_error(self):
        self.assertEqual(FitScore.mean_square_error(self._y_true, self._y_pred),
                         mean_squared_error(self._y_true, self._y_pred))
        self.assertEqual(FitScore.mean_square_error(self._y_true, 0),
                         mean_squared_error(self._y_true, np.zeros(len(self._y_true))))

    def test_r_square(self):
        self.assertEqual(FitScore.r_square(self._y_true, self._y_pred),
                         r2_score(self._y_true, self._y_pred))
        self.assertEqual(FitScore.r_square(self._y_true, self._y_pred),
                         self._result.rsquared)
        self.assertEqual(FitScore.r_square(self._y_true, 0),
                         r2_score(self._y_true, np.zeros(len(self._y_true))))

    def test_adjusted_r_square(self):
        self.assertEqual(FitScore.adjusted_r_square(self._y_true, self._y_pred, self._X.shape[1] - 1),
                         self._result.rsquared_adj)