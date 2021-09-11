import unittest
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from StatisticalLearning.ModelAssessment.ModelScore import ModelScore
from sklearn.metrics import mean_squared_error, r2_score


class TEST_FitScore(unittest.TestCase):

    def setUp(self):

        warnings.filterwarnings('ignore')

        self._X = sm.add_constant(pd.DataFrame([[1], [2], [3], [4], [5]]))
        self._y_true = pd.Series([2, 3.2, 4.1, 6.5, 5.5])
        self._result = sm.OLS(self._y_true, self._X).fit()
        self._y_pred = self._result.predict(self._X)

    def test_sum_square_error(self):
        self.assertAlmostEqual(ModelScore.sum_square_error(self._y_true, self._y_pred),
                               mean_squared_error(self._y_true, self._y_pred) * len(self._y_true))
        self.assertAlmostEqual(ModelScore.sum_square_error(self._y_true, 0),
                               mean_squared_error(self._y_true, np.zeros(len(self._y_true))) * len(self._y_true))

    def test_mean_square_error(self):
        self.assertEqual(ModelScore.mean_square_error(self._y_true, self._y_pred),
                         mean_squared_error(self._y_true, self._y_pred))
        self.assertEqual(ModelScore.mean_square_error(self._y_true, 0),
                         mean_squared_error(self._y_true, np.zeros(len(self._y_true))))

    def test_r_square(self):
        self.assertEqual(ModelScore.r_square(self._y_true, self._y_pred),
                         r2_score(self._y_true, self._y_pred))
        self.assertEqual(ModelScore.r_square(self._y_true, self._y_pred),
                         self._result.rsquared)
        self.assertEqual(ModelScore.r_square(self._y_true, 0),
                         r2_score(self._y_true, np.zeros(len(self._y_true))))

    def test_adjusted_r_square(self):
        self.assertEqual(ModelScore.adjusted_r_square(self._y_true, self._y_pred, self._X.shape[1] - 1),
                         self._result.rsquared_adj)

    def test_aic_linear(self):
        self.assertAlmostEqual(ModelScore.aic_linear(self._y_true, self._y_pred, self._X.shape[1] - 1), self._result.aic)

    def test_confusion_matrix(self):
        y_pred = pd.Series([1, 1, 1, 0, 0, 1, 0, 0, 0, 1])
        y_true = pd.Series([1, 1, 0, 0, 0, 1, 1, 0, 0, 0])
        result = ModelScore.confusion_matrix(y_true, y_pred)
        self.assertEqual(result['accuracy'], 0.7)
        self.assertEqual(result['precision'], 0.6)
        self.assertEqual(result['recall'], 0.75)
        self.assertEqual(result['F_score'], 2 / 3)
        self.assertEqual(result['confusion_matrix'], [[3, 1], [2, 4]])