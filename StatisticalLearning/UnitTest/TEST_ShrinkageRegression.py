import unittest
import pandas as pd
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from StatisticalLearning.ModelAssessment.ModelScore import ModelScore
from StatisticalLearning.LinearModel.ShrinkageRegression import PCRegression, PartialLeastSquare


class TEST_PCRegression(unittest.TestCase):

    def setUp(self):
        X, y = datasets.load_diabetes(return_X_y=True)
        self._X, self._y = pd.DataFrame(X), pd.Series(y)

    def test_principal_component_regression(self):
        pcr = PCRegression(n_components=10).fit(self._X, self._y)
        prediction_pcr = pcr.predict(self._X)
        mse_pcr = ModelScore.mean_square_error(self._y, prediction_pcr)
        lr = LinearRegression().fit(self._X, self._y)
        prediction_lr = lr.predict(self._X)
        mse_lr = ModelScore.mean_square_error(self._y, prediction_lr)
        self.assertEqual(mse_lr, mse_pcr)


class TEST_PartialLeastSquare(unittest.TestCase):

    def setUp(self):
        X, y = datasets.load_diabetes(return_X_y=True)
        self._X, self._y = pd.DataFrame(X), pd.Series(y)

    # Validation result from Matlab PSL1 algorithm
    def test_partial_least_square(self):
        pls = PartialLeastSquare(n_components=1).fit(self._X, self._y)
        prediction_pls = pls.predict(self._X)
        mse_pls = ModelScore.mean_square_error(self._y, prediction_pls)
        self.assertAlmostEqual(mse_pls, 26664.8108, places=4)



