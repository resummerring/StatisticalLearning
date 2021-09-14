import unittest
import pandas as pd
import warnings
from sklearn import datasets
from sklearn.linear_model import LinearRegression, Ridge
from StatisticalLearning.ModelAssessment.ModelScore import ModelScore
from StatisticalLearning.LinearModel.ShrinkageRegression import PCRegression, PartialLeastSquare, RidgeRegression


class TEST_PCRegression(unittest.TestCase):

    def setUp(self):

        warnings.filterwarnings('ignore')

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

        warnings.filterwarnings('ignore')

        X, y = datasets.load_diabetes(return_X_y=True)
        self._X, self._y = pd.DataFrame(X), pd.Series(y)

    # Validation result from Matlab PSL1 algorithm
    def test_partial_least_square(self):
        pls = PartialLeastSquare(n_components=1).fit(self._X, self._y)
        prediction_pls = pls.predict(self._X)
        mse_pls = ModelScore.mean_square_error(self._y, prediction_pls)
        self.assertAlmostEqual(mse_pls, 26664.8108, places=4)


class TEST_RidgeRegression(unittest.TestCase):

    def setUp(self):

        warnings.filterwarnings('ignore')

        X, y = datasets.load_diabetes(return_X_y=True)
        self._X, self._y = pd.DataFrame(X), pd.Series(y)

        self._ridge = Ridge(alpha=0.5, fit_intercept=True).fit(self._X, self._y)
        self._prediction = self._ridge.predict(self._X)
        self._error = ModelScore.mean_square_error(self._y, self._prediction)

    def test_normal_equation(self):

        ridge = RidgeRegression(fit_intercept=True)
        ridge = ridge.fit(self._X, self._y, solver='NormalEquation', shrinkage=0.5)

        prediction = ridge.predict(self._X)
        error = ModelScore.mean_square_error(self._y, prediction)
        self.assertAlmostEqual(error, self._error, places=1)

    def test_SVD(self):
        ridge = RidgeRegression(fit_intercept=True)
        ridge = ridge.fit(self._X, self._y, solver='SVD', shrinkage=0.5)

        prediction = ridge.predict(self._X)
        error = ModelScore.mean_square_error(self._y, prediction)
        self.assertAlmostEqual(error, self._error, places=1)






