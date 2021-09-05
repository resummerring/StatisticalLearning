import unittest
import numpy as np
import pandas as pd
from statsmodels import api
from sklearn import datasets
from sklearn import linear_model
from StatisticalLearning.LinearModel.LinearRegression import LinearRegression


class TEST_LinearRegression(unittest.TestCase):

    def setUp(self):
        X, y = datasets.load_diabetes(return_X_y=True)
        self._X, self._y = pd.DataFrame(X), pd.Series(y)
        self._lr = linear_model.LinearRegression(fit_intercept=True, copy_X=True).fit(X, y)

    def test_normal_equation(self):
        lr = LinearRegression(fit_intercept=True).fit(self._X, self._y, solver='NormalEquation')
        self.assertAlmostEqual(lr.intercept, self._lr.intercept_)
        self.assertAlmostEqual(np.linalg.norm(lr.coef - self._lr.coef_, ord=2), 0)

        lr = LinearRegression(fit_intercept=False).fit(self._X, self._y, solver='NormalEquation')
        self.assertEqual(lr.intercept, None)
        self.assertAlmostEqual(np.linalg.norm(lr.coef - self._lr.coef_, ord=2), 0)

    def test_svd_decomposition(self):
        lr = LinearRegression(fit_intercept=True).fit(self._X, self._y, solver='SVD')
        self.assertAlmostEqual(lr.intercept, self._lr.intercept_)
        self.assertAlmostEqual(np.linalg.norm(lr.coef - self._lr.coef_, ord=2), 0)

    def test_t_stat_and_p_value(self):
        lr = LinearRegression(fit_intercept=True).fit(self._X, self._y, solver='NormalEquation')
        sm_lr = api.OLS(self._y, api.add_constant(self._X)).fit()
        self.assertAlmostEqual(np.linalg.norm(lr.coef_t_stat - pd.Series(sm_lr.tvalues).iloc[1:], ord=2), 0, places=5)
        self.assertAlmostEqual(lr.intercept_t_stat, pd.Series(sm_lr.tvalues).iloc[0], places=5)
        self.assertAlmostEqual(np.linalg.norm(lr.coef_p_value - pd.Series(sm_lr.pvalues).iloc[1:], ord=2), 0, places=5)
        self.assertAlmostEqual(lr.intercept_p_value, pd.Series(sm_lr.pvalues).iloc[0], places=5)

    def test_gradient_descent(self):

        # Simulate random data for y = 3x_1 + 2x_2 + 1
        X = pd.DataFrame(np.random.randn(1000, 2))
        y = X.apply(lambda row: 3 * row[0] + 2 * row[1], axis=1) + 0.01 * np.random.randn(1000) + 1

        sklearn_lr = linear_model.LinearRegression(fit_intercept=True, copy_X=True).fit(X, y)
        lr = LinearRegression(fit_intercept=True).fit(X, y, solver='GradientDescent', param_tol=1e-6, func_tol=1e-10,
                                                      learning_rate=0.2, max_iter=1000, x0=pd.Series([0, 1, 2]))

        self.assertAlmostEqual(lr.intercept, sklearn_lr.intercept_, places=5)
        self.assertAlmostEqual(np.max([abs(sklearn_lr.coef_[i] - lr.coef.iloc[i]) for i in [0, 1]]), 0, places=5)

    def test_stochastic_gradient_descent(self):

        # Simulate random data for y = 3x_1 + 2x_2 + 1
        X = pd.DataFrame(np.random.randn(1000, 2))
        y = X.apply(lambda row: 3 * row[0] + 2 * row[1], axis=1) + 0.01 * np.random.randn(1000) + 1

        sklearn_lr = linear_model.LinearRegression(fit_intercept=True, copy_X=True).fit(X, y)
        lr = LinearRegression(fit_intercept=True).fit(X, y, solver='StochasticGradientDescent', func_tol=1e-10,
                                                      learning_rate=4e-4, max_epoc=100, x0=pd.Series([0, 1, 2]))
        self.assertAlmostEqual(lr.intercept, sklearn_lr.intercept_, places=4)
        self.assertAlmostEqual(np.max([abs(sklearn_lr.coef_[i] - lr.coef.iloc[i]) for i in [0, 1]]), 0, places=4)
