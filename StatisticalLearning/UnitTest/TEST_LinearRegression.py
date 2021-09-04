import unittest
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn import linear_model
from StatisticalLearning.LinearModel.LineaerRegression import LinearRegression


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

    def test_gradient_descent(self):

        # Simulate random data for y = 3x_1 + 2x_2 + 1
        X = pd.DataFrame(10 * np.random.randn(1000, 2))
        y = X.apply(lambda row: 3 * row[0] + 2 * row[1], axis=1) + np.random.randn(1000) + 1

        sklearn_lr = linear_model.LinearRegression(fit_intercept=True, copy_X=True).fit(X, y)
        lr = LinearRegression(fit_intercept=True).fit(X, y, solver='GradientDescent', param_tol=1e-6, func_tol=1e-6,
                                                      learning_rate=0.001, max_iter=10000)

        self.assertAlmostEqual(lr.intercept, sklearn_lr.intercept_, places=2)
        self.assertAlmostEqual(np.linalg.norm(lr.coef - sklearn_lr.coef_, ord=2), 0, places=4)
