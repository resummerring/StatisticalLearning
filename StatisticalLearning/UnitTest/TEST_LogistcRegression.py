import unittest
import pandas as pd
import statsmodels.api as sm
from sklearn import datasets
from sklearn import linear_model
from StatisticalLearning.Preprocess import Preprocessor
from StatisticalLearning.LinearModel.Classification import LogisticRegression


class TEST_LogisticRegression(unittest.TestCase):

    def setUp(self):
        X, y = datasets.load_breast_cancer(return_X_y=True)
        self._X, _, _ = Preprocessor.normalize(pd.DataFrame(X).iloc[:, :5])
        self._y = pd.Series(y).apply(lambda num: float(num))
        self._lr = linear_model.LogisticRegression(fit_intercept=True).fit(self._X, self._y)
        self._coef, self._intercept = self._lr.coef_, self._lr.intercept_

        coef = pd.Series([self._intercept.squeeze()] + list(self._coef.squeeze()))
        self._optimal_error = LogisticRegression.func(coef, (sm.add_constant(self._X), self._y))
        print(self._optimal_error)

    def test_gradient_descent(self):
        lr = LogisticRegression(fit_intercept=True)
        lr = lr.fit(self._X, self._y, solver='GradientDescent', momentum=0.99, param_tol=1e-4, func_tol=1e-5,
                    learning_rate=1.5, max_iter=5000)

        # Using optimization result from scipy.optimize
        coef = pd.Series([lr.intercept] + list(lr.coef))
        optimal_error = LogisticRegression.func(coef, (sm.add_constant(self._X), self._y))
        self.assertAlmostEqual(optimal_error, 0.14870229, places=5)
