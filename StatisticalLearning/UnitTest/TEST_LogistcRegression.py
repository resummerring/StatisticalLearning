import unittest
import warnings
import pandas as pd
import statsmodels.api as sm
from sklearn import datasets
from sklearn import linear_model
from StatisticalLearning.Transform.Preprocess import Preprocessor
from StatisticalLearning.LinearModel.Classification import LogisticRegression


class TEST_LogisticRegression(unittest.TestCase):

    def setUp(self):

        warnings.filterwarnings('ignore')

        X, y = datasets.load_breast_cancer(return_X_y=True)
        self._X, _, _ = Preprocessor.normalize(pd.DataFrame(X).iloc[:, :5])
        self._y = pd.Series(y).apply(lambda num: float(num))
        self._lr = linear_model.LogisticRegression(fit_intercept=True).fit(self._X, self._y)
        self._coef, self._intercept = self._lr.coef_, self._lr.intercept_

        coef = pd.Series([self._intercept.squeeze()] + list(self._coef.squeeze()))
        self._optimal_error = LogisticRegression.func(coef, (sm.add_constant(self._X), self._y))

    def test_gradient_descent(self):
        lr = LogisticRegression(fit_intercept=True)
        lr = lr.fit(self._X, self._y, solver='GradientDescent', momentum=0.99, param_tol=1e-4, func_tol=1e-5,
                    learning_rate=1.5, max_iter=5000)

        # Using optimization result from scipy.optimize
        coef = pd.Series([lr.intercept] + list(lr.coef))
        optimal_error = LogisticRegression.func(coef, (sm.add_constant(self._X), self._y))
        self.assertTrue(optimal_error <= self._optimal_error)
        self.assertAlmostEqual(optimal_error, 0.14870229, places=5)

        pred_sklearn = self._lr.predict(self._X)
        pred_lr = lr.predict(self._X).apply(lambda x: 1 if x >= 0.5 else 0)
        self.assertTrue((pred_lr - pd.Series(pred_sklearn.squeeze())).abs().sum() <= 20)

    def test_stochastic_gradient_descent(self):
        lr = LogisticRegression(fit_intercept=True)
        lr = lr.fit(self._X, self._y, solver='StochasticGradientDescent',
                    func_tol=1e-6, learning_rate=0.03, max_epoc=1000)

        coef = pd.Series([lr.intercept] + list(lr.coef))
        optimal_error = LogisticRegression.func(coef, (sm.add_constant(self._X), self._y))
        self.assertTrue(optimal_error <= self._optimal_error)
