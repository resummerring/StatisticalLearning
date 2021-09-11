import unittest
import warnings
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate
from StatisticalLearning.ModelAssessment.ModelScore import ModelScore
from StatisticalLearning.LinearModel.LinearRegression import LinearRegression as linear_regression
from StatisticalLearning.ModelAssessment.CrossValidation import ValidationSet, KFoldValidation, LeaveOneOut


class TEST_CrossValidation(unittest.TestCase):

    def setUp(self):

        warnings.filterwarnings('ignore')

        X, y = datasets.load_diabetes(return_X_y=True)
        self._X, self._y = pd.DataFrame(X), pd.Series(y)
        self._lr = LinearRegression(fit_intercept=True, copy_X=True)

        pred = self._lr.fit(self._X, self._y).predict(self._X)
        self._mse = ModelScore.mean_square_error(self._y, pred)

    def test_validation_set(self):
        validation_set = ValidationSet(self._X, self._y, self._lr, ModelScore.mean_square_error)
        validation_set = validation_set.validate(train_ratio=0.7, seed=0)
        self.assertTrue(0.9 * self._mse <= validation_set.mean_score <= 1.1 * self._mse)

    def test_leave_one_out(self):
        leave_one_out = LeaveOneOut(self._X, self._y, self._lr, ModelScore.mean_square_error)
        leave_one_out = leave_one_out.validate()

        validator = cross_validate(self._lr, self._X, self._y, cv=self._X.shape[0], scoring='neg_mean_squared_error')
        self.assertEqual(leave_one_out.mean_score, -np.mean(validator['test_score']))

    def test_kfold_validation(self):
        kfold = KFoldValidation(self._X, self._y, self._lr, ModelScore.mean_square_error)
        kfold = kfold.validate(k=5, seed=0)
        self.assertTrue(0.9 * self._mse <= kfold.mean_score <= 1.1 * self._mse)

    def test_parameter_pass(self):
        X = pd.DataFrame(np.random.randn(1000, 2))
        y = X.apply(lambda row: 3 * row[0] + 2 * row[1], axis=1) + 0.01 * np.random.randn(1000) + 1

        lr = linear_regression(fit_intercept=True)
        kfold = KFoldValidation(X, y, lr, ModelScore.mean_square_error)
        kfold = kfold.validate(solver='GradientDescent', param_tol=1e-6, func_tol=1e-10,
                               learning_rate=0.2, max_iter=1000, x0=pd.Series([0, 1, 2]))

        self.assertAlmostEqual(kfold.mean_score, 0.0001, places=3)
