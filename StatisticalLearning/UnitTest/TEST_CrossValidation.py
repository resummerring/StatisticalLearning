import unittest
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate
from StatisticalLearning.FitScore import FitScore
from StatisticalLearning.CrossValidation.CrossValidation import ValidationSet, KFoldValidation, LeaveOneOut


class TEST_CrossValidation(unittest.TestCase):

    def setUp(self):
        X, y = datasets.load_diabetes(return_X_y=True)
        self._X, self._y = pd.DataFrame(X), pd.Series(y)
        self._lr = LinearRegression(fit_intercept=True, copy_X=True)

        pred = self._lr.fit(self._X, self._y).predict(self._X)
        self._mse = FitScore.mean_square_error(self._y, pred)

    def test_validation_set(self):
        validation_set = ValidationSet(self._X, self._y, self._lr, FitScore.mean_square_error)
        validation_set = validation_set.validate(train_ratio=0.7, seed=0)
        self.assertTrue(0.9 * self._mse <= validation_set.mean_score <= 1.1 * self._mse)

    def test_leave_one_out(self):
        leave_one_out = LeaveOneOut(self._X, self._y, self._lr, FitScore.mean_square_error)
        leave_one_out = leave_one_out.validate()

        validator = cross_validate(self._lr, self._X, self._y, cv=self._X.shape[0], scoring='neg_mean_squared_error')
        self.assertEqual(leave_one_out.mean_score, -np.mean(validator['test_score']))

    def test_kfold_validation(self):
        kfold = KFoldValidation(self._X, self._y, self._lr, FitScore.mean_square_error)
        kfold = kfold.validate(k=5, seed=0)
        self.assertTrue(0.9 * self._mse <= kfold.mean_score <= 1.1 * self._mse)







