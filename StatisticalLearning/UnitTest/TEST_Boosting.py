import unittest
import warnings
import pandas as pd
from sklearn import datasets
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from StatisticalLearning.Ensemble.Boosting import BoostRegressor
from StatisticalLearning.ModelAssessment.ModelScore import ModelScore


class TEST_Boosting(unittest.TestCase):

    def setUp(self):

        warnings.filterwarnings('ignore')

        X, y = datasets.load_diabetes(return_X_y=True)
        self._X, self._y = pd.DataFrame(X), pd.Series(y)

    def test_linear_regression(self):

        lr = LinearRegression(fit_intercept=True, copy_X=True)
        booster = BoostRegressor(lr, learning_rate=0.05).fit(self._X, self._y)
        pred = booster.predict(self._X)
        error = ModelScore.mean_square_error(self._y, pred)

        # Sklearn gradient booster gives error = 2833
        self.assertTrue(2800 <= error <= 2900)

    def test_decision_tree(self):

        tree_limit = DecisionTreeRegressor(max_depth=1)
        booster = BoostRegressor(tree_limit, n_estimators=500, learning_rate=0.01).fit(self._X, self._y)
        pred = booster.predict(self._X)
        error = ModelScore.mean_square_error(self._y, pred)

        # Sklearn gradient booster gives error = 2744
        self.assertTrue(2700 <= error <= 2800)

        tree_unlimit = DecisionTreeRegressor()
        booster = BoostRegressor(tree_unlimit, n_estimators=500, learning_rate=0.01).fit(self._X, self._y)
        pred = booster.predict(self._X)
        error = ModelScore.mean_square_error(self._y, pred)

        # Overfitting test case, training error should be close to 0
        self.assertTrue(0 <= error <= 1)
