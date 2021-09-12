import unittest
import warnings
import pandas as pd
from sklearn import datasets
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from StatisticalLearning.Ensemble.Bagging import BaggingRegressor
from StatisticalLearning.ModelAssessment.ModelScore import ModelScore
from StatisticalLearning.LinearModel.LinearRegression import LinearRegression as linear_regression


class TEST_Bagging(unittest.TestCase):

    def setUp(self):

        warnings.filterwarnings('ignore')

        X, y = datasets.load_diabetes(return_X_y=True)
        self._X, self._y = pd.DataFrame(X), pd.Series(y)

    def test_sklearn_linear_regression(self):

        lr = LinearRegression(fit_intercept=True, copy_X=True)
        bagging = BaggingRegressor(estimator=lr, n_estimator=500)
        bagging = bagging.fit(self._X, self._y)
        prediction = bagging.predict(self._X)

        # Sklearn bagging regressor gives oob error = 3005
        self.assertTrue(2950 <= bagging.out_of_bag_error <= 3050)

        # Sklearn bagging regressor gives train error - 2905
        self.assertTrue(2850 <= ModelScore.mean_square_error(self._y, prediction) <= 2950)

    def test_homemade_linear_regression(self):

        lr = linear_regression(fit_intercept=True)
        bagging = BaggingRegressor(estimator=lr, n_estimator=500)
        bagging = bagging.fit(self._X, self._y, solver='NormalEquation')
        prediction = bagging.predict(self._X)

        # Sklearn bagging regressor gives oob error = 3005
        self.assertTrue(2950 <= bagging.out_of_bag_error <= 3050)

        # Sklearn bagging regressor gives train error - 2905
        self.assertTrue(2850 <= ModelScore.mean_square_error(self._y, prediction) <= 2950)

    def test_sklearn_decision_tree(self):

        tree = DecisionTreeRegressor()
        bagging = BaggingRegressor(estimator=tree, n_estimator=500)
        bagging = bagging.fit(self._X, self._y)
        prediction = bagging.predict(self._X)

        # Expected higher bias but much lower training error
        # Decision tree can handle nonlinear pattern and give lower train error
        self.assertTrue(bagging.out_of_bag_error > 3000)
        self.assertTrue(ModelScore.mean_square_error(self._y, prediction) <= 1000)






