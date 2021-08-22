import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from itertools import combinations
from typing import Union, List
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate
from FitQualityCriterion import FitScore as metric


class BestSubset(ABC):

    #######################################################################################################
    # Best subset algorithm:                                                                              #
    # Step 1: Denote M_0 as the null model, which uses sample mean as prediction                          #
    # Step 2: For k = 1, 2, ..., p:                                                                       #
    #           (a) Find all C(k, p) k-feature combinations and fit a regression model                    #
    #           (b) Denote M_k as the best model with k features based on RSS/MSE/R^2 on training set     #
    # Step 3: Select a single best model out of M_0, M_1, ..., M_p based on cross validated prediction    #
    #         error (AIC/BIC/R^2-adj) on training set                                                     #
    #######################################################################################################

    def __init__(self,
                 X: pd.DataFrame,
                 y: pd.Series):
        self._X = X
        self._y = y

    def _null_model_score(self) -> float:

        """
        Find the AIC score for model with only intercept term
        """

        lr = LinearRegression(fit_intercept=False, copy_X=False)
        intercept = pd.DataFrame(data=np.ones((self._X.shape[0], 1)))
        best_score = cross_validate(lr, intercept, self._y, 10, 'aic')['test_score']

        return best_score

    @abstractmethod
    def find_best_model_with_fixed_size(self, k: int) -> Union[List[int], None]:

        """
        Abstract method to find best model with k features. Must be overriden.

        :param k: int, number of features
        :return: Union[List[int], None], column index of best k features in X
        """
        return

    @abstractmethod
    def find_best_model(self) -> Union[List[int], None]:

        """
        Abstract method to find the optimal model from best models with different
        number of features. Must be overriden.

        :return: Union[List[int], None], column index of optimal subset features in X
        """
        return


class LinearRegressionBestSubset(BestSubset):

    """
    Best subset algorithm using linear regression as estimator
    """

    def __init__(self,
                 X: pd.DataFrame,
                 y: pd.Series):

        super().__init__(X, y)

    def find_best_model_with_fixed_size(self, k: int) -> Union[List[int], None]:

        best_index, best_mse = None, float('inf')

        for sub_index in combinations(list(range(self._X.shape[1])), k):

            subset = self._X.iloc[:, list(sub_index)]
            lr = LinearRegression(fit_intercept=True, copy_X=True).fit(subset, self._y)
            mean_squared_error = metric.mean_square_error(self._y, lr.predict(subset))

            if mean_squared_error < best_mse:
                best_index = list(sub_index)
                best_mse = mean_squared_error

        return best_index

    def find_best_model(self) -> Union[List[int], None]:

        lr = LinearRegression(fit_intercept=True, copy_X=True)
        global_best_score, global_best_index = self._null_model_score(), None

        for k in range(self._X.shape[1]):

            local_best_index = self.find_best_model_with_fixed_size(k)
            subset = self._X.iloc[:, local_best_index]
            local_best_score = cross_validate(lr, subset, self._y, 10, 'aic')['test_score']
            print(f'Best model with {k} features: AIC = {local_best_score}, best index = {local_best_index}')

            if local_best_score < global_best_score:
                global_best_index = local_best_index

        return global_best_index














