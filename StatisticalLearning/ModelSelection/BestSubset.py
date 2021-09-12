import pandas as pd
import numpy as np
from typing import Union, List
from itertools import combinations
from sklearn.linear_model import LinearRegression
from StatisticalLearning.Toolbox.Logger import Logger
from StatisticalLearning.ModelAssessment.ModelScore import ModelScore
from StatisticalLearning.ModelAssessment.CrossValidation import KFoldValidation


logger = Logger().get_logger(level='info')


class BestSubset:

    """
    Base class for best subset regression model selection methods

    Best subset algorithm:
       Step 1: Denote M_0 as the null model, which uses sample mean as prediction
       Step 2: For k = 1, 2, ..., p:
                 (a) Find all C(k, p) k-feature combinations and fit a regression model
                 (b) Denote M_k as the best model with k features based on RSS/MSE/R^2 on training set
       Step 3: Select a single best model out of M_0, M_1, ..., M_p based on cross validated prediction
               error (AIC/BIC/R^2-adj) on training set

    Limitation:
       (1) Computationally expensive
       (2) Potentially overfitting

    Solution:
       Heap package in R is proved to successfully handle both issues
    """

    def __init__(self,
                 X: pd.DataFrame,
                 y: pd.Series,
                 model: object):

        """
        :param X: pd.DataFrame, data matrix
        :param y: pd.Series, response vector
        :param model: object, model with fit() and prediction() implementation
        """
        self._X, self._y, self._model = X, y, model

    # ====================
    #  Private
    # ====================

    def _null_model_score(self) -> float:

        """
        Find the AIC score for model with only intercept term
        """

        lr = LinearRegression(fit_intercept=False, copy_X=False)
        intercept = pd.DataFrame(data=np.ones((self._X.shape[0], 1)))
        scorer = ModelScore.make_scorer(ModelScore.adjusted_r_square, nbr_features=0)
        validator = KFoldValidation(X=intercept, y=self._y, model=lr, scorer=scorer)
        best_score = validator.validate().mean_score

        return best_score

    # ====================
    #  Public
    # ====================

    def find_best_model_with_fixed_size(self, k: int, **kwargs) -> Union[List[int], None]:

        """
        Find best model with k features.

        :param k: int, number of features
        :return: Union[List[int], None], column index of best k features in X
        """

        best_index, best_mse = None, float('inf')

        for sub_index in combinations(list(range(self._X.shape[1])), k):

            subset = self._X.iloc[:, list(sub_index)]
            model = self._model.fit(subset, self._y, **kwargs)
            mean_squared_error = ModelScore.mean_square_error(self._y, model.predict(subset))

            if mean_squared_error < best_mse:
                best_index = list(sub_index)
                best_mse = mean_squared_error

        return best_index

    def find_best_model(self, **kwargs) -> Union[List[int], None]:

        """
        Abstract method to find the optimal model from best models with different
        number of features. Must be overriden.

        :return: Union[List[int], None], column index of optimal subset features in X
        """

        global_best_score, global_best_index = self._null_model_score(), None
        logger.info(f'Best model with intercept only: Adj-R2 = {global_best_score}')

        for k in range(1, self._X.shape[1] + 1):

            local_best_index = self.find_best_model_with_fixed_size(k, **kwargs)
            subset = self._X.iloc[:, local_best_index]
            scorer = ModelScore.make_scorer(ModelScore.adjusted_r_square, nbr_features=subset.shape[1])
            validator = KFoldValidation(X=subset, y=self._y, model=self._model, scorer=scorer)
            local_best_score = validator.validate(**kwargs).mean_score
            logger.info(f'Best model with {k} features: Adj-R2 = {local_best_score}, best index = {local_best_index}')

            if local_best_score > global_best_score:
                global_best_index = local_best_index
                global_best_score = local_best_score

        return global_best_index
