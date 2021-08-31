import numpy as np
import pandas as pd
from typing import Union, List
from abc import ABC, abstractmethod
import statsmodels.api as sm
from sklearn.metrics import make_scorer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate
from StatisticalLearning.ModelSelection.FitScore import FitScore


class StepwiseSelection(ABC):

    """
    Base class for stepwise model selection methods including:
    (1) forward stepwise
    (2) backward stepwise

    Forward stepwise algorithm:
       Step 1: Denote M_0 as the null model, which uses sample mean as prediction
       Step 2: For k = 1, 2, ..., p:
                 (a) Consider all p-k-1 options to add an extra feature and fit the regression model
                 (b) Denote M_k as the best model among the p-k-1 models above based on smallest SSR or
                     highest R2 on training set
       Step 3: Select a single best model out of M_0, M_1, ..., M_p based on cross validated prediction
               error (AIC/BIC/R^2-adj) on training set

    Backward stepwise algorithm:
       Step 1: Denote M_p as the full model, which uses all p features to fit the regression model
       Step 2: For k = p-1, p-2, ..., 1:
                 (a) Consider all k options to remove a feature from existing model and fit a new
                     regression model with k-1 features
                 (b) Denote M_k as the best model among the k models above based on smallest SSR or
                     highest R2 on training set
       Step 3: Select a single best model out of M_0, M_1, ..., M_p based on cross validated prediction
               error (AIC/BIC/R^2-adj) on training set

    Bi-directional stepwise algorithm:
       Step 1: Denote M_0 as the null model, which uses sample mean as prediction, best feature set is empty
       Step 2: While true:
                 if current model contains k features:
                 (a) Forward step: consider all p-k-1 options to add an extra feature and fit the regression model,
                                   find the best model all p-k-1 models
                 (b) Backward step: prune all features that has p-value higher than a pre-specified threshold

               End if no all variables in current model have reasonable low p value and any extra added feature
               will incur a p-value of itself to be higher than the threshold

    Limitation:
       (1) Computationally cheap
       (2) Sub-optimal: trade slightly higher bias with lower variance
       (3) Discrete: add or remove variables in a discrete way therefore often incur high variance
       (4) Forward selection is a greedy algorithm which might include features early that might become
           redundant later on
       (5) Forward selection can be used even when p > N but backward selection can only be used when N > p

     Solution:
      (1) Hybrid forward and backward: each step either add or remove a feature to minimize AIC
    """

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
        scorer = make_scorer(FitScore.adjusted_r_square, nbr_features=0, greater_is_better=True)
        best_score = cross_validate(estimator=lr, X=intercept, y=self._y, cv=10, scoring=scorer)['test_score'].mean()

        return best_score

    @abstractmethod
    def find_best_model_next_step(self, last_best_index: List[int]) -> Union[List[int], None]:

        """
        Abstract method to find next best model by adding an extra feature. Must be overriden.

        :param last_best_index: List[int], best feature index from last step
        :return: Union[List[int], None], column index of best features at this step
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


class LinearRegressionStepwiseForward(StepwiseSelection):

    """
    Stepwise forward selection method using linear regression as estimator
    """

    def __init__(self,
                 X: pd.DataFrame,
                 y: pd.Series):
        super().__init__(X, y)

    def find_best_model_next_step(self, last_best_index: List[int]) -> Union[List[int], None]:

        full_candidates = set(list(range(self._X.shape[1])))
        candidates = full_candidates - set(last_best_index) if last_best_index else full_candidates

        best_index, best_mse = None, float('inf')

        for index in candidates:

            included = last_best_index + [index]
            subset = self._X.iloc[:, included]
            lr = LinearRegression(fit_intercept=True, copy_X=True).fit(subset, self._y)
            mean_squared_error = FitScore.mean_square_error(self._y, lr.predict(subset))

            if mean_squared_error < best_mse:
                best_index = included
                best_mse = mean_squared_error

        return sorted(best_index)

    def find_best_model(self) -> Union[List[int], None]:

        lr = LinearRegression(fit_intercept=True, copy_X=True)
        global_best_score, global_best_index, local_best_index = self._null_model_score(), None, []
        print(f'Best model with intercept only: Adj-R2 = {global_best_score}')

        for k in range(self._X.shape[1]):

            local_best_index = self.find_best_model_next_step(local_best_index)
            subset = self._X.iloc[:, local_best_index]
            scorer = make_scorer(FitScore.adjusted_r_square, nbr_features=subset.shape[1], greater_is_better=True)
            local_best_score = cross_validate(estimator=lr, X=subset, y=self._y, cv=10,
                                              scoring=scorer)['test_score'].mean()
            print(f'Best model with {k + 1} features: Adj-R2 = {local_best_score}, best index = {local_best_index}')

            if local_best_score > global_best_score:
                global_best_index = local_best_index
                global_best_score = local_best_score

        return global_best_index


class LinearRegressionStepwiseBackward(StepwiseSelection):

    """
    Stepwise backward selection method using linear regression as estimator
    """

    def __init__(self,
                 X: pd.DataFrame,
                 y: pd.Series):
        super().__init__(X, y)

    def find_best_model_next_step(self, last_best_index: List[int]) -> Union[List[int], None]:

        full_candidates = set(last_best_index)
        best_index, best_mse = None, float('inf')

        for index in last_best_index:

            included = list(full_candidates - {index})
            subset = self._X.iloc[:, included]
            lr = LinearRegression(fit_intercept=True, copy_X=True).fit(subset, self._y)
            mean_squared_error = FitScore.mean_square_error(self._y, lr.predict(subset))

            if mean_squared_error < best_mse:
                best_index = included
                best_mse = mean_squared_error

        return sorted(best_index)

    def find_best_model(self) -> Union[List[int], None]:

        lr = LinearRegression(fit_intercept=True, copy_X=True)

        scorer = make_scorer(FitScore.adjusted_r_square, nbr_features=self._X.shape[1], greater_is_better=True)
        global_best_score = cross_validate(estimator=lr, X=self._X, y=self._y, cv=10,
                                           scoring=scorer)['test_score'].mean()
        global_best_index, local_best_index = list(range(self._X.shape[1])), list(range(self._X.shape[1]))
        print(f'Best model with {self._X.shape[1]} features: Adj-R2 = {global_best_score}')

        for k in range(self._X.shape[1] - 1, 0, -1):

            local_best_index = self.find_best_model_next_step(local_best_index)
            subset = self._X.iloc[:, local_best_index]
            scorer = make_scorer(FitScore.adjusted_r_square, nbr_features=subset.shape[1], greater_is_better=True)
            local_best_score = cross_validate(estimator=lr, X=subset, y=self._y, cv=10,
                                              scoring=scorer)['test_score'].mean()
            print(f'Best model with {k} features: Adj-R2 = {local_best_score}, best index = {local_best_index}')

            if local_best_score > global_best_score:
                global_best_index = local_best_index
                global_best_score = local_best_score

        local_best_score = self._null_model_score()
        print(f'Best model with intercept only: Adj-R2 = {local_best_score}')
        if local_best_score > global_best_score:
            global_best_index = local_best_index

        return global_best_index


class LinearRegressionStepwiseBidirectional(StepwiseSelection):

    """
    Stepwise bi-directional selection method using linear regression as estimator
    """

    def __init__(self,
                 X: pd.DataFrame,
                 y: pd.Series):
        super().__init__(X, y)

    def find_best_model_next_step(self, last_best_index: List[int]) -> Union[List[int], None]:

        full_candidates = set(list(range(self._X.shape[1])))
        candidates = full_candidates - set(last_best_index) if last_best_index else full_candidates

        best_index, best_mse = None, float('inf')

        # Forward by adding new feature
        for index in candidates:

            included = last_best_index + [index]
            subset = self._X.iloc[:, included]
            lr = LinearRegression(fit_intercept=True, copy_X=True).fit(subset, self._y)
            mean_squared_error = FitScore.mean_square_error(self._y, lr.predict(subset))

            if mean_squared_error < best_mse:
                best_index = included
                best_mse = mean_squared_error

        # Backward by pruning redundant features
        subset = sm.add_constant(self._X.iloc[: best_index])
        result = sm.OLS(self._y, subset).fit()
        keep_index = [i - 1 for i in range(1, subset.shape[1]) if result.pvalues[i] <= 0.05]
        return sorted(list(np.array(best_index)[keep_index]))

    def find_best_model(self) -> Union[List[int], None]:

        lr = LinearRegression(fit_intercept=True, copy_X=True)
        global_best_score, global_best_index, local_best_index = self._null_model_score(), None, []

        while True:

            local_best_index_next = self.find_best_model_next_step(local_best_index)

            subset = self._X.iloc[:, local_best_index]
            scorer = make_scorer(FitScore.adjusted_r_square, nbr_features=subset.shape[1], greater_is_better=True)
            local_best_score = cross_validate(estimator=lr, X=subset, y=self._y, cv=10,
                                              scoring=scorer)['test_score'].mean()

            if local_best_score > global_best_score:
                global_best_index = local_best_index
                global_best_score = local_best_score

            if local_best_index_next == local_best_index or local_best_index_next == range(self._X.shape[1]):
                return global_best_index

            local_best_index = local_best_index_next

