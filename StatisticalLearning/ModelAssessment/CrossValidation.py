from __future__ import annotations

import random
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Callable, List, Union
from sklearn.model_selection import KFold


class CrossValidation(ABC):

    def __init__(self,
                 X: pd.DataFrame,
                 y: pd.Series,
                 model: object,
                 scorer: Callable):
        """
       :param X: pd.DataFrame, n * p where n = #samples and p = #features
       :param y: pd.Series, n * 1 where n = #samples
       :param model: object, a model with fit() and predict() implementation
       :param scorer: Callable, a callable to compute score from y_true and y_pred
       """

        self._X, self._y = X, y
        self._model, self._scorer = model, scorer
        self._scores, self._mean_score = [], None

    # ====================
    #  Public
    # ====================

    @abstractmethod
    def validate(self, **kwargs):
        """
        An abstract method that all child classes should override.
        Main function where cross validation scores are computed.
        """
        raise NotImplementedError

    @property
    def scores(self) -> List[float]:
        return self._scores

    @property
    def mean_score(self) -> float:
        return self._mean_score


class ValidationSet(CrossValidation):

    """
    Validation set algorithm:
        (1) Given a dataset, split the full dataset into a train set and test set by randomly
            picking a certain amount of samples into train set and the others into test set
        (2) Train the model on the train set
        (3) Compute test error on the test set

    Advantage:
    (1) The model only needs to be fitted once

    Limitation:
    (1) Random split will give different test error each time
    (2) Not all information is used for training, which results in potential increased bias
    """

    def __init__(self,
                 X: pd.DataFrame,
                 y: pd.Series,
                 model: object,
                 scorer: Callable):

        super().__init__(X, y, model, scorer)

    # ====================
    #  Public
    # ====================

    def validate(self, train_ratio: float = 0.7, seed: int = 0, **kwargs) -> ValidationSet:

        """
        :param train_ratio: float: proportion to be included in training set
        :param seed: int, random number generator seed
        """

        n_samples = self._X.shape[0]
        full_candidates = set(range(n_samples))
        n_train = int(train_ratio * n_samples)

        random.seed(seed)
        train_index = set(random.sample(full_candidates, n_train))
        test_index = full_candidates - train_index
        train_index, test_index = list(train_index), list(test_index)
        X_train, y_train = self._X.iloc[train_index, :], self._y.iloc[train_index]
        X_test, y_test = self._X.iloc[test_index, :], self._y.iloc[test_index]

        model_trained = self._model.fit(X_train, y_train, **kwargs)
        pred = model_trained.predict(X_test)
        self._scores.append(self._scorer(y_test, pred))
        self._mean_score = np.mean(self._scores)

        return self


class LeaveOneOut(CrossValidation):

    """
    LOOCV (leave one out cross validation) algorithm:
        for i = 1, ..., n_samples:
            (1) remove sample i from dataset
            (2) train the model with samples [1, ..., i-1, i+1, ...,n_samples]
            (3) predict y_i using above trained model and calculate test error
        cross-validated test error = mean of test error

    Advantage:
    (1) Less bias since almost all data are used in training
    (2) Less randomness in terms of different train/test splits

    Limitation:
    Computationally expense. n times of fitting are needed.
    """

    def __init__(self,
                 X: pd.DataFrame,
                 y: pd.Series,
                 model: object,
                 scorer: Callable):

        super().__init__(X, y, model, scorer)

    # ====================
    #  Public
    # ====================

    def validate(self, **kwargs) -> LeaveOneOut:

        n_samples = self._X.shape[0]
        full_candidates = set(range(n_samples))

        for test_index in range(n_samples):

            train_index = list(full_candidates - {test_index})
            X_train, y_train = self._X.iloc[train_index, :], self._y.iloc[train_index]
            X_test, y_test = pd.DataFrame(self._X.iloc[test_index, :]).T, self._y[test_index]

            model_trained = self._model.fit(X_train, y_train, **kwargs)
            pred = model_trained.predict(X_test).squeeze()
            self._scores.append(self._scorer(y_test, pred))

        self._mean_score = np.mean(self._scores)

        return self


class KFoldValidation(CrossValidation):

    """
    K-fold cross validation algorithm:
        (1) Randomly but equally divide n samples into k buckets
        (2) For i = 1, 2, ..., k:
                use the all buckets except the i-th bucket as train set to fit the model
                use the fitted model to predict and compute test error on the i-th bucket
        (3) Take the average of test error on each bucket as the final test error

    Advantage:
        (1) Balance between high variability and computation cost
    """

    def __init__(self,
                 X: pd.DataFrame,
                 y: pd.Series,
                 model: object,
                 scorer: Callable):

        super().__init__(X, y, model, scorer)

    # ====================
    #  Public
    # ====================

    def validate(self, k: int = 5, seed: int = 0, **kwargs) -> KFoldValidation:

        """
        :param k: int, number of folds to use in cross validation
        :param seed: int, random number generator seed
        """

        random.seed(seed)

        n_samples = self._X.shape[0]
        full_candidates = set(np.arange(n_samples))
        shuffled_candidates = list(np.arange(n_samples))
        random.shuffle(shuffled_candidates)
        kfolds = [set(fold) for fold in np.array_split(shuffled_candidates, k)]

        for i in range(k):
            train_index, test_index = list(full_candidates - kfolds[i]), list(kfolds[i])
            X_train, y_train = self._X.iloc[train_index, :], self._y.iloc[train_index]
            X_test, y_test = self._X.iloc[test_index, :], self._y.iloc[test_index]

            model_trained = self._model.fit(X_train, y_train, **kwargs)
            pred = model_trained.predict(X_test)
            self._scores.append(self._scorer(y_test, pred))

        self._mean_score = np.mean(self._scores)

        return self


class PurgedKFoldValidation(CrossValidation):

    """
    The assumption underlying the usefulness of cross validation is that samples are IID to
    each other. Otherwise, cross validation tend to overfit and leakage appears in the presence
    of irrelevant features.

    Solution:
    (1) Purge: when test set is surrounded by training sets, remove the boundary samples in the
               training set which might use information contained the test set
    (2) Embargo: for each training set immediately after a test set, put an embargo period (can
                 be as small as 0.01 * N) to separate training set and test set
    """

    def __init__(self,
                 X: pd.DataFrame,
                 y: pd.Series,
                 model: object,
                 scorer: Callable):

        super().__init__(X, y, model, scorer)

    # ====================
    #  Public
    # ====================

    def validate(self, k: int, time: pd.Series, embargo: float, **kwargs) -> PurgedKFoldValidation:

        """
        :param k: int, number of splits
        :param time: pd.Series, index is start time and value is end time of training samples
        :param embargo: float, percentage of embargo period as proportion of total number of samples
        """

        split_gen = self.PurgedSplit(k, time, embargo)

        for train_index, test_index in split_gen.split(self._X):

            X_train, y_train = self._X.iloc[train_index, :], self._y.iloc[train_index]
            X_test, y_test = self._X.iloc[test_index, :], self._y.iloc[test_index]

            model_trained = self._model.fit(X_train, y_train)
            pred = model_trained.predict(X_test)
            self._scores.append(self._scorer(y_test, pred))

        self._mean_score = np.mean(self._scores)

        return self

    class PurgedSplit(KFold):

        """
        Construct generator to yield sample indices in the train and test sets
        after embargo period and purging are applied on original training sets.
        """

        def __init__(self,
                     n_splits: int = 5,
                     time: Union[pd.Series, None] = None,
                     embargo: float = 0.):

            """
            :param n_splits: int, number of splits
            :param time: Union[pd.Series, None], index is start time and value is end time for training samples
            :param embargo: float, percentage of embargo period as proportion of total number of samples
            """

            super().__init__(n_splits, shuffle=False, random_state=None)
            self._time, self._embargo = time, embargo

        # ====================
        #  Private
        # ====================

        def _get_train_index(self, test_times: pd.Series) -> pd.Series:
            """
            Assume a training sample spans [t(i0, t(i1))] and a test sample spans [t(j0), t(j1)]. Then
            sample i and sample j overlap with each other if and only if any of following cases is true:
            (1) t(j0) <= t(i0) <= t(j1)
            (2) t(j0) <= t(i1) <= t(j1)
            (3) t(i0) <= t(j0) <= t(j1) <= t(i1)

            :param test_times: pd.Series, index is start time and value is end time for test samples
            """

            output = self._time.copy(deep=True)
            for i, j in test_times.iteritems():
                case1 = output[(i <= output.index) & (output.index <= j)]
                case2 = output[(i <= output) & (output <= j)].index
                case3 = output[(output.index <= i) & (j <= output)].index
                output = output.drop(case1.union(case2).union(case3))

            return output

        def _get_embargo_period(self) -> pd.Series:
            """
            Apply embargo period onto original dataset
            """

            step = int(self._time.shape[0] * self._embargo)

            if step == 0:
                embargo = pd.Series(self._time, index=self._time)
            else:
                embargo = pd.Series(self._time[step:], index=self._time[:-step])
                embargo = embargo.append(pd.Series(self._time[-1], index=self._time[-step:]))

            return embargo

        # ====================
        #  Public
        # ====================

        def split(self, X: pd.DataFrame, y: pd.Series = None, groups=None):
            """
            Override base class method. Split dataset into training and test set.
            Apply embargo period first and then purge the training set splits.
            """

            if (X.index == self._time.index).sum() != len(self._time):
                raise ValueError("Data matrix and time events must have same index.")

            indices = np.arange(X.shape[0])
            embargo = int(X.shape[0] * self._embargo)

            test_start = [(x[0], x[-1] + 1) for x in np.array_split(np.arange(X.shape[0]), self.n_splits)]

            for i, j in test_start:

                t0 = self._time.index[i]
                test_index = indices[i: j]

                max_idx = self._time.index.searchsorted(self._time[test_index].max())
                train_index = self._time.index.searchsorted(self._time[self._time <= t0].index)
                train_index = np.concatenate((train_index, indices[max_idx + embargo:]))

                yield train_index, test_index
