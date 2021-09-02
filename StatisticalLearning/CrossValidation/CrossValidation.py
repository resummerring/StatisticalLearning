from __future__ import annotations

import random
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Protocol, Callable, List


class Model(Protocol):
    """
    A protocol class with fit and predict function required
    """
    def fit(self, X: pd.DataFrame, y: pd.Series) -> Model: ...
    def predict(self, X: pd.DataFrame) -> pd.Series: ...


class CrossValidation(ABC):

    def __init__(self,
                 X: pd.DataFrame,
                 y: pd.Series,
                 model: Model,
                 scorer: Callable):
        """
       :param X: pd.DataFrame, n * p where n = #samples and p = #features
       :param y: pd.Series, n * 1 where n = #samples
       :param model: Model, a model with fit() and predict() implementation
       :param scorer: Callable, a callable to compute score from y_true and y_pred
       """

        self._X, self._y = X, y
        self._model, self._scorer = model, scorer
        self._scores, self._mean_score = [], None

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
                 model: Model,
                 scorer: Callable):

        super().__init__(X, y, model, scorer)

    def validate(self, train_ratio: float = 0.7, seed: int = 0) -> ValidationSet:

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

        model_trained = self._model.fit(X_train, y_train)
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
                 model: Model,
                 scorer: Callable):

        super().__init__(X, y, model, scorer)

    def validate(self) -> LeaveOneOut:

        n_samples = self._X.shape[0]
        full_candidates = set(range(n_samples))

        for test_index in range(n_samples):

            train_index = list(full_candidates - {test_index})
            X_train, y_train = self._X.iloc[train_index, :], self._y.iloc[train_index]
            X_test, y_test = pd.DataFrame(self._X.iloc[test_index, :]).T, self._y[test_index]

            model_trained = self._model.fit(X_train, y_train)
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
                 model: Model,
                 scorer: Callable):

        super().__init__(X, y, model, scorer)

    def validate(self, k: int = 5, seed: int = 0) -> KFoldValidation:

        """
        :param k: int, number of folds to use in cross validation
        :param seed: int, random number generator seed
        """

        n_samples = self._X.shape[0]
        full_candidates = set(np.arange(n_samples))
        shuffled_candidates = list(np.arange(n_samples))
        random.shuffle(shuffled_candidates)
        kfolds = [set(fold) for fold in np.array_split(shuffled_candidates, k)]

        for i in range(k):
            train_index, test_index = list(full_candidates - kfolds[i]), list(kfolds[i])
            X_train, y_train = self._X.iloc[train_index, :], self._y.iloc[train_index]
            X_test, y_test = self._X.iloc[test_index, :], self._y.iloc[test_index]

            model_trained = self._model.fit(X_train, y_train)
            pred = model_trained.predict(X_test)
            self._scores.append(self._scorer(y_test, pred))

        self._mean_score = np.mean(self._scores)

        return self
