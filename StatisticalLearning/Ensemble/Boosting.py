from __future__ import annotations

import copy
import random
import pandas as pd


class BoostRegressor:

    """
    Gradient Boost Regression Algorithm (least square loss function):
        (1) Initialize F_0 = mean(y)
        (2) For i = 1, 2, ..., M:
                * compute pseudo-residual r = y - y_pred
                * construct weak estimator F_i and fit on (X, r)
                * update model F = F_0 + a * F_1 + ... + a * F_i
        (3) Predict using F(x_k) = F_0 + a * F_1(x_k) + ... + F_M(x_k)

    Note:
    (1) Learning rate a is typically between 0.001 and 0.01 to prevent overfitting
    (2) Optimal number of weak estimators is generated through cross-validation
    (3) Weak estimator is typically chosen as 1-depth decision tree

    Cons:
    (1) Extreme sensitive to outliers since each estimators is built on previous residuals
    (2) Slower than bagging since weak estimators are build sequentially instead of simultaneously
    (3) Prone to overfit, therefore low-level learning rate and decision tree are needed

    More general boosting algorithm:
    (1) Generate training set by random sampling with replacement with weights (uniform for the first one)
    (2) Fit an estimator using training set. Keep if accuracy meets threshold, otherwise discard
    (3) Give more weight to misclassified samples and less weight to correctly classified observations
    (4) Repeat the above until N estimators are found.
    (5) Ultimate prediction is the weighted average of individual predictions
    """

    def __init__(self,
                 estimator: object,
                 n_estimators: int = 100,
                 learning_rate: float = 0.01,
                 subsample: float = 1.0):

        """
        :param estimator: object, weak estimator
        :param n_estimators: int, number of weak estimators
        :param learning_rate: float, shrinkage coefficient for each subsequent weak estimator
        :param subsample: float, proportion of data to be used to fit subsequent weak estimator
        """

        self._estimator, self._n_estimators = estimator, n_estimators
        self._learning_rate, self._subsample = learning_rate, subsample
        self._average, self._last_prediction, self._estimators = None, None, []

    # ====================
    #  Private
    # ====================

    def _clean_up(self):
        self._average, self._last_prediction, self._estimators = None, None, []

    # ====================
    #  Public
    # ====================

    def fit(self, X: pd.DataFrame, y: pd.Series, seed: int = 0, **kwargs) -> BoostRegressor:
        """
        Fit method
        """

        self._clean_up()

        X, y = X.reset_index(drop=True), y.reset_index(drop=True)

        random.seed(seed)

        full_sample, nbr_sample = tuple(range(X.shape[0])), int(self._subsample * X.shape[0])

        # Initialization: F_0 = mean(y)
        self._average, self._last_prediction = y.mean(), pd.Series([y.mean()] * len(y))

        for i in range(self._n_estimators):

            # Compute pseudo-residuals
            residuals = y - self._last_prediction

            # Generate sub-samples without replacement
            partial = random.sample(full_sample, k=nbr_sample)
            X_train, y_train = X.iloc[partial, :], residuals[partial]

            # Train new weak estimator
            estimator = copy.deepcopy(self._estimator)
            estimator = estimator.fit(X_train, y_train, **kwargs)
            self._estimators.append(estimator)

            # Update prediction
            prediction = pd.Series(estimator.predict(X)).reset_index(drop=True)
            self._last_prediction += self._learning_rate * prediction

        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Predict method
        """

        result = pd.Series([self._average] * X.shape[0])

        for i in range(self._n_estimators):

            current_prediction = pd.Series(self._estimators[i].predict(X)).reset_index(drop=True)
            result += self._learning_rate * current_prediction

        return result







