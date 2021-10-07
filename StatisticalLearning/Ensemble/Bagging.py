from __future__ import annotations

import copy
import random
import pandas as pd
from typing import Union
from StatisticalLearning.ModelAssessment.ModelScore import ModelScore


class BaggingRegressor:

    """
    Bagging Algorithm:
        (1) Resample n different training datasets by bootstrapping
            (samples are sampled with replacement and features are sample without replacement)
        (2) Build n different estimators based on the above n different training set
            (and it's possible that each estimator is allowed to use only partial features)
        (3) Predict n different outcomes and taking the average as the final prediction
            (regression will take average while classification will take majority vote)

    Random Forest is bagging with base estimator as CART and using partial features for each tree

    Pros:
        (1) Variance is largely reduced at the cost of slightly increasing the bias
        (2) Natural out-of-bag error is available instead of cross-validation

    Var(bagging) = \sigma^2 * (\rho + (1 - \rho) / N)
    Here \rho is the average correlation among all estimators. When \rho -> 1, Var(bagging) -> \sigma^2,
    which means the variance is barely reduced. Hence, if the training set of different estimators are
    highly correlated, then the variance reduction effect might not be significant any more, which is
    particularly the case with finance data where samples are not IID and have high correlation. Even
    worse, the out of bag score will be inflated when using redundant observations.

    Solution:
    (1) Set max_samples for each sub-estimator to be the average uniqueness of the observed samples
    (2) Apply sequential bootstrapping method
    (3) Apply stratified k-fold cross validation

    Generally, bagging is preferred compared to boosting since financial applications is more likely to
    overfit rather than underfit and bagging can be paralleled while boosting needs to be fit sequentially.

    Machine learning algorithms such as SVM that doesn't scale well with sample size becomes more capable of
    learning large training set when combine with bagging. For example, using SVM as the base estimator and
    set early-stop condition (max iteration or tol) and parallelize N estimators will make it easier for
    SVM to learn large training set. (Individual SVM might take long long time to converge, if any, and the
    solution is not guaranteed to be a global optimum)
    """

    def __init__(self,
                 estimator: object,
                 n_estimator: int = 100,
                 max_samples: Union[int, float] = 1.0,
                 max_features: Union[int, float] = 1.0):

        """
        :param estimator: object, model with fit() and predict() implementation
        :param n_estimator: int, # estimators to use for averaging the prediction
        :param max_samples: Union[int, float], determine #samples to be used to train each estimator
        :param max_features: Union[int, float], determine #features to be used to train each estimator
        """

        self._estimator, self._n_estimator = estimator, n_estimator
        self._max_samples, self._max_features = max_samples, max_features
        self._sample_container, self._feature_container = None, None
        self._trained_models, self._out_of_bag_error = [], None

    # ====================
    #  Private
    # ====================

    def _bootstrap(self, total_samples: int, total_features: int, seed: int):

        """
        Generate bootstrapped datasets

        :param total_samples: int, total number of samples to bootstrap from
        :param total_features: int, total number of features to bootstrap from
        :param seed: int, random number generator seed
        """

        random.seed(seed)

        nbr_samples, nbr_features = self._max_samples, self._max_features

        if isinstance(self._max_samples, float):
            nbr_samples = int(self._max_samples * total_samples)
        if isinstance(self._max_features, float):
            nbr_features = int(self._max_features * total_features)

        # Determine #samples and #features to train each estimator
        sample_backet, feature_backet = tuple(range(total_samples)), tuple(range(total_features))

        # bootstrap samples with replacement and features without replacement
        self._sample_container = [random.choices(sample_backet, k=nbr_samples) for _ in range(self._n_estimator)]
        self._feature_container = [random.sample(feature_backet, k=nbr_features) for _ in range(self._n_estimator)]

    def _clean_up(self):
        self._sample_container, self._feature_container, self._trained_models = None, None, []

    # ====================
    #  Public
    # ====================

    @property
    def out_of_bag_error(self) -> float:
        """
        Out-of-bag mean squared error
        """

        if self._out_of_bag_error is None:
            raise ValueError('Model has not been fitted yet.')
        return self._out_of_bag_error

    def fit(self, X: pd.DataFrame, y: pd.Series, seed: int = 0, **kwargs) -> BaggingRegressor:
        """
        Fit method
        """

        self._clean_up()

        X, y = X.reset_index(drop=True), y.reset_index(drop=True)

        total_samples, total_features = X.shape[0], X.shape[1]
        self._bootstrap(total_samples, total_features, seed)

        full_candidates = set(range(X.shape[0]))

        # Container to collect all out-of-bag predictions
        oob_predictions = pd.DataFrame(index=list(range(X.shape[0])), columns=list(range(self._n_estimator)))

        for i in range(self._n_estimator):

            estimator = copy.deepcopy(self._estimator)

            # Fit model on bootstrapped dataset
            sample_index, feature_index = self._sample_container[i], self._feature_container[i]
            X_train, y_train = X.iloc[sample_index, feature_index], y[sample_index]
            self._trained_models.append(estimator.fit(X_train, y_train, **kwargs))

            # Out-of-bag prediction
            val_index = list(full_candidates - set(sample_index))
            X_val = X.iloc[val_index, feature_index]
            prediction = pd.Series(self._trained_models[-1].predict(X_val))
            prediction.index = val_index
            oob_predictions[i] = oob_predictions[i].fillna(prediction)

        oob_prediction = oob_predictions.mean(axis=1, skipna=True)
        self._out_of_bag_error = ModelScore.mean_square_error(y, oob_prediction)

        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Predict method
        """

        predictions = []

        for i in range(self._n_estimator):
            feature_index = self._feature_container[i]
            X_test = X.iloc[:, feature_index]
            predictions.append(self._trained_models[i].predict(X_test))

        return pd.DataFrame(predictions).mean(axis=0)


class BaggingClassifier:

    """
    Bagging Algorithm:
        (1) Resample n different training datasets by bootstrapping
            (samples are sampled with replacement and features are sample without replacement)
        (2) Build n different estimators based on the above n different training set
            (and it's possible that each estimator is allowed to use only partial features)
        (3) Predict n different labels and use the majority vote as the final prediction
            (regression will take average while classification will take majority vote)

    Random Forest is bagging with base estimator as CART and using partial features for each tree

    Pros:
        (1) Variance is largely reduced at the cost of slightly increasing the bias
        (2) Natural out-of-bag error is available instead of cross-validation
    """

    def __init__(self,
                 estimator: object,
                 n_estimator: int = 100,
                 max_samples: Union[int, float] = 1.0,
                 max_features: Union[int, float] = 1.0):

        """
        :param estimator: object, model with fit() and predict() implementation
        :param n_estimator: int, # estimators to use for averaging the prediction
        :param max_samples: Union[int, float], determine #samples to be used to train each estimator
        :param max_features: Union[int, float], determine #features to be used to train each estimator
        """

        self._estimator, self._n_estimator = estimator, n_estimator
        self._max_samples, self._max_features = max_samples, max_features
        self._sample_container, self._feature_container = None, None
        self._trained_models, self._out_of_bag_metric = [], None

    # ====================
    #  Private
    # ====================

    def _bootstrap(self, total_samples: int, total_features: int, seed: int):

        """
        Generate bootstrapped datasets

        :param total_samples: int, total number of samples to bootstrap from
        :param total_features: int, total number of features to bootstrap from
        :param seed: int, random number generator seed
        """

        random.seed(seed)

        nbr_samples, nbr_features = self._max_samples, self._max_features

        if isinstance(self._max_samples, float):
            nbr_samples = int(self._max_samples * total_samples)
        if isinstance(self._max_features, float):
            nbr_features = int(self._max_features * total_features)

        # Determine #samples and #features to train each estimator
        sample_backet, feature_backet = tuple(range(total_samples)), tuple(range(total_features))

        # bootstrap samples with replacement and features without replacement
        self._sample_container = [random.choices(sample_backet, k=nbr_samples) for _ in range(self._n_estimator)]
        self._feature_container = [random.sample(feature_backet, k=nbr_features) for _ in range(self._n_estimator)]

    def _clean_up(self):
        self._sample_container, self._feature_container, self._trained_models = None, None, []

    # ====================
    #  Public
    # ====================

    @property
    def out_of_bag_metric(self) -> dict:
        """
        Out-of-bag classification metrics
        """

        if self._out_of_bag_metric is None:
            raise ValueError('Model has not been fitted yet.')
        return self._out_of_bag_metric

    def fit(self, X: pd.DataFrame, y: pd.Series, seed: int = 0, **kwargs) -> BaggingClassifier:
        """
        Fit method
        """

        self._clean_up()

        X, y = X.reset_index(drop=True), y.apply(lambda num: int(num)).reset_index(drop=True)

        total_samples, total_features = X.shape[0], X.shape[1]
        self._bootstrap(total_samples, total_features, seed)

        full_candidates = set(range(X.shape[0]))

        # Container to collect all out-of-bag predictions
        oob_predictions = pd.DataFrame(index=list(range(X.shape[0])), columns=list(range(self._n_estimator)))

        for i in range(self._n_estimator):

            estimator = copy.deepcopy(self._estimator)

            # Fit model on bootstrapped dataset
            sample_index, feature_index = self._sample_container[i], self._feature_container[i]
            X_train, y_train = X.iloc[sample_index, feature_index], y[sample_index]
            self._trained_models.append(estimator.fit(X_train, y_train, **kwargs))

            # Out-of-bag prediction
            val_index = list(full_candidates - set(sample_index))
            X_val = X.iloc[val_index, feature_index]
            prediction = pd.Series(self._trained_models[-1].predict(X_val))
            prediction.index = val_index
            oob_predictions[i] = oob_predictions[i].fillna(prediction)

        oob_prediction = oob_predictions.mode(axis=1, dropna=True).squeeze()
        assert isinstance(oob_prediction, pd.Series) , "Multiple values returned for majority vote"
        self._out_of_bag_metric = ModelScore.confusion_matrix(y, oob_prediction)

        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Predict method
        """

        predictions = []

        for i in range(self._n_estimator):
            feature_index = self._feature_container[i]
            X_test = X.iloc[:, feature_index]
            predictions.append(self._trained_models[i].predict(X_test))

        result = pd.DataFrame(predictions).mode(axis=0, dropna=True).squeeze()
        assert isinstance(result, pd.Series), "Multiple values returned for majority vote"
        return result
