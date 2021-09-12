from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from StatisticalLearning.Preprocess import Preprocessor


class PCRegression:

    """
    Principal component regression:
    (1) Standardize data matrix to zero mean and unit variance
    (2) Decompose scaled data matrix to retrieve principal components
    (3) Build linear regression model on principal components
    (4) Predict by projecting test data onto fitted principal component directions

    Comment: PCR is a continuous version of Ridge regression
        (1) Ridge regression shrinks principal components based on the size of corresponding eigenvalues
        (2) PCR shrinks principal components by dropping the least important ones based on eigenvalues

    Limitation:
    (1) PCR works well when the first several principal components can capture most of the variance
    (2) Principal components are constructed in an unsupervised way, meaning the principal components are best explain
        most of the variance of the data itself but it might not explain the variance of the response

    """

    def __init__(self, n_components: int):
        """
        :param n_components: int, number of dimensions we want to keep
        """

        self._n_components = n_components
        self._lr, self._pca = None, None

    # ====================
    #  Private
    # ====================

    def _clean_up(self):
        self._lr, self._pca = None, None

    # ====================
    #  Public
    # ====================

    def fit(self, X: pd.DataFrame, y: pd.Series) -> PCRegression:
        """
        Fit method
        """

        self._clean_up()

        X, y = X.reset_index(drop=True), y.reset_index(drop=True)

        X_scaled, _, _ = Preprocessor.normalize(X)
        self._pca = PCA(n_components=self._n_components)
        X_reduced = self._pca.fit_transform(X_scaled)
        self._lr = LinearRegression(fit_intercept=True, copy_X=True).fit(X_reduced, y)
        return self

    def predict(self, X_test) -> pd.Series:
        """
        Predict method
        """

        if self._pca is None or self._lr is None:
            raise ValueError("Model hasn't been fitted yet")

        X_test_scaled, _, _ = Preprocessor.normalize(X_test)
        X_reduced_test = self._pca.transform(X_test_scaled)
        return self._lr.predict(X_reduced_test)


class PartialLeastSquare:

    """
    Partial Least Square Algorithm (1D):
    (1) w = (X.T @ y) / ||X.T @ y||
    (2) t = X @ w
    (3) P = X.T @ t / ||t||
    (4) X = X - t @ P
    (5) q = t.T @ y / ||t||
    (6) y = y - t @ q

    Partial Least Square algorithm seeks to find direction that have high variance and high correlations
    with response while principal component regression only finds directions with high variance in predictors.
    Moreover, partial least  square can easily handle situations where #samples < #feature.

    Principal Component Regression:
        max_a Var(X @ a) s.t. ||a|| = 1 and a.T @ S @ v = 0
    Partial Least Square:
        max_a Corr^2(y, X @ a) * Var(X @ a) s.t. ||a|| = 1 and a.T @ S @ v = 0
    Here S is the covariance matrix. New direction z = X @ a is ensured to be orthogonal to previous direction
    by the condition a.T @ S @ v = 0

    In most cases, variance will dominate and partial least square will behave similar to Principal component
    regression and ridge regression.
    """

    def __init__(self, n_components: int):
        """
        :param n_components: int, number of dimensions we want to keep
        """

        self._n_components = n_components
        self._B, self._mean, self._std = None, None, None

    # ====================
    #  Private
    # ====================

    def _clean_up(self):
        self._B, self._mean, self._std = None, None, None

    # ====================
    #  Public
    # ====================

    def fit(self, X: pd.DataFrame, y: pd.Series) -> PartialLeastSquare:
        """
        Fit method
        """

        self._clean_up()

        X, y = X.reset_index(drop=True), y.reset_index(drop=True)

        X_scaled, self._mean, self._std = Preprocessor.normalize(X)

        # Loading matrix
        W, P, q = [], [], []

        # Initialization
        X_k, y = np.array(X_scaled), np.array(y).reshape(-1, 1)
        for k in range(self._n_components):

            w_k = (X_k.T @ y) / np.linalg.norm(X_k.T @ y, ord=2)
            t_k = (X_k @ w_k).reshape(-1, 1) / np.linalg.norm(X_k @ w_k, ord=2)
            p_k = (X_k.T @ t_k).reshape(-1, 1)
            q_k = t_k.T @ y

            if q_k.squeeze() == 0:
                break

            W.append(w_k.squeeze().tolist())
            P.append(p_k.squeeze().tolist())
            q.append(q_k.squeeze())

            X_k = X_k - t_k @ p_k.T

        W, P, q = np.array(W).T, np.array(P).T, np.array(q).reshape(-1, 1)

        self._B = (W @ np.linalg.inv(P.T @ W)) @ q

        return self

    def predict(self, X_test: pd.DataFrame) -> pd.Series:
        """
        Predict method
        """

        if self._B is None:
            raise ValueError("Model has not been fitted yet")

        X_test_scaled = (X_test - self._mean) / self._std
        return np.dot(X_test_scaled, self._B).squeeze()
