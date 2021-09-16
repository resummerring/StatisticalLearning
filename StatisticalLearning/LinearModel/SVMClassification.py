from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Callable, List
from cvxopt import matrix, solvers


class SVMClassifier:

    """
    Support Vector Classifier Algorithm:
        (1) Describe classification problem as to maximize the margin distance:
            Hard margin: min ||w|| -> min w.T @ w s.t. y(w.T @ w + b) >= 1
            Soft margin: min w.T @ w + C sum(epsilon) s.t. y(w.T @ w + b) >= 1 - epsilon
        (2) Feature primary problem to dual problem
        (3) Solve dual problem using quadratic programming

    Pros:
        (1) Soft margin classifier adds the flexibility to adjust bias-variance trade-off
        (2) Kernel trick can be easily applied to handle non-linear decision boundary
            The dual problem only depends on inner product of samples
    Cons:
        (1) Can only be used for two-class classification (one-versus-one and one-versus-all
            could be used to expand SVM to multi-class classification)
    """

    def __init__(self, kernel: Callable, C: float = None):
        """
        :param kernel: Callable, a valid kernel function
        :param C: float, soft margin penalty, if None hard margin
        """

        self._C, self._kernel = C, kernel
        self._alpha, self._sv_index, self._sv_x, self._sv_y, self._b = None, None, None, None, None

    # ====================
    #  Private
    # ====================

    def _clean_up(self):
        self._alpha, self._sv_index, self._sv_x, self._sv_y, self._b = None, None, None, None, None

    def _predict_individual(self, x: pd.Series) -> int:
        """
        Predict for one data point
        """

        x = x.to_numpy().reshape(1, -1)
        prediction = np.sum([alpha * sv_y * self._kernel(x, sv_x)
                             for alpha, sv_x, sv_y in zip(self._alpha, self._sv_x, self._sv_y)])

        return int(np.sign(prediction + self._b))

    # ====================
    #  Public
    # ====================

    @property
    def sv_index(self) -> List:
        """
        Row index of support vectors in data matrix
        """

        return list(self._sv_index)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> SVMClassifier:
        """
        Fit method
        """

        X, y = X.to_numpy(), y.apply(lambda num: float(num)).to_numpy().reshape(-1, 1)

        n_sample, n_feature = X.shape[0], X.shape[1]

        # Kernel inner product
        K = np.zeros((n_sample, n_sample))
        for i, x_i in enumerate(X):
            for j, x_j in enumerate(X):
                K[i, j] = self._kernel(x_i, x_j)

        # Constructing cvxopt matrix
        P = matrix(np.outer(y, y) * K)
        q = matrix(np.ones(n_sample) * -1)
        A = matrix(y, (1, n_sample))
        b = matrix(0.0)

        # Hard margin
        if self._C is None:
            G = matrix(np.diag(np.ones(n_sample) * -1))
            h = matrix(np.zeros(n_sample))
        # Soft margin
        else:
            G_std = np.diag(np.ones(n_sample) * -1)
            G_slack = np.identity(n_sample)
            G = matrix(np.vstack((G_std, G_slack)))

            h_std = np.zeros(n_sample)
            h_slack = np.ones(n_sample) * self._C
            h = matrix(np.hstack((h_std, h_slack)))

        # Support vectors
        result = solvers.qp(P, q, G, h, A, b, options={'show_progress': False})
        status, alpha = result['status'], np.ravel(result['x'])
        if result['status'] != 'optimal':
            raise ValueError(f'Quadratic programming failed: {status}')

        is_sv, sv_index = alpha > 1e-5, np.arange(len(alpha))[alpha > 1e-4]
        self._alpha, self._sv_index, self._sv_x, self._sv_y = alpha[is_sv], sv_index, X[is_sv], y[is_sv]

        # Intercept
        self._b = np.mean([self._sv_y[n] -
                           np.sum(self._alpha * self._sv_y * K[sv_index[n], is_sv]) for n in range(len(self._alpha))])

        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Predict method
        """

        if self._alpha is None:
            raise ValueError("Model has not been fitted yet.")

        return X.apply(lambda row: self._predict_individual(row), axis=1)
