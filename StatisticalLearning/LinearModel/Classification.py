from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from typing import Tuple, Union
from StatisticalLearning.Optimization.GradientOptimizer import GradientDescent, StochasticGradientDescent


class LogisticRegression:

    """
    Two-Class Logistic Regression: P(y = 1) = exp(X @ b) / [1 + exp(X @ b)]
    where X @ b = b_0 + b_1 * x_1 + b_2 * x_2 + ... + b_p * x_p
    """

    def __init__(self, fit_intercept: bool = True):

        self._fit_intercept = fit_intercept
        self._coef, self._intercept = None, None

    @property
    def coef(self) -> Union[pd.Series, np.ndarray, None]:
        return self._coef

    @property
    def intercept(self) -> Union[float, None]:
        return self._intercept

    @staticmethod
    def prob(coef: pd.Series, row: pd.Series) -> float:
        """
        Probability of Y = 1 given fitted coefficients and a data point
        P(y = 1) = exp(X @ b) / [1 + exp(X @ b)] = 1 / [1 + exp(-X @ b)]
        """

        return 1. / (1. + np.exp(-coef.to_numpy().dot(row.to_numpy())))

    @staticmethod
    def func(coef: pd.Series, data: Tuple[pd.DataFrame, pd.Series]) -> float:
        """
        Objective function, i.e. log likelihood function

        loss = -sum[y_i * log p_i + (1 - y_i) * log(1 - p_i)] / m
        """

        X_train, y_train = data
        prob = X_train.apply(lambda row: LogisticRegression.prob(coef, row), axis=1)
        return -np.mean(np.log(prob) * y_train + np.log(1 - prob) * (1 - y_train))

    @staticmethod
    def grad_gd(coef: pd.Series, data: Tuple[pd.DataFrame, pd.Series]) -> pd.Series:
        """
        Gradient for gradient descent
        """

        X_train, y_train = data
        prob = X_train.apply(lambda row: LogisticRegression.prob(coef, row), axis=1)
        return X_train.apply(lambda col: col @ (prob - y_train), axis=0) / X_train.shape[0]

    def fit(self, X: pd.DataFrame, y: pd.Series, solver: str = 'GradientDescent', **kwargs) -> LogisticRegression:

        """
        Model fitted by maximum likelihood method

        :param X: pd.DataFrame, n * p data matrix with n = #samples and p = #features
        :param y: pd.Series, n * 1 label vector
        :param solver: str, {'GradientDescent', 'StochasticGradientDescent'}
        """

        if self._fit_intercept:
            X = sm.add_constant(X)

        X, y = X.reset_index(drop=True), y.reset_index(drop=True)
        X.columns = list(range(X.shape[1]))

        coefs = pd.Series(np.zeros(X.shape[1]))

        if solver == 'GradientDescent':

            optimizer = GradientDescent(self.func, self.grad_gd)
            x0 = kwargs['x0'] if 'x0' in kwargs else coefs
            learning_rate = kwargs['learning_rate'] if 'learning_rate' in kwargs else 0.1
            momentum = kwargs['momentum'] if 'momentum' in kwargs else 0.8
            func_tol = kwargs['func_tol'] if 'func_tol' in kwargs else 1e-8
            param_tol = kwargs['param_tol'] if 'param_tol' in kwargs else 1e-8
            max_iter = kwargs['max_iter'] if 'max_iter' in kwargs else 1000
            result = optimizer.solve(x0=x0,
                                     learning_rate=learning_rate,
                                     momentum=momentum,
                                     func_tol=func_tol,
                                     param_tol=param_tol,
                                     max_iter=max_iter,
                                     data=(X, y))

            if result.success:
                coefs = result.optimum
            else:
                raise ValueError("Gradient descent failed.")
        else:
            raise NotImplementedError

        if self._fit_intercept:
            self._coef, self._intercept = coefs.iloc[1:].reset_index(drop=True), coefs.iloc[0]
        else:
            self._coef = coefs

        return self

