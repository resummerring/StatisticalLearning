from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from typing import Tuple, Union
from StatisticalLearning.Optimization.GradientOptimizer import GradientDescent, StochasticGradientDescent


class LinearRegression:

    def __init__(self, fit_intercept: bool = True):
        self._fit_intercept = fit_intercept
        self._coef, self._intercept = None, None

    @property
    def coef(self) -> Union[pd.Series, np.ndarray, None]:
        return self._coef

    @property
    def intercept(self) -> Union[float, None]:
        return self._intercept

    def fit(self, X: pd.DataFrame, y: pd.Series, solver: str = 'GradientDescent', **kwargs) -> LinearRegression:

        """
        :param X: pd.DataFrame, n * p data matrix with n = #samples and p = #features
        :param y: pd.Series, n * 1 label vector
        :param solver: str, {'GradientDescent', 'StochasticGradientDescent', 'NormalEquation', 'Adam', 'SVD'}
        """

        if self._fit_intercept:
            X = sm.add_constant(X)

        coefs = pd.Series(np.zeros(X.shape[1]))

        if solver == 'NormalEquation':

            try:
                coefs = pd.Series(np.linalg.inv(X.T @ X) @ X.T @ y)
            except np.linalg.LinAlgError:
                raise ValueError("Normal equation failed: singular data matrix.")

        elif solver == 'GradientDescent':

            def func(coef: pd.Series, data: Tuple[pd.DataFrame, pd.Series]) -> float:
                X_train, y_train = data
                X_train = np.array(X_train)
                return ((y_train - X_train @ coef).T @ (y_train - X_train @ coef)) / X_train.shape[0]

            def grad(coef: pd.Series, data: Tuple[pd.DataFrame, pd.Series]) -> pd.Series:
                X_train, y_train = data
                X_train = np.array(X_train)
                return (X_train.T @ (X_train @ coef - y_train)) / X_train.shape[0]

            optimizer = GradientDescent(func, grad)
            x0 = kwargs['x0'] if 'x0' in kwargs else coefs
            learning_rate = kwargs['learning_rate'] if 'learning_rate' in kwargs else 0.1
            func_tol = kwargs['func_tol'] if 'func_tol' in kwargs else 1e-8
            param_tol = kwargs['param_tol'] if 'param_tol' in kwargs else 1e-8
            max_iter = kwargs['max_iter'] if 'max_iter' in kwargs else 1000
            result = optimizer.solve(x0=x0,
                                     learning_rate=learning_rate,
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
            self._coef, self._intercept = coefs.iloc[1:], coefs.iloc[0]
        else:
            self._coef = coefs

        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:

        """
        :param X: pd.DataFrame, n * p data matrix with n = #samples and p = #features
        """

        if self._coef is not None:
            if self._fit_intercept:
                return np.array(X) @ self._coef + self._intercept
            else:
                return np.array(X) @ self._coef
        else:
            raise ValueError('Model has not been fitted yet.')
