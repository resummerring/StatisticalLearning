from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
import scipy.stats as stats
from typing import Tuple, Union
from StatisticalLearning.Optimization.GradientOptimizer import GradientDescent, StochasticGradientDescent


class LinearRegression:

    """
    Multiple Linear Regression: y = b_0 + b_1 * x_1 + b_2 * x_2 + ... + b_p *x_p
    """

    def __init__(self, fit_intercept: bool = True):
        self._fit_intercept = fit_intercept
        self._degrees_freedom = None
        self._coef, self._intercept = None, None
        self._coef_t_stat, self._intercept_t_stat = None, None
        self._coef_p_value, self._intercept_p_value = None, None

    @property
    def coef(self) -> Union[pd.Series, np.ndarray, None]:
        return self._coef

    @property
    def intercept(self) -> Union[float, None]:
        return self._intercept

    @property
    def coef_t_stat(self) -> Union[pd.Series, np.ndarary, None]:
        return self._coef_t_stat

    @property
    def intercept_t_stat(self) -> Union[float, None]:
        return self._intercept_t_stat

    @property
    def degrees_freedom(self) -> Union[int, None]:
        return self._degrees_freedom

    @property
    def coef_p_value(self) -> Union[pd.Series, np.ndarray, None]:
        return self._coef_p_value

    @property
    def intercept_p_value(self) -> Union[float, None]:
        return self._intercept_p_value

    def _clean_up(self):

        """
        The clean function will be applied whenever the fit function is utilized.
        All private variables will be reset to None.
        """

        self._coef, self._intercept = None, None
        self._degrees_freedom = None
        self._coef_t_stat, self._intercept_t_stat = None, None
        self._coef_p_value, self._intercept_p_value = None, None

    @staticmethod
    def func(coef: pd.Series, data: Tuple[pd.DataFrame, pd.Series]) -> float:
        """
        Objective function for gradient descent and stochastic gradient descent
        """

        X_train, y_train = data
        X_train = np.array(X_train)
        return ((y_train - X_train @ coef).T @ (y_train - X_train @ coef)) / X_train.shape[0]

    @staticmethod
    def grad_gd(coef: pd.Series, data: Tuple[pd.DataFrame, pd.Series]) -> pd.Series:
        """
        Gradient for gradient descent
        """

        X_train, y_train = data
        X_train = np.array(X_train)
        return (X_train.T @ (X_train @ coef - y_train)) / X_train.shape[0]

    @staticmethod
    def grad_sgd(coef: pd.Series, data: Tuple[pd.Series, float]) -> pd.Series:
        """
        Gradient for stochastic gradient descent
        """

        X_train, y_train = data
        X_train = np.array(X_train).reshape(1, -1)
        return pd.Series((X_train.T @ (X_train @ coef - y_train)))

    def fit(self, X: pd.DataFrame, y: pd.Series, solver: str = 'GradientDescent', **kwargs) -> LinearRegression:

        """
        :param X: pd.DataFrame, n * p data matrix with n = #samples and p = #features
        :param y: pd.Series, n * 1 label vector
        :param solver: str, {'GradientDescent', 'StochasticGradientDescent', 'NormalEquation', SVD'}
        """

        self._clean_up()

        self._degrees_freedom = X.shape[0] - X.shape[1]

        X, y = X.reset_index(drop=True), y.reset_index(drop=True)
        X.columns = list(range(X.shape[1]))

        if self._fit_intercept:
            X = sm.add_constant(X)

        coefs = pd.Series(np.zeros(X.shape[1]))

        if solver == 'NormalEquation':

            try:
                coefs = pd.Series(np.linalg.inv(X.T @ X) @ X.T @ y)
            except np.linalg.LinAlgError:
                raise ValueError("Normal equation failed: singular data matrix.")

        elif solver == 'GradientDescent':

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

        elif solver == 'StochasticGradientDescent':

            optimizer = StochasticGradientDescent(self.func, self.grad_sgd)
            x0 = kwargs['x0'] if 'x0' in kwargs else coefs
            learning_rate = kwargs['learning_rate'] if 'learning_rate' in kwargs else 0.1
            func_tol = kwargs['func_tol'] if 'func_tol' in kwargs else 1e-8
            max_epoc = kwargs['max_epoc'] if 'max_epoc' in kwargs else 100
            result = optimizer.solve(x0=x0,
                                     learning_rate=learning_rate,
                                     func_tol=func_tol,
                                     max_epoc=max_epoc,
                                     data=(X, y))

            if result.success:
                coefs = result.optimum
            else:
                raise ValueError('Stochastic gradient descent failed.')

        elif solver == 'SVD':
            try:
                U, S, Vt = np.linalg.svd(X, full_matrices=False)
                coefs = pd.Series(Vt.T @ np.linalg.inv(np.diag(S)) @ U.T @ y)
            except np.linalg.LinAlgError:
                raise ValueError("SVD failed: unsuccessful SVD decomposition.")

        else:
            raise NotImplementedError("No implementation for the input solver.")

        residual_var = np.sum(np.square(np.array(X) @ coefs - y)) / (X.shape[0] - X.shape[1])
        try:
            coefs_std = pd.Series(np.sqrt(np.diag(np.linalg.inv(X.T @ X)) * residual_var))
        except np.linalg.LinAlgError:
            raise ValueError("t-test failed: singular data matrix.")

        t_stats = coefs / coefs_std
        p_values = pd.Series([2 * (1 - stats.t.cdf(abs(t_stat), self.degrees_freedom)) for t_stat in t_stats])

        if self._fit_intercept:
            self._coef, self._intercept = coefs.iloc[1:].reset_index(drop=True), coefs.iloc[0]
            self._coef_t_stat, self._intercept_t_stat = t_stats.iloc[1:].reset_index(drop=True), t_stats.iloc[0]
            self._coef_p_value, self._intercept_p_value = p_values.iloc[1:].reset_index(drop=True), p_values.iloc[0]
        else:
            self._coef = coefs
            self._coef_t_stat = t_stats
            self._coef_p_value = p_values

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
