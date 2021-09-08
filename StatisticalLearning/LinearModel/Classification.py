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

    Limitation:
    (1) Can only separate classed with linear boundaries which are unrealistic in practice
    (2) Prone to overfit with high-dimensional data where L1 / L2 regularization is needed
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

    @staticmethod
    def grad_sgd(coef: pd.Series, data: Tuple[pd.Series, float]) -> pd.Series:
        """
        Gradient for stochastic gradient descent
        """

        X_train, y_train = data
        prob = LogisticRegression.prob(coef, X_train)
        return pd.Series((prob - y_train) * X_train)

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

        elif solver == 'StochasticGradientDescent':

            optimizer = StochasticGradientDescent(self.func, self.grad_sgd)
            x0 = kwargs['x0'] if 'x0' in kwargs else coefs
            learning_rate = kwargs['learning_rate'] if 'learning_rate' in kwargs else 0.1
            func_tol = kwargs['func_tol'] if 'func_tol' in kwargs else 1e-8
            max_epoc = kwargs['max_epoc'] if 'max_epoc' in kwargs else 1000
            result = optimizer.solve(x0=x0,
                                     learning_rate=learning_rate,
                                     func_tol=func_tol,
                                     max_epoc=max_epoc,
                                     data=(X, y))

            if result.success:
                coefs = result.optimum
            else:
                raise ValueError("Stochastic gradient descent failed.")

        else:
            raise NotImplementedError("No implementation for the input solver.")

        if self._fit_intercept:
            self._coef, self._intercept = coefs.iloc[1:].reset_index(drop=True), coefs.iloc[0]
        else:
            self._coef = coefs

        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:

        """
        Output the predicted probability P(y = 1 | X)

        :param X: pd.DataFrame, n * p data matrix with n = #samples and p = #features
        """

        if self._coef is not None:
            X = sm.add_constant(X) if self._fit_intercept else X
            coef = pd.Series([self._intercept] + list(self._coef)) if self._fit_intercept else self._coef
            prob = X.apply(lambda row: self.prob(coef, row), axis=1)
            return prob

        else:
            raise ValueError('Model has not been fitted yet.')


class LinearDiscriminantAnalysis:

    """
    Linear Discriminant Analysis:
        (1) Estimate class population distribution: pi_k = N_k / N where N_k = #{i: y(i) = k}
        (2) Estimate class multivariate mean vector: mu_k = mean(x(i)) for y(i) = k
        (3) Estimate multivariate covariance matrix: cov = sum(cov_k) / N-K where cov_k is in-class covariance matrix

        Max Bayes posterior probability to find the predicted class k

    Assumption:
        Prior joint distribution of predictors is multivariate normal distribution and all classes share
        the same covariance matrix.
    """

    def __init__(self):
        self._pi_k, self._mu_k, self._cov, self._cov_inv = None, None, None, None

    def prob(self, x: pd.Series, mu: pd.Series, pi: float) -> float:

        """
        Find the probability that the test point belongs to a particular class

        :param x: pd.Series, test point
        :param mu: pd.Series, mean vector for class k
        :param pi: float, population prob for class k
        """

        return x.T @ self._cov_inv @ mu - 0.5 * mu.T @ self._cov_inv @ mu + np.log(pi)

    def _find_best_class(self, x: pd.Series) -> int:

        """
        Find the best class for a given test point

        :param x: pd.Series, test point
        """

        optimal_score, optimal_class = float('-inf'), None
        for k in self._pi_k.keys():
            prob_k = self.prob(x, self._mu_k[k], self._pi_k[k])
            if prob_k >= optimal_score:
                optimal_score, optimal_class = prob_k, k

        return optimal_class

    def fit(self, X: pd.DataFrame, y: pd.Series) -> LinearDiscriminantAnalysis:

        """
        :param X: pd.DataFrame, n * p data matrix with n = #samples and p = #features
        :param y: pd.Series, n * 1 label vector
        """

        # Make sure labels are encoded as integers
        try:
            y = y.apply(lambda s: int(s))
        except ValueError:
            raise ValueError("Labels need to be encoded as integers first.")

        # Population class distribution
        pi_k = (y.value_counts() / y.shape[0]).to_dict()

        # Gaussian mean vector estimation
        mu_k = {k: X[y == k].mean(axis=0) for k in pi_k.keys()}

        # Gaussian covariance matrix estimation
        cov_k = [np.cov(X[y == k], rowvar=False, bias=True) * X[y == k].shape[0] for k in pi_k.keys()]
        cov = pd.DataFrame(np.sum(cov_k, axis=0) / (X.shape[0] - len(cov_k)))

        self._pi_k, self._mu_k, self._cov = pi_k, mu_k, cov

        try:
            self._cov_inv = np.linalg.inv(self._cov)
        except np.linalg.LinAlgError:
            raise ValueError("Singular matrix cannot be inverted.")

        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:

        """
        :param X: pd.DataFrame, test data set
        """

        return X.apply(lambda row: self._find_best_class(row), axis=1)















