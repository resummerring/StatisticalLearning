import numpy as np
import pandas as pd
from copy import deepcopy
from typing import Callable, Union, Tuple, List
from StatisticalLearning.Optimization.Optimization import OptimizationResult, Optimizer


class GradientDescent(Optimizer):

    """
    Gradient Descent Algorithm:
        To minimize F(x):
            compute delta = - a * F'(x) + m * delta with momentum
            update x = x + delta until x converges with learning rate a and gradient F'

        In machine learning, we will compute the gradient descent based on the entire dataset

    Pros:
    (1) Less computational cost
    (2) Capable of handling large number of features

    Cons:
    (1) Possible to be stuck in local minima
    (2) Need proper learning rate to converge
    (3) Data needs to be preprocessed to same scale to speed convergence
    """

    def __init__(self,
                 function: Callable,
                 gradient: Callable):

        """
        Note, the gradient function should be applied to the entire dataset if given
        """

        super().__init__(function)
        self._gradient = gradient

    def solve(self,
              x0: Union[float, np.ndarray, pd.Series],
              learning_rate: float = 0.1,
              momentum: float = 0.8,
              data: Union[Tuple[pd.DataFrame, pd.Series], None] = None,
              param_tol: float = 1e-8,
              func_tol: float = 1e-8,
              max_iter: int = 1000) -> OptimizationResult:

        """
        Note: Data type of initial guess should be consistent with the return type of gradient function

        :param x0: Union[float, np.ndarray, pd.Series], initial guess for optimum
        :param learning_rate: float, learning rate for convergence
        :param momentum: float, momentum for updating gradient
        :param data: Union[Tuple[pd.DataFrame, pd.Series], None], data matrix and label vector
        :param param_tol: float: convergence tolerance for ||param_current - param_next||
        :param func_tol: float: convergence tolerance for |func_current - func_next|
        :param max_iter: int, max allowed iterations
        """

        param, func = deepcopy(x0), self._function(x0, data)

        # Container to store parameters and functions at each step
        param_container, func_container = [param], [func]
        try:
            delta = pd.Series(np.zeros(len(param)))
        except TypeError:
            delta = 0

        it, func_diff, param_diff = 0, 1e8, 1e8

        while it < max_iter and (func_diff > func_tol or param_diff > param_tol):

            delta = learning_rate * self._gradient(param, data) + momentum * delta
            param = param - delta
            param_container.append(param)
            func_container.append(self._function(param, data))

            func_diff = abs(func_container[-1] - func_container[-2])

            try:
                param_diff = np.linalg.norm(param_container[-1] - param_container[-2], ord=2)
            except ValueError:
                param_diff = abs(param_container[-1] - param_container[-2])

            it += 1

            print(f"Iteration {it} with objective error = {func_container[-1]}")

        success = True if it < max_iter else False

        return OptimizationResult(success, param, func_container[-1], param_container, func_container, it)


class StochasticGradientDescent(Optimizer):

    """
    Stochastic Gradient Descent Algorithm:
        To minimize F(x):
            for epoc = 1, 2, ..., max_epoc:
                (1) random shuffle data matrix and label X, y
                (2) for sample i = 1, 2, ..., n
                        update x = x - a * F'(x, i) where F' is the gradient function for a single sample point
                (3) end if convergence condition is reached

    Pros:
    (1) Much faster convergence
    (2) Less computational cost

    Cons:
    (1) Convergence path is not stable / monotonic
    """

    def __init__(self,
                 function: Callable,
                 gradient: Callable):

        """
        Note, the gradient function should be applied to one sample point a time
        """

        super().__init__(function)
        self._gradient = gradient

    def solve(self,
              x0: Union[float, np.ndarray, pd.Series],
              data: Tuple[pd.DataFrame, pd.Series],
              learning_rate: float = 0.1,
              func_tol: float = 1e-8,
              max_epoc: int = 100,
              seeds: Union[List[int], None] = None) -> OptimizationResult:

        """
        Note: Data type of initial guess should be consistent with the return type of gradient function

        :param x0: Union[float, np.ndarray, pd.Series], initial guess for optimum
        :param data: Tuple[pd.DataFrame, pd.Series], data matrix and label vector
        :param learning_rate: float, learning rate for convergence
        :param func_tol: float: convergence tolerance for |func_current - func_next|
        :param max_epoc: int, max allowed iteration epocs
        :param seeds: Union[List[int], None], random seeds used to shuffle datasets each epoc
        """

        (X, y), seeds = data, seeds if seeds is not None else list(range(max_epoc))

        param, func = deepcopy(x0), self._function(x0, data)

        # Container to store parameters and functions at each step
        param_container, func_container = [param], [func]

        epoc, func_diff = 0, 1e8

        while epoc < max_epoc and func_diff > func_tol:

            np.random.seed(seeds[epoc])
            shuffled_index = np.random.permutation(X.shape[0])
            X_epoc, y_epoc = X.iloc[shuffled_index, :], y.iloc[shuffled_index]

            for i in range(X_epoc.shape[0]):
                x_i, y_i = X_epoc.iloc[i, :], y_epoc.iloc[i]
                gard = self._gradient(param, (x_i, y_i))
                param = param - learning_rate * self._gradient(param, (x_i, y_i))

            param_container.append(param)
            func_container.append(self._function(param, data))
            func_diff = abs(func_container[-1] - func_container[-2])

            print(f"Epoc {epoc} with objective error = {func_container[-1]}")

            epoc += 1

        success = True if epoc < max_epoc else False

        return OptimizationResult(success, param, func_container[-1], param_container, func_container, epoc)
