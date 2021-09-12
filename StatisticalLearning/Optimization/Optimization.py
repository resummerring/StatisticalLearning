import numpy as np
import pandas as pd
from typing import Callable, Union, List
from abc import ABC, abstractmethod


class OptimizationResult:
    """
    A container class to store the optimization results
    """

    def __init__(self,
                 success: bool,
                 optimum: Union[float, np.ndarray, pd.Series],
                 minimum: float,
                 optimum_history: Union[None, List[Union[float, np.ndarray, pd.Series]]],
                 minimum_history: Union[None, List[float]],
                 it: int):
        """
        :param success: bool, whether or not the optimization is completed successfully
        :param optimum: Union[float, np.ndarray, pd.Series], the local / global minimum point
        :param minimum: float, the local / global minimum value
        :param optimum_history: Union[None, List[Union[float, np.ndarray, pd.Series]]],
                                a container that records the entire path of the local / global minimum point
        :param minimum_history: Union[None, List[float]],
                                a container that records the entire path of local / global minimum value
        :param it: int, number of iterations spent
        """

        self._success = success
        self._optimum = optimum
        self._minimum = minimum
        self._optimum_history = optimum_history
        self._minimum_history = minimum_history
        self._it = it

    # ====================
    #  Public
    # ====================

    @property
    def success(self) -> bool:
        return self._success

    @property
    def optimum(self) -> Union[float, np.ndarray, pd.Series]:
        return self._optimum

    @property
    def minimum(self) -> float:
        return self._minimum

    @property
    def optimum_path(self) -> Union[None, List[Union[float, np.ndarray, pd.Series]]]:
        return self._optimum_history

    @property
    def minimum_path(self) -> Union[None, List[float]]:
        return self._minimum_history

    @property
    def it(self) -> int:
        return self._it


class Optimizer(ABC):

    """
    Abstract base class for optimization algorithms (minimization in particular).
    """

    def __init__(self,
                 function: Callable):

        """
        Note, the function callable should always be applied on the entire dataset

        :param function: Callable, a callable which takes to-be-optimized np.ndarray variables and return float
        """

        self._function = function

    # ====================
    #  Public
    # ====================

    @abstractmethod
    def solve(self, **kwargs) -> OptimizationResult:
        """
        An abstract function that must be overriden. Main function for solving the optimization problem.
        """
        raise NotImplementedError
