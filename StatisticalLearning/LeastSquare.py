import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Union
from scipy.optimize import minimize


class LeastSquare(ABC):

    """
    Base least square class
    """

    def __init__(self,
                 X: Union[pd.DataFrame, np.ndarray],
                 y: Union[pd.Series, np.ndarray]):
        """
        :param X: pd.DataFrame, independent variables, m * n where m is #samples and n is #features
        :param y: pd.Series, dependent variable, m * 1 where m is #sample
        """

        self._X = X
        self._y = y

    @abstractmethod
    def sum_square_residual(self, **kwargs) -> Union[float, None]:
        """
        Compute sum of squared residuals
        """
        return None

    @abstractmethod
    def optimize(self, **kwargs) -> Union[np.ndarray, None]:
        """
        Minimize sum of squared residuals
        """
        return None


class LinearLeastSquare(LeastSquare):

    """
    Least square class for linear models where fitted function in the form:
    f(x) = a0 + a1 * x1 + ... + an * xn
    """

    def __init__(self,
                 X: Union[pd.DataFrame, np.ndarray],
                 y: Union[pd.Series, np.ndarray]):

        super().__init__(X, y)

    def sum_square_residual(self, coef: Union[pd.Series, np.ndarray]) -> float:
        residual = self._X.apply(lambda sample: np.dot(sample, coef), axis=1)
        return np.sum(np.square(residual - self._y)).squeeze()

    def optimize(self, init_guess) -> Union[np.ndarray, None]:
        result = minimize(fun=self.sum_square_residual, x0=init_guess, method='BFGS')
        if result.success:
            return result.x
        return None

