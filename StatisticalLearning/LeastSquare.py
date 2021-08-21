import numpy as np
import pandas as pd
from typing import Union, Callable


class LeastSquare:

    def __init__(self,
                 X: Union[pd.DataFrame, np.ndarray],
                 y: Union[pd.Series, np.ndarray],
                 F: Callable):
        """
        :param X: Union[pd.DataFrame, np.ndarray], independent variables, each column is a variable
        :param y: Union[pd.Series, np.ndarray], in
        :param F:
        """
        self._X = X
        self._y = y
        self._F = F

