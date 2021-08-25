import numpy as np
import pandas as pd
from typing import Union, List


class LinearRegressionStepwiseForward:

    """
    Stepwise forward selection method using linear regression as estimator
    """

    def __init__(self,
                 X: pd.DataFrame,
                 y: pd.Series):
        super().__init__(X, y)

    def find_best_model_with_fixed_size(self, k: int, last_best_index: List[int]) -> Union[List[int], None]:
        return

    def find_best_model(self) -> Union[List[int], None]:
        return

