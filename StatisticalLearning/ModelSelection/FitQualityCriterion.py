import numpy as np
import pandas as pd
from typing import Union


class FitScore:

    """
    Collection of different metrics to measure the fit quality of a statistical model
    """

    @staticmethod
    def sum_square_error(y_true: Union[pd.Series, np.ndarray],
                         y_pred: Union[pd.Series, np.ndarray, float]) -> float:
        return np.sum(np.square(y_true - y_pred)).squeeze()

    @staticmethod
    def mean_square_error(y_true: Union[pd.Series, np.ndarray],
                          y_pred: Union[pd.Series, np.ndarray, float]) -> float:
        return np.mean(np.square(y_true - y_pred)).squeeze()

    @staticmethod
    def r_square(y_true: Union[pd.Series, np.ndarray],
                 y_pred: Union[pd.Series, np.ndarray, float]) -> float:
        sst = np.sum(np.square(y_true - np.mean(y_true)))
        ssr = np.sum(np.square(y_true - y_pred))
        return 1 - ssr / sst

    @staticmethod
    def adjusted_r_square(y_true: Union[pd.Series, np.ndarray],
                          y_pred: Union[pd.Series, np.ndarray, float],
                          nbr_features: int) -> float:
        nbr_samples = len(y_true)
        return 1 - (1 - FitScore.r_square(y_true, y_pred)) * (nbr_samples - 1) / (nbr_samples - nbr_features - 1)