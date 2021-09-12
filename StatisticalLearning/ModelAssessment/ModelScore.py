import numpy as np
import pandas as pd
from typing import Union, Callable


class ModelScore:

    """
    Collection of different metrics to measure the fit quality of a statistical model
    """

    # ====================
    #  Public
    # ====================

    @staticmethod
    def sum_square_error(y_true: Union[pd.Series, np.ndarray],
                         y_pred: Union[pd.Series, np.ndarray, float]) -> float:
        """
        SSR = sum[(y_true[i] - y_pred[i]) ** 2]
        """
        return np.sum(np.square(y_true - y_pred)).squeeze()

    @staticmethod
    def mean_square_error(y_true: Union[pd.Series, np.ndarray],
                          y_pred: Union[pd.Series, np.ndarray, float]) -> float:
        """
        MSE =  mean[(y_true[i] - y_pred[i]) ** 2]
        """
        return np.mean(np.square(y_true - y_pred)).squeeze()

    @staticmethod
    def r_square(y_true: Union[pd.Series, np.ndarray],
                 y_pred: Union[pd.Series, np.ndarray, float]) -> float:
        """
        SST = sum[(y_true[i]) - mean(y_true) ** 2]
        R2 = 1 - SSR / SST
        """
        sst = np.sum(np.square(y_true - np.mean(y_true)))
        ssr = np.sum(np.square(y_true - y_pred))
        return 1 - ssr / sst

    @staticmethod
    def adjusted_r_square(y_true: Union[pd.Series, np.ndarray],
                          y_pred: Union[pd.Series, np.ndarray, float],
                          nbr_features: int) -> float:
        """
        Adj-R2 = 1 - (1 - R2) * (N - 1) / (N - p - 1)
        """
        nbr_samples = len(y_true)
        return 1 - (1 - ModelScore.r_square(y_true, y_pred)) * (nbr_samples - 1) / (nbr_samples - nbr_features - 1)

    @staticmethod
    def aic_linear(y_true: Union[pd.Series, np.ndarray],
                   y_pred: Union[pd.Series, np.ndarray, float],
                   nbr_features: int) -> float:
        """
        AIC = 2k - 2ln(L) where k is number of predictors and L is likelihood
        Under Gaussian distribution assumption, AIC can be simplified as:
        AIC = 2k + nlog(SSR) + C where C = n(log(2\pi) - log(n)) + n
        """
        nbr_samples = len(y_true)
        ssr = ModelScore.sum_square_error(y_true, y_pred)
        return nbr_samples * (np.log(2 * np.pi) + np.log(ssr) - np.log(nbr_samples)) \
            + 2 * (nbr_features + 1) + nbr_samples

    @staticmethod
    def confusion_matrix(y_true: Union[pd.Series, np.ndarray],
                         y_pred: Union[pd.Series, np.ndarray]) -> dict:

        """
        Accuracy = (TP + TN) / (TP + TN + FP + FN)
        Precision = TP / (TP + FP)
        Recall = TP / (TP + FN)
        F1-Score = 2 / (1 / Precision + 1 / Recall)
        """

        result = {}

        TP, TN, FP, FN = 0, 0, 0, 0
        for actual, pred in zip(y_true, y_pred):
            if actual == 1 and pred == 1:
                TP += 1
            elif actual == 0 and pred == 0:
                TN += 1
            elif actual == 1 and pred == 0:
                FN += 1
            elif actual == 0 and pred == 1:
                FP += 1
            else:
                raise ValueError('Only binary classification is accepted.')

        result['accuracy'] = (TP + TN) / (TP + TN + FP + FN)
        result['precision'] = TP / (TP + FP)
        result['recall'] = TP / (TP + FN)
        result['F_score'] = 2 / (1 / result['precision'] + 1 / result['recall'])
        result['confusion_matrix'] = [[TP, FN], [FP, TN]]

        return result

    @staticmethod
    def make_scorer(func: Callable, **kwargs):
        def wrapper(y_true, y_pred):
            return func(y_true, y_pred, **kwargs)
        return wrapper

