import numpy as np
import pandas as pd
from typing import Union
from abc import ABC, abstractmethod
from sklearn.linear_model import LinearRegression


class RFModel(ABC):

    """
    An abstract interface class for risk factor model:
        (1) Time-series risk factor model: Fama-French Model
        (2) Cross-sectional risk factor model: BARRA Model

    r(i) = \sum_k b(i, k) * f(k) + s(i) for k = 1,...,K and i = 1,...,N
        where b(i, k) is stock i exposure on risk factor k and s(i) is idiosyncratic risk of stock i

    In matrix format: r = Bf + s where r is N * 1, B is N * K, f is K * 1 and s is N * 1
    Assume: r is de-mean and r is not factor return f is not correlated with residual return s
    Var(r) = Var(Bf + s) -> Var(r) = E[(r-r_mean)(r-r_mean)^T] = E(r r^T) = E[(Bf + s)(Bf + s)^T]
    = B E(f f^T) B^T + E(s s^T) = BFB + S where F, S is respective covariance matrix of factor return and residual

    Given factor return f and factor exposure:
        F = cov(f)
        s = r - Bf -> S = cov(s)
    """

    def __init__(self,
                 stock_return: pd.DataFrame,
                 factor_return: Union[pd.DataFrame, None] = None,
                 factor_exposure: Union[pd.DataFrame, None] = None):
        """
        Assuming N stocks, K risk factors and T periods

        :param stock_return: pd.DataFrame, T * N, return of each stock
        :param factor_return: pd.Series, T * K, return of each factor
        :param factor_exposure: pd.DataFrame, N * K, exposure of a stock on each risk factor
        """

        self._stock_return = stock_return
        self._factor_return = factor_return
        self._factor_exposure = factor_exposure

    @property
    def factor_return(self) -> Union[None, pd.DataFrame]:
        """
        Return T * K factor return matrix
        """
        return self._factor_return

    @property
    def factor_exposure(self) -> Union[None, pd.DataFrame]:
        """
        Return N * K factor exposure matrix
        :return:
        """
        return self._factor_exposure

    @property
    def covariance_matrix(self) -> pd.DataFrame:
        """
        Return N * N stock covariance matrix
        """
        if self._factor_return is None or self._factor_exposure is None:
            raise ValueError("Model hasn't been fitted yet")

        factor_return_cov = self._factor_return.cov(ddof=1)
        residual_return_cov = (self._stock_return - self._factor_return.T @ self._factor_exposure).cov(ddof=1)
        return self._factor_exposure @ factor_return_cov @ self._factor_exposure.T + residual_return_cov

    @abstractmethod
    def fit(self):
        """
        Main function to fit risk factor multiple regression
        """

        raise NotImplementedError


class TimeSeriesRFModel(RFModel):
    """
    Given factor return, estimate factor exposure:
        for each stock i, fit a multiple linear regression:
            r(i, t) = \sum_k b(i, k) * f(k, t) + s(i) for k = 1,...,K
                  -> b(i, k) for k = 1,...,K is the factor exposure of stock i

    'Time series' comes from the fact that for each stock, the factor exposure is computed
    through multiple linear regression with time series return data
    """

    def __init__(self, stock_return: pd.Data, factor_return: pd.DataFrame):
        super().__init__(stock_return=stock_return, factor_return=factor_return, factor_exposure=None)

    def fit(self):
        """
        Main fit function for time series risk factor model:
            loop over stocks to conducting multiple linear regression with time series return data
        """

        factor_exposure = pd.DataFrame(columns=self._factor_return.columns)

        lr = LinearRegression(fit_intercept=True)
        columns = self._stock_return.columns

        for i, stock in enumerate(columns):
            lr = lr.fit(X=self._factor_return, y=self._stock_return[stock])
            factor_exposure.iloc[i, :] = np.array(lr.coef_).reshape(1, -1)

        self._factor_exposure = factor_exposure
