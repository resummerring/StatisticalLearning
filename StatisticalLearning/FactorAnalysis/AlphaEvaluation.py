import numpy as np
import pandas as pd
from typing import Callable
from StatisticalLearning.FactorAnalysis.AlphaFactor import AlphaFactor


class UniAlphaEval:
    """
    A class to evaluate univariate alpha factor performance including:
    (1) Forward factor return
    """

    def __init__(self,
                 factor: AlphaFactor,
                 stock_return: pd.DataFrame,
                 transformer: Callable = lambda alpha: alpha):
        """
        :param factor: AlphaFactor, alpha factor with stock universe I(1) - I(N) and time periods T(0 - T(M-1)
        :param stock_return: pd.DataFrame, stock universe I(1) - I(N) return on sorted time periods T(1) - T(M)
        :param transformer: Callable, a transformer applied on alpha factor to construct portfolio weights
        """

        self._factor = factor
        self._transformer = transformer
        self._stock_return = stock_return

        assert factor.alpha.shape == self._stock_return.shape, f"Dimension mismatch: weight with shape " \
                                f"{factor.alpha.shape} while stock return with shape {self._stock_return.shape}"

    def forward_factor_return(self) -> pd.Series:
        """
        Calculate forward portfolio: p(t) = sum_i w(i, t) * r(i, t)
        """

        weight = self._factor.weight(transformer=self._transformer)
        forward_return = weight.mul(self._stock_return).sum(axis=1).values
        return pd.Series(forward_return, index=self._stock_return.index)

    def sharpe_ratio(self, frequency: int = 1) -> float:
        """
        Calculate portfolio sharpe ratio = mean_t p(t) / std_t p(t)

        :param frequency: int, frequency of time period, daily = 1, monthly = 12, yearly = 252
        """

        forward_return = self.forward_factor_return()
        sharpe_ratio = forward_return.mean() / forward_return.std()
        return sharpe_ratio * np.sqrt(252) / np.sqrt(frequency)

    def rank_IC(self) -> pd.Series:
        """
        Calculate portfolio rank information coefficient = Corr(Rank(alpha(t-1)), Rank(p(t)))
        Rank IC is more robust than pearson correlation since we care more about the right
        direction instead of the exact return.
        """

        alpha_rank = self._factor.alpha.apply(
            lambda row: row.rank(axis=0, method='first', na_option='keep', ascending=True), axis=1)

        return_rank = self._stock_return.apply(
            lambda row: row.rank(axis=0, method='first', na_option='keep', ascending=True), axis=1)

        rank_corr = [np.corrcoef(alpha_rank.iloc[i, :], return_rank.iloc[i, :]) for i in range(alpha_rank.shape[0])]
        return pd.Series(rank_corr, index=self._stock_return.index)

