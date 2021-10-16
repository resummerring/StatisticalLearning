import numpy as np
import pandas as pd
from typing import Callable
from StatisticalLearning.FactorAnalysis.AlphaFactor import AlphaFactor, AlphaVector


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

    def rank_info_ratio(self) -> pd.Series:
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

    def rank_auto_corr(self) -> pd.Series:
        """
        Calculate factor rank auto-correlation(t) = Corr(Rank(alpha(t-1)), Rank(alpha(t)))
        Higher rank auto-correlation -> lower turnover -> low transaction cost + liquidity
        """

        alpha_rank = self._factor.alpha.apply(
            lambda row: row.rank(axis=0, method='first', na_option='keep', ascending=True), axis=1)

        autocorr = []
        for i in range(alpha_rank.shape[0] - 1):
            last_alpha, current_alpha = alpha_rank.iloc[i, :], alpha_rank.iloc[i + 1, :]
            autocorr.append(np.corrcoef(last_alpha, current_alpha))

        return pd.Series(autocorr, index=alpha_rank.index.values.tolist()[1:])

    def turnover(self):
        """
        Calculate portfolio turnover(t) = sum_i [weight(t, i) - weight(t - 1, i)]
        High turn over -> high transaction cost
        """

        weight = self._factor.weight(transformer=self._transformer)

        turn_over = []
        for i in range(weight.shape[0] - 1):
            last_weight, current_weight = weight.iloc[i, :], weight.iloc[i + 1, :]
            turn_over.append(np.sum(np.abs(last_weight - current_weight)))

        return pd.Series(turn_over, index=weight.idnex.values.tolist()[1:])

    def transfer_coef(self, optimal_weights: pd.DataFrame):
        """
        Calculate transfer coefficient between alpha vector and optimized portfolio weight.
        The higher the correlation, the more influence alpha vector takes into the portfolio
        and the less likely risk factor and alpha factors are correlated.

        :param optimal_weights: pd.Series, index are time periods, columns are stock universe,
        values are stock weights in optimized portfolio
        """

        alpha = self._factor.weight(transformer=self._transformer)

        assert optimal_weights.shape == alpha.shape, f"Dimension mismatch: portfolio " \
            f"weight with shape {optimal_weights.shape} while alpha factor with shape {alpha.shape}"

        coef = []
        for i in range(optimal_weights.shape[0]):
            coef.append(np.corrcoef(optimal_weights.iloc[i, :], alpha.iloc[i, :]))

        return pd.Series(coef, index=alpha.index)

    def conditional_alpha(self, factor: AlphaFactor) -> AlphaFactor:

        assert self._factor.alpha.index == factor.alpha.index, "Time period mismatch between alpha factors"

        combined_alpha, index = [], factor.alpha.index.values.tolist()
        for i in range(factor.alpha.shape[0]):
            combined_alpha.append(
                AlphaVector(vector=self._factor.alpha.iloc[i, :].mul(factor.alpha.iloc[i, :]), index=index[i]))

        return AlphaFactor(combined_alpha)
