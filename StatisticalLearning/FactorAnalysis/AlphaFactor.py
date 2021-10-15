from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Union, List, Any, Callable


class AlphaVector:

    """
    An alpha vector is a vector of alpha values at a single time period T(i) over stock universe I(1) - I(N)

    Alpha factor: drivers of mean return
    Risk factor: drivers of volatility

    Notice, static stock universe should be avoided since it introduces look-ahead bias: survivorship bias.
    Instead, a dynamic stock universe can be used such as the component stocks of an index, such as S&P 500
    """

    def __init__(self, vector: pd.Series, index: Any):
        """
        :param vector: pd.Series, index is stock tick and value is alpha value
        :param index: Any, an indicator for the time period of alpha vector with <,>,= overriden
        """

        self._vector = vector
        self._index = index

    # ====================
    #  Public
    # ====================

    @property
    def vector(self) -> pd.Series:
        """
        Return alpha value vector
        """
        return self._vector

    @property
    def index(self) -> Any:
        """
        Return vector time period
        """
        return self._index

    def market_neutral(self) -> AlphaVector:
        """
        x(i) = x(i) - mean_i x(i) -> mean_i x(i) = 0

        Make portfolio dollar neutral (dollars long = dollars short) and therefore approximately market-neutral,
        excluding the influence from overall market when testing the predictive power of the factor.
        """

        self._vector = self._vector - self._vector.mean(skipna=True)
        return self

    def leverage_neutral(self) -> AlphaVector:
        """
        x(i) = x(i) / sum_i |x(i)| -> sum_i |x(i)| = 1

        Make portfolio leverage-neutral (leverage ratio = sum of positions / notional = 1) and
        Make different factors more comparable to each other by forcing same amount of dollar.
        """

        self._vector = self._vector / self._vector.abs().sum(skipna=True)
        return self

    def re_scale(self) -> AlphaVector:
        """
        x(i) = x(i) / std_i x(i) -> std_i x(i) = 1

        Make portfolio unit variance (standard deviation = 1) and make factors with different scale
        more comparable to each other.
        """

        self._vector = self._vector / np.std(self._vector, ddof=1)
        return self

    def sector_neutral(self, sector_map: dict) -> AlphaVector:
        """
        :param sector_map: dict, a mapping from stock tick to corresponding sector

        x(i) = x(i) - mean_j x(j) where j is sector universe
        Make portfolio sector-neutral and less sensitive to the variation of sector movement
        """

        factor = pd.DataFrame(columns=['Return', 'Sector'])
        factor['Return'] = self._vector
        factor['Sector'] = factor['Return'].apply(lambda tick: sector_map[tick])

        sector_mean = factor.groupby('Sector')['Return'].mean()
        factor['Return'] = factor.apply(lambda row: row['Return'] - sector_mean[row['Sector']])

        self._vector = factor['Return']

        return self

    def winsorize(self,
                  cutoff: float = 0.9,
                  hard_lower: Union[None, float] = None,
                  hard_upper: Union[None, float] = None) -> AlphaVector:
        """
        Remove outlier outsize of [cutoff, 1 - cutoff] of the factor values

        :param cutoff: float, soft threshold to determine outliers based on quantile
        :param hard_lower: Union[None, float], hard lower threshold to apply if given
        :param hard_upper: Union[None, float], hard upper threshold to apply if given
        """

        q_lower = self._vector.quantile(q=cutoff, interpolation='linear')
        q_upper = self._vector.quantile(q=1 - cutoff, interpolation='linear')

        q_lower = q_lower if hard_lower is None else hard_lower
        q_upper = q_upper if hard_upper is None else hard_upper

        self._vector[self._vector < q_lower] = q_lower
        self._vector[self._vector > q_upper] = q_upper

        return self

    def rank(self) -> AlphaVector:
        """
        Transform alpha signal to be less sensitive to noise and outliers and avoid excessive trades
        Cannot make alpha signal comparable if generated from different stock universe
        """

        self._vector = self._vector.rank(axis=0, method='first', na_option='keep', ascending=True)
        return self

    def z_score(self) -> AlphaVector:
        """
        Standardize to zero mean and unit variance
        Useful when alpha signals are generated from different stock universe
        """

        return self.market_neutral().re_scale()


class AlphaFactor:

    def __init__(self, vectors: List[AlphaVector]):
        """
        :param vectors: List[AlphaVector], a collection of alpha factors from different time periods

        Assume all alpha vector comes from the same stock universe
        """

        vectors.sort(key=lambda vector: vector.index, reverse=False)

        self._vectors = vectors
        self._alpha = self.weight(lambda alpha: alpha)

    # ====================
    #  Public
    # ====================

    @property
    def alpha(self) -> pd.DataFrame:
        """
        Return T * N alpha factor matrix
        """

        return self._alpha

    def weight(self, transformer: Callable) -> pd.DataFrame:
        """
        Transform alpha vector into portfolio weights
        """

        columns = list(self._vectors[0].index)
        index = [vector.index for vector in self._vectors]

        weights = pd.DataFrame(columns=columns, index=index)
        for i, vector in enumerate(self._vectors):
            weights.iloc[i, :] = transformer(vector.vector)

        return weights / weights.sum(axis=1)

    def smooth(self, columns: Union[List[str], None], weight: List[float]) -> AlphaFactor:
        """
        Smooth alpha factors are by using weighted moving average

        :param columns: Union[List[str], None], subset of columns to apply moving average
        :param weight: List[float], weight to use for the rolling window
        """

        if columns is None:
            self._alpha = self._alpha.apply(
                lambda col: col.rolling.mean(window=len(weight))
                .apply(lambda window: np.dot(window, weight) / sum(weight)), axis=0)
        else:
            self._alpha[columns] = self._alpha[columns].apply(
                lambda col: col.rolling.mean(window=len(weight))
                .apply(lambda window: np.dot(window, weight) / sum(weight)), axis=0)

        return self
