from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Union


class Factor:

    """
    A factor is a numeric value, one for each stock, potentially predictive of performance

    Alpha factor: drivers of mean return
    Risk factor: drivers of volatility
    """

    def __init__(self, factor: pd.Series):
        """
        :param factor: pd.Series, index is stock tick and value is alpha value
        """

        self._factor = factor

    def market_neutral(self) -> Factor:
        """
        x(i) = x(i) - mean_i x(i) -> mean_i x(i) = 0

        Make portfolio dollar neutral (dollars long = dollars short) and therefore approximately market-neutral,
        excluding the influence from overall market when testing the predictive power of the factor.
        """

        self._factor = self._factor - self._factor.mean(skipna=True)
        return self

    def leverage_neutral(self) -> Factor:
        """
        x(i) = x(i) / sum_i |x(i)| -> sum_i |x(i)| = 1

        Make portfolio leverage-neutral (leverage ratio = sum of positions / notional = 1) and
        Make different factors more comparable to each other by forcing same amount of dollar.
        """

        self._factor = self._factor / self._factor.abs().sum(skipna=True)
        return self

    def re_scale(self) -> Factor:
        """
        x(i) = x(i) / std_i x(i) -> std_i x(i) = 1

        Make portfolio unit variance (standard deviation = 1) and make factors with different scale
        more comparable to each other.
        """

        self._factor = self._factor / np.std(self._factor, ddof=1)
        return self

    def sector_neutral(self, sector_map: dict) -> Factor:
        """
        :param sector_map: dict, a mapping from stock tick to corresponding sector

        x(i) = x(i) - mean_j x(j) where j is sector universe
        Make portfolio sector-neutral and less sensitive to the variation of sector movement
        """

        factor = pd.DataFrame(columns=['Return', 'Sector'])
        factor['Return'] = self._factor
        factor['Sector'] = factor['Return'].apply(lambda tick: sector_map[tick])

        sector_mean = factor.groupby('Sector')['Return'].mean()
        factor['Return'] = factor.apply(lambda row: row['Return'] - sector_mean[row['Sector']])

        self._factor = factor['Return']

        return self

    def winsorize(self,
                  cutoff: float = 0.9,
                  hard_lower: Union[None, float] = None,
                  hard_upper: Union[None, float] = None) -> Factor:
        """
        Remove outlier outsize of [cutoff, 1 - cutoff] of the factor values

        :param cutoff: float, soft threshold to determine outliers based on quantile
        :param hard_lower: Union[None, float], hard lower threshold to apply if given
        :param hard_upper: Union[None, float], hard upper threshold to apply if given
        """

        q_lower = self._factor.quantile(q=cutoff, interpolation='linear')
        q_upper = self._factor.quantile(q=1-cutoff, interpolation='linear')

        q_lower = q_lower if hard_lower is None else hard_lower
        q_upper = q_upper if hard_upper is None else hard_upper

        self._factor[self._factor < q_lower] = q_lower
        self._factor[self._factor > q_upper] = q_upper

        return self

    def rank(self) -> Factor:
        """
        Transform alpha signal to be less sensitive to noise and outliers and avoid excessive trades
        """

        self._factor = self._factor.rank(axis=0, method='first', na_option='keep', ascending=True)
        return self

    @property
    def factor(self) -> pd.Series:
        """
        Return factor time series
        """
        return self._factor
