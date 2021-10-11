import pandas as pd


class Factor:

    """
    A factor is a numeric value, one for each stock, potentially predictive of performance

    Alpha factor: drivers of mean return
    Risk factor: drivers of volatility
    """

    def __init__(self, factor: pd.Series):
        """
        :param factor: pd.Series, raw factor series for portfolio universe
        """

        self._factor = self._standardize(factor)

    @staticmethod
    def _standardize(factor: pd.Series) -> pd.Series:
        """
        De-mean and re-scale raw factor series:
            (1) De-mean: x(i) = x(i) - mean_i x(i) -> mean_i x(i) = 0
            (2) Re-scale: x(i) = x(i) / sum_i |x(i)| -> sum_i |x(i)| = 1

        De-mean: make portfolio dollar neutral (dollars long = dollars short) and therefore approximately
        market-neutral, excluding the influence from overall market when testing the predictive power of the factor.

        Re-scale: make portfolio leverage-neutral (leverage ratio = sum of positions / notional = 1) and
        make different factors more comparable to each other by forcing same amount of dollar.

        :param factor: pd.Series, raw factor series for portfolio universe
        """

        # Step 1: de-mean
        demean = factor - factor.mean(skipna=True)

        # Step 2: re-scale
        rescale = demean / demean.abs().sum(skipna=True)

        return rescale

    @property
    def factor(self) -> pd.Series:
        """
        Return factor time series
        """
        return self._factor
