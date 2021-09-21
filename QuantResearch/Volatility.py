import numpy as np
import pandas as pd


def annual_volatility(series: pd.Series, already_return: bool, frequency: int) -> float:
    """
    Compute annualized volatility

    :param series: pd.Series, price or return data
    :param already_return: bool, True if series is already return data
    :param frequency: int, annual: 1, month: 12, week: 53, day: 252
    """
    series = series if already_return else np.log(series).diff().dropna()
    series = series.reset_index(drop=True)
    volatility = np.var(series, ddof=1).squeeze()
    return volatility * np.sqrt(frequency)


def rolling_annual_volatility(series: pd.Series, already_return: bool, frequency: int, window: int) -> pd.Series:
    """
    Compute annualized volatility with a rolling window

    :param series: pd.Series, price or return data
    :param already_return: bool, True if series is already return data
    :param frequency: int, annual: 1, month: 12, week: 53, day: 252
    :param window: int, length of rolling window
    """

    series = series if already_return else np.log(series).diff().dropna()
    series = series.reset_index(drop=True)
    return series.rolling(window=window, center=False)\
        .apply(annual_volatility, already_return=True, frequency=frequency)


def ewma_annual_volatility(series: pd.Series, already_return: bool, frequency: int, n_periods: int, coef: float):
    """
    Compute annualized volatility with exponentially weighted moving average

    :param series: pd.Series, price or return data
    :param already_return: bool, True if series is already return data
    :param frequency: int, annual: 1, month: 12, week: 53, day: 252
    :param n_periods: int, periods used to compute ewma
    :param coef: float, exponential coefficient
    """

    series = series if already_return else np.log(series).diff().dropna()
    series = series.reset_index(drop=True)
    weights = np.array([coef ** k for k in range(n_periods)])
    variance = np.dot(series[-n_periods:], weights) / np.sum(weights)
    return np.sqrt(frequency * variance)

