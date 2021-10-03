import pandas as pd
import seaborn as sns
from scipy import stats
from typing import Tuple, List
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import het_breuschpagan


sns.set_style('darkgrid')


# ======================
#  Normality Test
# ======================


def test_normality(data: pd.Series, plot: bool = True) -> Tuple[float, float]:
    """
    Conduct visualization and statistical normality tests including:
    (1) Shapiro-Wilk test
    (2) Kolmogorov–Smirnov test
    Produce p-value for the above two tests

    H0: Random variable follows Gaussian distribution
    H1: Random variable doesn't follow Gaussian distribution
    If p < 0.05: reject H0 -> random variable doesn't follow Gaussian
    If p >= 0.05, cannot reject H0 -> random variable follows Gaussian

    Shapiro-Wilk has the best power of significant among all normality tests.
    Power: probability of correctly rejecting false H0 given H1 is right.
    Type II error: probability of wrongly failing to reject false H0 given H1 is right.
    Power = 1 - Type II error -> Max Power = Min Type II error

    Kolmogorov–Smirnov test is used to test whether or not two distributions are identical.

    :param data: pd.Series, 1d array like data for normality test
    :param plot: bool, whether to produce plots
    """

    if plot:

        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

        # Distribution
        sns.distplot(data, ax=ax1)
        ax1.set_title('Distribution')

        # Quantile-Quantile
        stats.probplot(data, plot=ax2)
        ax2.set_title('QQ - Plot')

        plt.show()
        plt.close()

    # Shapiro-Wilk test
    shapiro_stat, shapiro_p_value = stats.shapiro(data)

    # Kolmogorov–Smirnov test
    kolmo_stat, kolmo_p_value = stats.kstest(data, 'norm')

    return shapiro_p_value, kolmo_p_value

# ======================
#  Homoscedasticity Test
# ======================


def test_homoscedasticity(data: pd.DataFrame, x: List[str], y: str, plot: bool = True) -> float:
    """
    Conduct visualization and statistical test for homoscedasticity:
    (1) Breusch–Pagan test
    Produce p-value for the above test.

     Breusch–Pagan test is done by regressing residuals on independent variables:
        r^2 = \beta @ X
    H0: beta_1 = beta_2 = ... = beta_k = 0
    H1: At least exists a beta_i != 0

    If p-value < 0.05, reject null hypothesis -> heteroscedasticity
    If p-value >= 0.05, cannot reject null hypothesis -> homoscedasticity


    :param data: pd.DataFrame, data frame with independent and dependent variables
    :param x: List[str], list of columns names for independent variables
    :param y: str, column name for dependent variable
    :param plot: bool, whether or not to produce plots
    """

    lr = sm.OLS(endog=data[y], exdog=data[x]).fit()
    fitted_y = pd.Series(lr.predict(x=data[x]))

    if plot:

        df = pd.DataFrame()
        df['fitted'] = fitted_y
        df['residual'] = lr.resid

        plt.figure(figsize=(12, 8))
        sns.scatterplot(data=df, x='fitted', y='residual')
        plt.title('Residual v.s. Fitted')
        plt.show()
        plt.close()

    return het_breuschpagan(lr.resid, exog_het=data[x])[1]

# ======================
#  Stationarity Test
# ======================


def test_stationarity(data: pd.Series, max_order: int = 5, plot: bool = True) -> int:
    """
    Conduct visualization and statistical test for stationarity:
    (1) ADF test
    Produce order in I(k)

    :param data: pd.Series, time series data to test
    :param max_order: int, max order for test purpose
    :param plot: bool, whether or not to generate plots

    ADF test is to designed to exam whether ot not a unit root exist:
    H0: A unit root exists
    H1: Process is stationary
    Intuition behind this is, if the process is characterized by a unit root, then the lagged features
    are not significant enough to predict the next period. If such unit root doesn't exist, then lagged
    features are more likely to be the driving force and therefore non-stationary.
    """

    order, stationary, data = -1, False, data.reset_index(drop=True)

    while not stationary and order <= max_order:

        p_value, order = adfuller(data)[1], order + 1
        stationary = True if p_value < 0.05 else False

        if plot:
            plt.figure(figsize=(12, 8))
            sns.lineplot(data)
            plt.title(f'I{order} model with p-value = {p_value}')
            plt.show()
            plt.close()

        data = data.diff().dropna()

    return order
