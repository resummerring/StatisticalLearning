import numpy as np
import pandas as pd
from typing import Union, Tuple
from scipy.optimize import minimize
from sklearn.neighbors import KernelDensity


class SignalEnhance:
    """
    Assume a matrix of IID observation generated from underlying process with mean = 0 and
        variance = sigma^2, then the covariance matrix C = X.T @ X / T has eigenvalues lambd that
        asymptotically converges to Marcenko-Pastur distribution with the following pdf function:
            f(x) = (T / N) * [(x_max - x) * (x - x_min) / (2 * pi * x * sigma^2)] if x_min <= x <= m_max else 0
        All eigenvalues x \in [0, x_max] will be associated with noise and {x | x > x_max} are signals.

    For a matrix that consists of both noise and signals, we can calibrate the eigenvalues of its correlation
    matrix to Marcenko-Pastur distribution to find the optimal sigma that minimizes the sum of squared error
    between theoretical pdf value and fitted KDE pdf values. With fitted sigma and max eigenvalue limit, we can
    differentiate noises from signals. Fitted sigma^2 can be regarded as the proportion of variance that can be
    explained by the noise (randomness) of the data matrix.
    """

    def __init__(self, matrix: np.ndarray, band_width: float = 0.25, kernel: str = 'gaussian', n_steps: int = 1000):

        """
        :param matrix: np.ndarray, T * N observation matrix with T > N
        :param band_width: float, band width for fitting kernel density, by default 0.25
        :param kernel: str, distribution for fitting kernel density, by default gaussian
        :param n_steps: int, number of discrete pdf values to evaluate for optimization
        """

        self._matrix = matrix
        self._band_width = band_width
        self._kernel = kernel
        self._n_steps = n_steps
        self._q = self._matrix.shape[0] / self._matrix.shape[1]

    # ====================
    #  Private
    # ====================

    def _marcenko_pastur_pdf(self, sigma: float) -> pd.Series:
        """
        Compute Marcenko-Pastur pdf values at positions np.linspace(x_min, x_max, n_steps)

        :param sigma: float, standard deviation of underlying generating process

        """
        eval_max = np.square(sigma) * np.square(1 + 1. / np.sqrt(self._q)).squeeze()
        eval_min = np.square(sigma) * np.square(1 - 1. / np.sqrt(self._q)).squeeze()
        eval = np.linspace(eval_min, eval_max, self._n_steps).squeeze()
        pdf = self._q * np.sqrt((eval_max - eval) * (eval - eval_min)) / (2 * np.pi * eval * np.square(sigma))
        return pd.Series(pdf, index=eval)

    def _fit_empirical_kde(self, lambds: np.ndarray, val: Union[None, np.ndarray] = None) -> pd.Series:
        """
        Fit 1D kernel density of lambds and evaluate fitted distribution on test points

        :param lambds: np.ndarray, 1D points on which kernel density will be fitted
        :param val: Union[None, np.ndarray], test points on which pdf will be returned, if None use lambds
        """

        lambds = lambds.reshape(-1, 1)
        val = np.unique(lambds).reshape(-1, 1) if val is None else val.reshape(-1, 1)
        kde = KernelDensity(kernel=self._kernel, bandwidth=self._band_width).fit(lambds)
        return pd.Series(np.exp(kde.score_samples(np.array(val))), index=val.flatten())

    def _pdf_error(self, sigma: float, lambds: np.ndarray) -> float:
        """
        Objective function to fit any covariance matrix to Marcenko-Pastur distribution
        by minimizing the squared sum of residuals between original pdf and fitted kde pdf

        :param sigma: float, standard deviation of underlying generating process
        :param lambds: np.ndarray, 1D points on which kernel density will be fitted
        """

        pdf_mp = self._marcenko_pastur_pdf(sigma=sigma)
        pdf_kde = self._fit_empirical_kde(lambds=lambds, val=pdf_mp.index.values)
        return np.sum(np.square(pdf_mp - pdf_kde)).squeeze()

    def _fit_max_eval(self, lambds: np.ndarray) -> Tuple[float, float]:
        """
        Calibrate covariance matrix eigenvalues towards Marcenko-Pastur distribution to get max eigenvalue and sigma

        :param lambds: pd.Series, 1D points on which kernel density will be fitted
        """
        minimizer = minimize(lambda *x: self._pdf_error(*x), x0=np.array([0.5]),
                             args=(lambds,), bounds=((1e-5, 1 - 1e-5),))

        sigma = minimizer['x'][0] if minimizer['success'] else 1
        eval_max = np.sqrt(sigma) * np.square(1 + 1. / np.sqrt(self._q))
        return eval_max, sigma

    # ====================
    #  Public
    # ====================

    @staticmethod
    def get_sorted_pca(matrix: np.ndarray):
        """
        Return descendingly sorted principal components by eigenvalues of a matrix

        :param matrix: np.ndarray, a square matrix that can be eig-decomposed
        """

        try:
            eval, evec = np.linalg.eigh(matrix)
        except np.linalg.LinAlgError:
            raise ValueError("Input matrix cannot be eigen-decomposed.")

        order = eval.argsort()[::-1]
        eval, evec = eval[order], evec[:, order]
        return np.diagflat(eval), evec

    @staticmethod
    def cov_to_corr(cov: np.ndarray) -> np.ndarray:
        """
        Transform a covariance matrix into a correlation matrix

        :param cov: np.ndarray, covariance matrix
        """

        std = np.sqrt(np.diag(cov))
        corr = cov / np.outer(std, std)
        corr[corr < - 1], corr[corr > 1] = -1, 1

        return corr

    def calibrate(self) -> Tuple[float, float]:
        """
        Main function to calibrate observation matrix to find optimal sigma
        """

        cov = np.cov(self._matrix, rowvar=False)
        corr = self.cov_to_corr(cov)
        eval, evec = self.get_sorted_pca(corr)
        e_max, sigma = self._fit_max_eval(np.diag(eval))

        return e_max, sigma
