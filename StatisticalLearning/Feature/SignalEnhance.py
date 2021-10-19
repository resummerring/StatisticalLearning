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

    After the signal and noise components are identified, denoising and detoning techniques can be used to process
    the correlation matrix and generated a more robust, well-conditioned, well-informed correlation matrix.
    """

    def __init__(self, matrix: np.ndarray, q: Union[float, None] = None, is_cov: bool = False,
                 band_width: float = 0.25, kernel: str = 'gaussian', n_steps: int = 1000):

        """
        :param matrix: np.ndarray, T * N observation matrix with T > N if is_cov = False else N * N covariance matrix
        :param q: Union[float, None], if is_cov = true, q must be provided, otherwise q will be deduced from matrix
        :param is_cov, bool, if true then covariance matrix is provided else observation matrix is provided
        :param band_width: float, band width for fitting kernel density, by default 0.25
        :param kernel: str, distribution for fitting kernel density, by default gaussian
        :param n_steps: int, number of discrete pdf values to evaluate for optimization
        """

        # Observation matrix property
        self._matrix = matrix if not is_cov else None
        self._q = self._matrix.shape[0] / self._matrix.shape[1] if not is_cov else q

        if self._q is None:
            raise ValueError("If covariance matrix is provided, q is also needed")

        # Original covariance / correlation matrix property
        self._original_cov = np.cov(self._matrix, rowvar=False) if not is_cov else matrix
        self._original_corr = self.cov_to_corr(self._original_cov)
        self._original_corr_eval, self._original_corr_evec = self.get_sorted_pca(self._original_corr)

        # Denoised correlation matrix property
        self._denoised_corr = None

        # Hyper-parameter tuning KDE
        self._band_width = band_width
        self._kernel = kernel
        self._n_steps = n_steps

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

    def _fit_max_eval(self) -> Tuple[float, float]:
        """
        Calibrate covariance matrix eigenvalues towards Marcenko-Pastur distribution to get max eigenvalue and sigma
        """

        lambds = np.diag(self._original_corr_eval)

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

    def denoise(self, method: str = 'Constant Residual', alpha: Union[float, None] = 1.):
        """
        Denoise correlation matrix according to calibrated sigma and max eigenvalue threshold:
        Assume {eval} i = 1,...,N are the descendingly sorted eigenvalues from correlation matrix,
        find k such that eval(k) > eval_max -> signal and eval(k+1) <= eval_max -> noise

        1. Constant Residual:
            (1) set all {eval(i) | eval(i) < eval_max} = mean({eval(i) | eval(i) < eval_max}) -> {new_eval}
            (2) denoised correlation = rescaled(evec @ eval_new @ evec.T)

            Trace is preserved, minimum eval is increased -> condition number is improved

        2. Target Shrinkage:
            (1) signal_component = evec_signal @ eval_signal @ evec_signal.T
            (2) noise_component = evec_noise @ eval_noise @ evec_noise.T
            (3) denoised correlation = signal_component + alpha * noise_component + (1 - alpha) * diag(noise_component)

            alpha = 1 -> no shrinkage, alpha = 0 -> total shrinkage

        :param method: str, shrinkage method {'Constant Residual', 'Target Shrinkage'}
        :param alpha: Union[float, None], only used with Target Shrinkage, 1 = no shrinkage and 0 = total shrinkage
        """

        eval_max, sigma = self._fit_max_eval()
        lambds = np.diag(self._original_corr_eval).copy()

        n_factors = next(index for index, val in enumerate(lambds) if val < eval_max)

        if method == 'Constant Residual':

            lambds[n_factors:] = lambds[n_factors:].sum() / float(lambds.shape[0] - n_factors)
            denoised_corr_eval = np.diag(lambds)
            denoised_corr = np.dot(self._original_corr_evec, denoised_corr_eval).dot(self._original_corr_evec.T)

            # Rescale diagonal element to be 1
            self._denoised_corr = self.cov_to_corr(denoised_corr)

        elif method == 'Target Shrinkage':

            lambds_signal = self._original_corr_eval[:n_factors, :n_factors]
            evec_signal = self._original_corr_evec[:, :n_factors]
            lambds_noise = self._original_corr_eval[n_factors:, n_factors:]
            evec_noise = self._original_corr_evec[:, n_factors:]

            if not lambds_signal.shape[0] == 0:
                corr_signal = np.dot(evec_signal, lambds_signal).dot(evec_signal.T)
                corr_noise = np.dot(evec_noise, lambds_noise).dot(evec_noise.T)
                self._denoised_corr = corr_signal + alpha * corr_noise + (1 - alpha) * np.diag(np.diag(corr_noise))

        else:
            raise ValueError("Accepted shrinkage methods are: (1) Constant Residual (2) Target Shrinkage")
