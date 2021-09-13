import numpy as np
from numba import njit
from StatisticalLearning.Transform.Distance import Distance


class Kernel:
    """
    All calculation are accelerated using not-just-in-time compilation
    Kernel methods are convenient to map non-linear classification problem to linear problem

    Kernel methods including:
    (1) Linear
    (2) Polynomial
    (3) Gaussian
    (4) Exponential
    (5) Epanechnikov
    (6) Tri-cube
    """

    @staticmethod
    @njit
    def linear(x: np.ndarray, y: np.ndarray, c: float = 0) -> float:
        """
        K(x, y) = x.T @ y + c
        """

        x, y = x.reshape(-1, 1), y.reshape(-1, 1)
        assert x.shape == y.shape, "Input should have the same shape."

        return np.sum(np.multiply(x, y)) + c

    @staticmethod
    @njit
    def polynomial(x: np.ndarray, y: np.ndarray, a: float = 1, c: float = 0, d: float = 1) -> float:
        """
        K(x, y) = (a * x.T @ y + c) ^ d
        """

        x, y = x.reshape(-1, 1), y.reshape(-1, 1)
        assert x.shape == y.shape, "Input should have the same shape."

        return np.power(a * np.sum(np.multiply(x, y)) + c, d)

    @staticmethod
    @njit
    def gaussian(x: np.ndarray, y: np.ndarray, sigma: float) -> float:
        """
        K(x, y) = exp {- ||x - y||^2 / 2 sigma^2}
        """

        x, y = x.reshape(-1, 1), y.reshape(-1, 1)
        assert x.shape == y.shape, "Input should have the same shape."

        return np.exp(-np.sum(np.square(x - y)) / (2 * sigma * sigma))

    @staticmethod
    @njit
    def exponential(x: np.ndarray, y: np.ndarray, sigma: float) -> float:
        """
        K(x, y) = exp {- ||x - y|| / 2 sigma^2}
        """

        x, y = x.reshape(-1, 1), y.reshape(-1, 1)
        assert x.shape == y.shape, "Input should have the same shape."

        return np.exp(-np.sqrt(np.sum(np.square(x - y))) / (2 * sigma * sigma))

    @staticmethod
    @njit
    def epanechnikov(x: np.ndarray, y: np.ndarray, sigma: float) -> float:
        """
        K(x, y) = max[0.75 * (1 - (||x - y|| / lambda) ^ 2), 0]
        """

        x, y = x.reshape(-1, 1), y.reshape(-1, 1)
        assert x.shape == y.shape, "Input should have the same shape."

        return np.maximum(0.75 * (1 - np.sum(np.square(x - y)) / (sigma * sigma)), 0)

    @staticmethod
    @njit
    def tri_cube(x: np.ndarray, y: np.ndarray, sigma: float) -> float:
        """
        K(x, y) = max[(1 - | ||x - y|| / lambda |^3)^3, 0]
        """

        x, y = x.reshape(-1, 1), y.reshape(-1, 1)
        assert x.shape == y.shape, "Input should have the same shape."

        return np.maximum(np.power(1 - np.power(np.abs(np.sqrt(np.sum(np.square(x - y))) / sigma), 3), 3), 0)


