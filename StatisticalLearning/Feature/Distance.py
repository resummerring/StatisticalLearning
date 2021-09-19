import numpy as np
from numba import njit
from typing import Union


class Distance:

    """
    All calculation are accelerated using not-just-in-time compilation

    Distance measure between two vectors including:
    (1) Euclidean distance
    (2) Manhattan distance
    (3) Chebyshev distance
    (4) Minkowski distance
    (5) Cosine distance
    """

    # ====================
    #  Public
    # ====================

    @staticmethod
    @njit
    def euclidean(x: np.ndarray, y: np.ndarray) -> Union[float, np.ndarray]:
        """
        D(x, y) = sqrt[sum_i (x_i - y_i)^2]
        """

        x, y = x.reshape(-1, 1), y.reshape(-1, 1)
        assert x.shape == y.shape, "Input should have the same shape."

        return np.sqrt(np.sum(np.square(x - y)))

    @staticmethod
    @njit
    def manhattan(x: np.ndarray, y: np.ndarray) -> Union[float, np.ndarray]:
        """
        D(x, y) = sum_i (|x_i - y_i|)
        """

        x, y = x.reshape(-1, 1), y.reshape(-1, 1)
        assert x.shape == y.shape, "Input should have the same shape."

        return np.sum(np.abs(x - y))

    @staticmethod
    @njit
    def chebyshev(x: np.ndarray, y: np.ndarray) -> Union[float, np.ndarray]:
        """
        D(x, y) = max_i (|x_i - y_i|)
        """

        x, y = x.reshape(-1, 1), y.reshape(-1, 1)
        assert x.shape == y.shape, "Input should have the same shape."

        return np.max(np.abs(x - y))

    @staticmethod
    @njit
    def minkowski(x: np.ndarray, y: np.ndarray, p: float) -> Union[float, np.ndarray]:
        """
        D(x, y) = [sum_i (|x_i - y_i|^p)]^(1/p)
        """

        x, y = x.reshape(-1, 1), y.reshape(-1, 1)
        assert x.shape == y.shape, "Input should have the same shape."

        return np.power(np.sum(np.power(np.abs(x - y), p)), 1/p)

    @staticmethod
    @njit
    def cosine(x: np.ndarray, y: np.ndarray) -> Union[float, np.ndarray]:
        """
        D(x, y) = x @ y / (||x|| * ||y||)

        Cosine similarity = 1 - Cosine distance
        """

        x, y = x.reshape(-1, 1), y.reshape(-1, 1)
        assert x.shape == y.shape, "Input should have the same shape."

        norm_x, norm_y = np.sqrt(np.sum(np.square(x))), np.sqrt(np.sum(np.square(y)))
        return 1 - (np.sum(np.multiply(x, y)) / (norm_x * norm_y))
