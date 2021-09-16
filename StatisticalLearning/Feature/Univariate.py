from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from typing import Union, Tuple


class UnivariateBuilder:

    """
    A convenient preprocessing pipeline based on Builder Pattern to transform input feature
    in corresponding training set and test set simultaneously including:
    (1) Binarize
    (2) Quantize
    (3) Log Feature
    (4) Power Feature
    (5) Min-Max Scale
    (6) Standardize
    (7) L2 Scale
    """

    def __init__(self, train: pd.Series, test: Union[pd.Series, None] = None):
        """
        :param train: pd.Series, feature vector from training set
        :param test: Union[pd.Series, None], feature vector from test set
        """

        self._train = train.reset_index(drop=True)
        self._test = test.reset_index(drop=True) if test is not None else None

    def binarize(self,
                 threshold: float = 0.,
                 higher_positive: bool = True) -> UnivariateBuilder:
        """
        Convert numeric features/target to 0/1 such that:
            if higher_positive = True, value >= threshold -> 1 else 0
            if higher_positive = False, value <= threshold -> 1 else 0

        :param threshold: float, threshold to determine positive limit
        :param higher_positive: bool, indicator for positive direction
        """

        self._train = self._train.apply(lambda v: 1 if v >= threshold else 0) if higher_positive \
                  else self._train.apply(lambda v: 1 if v <= threshold else 0)

        if self._test is not None:
            self._test = self._test.apply(lambda v: 1 if v >= threshold else 0) if higher_positive \
                else self._test.apply(lambda v: 1 if v <= threshold else 0)

        return self

    def quantize(self,
                 bins: Union[np.ndarray, None] = None,
                 n_quantile: int = 10) -> UnivariateBuilder:
        """
        Divide numeric feature into intervals:
            if bins is given, pre-determined fixed intervals will be used
            if bins is None, adaptive quantile intervals will be used

        :param bins: Union[np.ndarray, None], 1d-array including fixed intervals
        :param n_quantile: int, number of quantiles to divide raw vector into
        """

        # Fixed-width Binning / Quantile Binning
        bins = bins if bins is not None else np.quantile(self._train, np.linspace(0, 1, n_quantile + 1))
        bins = np.sort(bins.reshape(1, -1)).squeeze()

        self._train = pd.Series(np.searchsorted(a=bins, v=self._train, side='right'))

        if self._test is not None:
            self._test = pd.Series(np.searchsorted(a=bins, v=self._test, side='right'))

        return self

    def log_transform(self) -> UnivariateBuilder:
        """
        Feature raw vector s.t. x -> log(x) for heavy-tailed distribution
        """

        assert np.min(self._train) > 0, 'Training feature must be positive for log transformation.'
        self._train = pd.Series(np.log(self._train))

        if self._test is not None:
            assert np.min(self._test) > 0, 'Test feature must be positive for log transformation.'
            self._train = pd.Series(np.log(self._test))

        return self

    def power_transform(self, lmbda: Union[float, None] = None) -> UnivariateBuilder:
        """
        Box-Cox transformation to make heavily tailed distribution more like Gaussian:
        x -> x ^ lmbda - 1 / lmbda if lmbda != 0 else log(x)

        :param lmbda: float, param for power transformation, if None, optimal param will be fitted
        """

        assert np.min(self._train) > 0, 'Training feature must be positive for Box-Cox transformation.'
        if lmbda is None:
            self._train, lmbda = pd.Series(stats.boxcox(self._train.to_numpy()))
        else:
            self._train = pd.Series(stats.boxcox(self._train.to_numpy(), lmbda=lmbda))

        if self._test is not None:
            assert np.min(self._test) > 0, 'Test feature must be positive for Box-Cox transformation.'
            self._test = pd.Series(stats.boxcox(self._test.to_numpy(), lmbda=lmbda))

        return self

    def min_max_scale(self) -> UnivariateBuilder:
        """
        Scale numeric feature to [0, 1]:
            x = [x - min(x)] / [max(x) - min(x)]
        """

        min_value, max_value = np.min(self._train), np.max(self._train)
        self._train = self._train.apply(lambda v: (v - min_value) / (max_value - min_value))

        if self._test is not None:
            self._test = self._test.apply(lambda v: (v - min_value) / (max_value - min_value))

        return self

    def standardize(self) -> UnivariateBuilder:
        """
        Scale numeric feature to zero mean and unit variance:
            x = [x - mean(x)] / std(x)
        """

        mean, std = np.mean(self._train), np.std(self._train)
        self._train = self._train.apply(lambda v: (v - mean) / std)

        if self._test is not None:
            self._test = self._test.apply(lambda v: (v - mean) / std)

        return self

    def l2_scale(self) -> UnivariateBuilder:
        """
        Scale numeric feature to make unit l2 norm:
            x = x / ||x||
        """

        norm = np.linalg.norm(self._train, ord=2)
        self._train /= norm

        if self._test is not None:
            self._test /= norm

        return self

    def build(self) -> Tuple[pd.Series, pd.Series]:
        """
        Build function to call to return transformed train and test
        """

        return self._train, self._test
