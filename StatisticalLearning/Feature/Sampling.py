import numpy as np
import pandas as pd
from typing import List


class SequentialBootStrap:
    """
    Assume time steps t = 0,...,T and samples i = 0,...,I.

    Two samples i and j are called concurrent at time t if both of them depend on the history
    [t - 1, t]. In that case, there is overlapping between sample i and sample j, therefore
    the samples are no longer IID (independently identically distributed).

    Denote 1(t,i) as the indicator that 1(t,i) = 1 if and only of [t0(i), t1(i)] overlaps
    with [t - 1, t]. Here t0(i) and t1(i) is the start and end time of the i-th sample
    and t - 1 and t is the start and end time of pre-determined unit interval.

    Denote c(t) = \sum_i 1(t, i) is the total number of samples concurrent at time t.
    Denote u(t, i) = 1(t, i) / c(t) is the uniqueness of sample i at time t.
    Denote u(i) = \sum_t u(t, i) / \sum_t 1(t, i) is the overall uniqueness of sample i.

    Sequential Bootstrapping:
        Instead of bootstrapping with equal probability, sequential bootstrapping will dynamically
        adjust the probability by considering the uniqueness of samples.

    (1) First, draw a random sample from uniform distribution -> K = [i]
    (2) While len(output) < desired_samples:
            try adding j to the output:
                K_ = K + [j]
                u(t, j) = 1(t, j) / \sum_k 1(t, k) for k in K_
                u(j) = \sum_t u(t, j) / \sum_t 1(t, j)
            p(j) = u(j) / \sum_j u(j) -> \sum_j p(j) = 1
            Randomly draw a sample with probability {p}

    The sequential bootstrap sample will be much closer to IID than samples from standard
    bootstrap method, which can be measure by average overall uniqueness = \sum_i u(i) / I.
    """

    def __init__(self, indicator: pd.DataFrame):
        """
        :param indicator: pd.DataFrame, T * I indicator matrix where indicator[t, i] = 1(t, i)
        """

        self._indicator = indicator

    # ====================
    #  Private
    # ====================

    @staticmethod
    def _get_avg_uniqueness(indicator: pd.DataFrame) -> pd.Series:
        """
        Compute average uniqueness for a sub-indicator matrix

        :param indicator: pd.DataFrame, a slice of indicator matrix
        """

        # c(t): total number of concurrent samples at t
        c = indicator.sum(axis=1)

        # u(t, i): uniqueness of sample i at time t
        u = indicator.div(c, axis=0)

        # u(i): average uniqueness of sample i
        avg_u = u[u > 0].mean()

        return avg_u

    # ====================
    #  Public
    # ====================

    def bootstrap(self, n_samples: float = 1.) -> List[int]:
        """
        Sequential bootstrapping algorithm to generate samples closer to IID

        :param n_samples: float, [0, 1] number of samples to generate as proportion to total samples
        """

        assert 0. <= n_samples <= 1., "Input must be between 0 and 1"

        n_samples = 1. if n_samples is not None else n_samples
        n_samples = int(n_samples * self._indicator.shape[1])

        candidates = self._indicator.columns

        output = []
        for _ in range(n_samples):

            avg_u = pd.Series()
            for col in candidates:
                indicator_added = self._indicator[output + [col]]
                avg_u.loc[i] = self._get_avg_uniqueness(indicator_added).iloc[-1]

            prob = avg_u / avg_u.sum()
            output.append(np.random.choice(candidates, p=prob))

        return output
