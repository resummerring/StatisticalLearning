import numpy as np
import pandas as pd
from typing import List


class DiscreteRV:
    """
    Discrete random variable
    """
    def __init__(self, value: np.ndarray, prob: np.ndarray):
        """
        :param value: np.ndarray, possible values
        :param prob: np.ndarray, corresponding probability
        """

        assert value.shape == prob.shape, f"Dimension mismatch: value has shape {value.shape} " \
                                          f"while prob has shape {prob.shape}"

        assert min(prob) >= 0, f"Value Error: input probability must be positive"

        self._value, self._prob = value.reshape(-1, 1), prob.reshape(-1, 1)

    @property
    def value(self) -> np.ndarray:
        return self._value

    @property
    def prob(self) -> np.ndarray:
        return self._prob


class CondDiscreteRV:
    """
    Conditional discrete random variable
    """
    def __init__(self, cond_prob: np.ndarray, cond_dist: List[DiscreteRV]):
        """
        :param cond_prob: np.ndarray, conditional probability
        :param cond_dist: np.ndarray[DiscreteRV], distribution given conditional value
        """

        cond_prob = cond_prob.reshape(-1, 1)
        assert len(cond_prob) == len(cond_dist), f"Dimension mismatch: conditional value has shape {len(cond_prob)}"\
                                                 f"while conditional distribution has shape {len(cond_dist)}"

        self._cond_prob = cond_prob
        self._cond_dist = np.array(cond_dist)

    @property
    def cond_prob(self) -> np.ndarray:
        return self._cond_prob

    @property
    def cond_dist(self) -> np.ndarray:
        return self._cond_dist


class InfoDistance:
    """
    An information theory based distance / similarity class

    Correlation based distance metrics have the following disadvantage:
    (1) It only quantifies the 'linear' codependency and neglects nonlinear relationship
    (2) It is very sensitive to and highly influenced by outliers
    (3) It might not work well on cases where multivariate normal doesn't apply
    """

    @staticmethod
    def entropy(rv: DiscreteRV) -> float:
        """
        H(X) = -sum_x [p(x) * log p(x)]
        Amount of uncertainty associated with random variable X

        H(X, Y) = -sum_xy [p(x, y) * log p(x, y)]
        Amount of uncertainty associated with joint random variable (X, Y)
        X and Y may not live on the same probability space for joint entropy

        H(X, Y) = H(Y, X), H(X, X) = H(X), H(X, Y) >= max[H(X), H(Y)], H(X, Y) <= H(X) + H(Y)

        :param rv: DiscreteRV, discrete random variable
        """
        return -np.dot(rv.prob, np.log(rv.prob))

    @staticmethod
    def cond_entropy(cond: CondDiscreteRV) -> float:
        """
        H(X|Y) = H(X, Y) - H(X) = - sum_y {p(y) * sum_x [p(x|Y = y) * log p(x|Y = y)]}
        Amount of uncertainty in X if Y is provided

        :param cond: CondDiscreteRV, conditional discrete random variable
        """

        entropy = np.array([InfoDistance.entropy(rv) for rv in cond.cond_dist])
        return -np.dot(cond.cond_prob, entropy)

    @staticmethod
    def KLDistance(p: DiscreteRV, q: DiscreteRV) -> float:
        """
        KL(p||q) = -sum_x p(x) log (q(x) / p(x))
        Divergence of distribution p away from benchmark distribution q

        Note, KL distance can only be measure by two random variables defined on the same probability space

        :param p: DiscreteRV, test distribution
        :param q: DiscreteRV, benchmark distribution
        """

        assert np.array_equal(p.value, q.value), "Input random variables are not define on the same probability space"

        return -np.dot(p.prob, np.log(np.divide(q.prob, p.prob)))

    @staticmethod
    def cross_entropy(p: DiscreteRV, q: DiscreteRV) -> float:
        """
        HC(p||q) = -sum_x p(x) * log q(x) = H(X) + KL(p||q)
        Uncertainty associated with X when information evaluated using wrong distribution q rather than p

        :param p: DiscreteRV, discrete random variable
        :param q: DiscreteRV, discrete random variable
        """

        assert np.array_equal(p.value, q.value), "Input random variables are not define on the same probability space"

        return -np.dot(p.prob, np.log(q.prob))
