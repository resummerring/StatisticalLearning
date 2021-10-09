import numpy as np
from typing import Generator


class HyperParamTuner:

    @staticmethod
    def grid_search(lower: float, upper: float, step: float) -> Generator[float]:
        """
        1D exhaustive grid search generator

        :param lower: float, lower bound of search space, inclusive
        :param upper: float, upper bound of search space, exclusive
        :param step: float, search step between lower and upper bound
        """

        for value in np.arange(lower, upper, step):
            yield value

    @staticmethod
    def random_search(lower: float, upper: float, seed: int = 0) -> Generator[float]:
        """
        1D uniform random search generator

        :param lower: float, lower bound of search space, inclusive
        :param upper: float, upper bound of search space, inclusive
        :param seed: int, random number generator seed, by default 0
        """

        np.random.seed(seed)
        yield np.random.uniform(lower, upper, 1).squeeze()

    @staticmethod
    def log_uniform_search(lower: float, upper: float, seed: int = 0) -> Generator[float]:
        """
        1D log-uniform random search generator

        :param lower: float, lower bound of search space, inclusive
        :param upper: float, upper bound of search space, inclusive
        :param seed: int, random number generator seed, by default 0
        """

        np.random.seed(seed)
        yield np.exp(np.random.uniform(np.log(lower), np.log(upper), 1)).squeeze()
