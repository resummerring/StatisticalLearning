import unittest
import numpy as np
import pandas as pd
from StatisticalLearning.LeastSquare import LinearLeastSquare


class TEST_LeastSquare(unittest.TestCase):

    def test_sum_square_error(self):
        X = pd.DataFrame([[1, 2, 3],
                          [2, 3, 4],
                          [3, 4, 5]])
        y = pd.Series([2, 2, 2])
        ls = LinearLeastSquare(X, y)
        self.assertEqual(ls.sum_square_residual(pd.Series([1, -1, 1])), 5)
        self.assertEqual(ls.sum_square_residual(pd.Series([-1, 1, 1])), 29)

    def test_optimize(self):
        X = pd.DataFrame(np.random.rand(100, 3))
        y = 2 * X.iloc[:, 0] - 3 * X.iloc[:, 1] + X.iloc[:, 2]
        ls = LinearLeastSquare(X, y)
        coef = ls.optimize(init_guess=pd.Series([1, -1, 1]))
        self.assertAlmostEqual(np.linalg.norm(coef - np.array([2, -3, 1])), 0, places=5)
