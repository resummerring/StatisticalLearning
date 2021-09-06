import unittest
import warnings
import numpy as np
import pandas as pd
from StatisticalLearning.Optimization.GradientOptimizer import GradientDescent


class TEST_GradientDescent(unittest.TestCase):

    def setUp(self):
        warnings.filterwarnings('ignore', category=FutureWarning)

    def test_simple_function(self):

        def f(x, data=None): return x ** 2 - 2 * x + 1
        def g(x, data=None): return 2 * x - 2

        optimizer = GradientDescent(f, g)
        result = optimizer.solve(x0=0, learning_rate=0.1)
        self.assertTrue(result.success)
        self.assertAlmostEqual(result.optimum, 1)
        self.assertAlmostEqual(result.minimum, 0)

    def test_array_function(self):

        def f(x, data=None): return (x[0] - x[1]) ** 2 + (x[0] + x[1] - 1) ** 2
        def g(x, data=None): return np.array([2 * (x[0] - x[1]) + 2 * (x[0] + x[1] - 1),
                                             -2 * (x[0] - x[1]) + 2 * (x[0] + x[1] - 1)])

        optimizer = GradientDescent(f, g)
        result = optimizer.solve(x0=np.array([0, 0]), learning_rate=0.1)
        self.assertTrue(result.success)
        self.assertAlmostEqual(result.optimum[0], 0.5)
        self.assertAlmostEqual(result.optimum[1], 0.5)
        self.assertAlmostEqual(result.minimum, 0)

    def test_series_function(self):

        def f(x, data=None): return (x[0] - x[1]) ** 2 + (x[0] + x[1] - 1) ** 2
        def g(x, data=None): return pd.Series([2 * (x[0] - x[1]) + 2 * (x[0] + x[1] - 1),
                                              -2 * (x[0] - x[1]) + 2 * (x[0] + x[1] - 1)])

        optimizer = GradientDescent(f, g)
        result = optimizer.solve(x0=pd.Series([0, 0]), learning_rate=0.1)
        self.assertTrue(result.success)
        self.assertAlmostEqual(result.optimum[0], 0.5)
        self.assertAlmostEqual(result.optimum[1], 0.5)
        self.assertAlmostEqual(result.minimum, 0)
