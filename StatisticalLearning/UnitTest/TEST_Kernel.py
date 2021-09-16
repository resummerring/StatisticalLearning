import time
import unittest
import numpy as np
from StatisticalLearning.Feature.Kernel import Kernel


class TEST_Kernel(unittest.TestCase):

    def setUp(self):

        self._x = np.array([1, 2, 3])
        self._y = np.array([2, 3, 4])

    def test_kernel(self):
        self.assertEqual(Kernel.linear(self._x, self._y), 20)
        self.assertEqual(Kernel.polynomial(self._x, self._y), 20)
        self.assertEqual(Kernel.gaussian(self._x, self._y, sigma=1), np.exp(-1.5))
        self.assertEqual(Kernel.exponential(self._x, self._y, sigma=1), np.exp(-np.sqrt(3) / 2))
        self.assertEqual(Kernel.epanechnikov(self._x, self._y, sigma=1), 0)
        self.assertEqual(Kernel.tri_cube(self._x, self._y, sigma=1), 0)

    def test_njit(self):
        X = np.random.rand(1000, 10)
        Y = np.random.rand(1000, 10)

        # Compilation step
        Kernel.gaussian(X[0], Y[0], sigma=1)

        # With not-just-in-time
        start = time.time()
        for x, y in zip(X, Y):
            Kernel.gaussian(x, y, sigma=1)
        end = time.time()

        # Without compilation, expected time is 0.02 seconds
        self.assertTrue(end - start < 1e-8)




