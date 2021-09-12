import time
import unittest
import warnings
import numpy as np
from scipy.spatial.distance import cosine
from StatisticalLearning.Transform.Distance import Distance


class TEST_Distance(unittest.TestCase):

    def setUp(self):

        warnings.filterwarnings('ignore')

        self._x = np.array([1, 3, 6, 5, 2])
        self._y = np.array([3, 1, 2, 5, 4])

    def test_distance(self):
        self.assertEqual(Distance.euclidean(self._x, self._y), np.sqrt(28))
        self.assertEqual(Distance.manhattan(self._x, self._y), 10.0)
        self.assertEqual(Distance.chebyshev(self._x, self._y), 4.0)
        self.assertEqual(Distance.minkowski(self._x, self._y, 2.0), np.sqrt(28))
        self.assertEqual(Distance.minkowski(self._x, self._y, 1.0), 10.0)
        self.assertEqual(Distance.cosine(self._x, self._y), cosine(self._x, self._y))

    def test_assertion(self):

        x = np.array([1, 2, 3, 4, 5])
        y = np.array([1, 2, 3, 4, 5, 6])
        self.assertRaises(AssertionError, Distance.euclidean, x, y)

    def test_njit(self):

        X = np.random.rand(1000, 10)
        Y = np.random.rand(1000, 10)

        # Compilation step
        Distance.euclidean(X[0], Y[0])

        # With not-just-in-time
        start = time.time()
        for x, y in zip(X, Y):
            Distance.euclidean(x, y)
        end = time.time()

        # Without compilation, expected time is 0.02 seconds
        self.assertTrue(end - start < 1e-8)




