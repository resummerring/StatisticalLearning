import unittest
import warnings
import pandas as pd
import numpy as np
from StatisticalLearning.Feature.Univariate import UnivariateProcessor


class TEST_UnivariateBuilder(unittest.TestCase):

    def setUp(self):
        warnings.filterwarnings('ignore')

    def test_single(self):

        train, test = pd.Series([1, 2, 3, 4, 5]), pd.Series([0, 3.5, 8])

        # Standardize
        train_,  test_ = UnivariateProcessor(train, test).standardize().build()
        expected_train = pd.Series([-np.sqrt(2), -1 / np.sqrt(2), 0, 1 / np.sqrt(2), np.sqrt(2)])
        expected_test = pd.Series([-1.5 * np.sqrt(2), 0.5 / np.sqrt(2), 2.5 * np.sqrt(2)])
        self.assertAlmostEqual(np.linalg.norm(train_ - expected_train), 0, places=10)
        self.assertAlmostEqual(np.linalg.norm(test_ - expected_test), 0, places=10)

        # Min-Max scale
        train_, test_ = UnivariateProcessor(train, test).min_max_scale().build()
        expected_train = pd.Series([0, 0.25, 0.5, 0.75, 1.0])
        expected_test = pd.Series([-0.25, 0.625, 1.75])
        self.assertAlmostEqual(np.linalg.norm(train_ - expected_train), 0, places=10)
        self.assertAlmostEqual(np.linalg.norm(test_ - expected_test), 0, places=10)

        # L2 scale
        train_, test_ = UnivariateProcessor(train, test).l2_scale().build()
        expected_train = train.apply(lambda v: v / np.sqrt(55))
        expected_test = test.apply(lambda v: v / np.sqrt(55))
        self.assertAlmostEqual(np.linalg.norm(train_ - expected_train), 0, places=10)
        self.assertAlmostEqual(np.linalg.norm(test_ - expected_test), 0, places=10)

        # Log transform
        train_, _ = UnivariateProcessor(train).log_transform().build()
        expected_train = pd.Series(np.log(train))
        self.assertAlmostEqual(np.linalg.norm(train_ - expected_train), 0, places=10)

        # Power transform
        train_, _ = UnivariateProcessor(train).power_transform(lmbda=1).build()
        expected_train = pd.Series([0, 1, 2, 3, 4])
        self.assertAlmostEqual(np.linalg.norm(train_ - expected_train), 0, places=10)

        train_, _ = UnivariateProcessor(train).power_transform(lmbda=2).build()
        expected_train = train.apply(lambda v: 0.5 * (v ** 2 - 1))
        self.assertAlmostEqual(np.linalg.norm(train_ - expected_train), 0, places=10)

        # Binarize
        train_, test_ = UnivariateProcessor(train, test).binarize(threshold=3.5).build()
        expected_train = pd.Series([0, 0, 0, 1, 1])
        expected_test = pd.Series([0, 1, 1])
        self.assertTrue(train_.equals(expected_train))
        self.assertTrue(test_.equals(expected_test))

        # Quantize
        train_, test_ = UnivariateProcessor(train, test).quantize(bins=np.array([-1, 2.5, 4.5, 9])).build()
        expected_train = pd.Series([1, 1, 2, 2, 3])
        expected_test = pd.Series([1, 2, 3])
        self.assertTrue(train_.equals(expected_train))
        self.assertTrue(test_.equals(expected_test))
