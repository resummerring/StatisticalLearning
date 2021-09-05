import unittest
import numpy as np
import pandas as pd
from StatisticalLearning.Preprocess import Preprocessor


class TEST_Preprocess(unittest.TestCase):

    def test_normalize(self):
        df = pd.DataFrame(2 + 3 * np.random.randn(1000, 10))
        df_out, _, _ = Preprocessor.normalize(df)
        mean, std = df_out.mean(axis=0), df_out.std(axis=0)
        self.assertAlmostEqual(np.linalg.norm(mean, ord=2), 0, places=12)
        self.assertAlmostEqual(np.linalg.norm(std - 1, ord=2), 0, places=12)

        df = pd.Series((2 + 3 * np.random.randn(1, 1000)).squeeze())
        df_out, _, _ = Preprocessor.normalize(df)
        mean, std = df_out.mean(axis=0), df_out.std(axis=0)
        self.assertAlmostEqual(mean, 0, places=12)
        self.assertAlmostEqual(std, 1, places=12)

    def test_standardize(self):
        df = pd.DataFrame([[-1, -2, 1], [1, 3, 2], [2, 4, 4], [-2, -1, 0]])
        df_out, _, _ = Preprocessor.standardize(df)
        self.assertAlmostEqual(np.linalg.norm(df_out.max(axis=0) - 1, ord=2), 0, places=12)
        self.assertAlmostEqual(np.linalg.norm(df_out.min(axis=0), ord=2), 0, places=12)

        df = pd.Series([-1, 0, 1, 2, 3, 4])
        df_out, _, _ = Preprocessor.standardize(df)
        self.assertEqual(np.linalg.norm(df_out - pd.Series([0, 0.2, 0.4, 0.6, 0.8, 1.]), ord=2), 0)

    def test_dummy_variables(self):
        df = pd.DataFrame([['male', 'red', 1], ['female', 'orange', 2], ['male', 'orange', 3], ['male', 'yellow', 4]],
                          columns=['sex', 'color', 'age'])
        df = Preprocessor.add_dummy_variables(df, ['sex', 'color'])
        expected = pd.DataFrame([[1, 1, 1, 0], [2, 0, 0, 1], [3, 1, 0, 1], [4, 1, 0, 0]],
                                columns=['age', 'sex_male', 'color_red', 'color_orange'])
        self.assertTrue(df.equals(expected))


