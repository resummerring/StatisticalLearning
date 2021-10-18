import unittest
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from StatisticalLearning.Feature.SignalEnhance import SignalEnhance

sns.set_style('darkgrid')


class TEST_SignalEnhance(unittest.TestCase):

    def setUp(self):
        self._random_matrix = np.random.normal(size=(10000, 1000))
        self._enhancer = SignalEnhance(matrix=self._random_matrix, band_width=0.01)

    def test_marcenko_pastur_pdf_against_kde_fit(self):

        eval, evec = self._enhancer.get_sorted_pca(np.corrcoef(self._random_matrix, rowvar=False))
        pdf_mp = self._enhancer._marcenko_pastur_pdf(sigma=1)
        pdf_kde = self._enhancer._fit_empirical_kde(lambds=np.diag(eval))

        sns.lineplot(x=pdf_mp.index, y=pdf_mp, label='MP pdf')
        sns.lineplot(x=pdf_kde.index, y=pdf_kde, label='KDE pdf')
        plt.show()
        plt.close()

        self.assertTrue(np.sum(np.square(pdf_mp - pdf_kde)).squeeze() <= 1e-5)

    def test_calibration(self):

        eval_max, sigma = self._enhancer.calibrate()
        self.assertAlmostEqual(sigma, 1, places=4)