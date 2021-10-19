import unittest
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from StatisticalLearning.Feature.SignalEnhance import SignalEnhance

sns.set_style('darkgrid')


class TEST_SignalEnhance(unittest.TestCase):

    def setUp(self):

        def get_random_cov(n_cols: int, n_factors: int):
            """
            Generate random covariance matrix
            """

            w = np.random.normal(size=(n_cols, n_factors))
            cov = np.dot(w, w.T) + np.diag(np.random.uniform(size=n_cols))
            return cov

        self._get_random_cov = get_random_cov

    def test_random_matrix(self):

        random_matrix = np.random.normal(size=(10000, 1000))
        enhancer = SignalEnhance(matrix=random_matrix, band_width=0.01)

        eval, evec = enhancer.get_sorted_pca(np.corrcoef(random_matrix, rowvar=False))
        pdf_mp = enhancer._marcenko_pastur_pdf(sigma=1)
        pdf_kde = enhancer._fit_empirical_kde(lambds=np.diag(eval))

        sns.lineplot(x=pdf_mp.index, y=pdf_mp, label='MP pdf')
        sns.lineplot(x=pdf_kde.index, y=pdf_kde, label='KDE pdf')
        plt.show()
        plt.close()

        self.assertTrue(np.sum(np.square(pdf_mp - pdf_kde)).squeeze() <= 1e-5)

        eval_max, sigma = enhancer._fit_max_eval()
        self.assertAlmostEqual(sigma, 1, places=4)

        enhancer.denoise(method='Constant Residual')
        enhancer.denoise(method='Target Shrinkage', alpha=0)

    def test_matrix_with_signal(self):

        alpha, n_cols, n_factor, q = 0.995, 1000, 100, 10
        cov = np.cov(np.random.normal(size=(n_cols * q, n_cols)), rowvar=False)
        cov = alpha * cov + (1 - alpha) * self._get_random_cov(n_cols, n_factor)
        corr = SignalEnhance.cov_to_corr(cov)

        enhancer = SignalEnhance(matrix=cov, is_cov=True, q=q, band_width=0.01)
        _, sigma = enhancer._fit_max_eval()

        eval, evec = enhancer.get_sorted_pca(corr)
        pdf_mp = enhancer._marcenko_pastur_pdf(sigma=sigma)
        pdf_kde = enhancer._fit_empirical_kde(lambds=np.diag(eval))

        sns.lineplot(x=pdf_mp.index, y=pdf_mp, label='MP pdf')
        sns.lineplot(x=pdf_kde.index, y=pdf_kde, label='KDE pdf')
        plt.show()
        plt.close()

        original_eval = enhancer._original_corr_eval
        original_eval = np.diag(original_eval)

        enhancer.denoise(method='Constant Residual')
        denoised_eval, _ = SignalEnhance.get_sorted_pca(enhancer._denoised_corr)
        denoised_eval = np.diag(denoised_eval)
        sns.lineplot(x=range(len(original_eval)), y=original_eval, label='Before denoise')
        sns.lineplot(x=range(len(denoised_eval)), y=denoised_eval, label='After denoise', linestyle='--')
        plt.show()
        plt.close()

        enhancer.denoise(method='Target Shrinkage', alpha=0)
        denoised_eval, _ = SignalEnhance.get_sorted_pca(enhancer._denoised_corr)
        denoised_eval = np.diag(denoised_eval)
        sns.lineplot(x=range(len(original_eval)), y=original_eval, label='Before denoise')
        sns.lineplot(x=range(len(denoised_eval)), y=denoised_eval, label='After denoise', linestyle='--')
        plt.show()
        plt.close()
