import unittest
import warnings
import pandas as pd
from sklearn import datasets
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from StatisticalLearning.LinearModel.Classification import LinearDiscriminantAnalysis


class TEST_LinearDiscriminantAnalysis(unittest.TestCase):

    def setUp(self):
        warnings.filterwarnings('ignore')

    def test_binary_classification(self):

        X, y = datasets.load_breast_cancer(return_X_y=True)
        X, y = pd.DataFrame(X), pd.Series(y)

        lda = LinearDiscriminantAnalysis(n_components=5).fit(X, y)
        sklearn_lda = LDA().fit(X, y)

        pred_lda = lda.predict(X)
        pred_sklearn = sklearn_lda.predict(X)

        self.assertTrue((pred_lda - pd.Series(pred_sklearn.squeeze())).abs().sum() == 0)

    def test_multiple_classification(self):

        X, y = datasets.load_digits(return_X_y=True)
        X, y = pd.DataFrame(X), pd.Series(y)

        lda = LinearDiscriminantAnalysis(n_components=5).fit(X, y)
        sklearn_lda = LDA().fit(X, y)

        pred_lda = lda.predict(X)
        pred_sklearn = sklearn_lda.predict(X)

        self.assertTrue((pred_lda - pd.Series(pred_sklearn.squeeze())).abs().sum() == 0)

    def test_transform(self):

        # TODO: Compare discriminant direction with sklearn
        X, y = datasets.load_digits(return_X_y=True)
        X, y = pd.DataFrame(X), pd.Series(y)

        lda = LinearDiscriminantAnalysis(n_components=5).fit(X, y)
        X_lda_tran = lda.transform(X.copy(deep=True)).applymap(lambda num: num.real)

        self.assertTrue(X_lda_tran.shape[0] == X.shape[0] and X_lda_tran.shape[1] == 5)


