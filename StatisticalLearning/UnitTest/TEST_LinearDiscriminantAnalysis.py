import unittest
import warnings
import pandas as pd
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from StatisticalLearning.LinearModel.Classification import LinearDiscriminantAnalysis


class TEST_LinearDiscriminantAnalysis(unittest.TestCase):

    def setUp(self):
        warnings.filterwarnings('ignore')

    def test_binary_classification(self):

        X, y = datasets.load_breast_cancer(return_X_y=True)
        X, y = pd.DataFrame(X), pd.Series(y)

        lr = LogisticRegression(fit_intercept=True).fit(X, y)
        lda = LinearDiscriminantAnalysis().fit(X, y)

        pred_lr = lr.predict(X)
        pred_lda = lda.predict(X)
        self.assertTrue((pred_lda - pd.Series(pred_lr.squeeze())).abs().sum() <= 30)


