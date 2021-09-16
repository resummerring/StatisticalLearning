import unittest
import warnings
import numpy as np
import pandas as pd
from sklearn import datasets
from StatisticalLearning.Feature.Kernel import Kernel
from StatisticalLearning.LinearModel.SVMClassification import SVMClassifier


class TEST_SVMClassifier(unittest.TestCase):

    def setUp(self):

        warnings.filterwarnings('ignore')

        X, y = datasets.load_breast_cancer(return_X_y=True)
        self._X, self._y = pd.DataFrame(X), pd.Series(y).apply(lambda label: label if label == 1 else -1)

    def test_linear_kernel_synthetic(self):

        X = pd.DataFrame(np.array([[3, 4], [1, 4], [2, 3], [6, -1], [7, -1], [5, -3], [2, 4]]))
        y = pd.Series(np.array([-1, -1, -1, 1, 1, 1, 1]))

        svm = SVMClassifier(C=10, kernel=Kernel.linear).fit(X, y)
        pred = svm.predict(X)

        self.assertTrue(pred.equals(pd.Series([-1, -1, -1, 1, 1, 1, -1])))
        self.assertTrue(svm.sv_index == [0, 2, 3, 6])
