import pandas as pd


class BaggingRegressor:

    def __init__(self, estimator: object, n_estimator: int):
        self._n_estimator = n_estimator

    def fit(self, X: pd.DataFrame, y: pd.Series):
        return