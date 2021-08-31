import pandas as pd
from typing import Tuple, Union


class Preprocessor:

    @staticmethod
    def standardize(df: Union[pd.DataFrame, pd.Series]) -> Tuple[Union[pd.DataFrame, pd.Series],
                                                                 Union[pd.Series, float],
                                                                 Union[pd.Series, float]]:
        """
        Standardize dataframe/series to zero mean and unit variance

        :param df: Union[pd.DataFrame, pd.Series], n * p where n is #samples and p is #features
        """

        mean, std = df.mean(axis=0), df.std(axis=0)
        return (df - mean) / std, mean, std
