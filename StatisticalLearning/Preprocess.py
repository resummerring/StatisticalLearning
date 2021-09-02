import pandas as pd
from typing import Tuple, Union


class Preprocessor:

    @staticmethod
    def normalize(df: Union[pd.DataFrame, pd.Series],
                  scale_mean: bool = True,
                  scale_std: bool = True) -> Tuple[Union[pd.DataFrame, pd.Series],
                                                   Union[pd.Series, float],
                                                   Union[pd.Series, float]]:
        """
        Normalize dataframe/series to zero mean and unit variance

        :param df: Union[pd.DataFrame, pd.Series], n * p where n is #samples and p is #features
        :param scale_mean: bool, whether or not to center mean zero
        :param scale_std: bool, whether or not to scale unit variance
        """

        df_out = df.copy(deep=True)
        mean, std = df.mean(axis=0), df.std(axis=0)
        df_out = df_out - mean if scale_mean else df_out
        df_out = df_out / std if scale_std else df_out
        return df_out, mean, std

    @staticmethod
    def standardize(df: Union[pd.DataFrame, pd.Series]) -> Tuple[Union[pd.DataFrame, pd.Series],
                                                                 Union[pd.Series, float],
                                                                 Union[pd.Series, float]]:
        """
        Standardize dataframe/series to lie between [0, 1]

        :param df: Union[pd.DataFrame, pd.Series], n * p where n is #samples and p is #features
        """

        df_out = df.copy(deep=True)
        df_max, df_min = df_out.max(axis=0), df_out.min(axis=0)
        df_out = (df_out - df_min) / (df_max - df_min)
        return df_out, df_max, df_min
