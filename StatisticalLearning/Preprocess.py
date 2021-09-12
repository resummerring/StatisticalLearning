import pandas as pd
from typing import Tuple, Union, List


class Preprocessor:

    # ====================
    #  Public
    # ====================

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

    @staticmethod
    def add_dummy_variables(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        """
        Transform categorical variable with k levels into k - 1 dummy variables

        :param df: pd.DataFrame, n * p data matrix where n = #samples and p = #features
        :param cols: List[str], column names of categorical variables in the data matrix
        """

        all_columns = df.columns

        for col in cols:

            if col not in all_columns:
                raise ValueError(f"Column {col} doesn't exist")

            levels = df[col].unique()

            # The last state will be left as baseline
            for level in levels[:-1]:
                df[f'{col}_{level}'] = df[col].apply(lambda state: 1 if state == level else 0)

            df.drop(col, axis=1, inplace=True)

        return df





