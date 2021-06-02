from typing import Type, Union

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class MeanEncoder(BaseEstimator, TransformerMixin):
    """
    Mean encoder with smoothing regularization
    """

    def __init__(self, alpha: int = 20):
        super().__init__()
        self.alpha = alpha
        self.cols_values = dict()
        self.global_mean = None

    def fit(self, x: pd.DataFrame, y: pd.Series):
        """
        Computing means and storing in class attributes
        :param x: data to fit
        :param y: labels of data
        :return: fitted exemplar of class
        """
        self.global_mean = y.mean()
        for col in x.columns:
            target_stat = y.groupby(x[col]).agg(["sum", "count"])
            col_dict = (
                (target_stat["sum"] + self.global_mean * self.alpha)
                / (target_stat["count"] + self.alpha)
            ).to_dict()
            self.cols_values[col] = col_dict
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transforming data categories to computed means
        :param df: data to transform
        :return: transformed data
        """
        for column in df.columns:
            df[column] = df[column].map(self.cols_values[column]).fillna(self.global_mean)
        return df


