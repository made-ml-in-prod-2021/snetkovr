import logging

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from .encoder import MeanEncoder


def build_categorical_pipeline() -> Pipeline:
    categorical_pipeline = Pipeline(
        [
            ("mean", MeanEncoder()),
            ("impute", SimpleImputer(missing_values=np.nan, strategy="most_frequent")),
        ]
    )
    return categorical_pipeline


def build_numerical_pipeline() -> Pipeline:
    num_pipeline = Pipeline(
        [("impute", SimpleImputer(missing_values=np.nan, strategy="mean"))]
    )
    return num_pipeline


def make_features(transformer: ColumnTransformer, df: pd.DataFrame) -> pd.DataFrame:
    logging.info(f"Transform dataframe with shape {df.shape}")
    return pd.DataFrame(transformer.transform(df))


def build_transformer(params) -> ColumnTransformer:
    transformer = ColumnTransformer(
        [
            (
                "categorical_pipeline",
                build_categorical_pipeline(),
                list(params.categorical_features),
            ),
            (
                "numerical_pipeline",
                build_numerical_pipeline(),
                list(params.numerical_features),
            ),
        ]
    )
    return transformer


def extract_target(df: pd.DataFrame, params) -> pd.Series:
    logging.info(f"Extract target {params.target}")
    target = df[params.target]
    return target


def serialize_transformer(transformer: ColumnTransformer, output: str) -> str:
    logging.info(f"Save transformer into {output}")
    with open(output, "wb") as f:
        joblib.dump(transformer, f)
    return output
