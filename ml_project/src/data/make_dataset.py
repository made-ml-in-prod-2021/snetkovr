from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


def read_data(path: str) -> pd.DataFrame:
    data = pd.read_csv(path)
    return data


def split_train_val_data(
        data: pd.DataFrame,
        params
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_data, validation_data = train_test_split(
        data,
        test_size=params.val_size,
    )
    return train_data, validation_data