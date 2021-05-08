import pytest
import os
import pandas as pd
import pickle
from typing import List

from tests.utils import create_fake_dataset
from src.schemas.schemas import Config, GeneralConfig, ValidationConfig
from hydra.experimental import compose, initialize
from typing import cast


@pytest.fixture()
def src_path() -> str:
    return "src"


@pytest.fixture()
def tests_path() -> str:
    return "tests"


@pytest.fixture()
def dataset_path() -> str:
    path = os.path.join(os.path.dirname(__file__), "data_sample.zip")
    data = create_fake_dataset()
    data.to_csv(path, compression="zip")
    return path


@pytest.fixture()
def general_config():
    return GeneralConfig(
        categorical_features=["thal", "slope", "sex", "cp", "fbs", "restecg", "exang", "ca"],
        numerical_features=["age", "trestbps", "chol", "thalach", "oldpeak"],
        target="target",
    )


@pytest.fixture()
def validation_config():
    return ValidationConfig(val_size=0.1)


@pytest.fixture()
def model_dir() -> str:
    return os.path.join(os.path.dirname(__file__), "model_test_dir")


