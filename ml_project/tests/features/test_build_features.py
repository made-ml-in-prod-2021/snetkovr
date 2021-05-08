import pandas as pd
from sklearn.compose import ColumnTransformer

from src.features.build_features import build_transformer, extract_target, make_features
from tests.utils import create_fake_dataset


def test_build_transformer(general_config):
    transformer = build_transformer(general_config)
    assert isinstance(transformer, ColumnTransformer)


def test_extract_target(general_config):
    dataset = create_fake_dataset()
    target = extract_target(dataset, general_config)
    assert isinstance(target, pd.Series)
    assert target.equals(dataset[general_config.target])


def test_make_features(general_config):
    dataset = create_fake_dataset()
    transformer = build_transformer(general_config)
    target = extract_target(dataset, general_config)
    transformer.fit(dataset, target)

    features = make_features(transformer, dataset)
    assert isinstance(features, pd.DataFrame)
    assert features.shape[1] == dataset.shape[1] - 1 # target_column
