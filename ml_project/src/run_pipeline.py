import json
import logging
import os
from typing import Dict, Tuple

import hydra
import pandas as pd
from hydra.utils import get_original_cwd
from omegaconf import DictConfig

from data import read_data, split_train_val_data
from features import (build_transformer, extract_target,
                      make_features, serialize_transformer)
from models.train import evaluate_model, predict_model, train_model, serialize_model

logger = logging.getLogger(__name__)


def get_original_cwd_hack():
    """Хак директории для правильного тестирования hydra."""
    try:
        return get_original_cwd()
    except ValueError:  # ValueError: get_original_cwd() must only be used after HydraConfig is initialized
        return os.getcwd()


def run(cfg: DictConfig) -> Tuple[str, Dict[str, float]]:
    logger.info(f"start train pipeline with params {cfg}")
    data = read_data(os.path.join(get_original_cwd_hack(), cfg.general.data_path))
    logger.info(f"data.shape is {data.shape}")
    train_df, val_df = split_train_val_data(
        data, cfg.validation
    )
    logger.info(f"Train shape {train_df.shape}\nValidation shape {val_df.shape}")
    logger.info(f"Process features...")
    transformer = build_transformer(cfg.general)
    train_target = extract_target(train_df, cfg.general)

    transformer.fit(train_df, train_target)
    train_features = make_features(transformer, train_df)

    model = train_model(
        train_features, train_target, cfg.model
    )
    logger.info("model trained!")

    val_features = make_features(transformer, val_df)
    val_target = extract_target(val_df, cfg.general)

    val_features_prepared = prepare_val_features_for_predict(
        train_features, val_features
    )
    logger.info(f"val_features.shape is {val_features_prepared.shape}")
    predicts = predict_model(
        model,
        val_features_prepared,
    )

    metrics = evaluate_model(
        predicts,
        val_target,
        cfg.general.threshold,
    )
    logger.info(metrics)

    with open(os.path.join(get_original_cwd_hack(), cfg.general.metric_path), "w+") as metric_file:
        json.dump(metrics, metric_file)
    logger.info(f"metrics is {metrics}")

    path_to_transformers = serialize_transformer(
        transformer,
        os.path.join(get_original_cwd_hack(), cfg.general.transformer_path),
    )

    path_to_model = serialize_model(
        model,
        os.path.join(get_original_cwd_hack(), cfg.general.models_path),
    )
    return path_to_model, metrics


def prepare_val_features_for_predict(
    train_features: pd.DataFrame, val_features: pd.DataFrame
):
    train_features, val_features = train_features.align(
        val_features, join="left", axis=1
    )
    val_features = val_features.fillna(0)
    return val_features


@hydra.main(config_path='ml_project/configs', config_name='config')
def main(cfg: DictConfig):
    run(cfg)


if __name__ == '__main__':
    main()