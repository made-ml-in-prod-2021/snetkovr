import os
import yaml
from typing import cast

from hydra.experimental import compose, initialize

from src.run_pipeline import run
from src.schemas.schemas import Config


def test_train_pipeline():
    with initialize(config_path="../configs", job_name="test_app"):
        config = compose(config_name="config")
        config = cast(Config, config)
        expected_model_path = config.general.models_path
        expected_experiment_metrics_path = config.general.metric_path
        run(config)
    for path in [
        expected_model_path,
        expected_experiment_metrics_path,
    ]:
        assert os.path.exists(path)
    with open(expected_experiment_metrics_path, "r") as metrics_file:
        metrics = yaml.load(metrics_file, Loader=yaml.FullLoader)
        assert metrics["recall"] > 0