import logging
import joblib
from typing import Dict, Union

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, recall_score, precision_score, f1_score
from hydra.utils import instantiate


def train_model(
    features: pd.DataFrame, target: pd.Series, model_cfg
):
    model = instantiate(model_cfg)
    model.fit(features, target)
    return model


def predict_model(
    model, features: pd.DataFrame
) -> np.ndarray:
    logging.info("Model predict")
    predicts = model.predict_proba(features)
    return predicts


def evaluate_model(
    proba: np.ndarray, target: pd.Series, threshold, digits=4
) -> Dict[str, float]:
    logging.info("Metric calculation")
    predicts = proba[:, 1] > threshold
    return {
        "roc-auc": round(roc_auc_score(target, proba[:, 1]), digits),
        "recall": round(recall_score(target, predicts), digits),
        "precision": round(precision_score(target, predicts), digits),
        "f1": round(f1_score(target, predicts), digits),
    }


def serialize_model(model, output: str) -> str:
    logging.info(f"Save model into {output}")
    with open(output, "wb") as f:
        joblib.dump(model, f)
    return output
