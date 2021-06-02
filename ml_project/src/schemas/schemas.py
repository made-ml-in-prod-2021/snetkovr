from dataclasses import dataclass, field
from typing import List, Any, Optional


@dataclass
class GeneralConfig:
    random_state: int = 42
    categorical_features: List[str] = field(default_factory=list)
    numerical_features: List[str] = field(default_factory=list)
    target: str = "rmd"
    threshold: float = 0.5
    data_path: str = "src/data/raw/heart.csv"
    metric_path: str = ""
    models_path: str = ""
    transformer_path: str = ""


@dataclass
class ValidationConfig:
    val_size: float = None


@dataclass
class RFConfig:
    max_depth: Optional[int] = None
    _target_: str = 'sklearn.ensemble.RandomForestClassifier'
    n_estimators: int = 100
    random_state: int = 42


@dataclass
class LogregConfig:
    _target_: str = 'sklearn.linear_model.LogisticRegression'
    penalty: str = 'l1'
    solver: str = 'liblinear'
    C: float = 1.0
    random_state: int = 42


@dataclass
class Config:
    model: Any = RFConfig()
    validation: ValidationConfig = ValidationConfig()
    general: GeneralConfig = GeneralConfig()
