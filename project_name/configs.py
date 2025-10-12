import os
from typing import Literal, Optional

from loguru import logger
from pydantic import BaseModel
import yaml


class ModelConfig(BaseModel):
    num_epoch: int = 3
    model_type: Literal["logreg", "random_forest", "decision_tree"]

    # Logistic Regression params
    C: Optional[float] = 1.0
    max_iter: Optional[int] = 100

    # Random Forest params
    n_estimators: Optional[int] = 100
    max_depth: Optional[int] = 10

    # Decision Tree params
    dt_max_depth: Optional[int] = 10
    criterion: Optional[str] = "gini"

    path_to_data: str = None
    random_state: int = 42
    test_size: float = 0.2
    target_column: str = 'label'

def load_config(path_to_config: str = None) -> ModelConfig:
    if path_to_config is None:
        path_to_config = os.getenv("PATH_TO_CONFIG")
    
    if path_to_config is None:
        return
    
    with open(path_to_config) as f:
        config_dict = yaml.safe_load(f)

    try:
        cfg = ModelConfig(**config_dict)
        return cfg
    except Exception:
        logger.error(f"Validation Error. Couldn load config from {path_to_config}, dict: {config_dict}")
        return None
