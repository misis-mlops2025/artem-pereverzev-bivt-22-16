from pydantic import BaseModel
from typing import Literal, Optional
import os
import yaml
from loguru import logger

class ModelConfig(BaseModel):
    num_epoch: int = 1
    model_type: Literal["logreg", "random_forest", "decision_tree"]

    # Logistic Regression params
    C: Optional[float] = 1.0
    max_iter: Optional[int] = 100

    # Random Forest params
    n_estimators: Optional[int] = 100
    max_depth: Optional[int] = None

    # Decision Tree params
    dt_max_depth: Optional[int] = None
    criterion: Optional[str] = "gini"

    path_to_data: str = None


def load_config(path_to_config: str = None) -> ModelConfig:
    if path_to_config is None:
        path_to_config = os.getenv("PATH_TO_CONFIG")
    
    if path_to_config is None:
        return
    
    with open("config.yaml") as f:
        config_dict = yaml.safe_load(f)

    try:
        cfg = ModelConfig(**config_dict)
    except Exception as e:
        logger.error(f"Validation Error. Couldn load config from {path_to_config}")
    return cfg
