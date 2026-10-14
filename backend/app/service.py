from __future__ import annotations

from functools import lru_cache

from crowd_anomaly import CrowdAnomalyModel

from .settings import get_settings


@lru_cache(maxsize=1)
def get_model() -> CrowdAnomalyModel:
    """Load the trained model once and reuse it for every request."""
    path = get_settings().model_path
    if not path.is_file():
        raise FileNotFoundError(f"Model artifact not found: {path}")
    return CrowdAnomalyModel.load(path)
