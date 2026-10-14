from functools import lru_cache
import os
from pathlib import Path
from src import CrowdAnomalyModel

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = ROOT / "artifacts" / "ped1.joblib"

@lru_cache(maxsize=1)
def get_model() -> CrowdAnomalyModel:
    model_path = Path(os.getenv("MODEL_PATH", str(DEFAULT_MODEL))).expanduser()
    if not model_path.is_file():
        raise FileNotFoundError(f"Model artifact not found: {model_path}")
    return CrowdAnomalyModel.load(model_path)
