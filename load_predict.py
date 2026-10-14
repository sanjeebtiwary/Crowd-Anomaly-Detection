"""Convenience entry point:  python load_predict.py [predict options]

Equivalent to:  python -m crowd_anomaly predict --source <video> --model <model>
With no arguments it scores ``test.avi`` using the model in ``MODEL_PATH`` (default
``artifacts/ped1.joblib``). Any arguments you pass replace the defaults
(see ``python -m crowd_anomaly predict -h``).
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))  # works even before `pip install -e .`

from crowd_anomaly.cli import main  # noqa: E402

if __name__ == "__main__":
    argv = sys.argv[1:]
    if not argv:
        model = Path(os.getenv("MODEL_PATH", ROOT / "artifacts" / "ped1.joblib"))
        if not model.is_file():
            sys.exit(f"No trained model at {model}. Train one first:\n"
                     "  python -m crowd_anomaly train --subset ped1 --model-out artifacts/ped1.joblib")
        argv = ["--source", "test.avi", "--model", str(model)]
    sys.exit(main(["predict", *argv]))
