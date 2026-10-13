"""Backward-compatible entry point:  python load_predict.py

Equivalent to:  python -m crowd_anomaly predict --source test.avi --model models/model.joblib
Any arguments you pass override the defaults (see `python -m crowd_anomaly predict -h`).
"""
import sys
from pathlib import Path

from crowd_anomaly.cli import main

if __name__ == "__main__":
    argv = sys.argv[1:]
    if not argv:
        argv = ["--source", "test.avi", "--model", "models/model.joblib"]
        if not Path("models/model.joblib").exists():
            sys.exit("No trained model at models/model.joblib. Train one first:\n"
                     "  python -m crowd_anomaly train --data-root <UCSD folder> --subset ped1 "
                     "--model-out models/model.joblib")
    sys.exit(main(["predict", *argv]))
