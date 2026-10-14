"""Download the UCSD Anomaly Dataset straight from Kaggle with ``kagglehub``.

Dataset page: https://www.kaggle.com/datasets/orvile/ucsd-anomaly-dataset/data
kagglehub caches the files (default ``~/.cache/kagglehub``), so repeat calls are instant.
If Kaggle asks for credentials, set KAGGLE_USERNAME / KAGGLE_KEY or run ``kagglehub.login()``.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

log = logging.getLogger(__name__)

KAGGLE_HANDLE = "orvile/ucsd-anomaly-dataset"


def download_dataset(handle: str = KAGGLE_HANDLE, force: bool = False) -> Path:
    """Download (or reuse the cached copy of) the dataset and return its local folder."""
    try:
        import kagglehub
    except ImportError as exc:
        raise RuntimeError("kagglehub is not installed. Run: pip install kagglehub") from exc
    path = Path(kagglehub.dataset_download(handle, force_download=force))
    log.info("Path to dataset files: %s", path)
    return path


def resolve_data_root(data_root: Optional[Union[str, Path]] = None) -> Path:
    """Use ``data_root`` if given, otherwise download the dataset from Kaggle."""
    return Path(data_root) if data_root else download_dataset()
