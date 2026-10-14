"""Runtime settings, read from environment variables (and a local ``.env`` if present)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

try:  # optional: python-dotenv is part of the `api` extra
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:  # pragma: no cover
    pass


def _model_path() -> Path:
    p = Path(os.getenv("MODEL_PATH", "artifacts/ped1.joblib")).expanduser()
    return p if p.is_absolute() else ROOT / p


@dataclass(frozen=True)
class Settings:
    model_path: Path
    cors_origins: tuple
    max_upload_mb: int
    min_segment_len: int


def get_settings() -> Settings:
    origins = tuple(x.strip() for x in os.getenv("CORS_ORIGINS", "*").split(",") if x.strip())
    return Settings(
        model_path=_model_path(),
        cors_origins=origins or ("*",),
        max_upload_mb=int(os.getenv("MAX_UPLOAD_MB", "100")),
        min_segment_len=int(os.getenv("MIN_SEGMENT_LEN", "3")),
    )
