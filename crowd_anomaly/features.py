"""Optical-flow feature extraction.

For every pair of consecutive frames we compute dense Farneback flow and pool it
on a ``grid_rows x grid_cols`` grid. Each cell keeps three numbers:
mean dx, mean dy and mean flow magnitude -> array of shape (T-1, rows, cols, 3).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from joblib import Parallel, delayed

from .config import Config
from .data import frame_size_of, iter_frames, prepare_frame

log = logging.getLogger(__name__)

N_CELL_FEATURES = 3


def pool_flow(flow: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """(H, W, 2) flow -> (rows, cols, 3) of [mean dx, mean dy, mean magnitude]."""
    mag = np.linalg.norm(flow, axis=2, keepdims=True)
    stacked = np.concatenate([flow, mag], axis=2).astype(np.float32)
    return cv2.resize(stacked, (cols, rows), interpolation=cv2.INTER_AREA)


def extract_features(frames: Iterable[np.ndarray], cfg: Config, frame_size: Tuple[int, int]) -> np.ndarray:
    """Flow features for a stream of raw frames. Returns (T-1, rows, cols, 3)."""
    feats: List[np.ndarray] = []
    prev: Optional[np.ndarray] = None
    for raw in frames:
        cur = prepare_frame(raw, frame_size, cfg.blur_ksize)
        if prev is not None:
            flow = cv2.calcOpticalFlowFarneback(
                prev, cur, None,
                cfg.flow_pyr_scale, cfg.flow_levels, cfg.flow_winsize,
                cfg.flow_iterations, cfg.flow_poly_n, cfg.flow_poly_sigma, 0,
            )
            feats.append(pool_flow(flow, cfg.grid_rows, cfg.grid_cols))
        prev = cur
    if not feats:
        raise ValueError("Need at least two frames to compute optical flow")
    return np.stack(feats)


def clip_features(source: Path, cfg: Config, frame_size: Tuple[int, int]) -> np.ndarray:
    """Features for one clip folder or video file."""
    return extract_features(iter_frames(source), cfg, frame_size)


def extract_many(
    sources: Sequence[Path], cfg: Config, frame_size: Optional[Tuple[int, int]] = None,
    n_jobs: int = 1, progress: bool = True,
) -> Tuple[List[np.ndarray], Tuple[int, int]]:
    """Features for many clips (optionally in parallel). Returns (list, frame_size)."""
    if frame_size is None:
        frame_size = tuple(cfg.frame_size) if cfg.frame_size else frame_size_of(sources[0])
    items = list(sources)
    try:
        from tqdm import tqdm
        items_iter = tqdm(items, desc="Extracting flow", unit="clip", disable=not progress)
    except ImportError:  # tqdm is optional
        items_iter = items
    if n_jobs == 1:
        out = [clip_features(s, cfg, frame_size) for s in items_iter]
    else:
        out = Parallel(n_jobs=n_jobs)(delayed(clip_features)(s, cfg, frame_size) for s in items_iter)
    return out, frame_size
