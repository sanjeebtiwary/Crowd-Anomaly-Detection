"""Optical-flow feature extraction.

For every pair of consecutive frames we compute dense Farneback flow and pool it on a
``grid_rows x grid_cols`` grid. Per cell we keep
  basic: mean dx, mean dy, mean magnitude                       -> 3 numbers
  rich : basic + max magnitude + magnitude std                  -> 5 numbers
giving an array of shape (T-1, rows, cols, n_features) per clip.
Extracted features can be cached on disk (they are the slow part).
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from joblib import Parallel, delayed

from .config import Config
from .data import frame_size_of, iter_frames, prepare_frame

log = logging.getLogger(__name__)

N_FEATURES = {"basic": 3, "rich": 5}


def _block_max(a: np.ndarray, rows: int, cols: int) -> np.ndarray:
    H, W = a.shape
    ye = np.linspace(0, H, rows + 1).astype(int)
    xe = np.linspace(0, W, cols + 1).astype(int)
    ye[1:] = np.maximum(ye[1:], ye[:-1] + 1)  # never empty
    xe[1:] = np.maximum(xe[1:], xe[:-1] + 1)
    out = np.maximum.reduceat(a, ye[:-1], axis=0)
    out = np.maximum.reduceat(out, xe[:-1], axis=1)
    return out[:rows, :cols]


def pool_flow(flow: np.ndarray, rows: int, cols: int, feature_set: str = "basic") -> np.ndarray:
    """(H, W, 2) flow -> (rows, cols, n_features); see module docstring."""
    mag = np.linalg.norm(flow, axis=2, keepdims=True)
    stacked = np.concatenate([flow, mag, mag ** 2], axis=2).astype(np.float32)
    cells = cv2.resize(stacked, (cols, rows), interpolation=cv2.INTER_AREA)  # means
    basic = cells[..., :3]
    if feature_set == "basic":
        return basic
    std = np.sqrt(np.maximum(cells[..., 3] - cells[..., 2] ** 2, 0.0))
    mx = _block_max(mag[..., 0], rows, cols)
    return np.concatenate([basic, mx[..., None], std[..., None]], axis=-1).astype(np.float32)


def extract_features(frames: Iterable[np.ndarray], cfg: Config, frame_size: Tuple[int, int]) -> np.ndarray:
    """Flow features for a stream of raw frames. Returns (T-1, rows, cols, n_features)."""
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
            feats.append(pool_flow(flow, cfg.grid_rows, cfg.grid_cols, cfg.feature_set))
        prev = cur
    if not feats:
        raise ValueError("Need at least two frames to compute optical flow")
    return np.stack(feats)


def _cache_path(cache_dir: Path, source: Path, cfg: Config, frame_size) -> Path:
    """Key = source identity (path, size, mtime) + every setting that affects features."""
    source = Path(source)
    if source.is_dir():
        files = [f for f in source.iterdir() if f.is_file()]
        ident = (len(files), max((f.stat().st_mtime_ns for f in files), default=0),
                 sum(f.stat().st_size for f in files))
    else:
        st = source.stat()
        ident = (st.st_size, st.st_mtime_ns)
    key = repr((str(source.resolve()), ident, tuple(frame_size), cfg.blur_ksize, cfg.flow_pyr_scale,
                cfg.flow_levels, cfg.flow_winsize, cfg.flow_iterations, cfg.flow_poly_n,
                cfg.flow_poly_sigma, cfg.grid_rows, cfg.grid_cols, cfg.feature_set))
    return Path(cache_dir) / f"{source.name}_{hashlib.sha1(key.encode()).hexdigest()[:16]}.npy"


def clip_features(source, cfg: Config, frame_size: Tuple[int, int],
                  cache_dir: Optional[Path] = None) -> np.ndarray:
    """Features for one clip folder or video file (optionally cached on disk)."""
    source = Path(source)
    if cache_dir is not None:
        cp = _cache_path(cache_dir, source, cfg, frame_size)
        if cp.exists():
            try:
                return np.load(cp)
            except Exception:  # corrupt/partial cache file -> recompute
                log.warning("Ignoring unreadable cache file %s", cp)
    feats = extract_features(iter_frames(source), cfg, frame_size)
    if cache_dir is not None:
        cp.parent.mkdir(parents=True, exist_ok=True)
        tmp = cp.with_suffix(".tmp.npy")
        np.save(tmp, feats)
        tmp.replace(cp)  # atomic: no half-written cache files
    return feats


def extract_many(
    sources: Sequence[Path], cfg: Config, frame_size: Optional[Tuple[int, int]] = None,
    n_jobs: int = 1, progress: bool = True, cache_dir: Optional[Path] = None,
) -> Tuple[List[np.ndarray], Tuple[int, int]]:
    """Features for many clips (optionally parallel / cached). Returns (list, frame_size)."""
    if frame_size is None:
        frame_size = tuple(cfg.frame_size) if cfg.frame_size else frame_size_of(sources[0])
    items = list(sources)
    try:
        from tqdm import tqdm
        items_iter = tqdm(items, desc="Extracting flow", unit="clip", disable=not progress)
    except ImportError:  # tqdm is optional
        items_iter = items
    if n_jobs == 1:
        out = [clip_features(s, cfg, frame_size, cache_dir) for s in items_iter]
    else:
        out = Parallel(n_jobs=n_jobs)(delayed(clip_features)(s, cfg, frame_size, cache_dir)
                                      for s in items_iter)
    return out, frame_size
