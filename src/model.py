"""Normality model: standardiser + Gaussian mixture over per-cell motion.

Every grid cell of every training frame is one sample ``[dx, dy, magnitude, (max, std), y, x]``
(position included, so the model knows that motion near the camera looks different
from motion far away). The anomaly score of a cell is its negative log-likelihood
under the mixture; the frame score is the mean of the ``top_k`` worst cells,
smoothed over time.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import joblib
import numpy as np
from scipy.ndimage import gaussian_filter1d
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from .config import Config

log = logging.getLogger(__name__)
FORMAT_VERSION = 1


def cell_samples(feats: np.ndarray) -> np.ndarray:
    """(T, R, C, F) -> (T, R, C, F+2) by appending normalised (y, x) cell centres."""
    T, R, C, _ = feats.shape
    ys, xs = np.meshgrid((np.arange(R) + 0.5) / R, (np.arange(C) + 0.5) / C, indexing="ij")
    pos = np.broadcast_to(np.stack([ys, xs], axis=-1), (T, R, C, 2))
    return np.concatenate([feats, pos], axis=-1).astype(np.float64)


def align_to_frames(per_pair: np.ndarray) -> np.ndarray:
    """Flow gives T-1 values for T frames; repeat the first so index i <-> frame i."""
    return np.concatenate([per_pair[:1], per_pair], axis=0)


def flags_to_segments(flags: np.ndarray, min_len: int = 1) -> List[Tuple[int, int]]:
    """Runs of True as inclusive (start, end) frame indices, dropping short runs."""
    segs, start = [], None
    for i, f in enumerate(np.append(flags.astype(bool), False)):
        if f and start is None:
            start = i
        elif not f and start is not None:
            if i - start >= min_len:
                segs.append((start, i - 1))
            start = None
    return segs


class CrowdAnomalyModel:
    def __init__(self, cfg: Optional[Config] = None):
        self.cfg = (cfg or Config()).validate()
        self.frame_size: Optional[Tuple[int, int]] = None
        self.scaler: Optional[StandardScaler] = None
        self.gmm: Optional[GaussianMixture] = None
        self.threshold_: Optional[float] = None
        self.cell_threshold_: Optional[float] = None
        self.n_components_: Optional[int] = None

    # ------------------------------------------------------------------ fit
    def _check_fitted(self):
        if self.gmm is None or self.scaler is None:
            raise RuntimeError("Model is not fitted")

    def _stack(self, clips: Sequence[np.ndarray]) -> np.ndarray:
        return np.concatenate([cell_samples(f).reshape(-1, f.shape[-1] + 2) for f in clips], axis=0)

    def fit(
        self,
        train_feats: Sequence[np.ndarray],
        frame_size: Tuple[int, int],
        val_feats: Optional[Sequence[np.ndarray]] = None,
    ) -> "CrowdAnomalyModel":
        """Fit on normal clips. If ``val_feats`` is None, ``val_fraction`` of the
        clips (by clip, never by frame) are held out to calibrate thresholds."""
        cfg = self.cfg
        train_feats = list(train_feats)
        rng = np.random.default_rng(cfg.seed)
        if val_feats is None:
            val_feats = []
            n_val = int(round(len(train_feats) * cfg.val_fraction))
            if cfg.val_fraction > 0 and len(train_feats) >= 3:
                n_val = max(n_val, 1)
            if n_val and len(train_feats) - n_val >= 1:
                order = rng.permutation(len(train_feats))
                val_idx = set(order[:n_val].tolist())
                val_feats = [f for i, f in enumerate(train_feats) if i in val_idx]
                train_feats = [f for i, f in enumerate(train_feats) if i not in val_idx]
        val_feats = list(val_feats)
        if not val_feats:
            log.warning("No held-out clips: thresholds calibrated on training data (optimistic).")
            val_feats = train_feats

        X = self._stack(train_feats)
        if len(X) > cfg.max_train_samples:
            X = X[rng.choice(len(X), cfg.max_train_samples, replace=False)]
        log.info("Fitting on %d cell samples from %d clips (%d held out)",
                 len(X), len(train_feats), len(val_feats))

        self.scaler = StandardScaler().fit(X)
        Xs = self.scaler.transform(X)

        k = cfg.n_components
        if cfg.auto_components:
            k = self._select_components(Xs)
        self.n_components_ = k
        self.gmm = GaussianMixture(
            n_components=k, covariance_type=cfg.covariance_type,
            reg_covar=cfg.reg_covar, max_iter=300, n_init=2, random_state=cfg.seed,
        ).fit(Xs)
        if not self.gmm.converged_:
            log.warning("GMM did not converge; consider more data or fewer components.")
        self.frame_size = tuple(frame_size)

        # Calibrate thresholds on held-out normal clips.
        frame_scores = np.concatenate([self.frame_scores(f) for f in val_feats])
        cell_scores = np.concatenate([self.cell_scores(f).ravel() for f in val_feats])
        self.threshold_ = float(np.percentile(frame_scores, cfg.threshold_percentile))
        self.cell_threshold_ = float(np.percentile(cell_scores, cfg.cell_threshold_percentile))
        log.info("Components=%d  frame threshold=%.3f  cell threshold=%.3f",
                 k, self.threshold_, self.cell_threshold_)
        return self

    def _select_components(self, Xs: np.ndarray) -> int:
        cfg = self.cfg
        rng = np.random.default_rng(cfg.seed)
        sub = Xs[rng.choice(len(Xs), min(len(Xs), 50_000), replace=False)]
        best_k, best_bic = None, np.inf
        for k in cfg.component_candidates:
            g = GaussianMixture(k, covariance_type=cfg.covariance_type,
                                reg_covar=cfg.reg_covar, random_state=cfg.seed).fit(sub)
            bic = g.bic(sub)
            log.info("  k=%d  BIC=%.1f", k, bic)
            if bic < best_bic:
                best_k, best_bic = k, bic
        return int(best_k)

    # -------------------------------------------------------------- scoring
    def cell_scores(self, feats: np.ndarray) -> np.ndarray:
        """Negative log-likelihood per cell: (T, R, C). Higher = more unusual."""
        self._check_fitted()
        T, R, C, _ = feats.shape
        Xs = self.scaler.transform(cell_samples(feats).reshape(-1, feats.shape[-1] + 2))
        return (-self.gmm.score_samples(Xs)).reshape(T, R, C)

    def frame_scores(self, feats: np.ndarray, smooth: bool = True) -> np.ndarray:
        """One score per flow pair (length T-1 for a T-frame clip)."""
        cells = self.cell_scores(feats).reshape(len(feats), -1)
        k = self.cfg.top_k
        top = np.partition(cells, -k, axis=1)[:, -k:].mean(axis=1)
        if smooth and self.cfg.smooth_sigma > 0:
            top = gaussian_filter1d(top, self.cfg.smooth_sigma, mode="nearest")
        return top

    def predict(self, feats: np.ndarray) -> dict:
        """Frame-aligned scores, flags and cell scores for one clip/video."""
        scores = align_to_frames(self.frame_scores(feats))
        return {
            "scores": scores,
            "flags": scores > self.threshold_,
            "cell_scores": align_to_frames(self.cell_scores(feats)),
        }

    # ------------------------------------------------------------ save/load
    def save(self, path) -> Path:
        self._check_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "format_version": FORMAT_VERSION,
            "config": self.cfg.to_dict(),
            "frame_size": self.frame_size,
            "scaler": self.scaler,
            "gmm": self.gmm,
            "threshold": self.threshold_,
            "cell_threshold": self.cell_threshold_,
            "n_components": self.n_components_,
        }, path, compress=3)
        return path

    @classmethod
    def load(cls, path) -> "CrowdAnomalyModel":
        """Load a saved model. Only load files you trust (joblib uses pickle)."""
        blob = joblib.load(path)
        if blob.get("format_version") != FORMAT_VERSION:
            raise ValueError(f"Unsupported model format: {blob.get('format_version')}")
        m = cls(Config.from_dict(blob["config"]))
        m.frame_size = tuple(blob["frame_size"])
        m.scaler, m.gmm = blob["scaler"], blob["gmm"]
        m.threshold_, m.cell_threshold_ = blob["threshold"], blob["cell_threshold"]
        m.n_components_ = blob["n_components"]
        return m
