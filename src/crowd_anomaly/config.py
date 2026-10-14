"""Central configuration. One flat dataclass, stored inside every saved model."""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Optional, Tuple


@dataclass
class Config:
    # --- Frame preprocessing -------------------------------------------------
    # (width, height). None = infer from the first training frame and store it
    # in the model, so any input video is resized to what the model was trained on.
    frame_size: Optional[Tuple[int, int]] = None
    blur_ksize: int = 3  # Gaussian blur before flow (0 = off); must be odd

    # --- Optical flow (cv2.calcOpticalFlowFarneback) -------------------------
    flow_pyr_scale: float = 0.5
    flow_levels: int = 3
    flow_winsize: int = 15
    flow_iterations: int = 3
    flow_poly_n: int = 5
    flow_poly_sigma: float = 1.2

    # --- Spatial pooling -----------------------------------------------------
    grid_rows: int = 6
    grid_cols: int = 8
    # "basic": mean dx, mean dy, mean magnitude per cell
    # "rich":  basic + max magnitude + magnitude std (better for small, fast objects)
    feature_set: str = "basic"

    # --- Normality model (Gaussian mixture over per-cell motion) -------------
    n_components: int = 8
    auto_components: bool = False  # pick n_components by BIC
    component_candidates: Tuple[int, ...] = (2, 4, 8, 12, 16)
    covariance_type: str = "full"
    reg_covar: float = 1e-4
    max_train_samples: int = 400_000

    # --- Scoring -------------------------------------------------------------
    top_k: int = 3  # frame score = mean of the k most anomalous cells
    smooth_sigma: float = 2.0  # temporal Gaussian smoothing of frame scores (0 = off)

    # --- Thresholds (chosen on held-out *normal* clips) ----------------------
    val_fraction: float = 0.2
    threshold_percentile: float = 99.0
    cell_threshold_percentile: float = 99.5

    seed: int = 0

    def validate(self) -> "Config":
        if self.blur_ksize and self.blur_ksize % 2 == 0:
            raise ValueError("blur_ksize must be odd (or 0 to disable)")
        if self.grid_rows < 1 or self.grid_cols < 1:
            raise ValueError("grid_rows and grid_cols must be >= 1")
        if not 1 <= self.top_k <= self.grid_rows * self.grid_cols:
            raise ValueError("top_k must be between 1 and grid_rows*grid_cols")
        if self.n_components < 1:
            raise ValueError("n_components must be >= 1")
        if not 0.0 <= self.val_fraction < 1.0:
            raise ValueError("val_fraction must be in [0, 1)")
        if self.feature_set not in {"basic", "rich"}:
            raise ValueError("feature_set must be 'basic' or 'rich'")
        if self.covariance_type not in {"full", "tied", "diag", "spherical"}:
            raise ValueError("invalid covariance_type")
        return self

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        known = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in d.items() if k in known}
        for key in ("frame_size", "component_candidates"):
            if kwargs.get(key) is not None:
                kwargs[key] = tuple(kwargs[key])
        return cls(**kwargs).validate()
