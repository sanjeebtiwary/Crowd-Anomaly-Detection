from __future__ import annotations

from pathlib import Path
from typing import Tuple

import cv2
import numpy as np


def _background(rng, h, w):
    bg = cv2.GaussianBlur(rng.normal(128, 40, (h, w)).astype(np.float32), (0, 0), 2.0)
    return np.clip(bg, 0, 255)


def _render_clip(rng, bg, n_frames, anomaly: Tuple[int, int] | None):
    h, w = bg.shape
    walkers = []
    for _ in range(rng.integers(3, 6)):
        bh, bw = int(rng.integers(12, 18)), int(rng.integers(6, 9))
        walkers.append({
            "tex": rng.normal(150, 60, (bh, bw)).astype(np.float32),
            "x": float(rng.integers(0, w)), "y": int(rng.integers(h // 3, h - bh - 2)),
            "v": float(rng.choice([-1, 1]) * rng.integers(1, 3)),
        })
    biker = {"tex": rng.normal(170, 70, (16, 28)).astype(np.float32), "y": h // 2, "v": 7.0}
    frames = []
    for t in range(n_frames):
        img = bg.copy()
        objs = [(wk["tex"], int(wk["x"]) % w, wk["y"]) for wk in walkers]
        if anomaly and anomaly[0] <= t < anomaly[1]:
            objs.append((biker["tex"], int((t - anomaly[0]) * biker["v"]) % w, biker["y"]))
        for tex, x, y in objs:
            th, tw = tex.shape
            xs = (np.arange(tw) + x) % w
            img[y : y + th][:, xs] = tex
        for wk in walkers:
            wk["x"] += wk["v"]
        img += rng.normal(0, 1.5, img.shape)
        frames.append(np.clip(img, 0, 255).astype(np.uint8))
    return frames


def make_synthetic_ucsd(
    root, subset: str = "ped1", n_train: int = 8, n_test: int = 4,
    n_frames: int = 50, size: Tuple[int, int] = (160, 120), n_mask_clips: int = 2, seed: int = 0,
) -> Path:
    """Create ``<root>/UCSD_Anomaly_Dataset/UCSD<subset>/{Train,Test}`` and return its path.

    Odd-numbered test clips contain an anomaly in frames 20-39 (1-based 21-40).
    """
    rng = np.random.default_rng(seed)
    w, h = size
    sub = Path(root) / "UCSD_Anomaly_Dataset" / f"UCSD{subset}"
    bg = _background(rng, h, w)

    def write(folder: Path, frames, ext=".tif"):
        folder.mkdir(parents=True, exist_ok=True)
        for i, f in enumerate(frames, 1):
            cv2.imwrite(str(folder / f"{i:03d}{ext}"), f)

    for i in range(1, n_train + 1):
        write(sub / "Train" / f"Train{i:03d}", _render_clip(rng, bg, n_frames, None))

    m_lines = []
    for i in range(1, n_test + 1):
        has_anom = i % 2 == 1
        frames = _render_clip(rng, bg, n_frames, (20, 40) if has_anom else None)
        write(sub / "Test" / f"Test{i:03d}", frames)
        m_lines.append(f"TestVideoFile{{{i}}}.gt_frame = {'[21:40]' if has_anom else '[]'};")
        if i <= n_mask_clips:
            masks = []
            for t in range(n_frames):
                m = np.zeros((h, w), np.uint8)
                if has_anom and 20 <= t < 40:
                    m[h // 2 : h // 2 + 16, :] = 255
                masks.append(m)
            write(sub / "Test" / f"Test{i:03d}_gt", masks, ext=".bmp")
    (sub / "Test" / f"UCSD{subset}.m").write_text("\n".join(m_lines) + "\n")
    return sub
