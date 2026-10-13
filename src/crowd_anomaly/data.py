"""Dataset discovery, frame/video reading and ground-truth parsing (UCSD layout)."""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple, Union

import cv2
import numpy as np

log = logging.getLogger(__name__)

IMAGE_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}
VIDEO_EXTS = {".avi", ".mp4", ".mov", ".mkv", ".mpg", ".mpeg", ".wmv", ".m4v"}

PathLike = Union[str, Path]


# --------------------------------------------------------------------------- #
# Dataset discovery
# --------------------------------------------------------------------------- #
def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def find_subset_dir(root: PathLike, subset: str) -> Path:
    """Locate the folder for 'ped1' / 'ped2' below ``root``.

    Accepts folders named ``UCSDped1``, ``UCSD_ped1`` or just ``ped1`` (case and
    punctuation ignored) and searches ``root`` plus up to five levels beneath it,
    so ``root`` can be the download folder, the dataset folder or the subset itself.
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Data root does not exist: {root.resolve()}")
    keys = {f"ucsd{subset.lower()}", subset.lower()}
    level = [root]
    for _ in range(6):  # root + up to five levels below it
        for p in level:
            if _norm(p.name) in keys and p.is_dir():
                return p
        level = [c for p in level if p.is_dir() for c in sorted(p.iterdir()) if c.is_dir()]
    found = sorted(p.name for p in root.iterdir() if p.is_dir())[:10]
    raise FileNotFoundError(
        f"Could not find a '{subset}' / 'UCSD{subset}' folder under {root.resolve()}. "
        f"Folders found there: {found}. Point the data root at the extracted dataset."
    )


def _natural_key(p: Path):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", p.name)]


def discover_clips(subset_dir: PathLike, split: str) -> List[Path]:
    """Return clip folders (Train001, Train002, ... / Test001, ...) in order."""
    subset_dir = Path(subset_dir)
    split_dir = next(
        (p for p in subset_dir.iterdir() if p.is_dir() and p.name.lower() == split.lower()),
        None,
    )
    if split_dir is None:
        raise FileNotFoundError(f"No '{split}' folder inside {subset_dir}")
    pat = re.compile(rf"{split}\d+", re.IGNORECASE)
    clips = sorted(
        (p for p in split_dir.iterdir() if p.is_dir() and pat.fullmatch(p.name)),
        key=_natural_key,
    )
    if not clips:
        raise FileNotFoundError(f"No clip folders like '{split}001' in {split_dir}")
    return clips


# --------------------------------------------------------------------------- #
# Frame reading
# --------------------------------------------------------------------------- #
def _to_uint8(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img
    if img.dtype == np.uint16:
        return (img / 257.0).round().astype(np.uint8)
    img = img.astype(np.float32)
    lo, hi = float(img.min()), float(img.max())
    if hi <= lo:
        return np.zeros(img.shape, np.uint8)
    return ((img - lo) / (hi - lo) * 255).astype(np.uint8)


def list_image_files(folder: PathLike) -> List[Path]:
    return sorted(
        (p for p in Path(folder).iterdir() if p.suffix.lower() in IMAGE_EXTS),
        key=_natural_key,
    )


def iter_frames(source: PathLike) -> Iterator[np.ndarray]:
    """Yield raw frames from a folder of images or a video file (restartable)."""
    source = Path(source)
    if source.is_dir():
        files = list_image_files(source)
        if not files:
            raise FileNotFoundError(f"No image files found in {source}")
        for f in files:
            img = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
            if img is None:
                log.warning("Skipping unreadable image: %s", f)
                continue
            yield _to_uint8(img)
    elif source.is_file():
        cap = cv2.VideoCapture(str(source))
        if not cap.isOpened():
            raise IOError(f"OpenCV could not open video: {source}")
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                yield frame
        finally:
            cap.release()
    else:
        raise FileNotFoundError(f"Source not found: {source}")


def source_fps(source: PathLike, default: float = 10.0) -> float:
    """Frame rate of a video file; image folders (UCSD is 10 fps) use ``default``."""
    source = Path(source)
    if source.is_file():
        cap = cv2.VideoCapture(str(source))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        if fps and fps > 1:
            return float(fps)
    return default


def frame_size_of(source: PathLike) -> Tuple[int, int]:
    """(width, height) of the first frame of a source."""
    first = next(iter_frames(source))
    return int(first.shape[1]), int(first.shape[0])


def prepare_frame(frame: np.ndarray, size: Tuple[int, int], blur_ksize: int = 3) -> np.ndarray:
    """Grayscale uint8, resized to ``size`` (w, h), optionally blurred."""
    if frame.ndim == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY if frame.shape[2] == 4 else cv2.COLOR_BGR2GRAY)
    if (frame.shape[1], frame.shape[0]) != tuple(size):
        frame = cv2.resize(frame, tuple(size), interpolation=cv2.INTER_AREA)
    if blur_ksize:
        frame = cv2.GaussianBlur(frame, (blur_ksize, blur_ksize), 0)
    return frame


# --------------------------------------------------------------------------- #
# Ground truth
# --------------------------------------------------------------------------- #
def parse_gt_m_file(path: PathLike) -> Dict[int, List[Tuple[int, int]]]:
    """Parse UCSD ``UCSDpedX.m``: ``TestVideoFile{1}.gt_frame = [60:152];``.

    Returns {clip_number: [(start, end), ...]} with 1-based inclusive frame ranges.
    """
    text = Path(path).read_text(errors="ignore")
    out: Dict[int, List[Tuple[int, int]]] = {}
    for m in re.finditer(r"\{\s*(\d+)\s*\}\s*\.\s*gt_frame\s*=\s*\[(.*?)\]", text, re.S):
        spans = []
        for a, b in re.findall(r"(\d+)(?:\s*:\s*(\d+))?", m.group(2)):
            start = int(a)
            spans.append((start, int(b) if b else start))
        out[int(m.group(1))] = spans
    return out


def _clip_number(clip_dir: Path) -> Optional[int]:
    m = re.search(r"(\d+)$", clip_dir.name)
    return int(m.group(1)) if m else None


def find_gt_m_file(subset_dir: PathLike) -> Optional[Path]:
    """Find the frame-level ground-truth ``.m`` file (e.g. ``UCSDped1.m``), if the download has one.

    Searches below the subset folder and, failing that, below its parent (some copies keep
    it next to the subset folder). Files named after the subset are preferred.
    """
    subset_dir = Path(subset_dir)
    key = _norm(subset_dir.name).replace("ucsd", "")
    for base in (subset_dir, subset_dir.parent):
        found = sorted(base.rglob("*.m")) if base.exists() else []
        found = [f for f in found if not f.name.startswith("._")]
        preferred = [f for f in found if key in _norm(f.stem)]
        if preferred or found:
            return (preferred or found)[0]
    return None


def load_frame_labels(
    clip_dir: PathLike,
    n_frames: int,
    gt_ranges: Optional[Dict[int, List[Tuple[int, int]]]] = None,
) -> Optional[np.ndarray]:
    """Per-frame binary labels (1 = anomalous) for a test clip, or None if unknown.

    Prefers the ``.m`` frame ranges (cover every test clip); falls back to the
    ``TestXXX_gt`` pixel masks (a frame is anomalous if its mask has any pixel set).
    """
    clip_dir = Path(clip_dir)
    num = _clip_number(clip_dir)
    if gt_ranges and num in gt_ranges:
        labels = np.zeros(n_frames, dtype=np.int8)
        for a, b in gt_ranges[num]:
            labels[max(a - 1, 0) : min(b, n_frames)] = 1
        return labels
    mask_dir = clip_dir.parent / f"{clip_dir.name}_gt"
    if mask_dir.is_dir():
        files = list_image_files(mask_dir)
        if files:
            labels = np.zeros(n_frames, dtype=np.int8)
            for i, f in enumerate(files[:n_frames]):
                m = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
                labels[i] = int(m is not None and bool(m.any()))
            return labels
    return None
