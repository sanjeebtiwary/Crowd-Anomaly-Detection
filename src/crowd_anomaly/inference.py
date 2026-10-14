"""Run a trained model on a video file or folder of frames; write CSV, plot, video."""
from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Dict

import cv2
import numpy as np

from .data import iter_frames, source_fps
from .features import clip_features
from .model import CrowdAnomalyModel, flags_to_segments

log = logging.getLogger(__name__)


def _annotate(frame, cell_scores, cell_thr, flagged, score, thr):
    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    out = frame.copy()
    h, w = out.shape[:2]
    mask = (cell_scores > cell_thr).astype(np.uint8)
    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
    if mask.any():
        red = out.copy()
        red[mask] = (0, 0, 255)
        out = cv2.addWeighted(red, 0.35, out, 0.65, 0)
    color = (0, 0, 255) if flagged else (0, 160, 0)
    cv2.rectangle(out, (0, 0), (w - 1, h - 1), color, 3 if flagged else 1)
    label = f"{'ANOMALY' if flagged else 'normal'}  score {score:.2f} / thr {thr:.2f}"
    cv2.putText(out, label, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return out


def _open_writer(path: Path, fps: float, size):
    for ext, fourcc in ((".mp4", "mp4v"), (".avi", "MJPG")):
        p = path.with_suffix(ext)
        wr = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*fourcc), fps, size)
        if wr.isOpened():
            return wr, p
    raise IOError("Could not open a video writer (tried mp4v and MJPG)")


def predict_source(
    model: CrowdAnomalyModel, source, out_dir=None, save_video: bool = True,
    save_plot: bool = True, min_segment_len: int = 3,
) -> Dict:
    """Score ``source`` and (optionally) write ``scores.csv``, ``scores.png``, ``annotated.mp4``."""
    source = Path(source)
    feats = clip_features(source, model.cfg, model.frame_size)
    pred = model.predict(feats)
    scores, flags = pred["scores"], pred["flags"]
    segments = flags_to_segments(flags, min_segment_len)
    result = {"scores": scores, "flags": flags, "segments": segments,
              "threshold": model.threshold_, "n_frames": len(scores)}

    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "scores.csv", "w", newline="") as fh:
            wr = csv.writer(fh, lineterminator="\n")
            wr.writerow(["frame", "score", "threshold", "is_anomaly"])
            for i, (s, f) in enumerate(zip(scores, flags)):
                wr.writerow([i, f"{s:.5f}", f"{model.threshold_:.5f}", int(f)])
        result["csv"] = out / "scores.csv"

        if save_plot:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(9, 3))
            ax.plot(scores, lw=1.3, label="anomaly score")
            ax.axhline(model.threshold_, color="r", ls="--", label="threshold")
            for a, b in segments:
                ax.axvspan(a, b + 1, color="orange", alpha=0.3)
            ax.set(xlabel="frame", ylabel="score", title=f"{source.name}: {len(segments)} anomalous segment(s)")
            ax.legend(loc="upper right")
            fig.tight_layout()
            fig.savefig(out / "scores.png", dpi=150)
            plt.close(fig)
            result["plot"] = out / "scores.png"

        if save_video:
            writer = None
            for i, raw in enumerate(iter_frames(source)):
                if i >= len(scores):
                    break
                frame = _annotate(raw, pred["cell_scores"][i], model.cell_threshold_,
                                  bool(flags[i]), scores[i], model.threshold_)
                if writer is None:
                    h, w = frame.shape[:2]
                    writer, vpath = _open_writer(out / "annotated", source_fps(source), (w, h))
                writer.write(frame)
            if writer is not None:
                writer.release()
                result["video"] = vpath
    return result
