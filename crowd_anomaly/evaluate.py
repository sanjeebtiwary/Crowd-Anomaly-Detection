"""Frame-level evaluation: ROC-AUC, EER, and precision/recall at the model threshold."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
from sklearn.metrics import (average_precision_score, precision_recall_fscore_support,
                             roc_auc_score, roc_curve)

from .config import Config
from .data import discover_clips, find_gt_m_file, find_subset_dir, load_frame_labels, parse_gt_m_file
from .features import extract_many
from .model import CrowdAnomalyModel

log = logging.getLogger(__name__)


def compute_eer(y_true: np.ndarray, scores: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(y_true, scores)
    fnr = 1 - tpr
    i = int(np.nanargmin(np.abs(fnr - fpr)))
    return float((fpr[i] + fnr[i]) / 2)


def evaluate_clips(
    model: CrowdAnomalyModel, clip_dirs: Sequence[Path], gt_ranges=None,
    n_jobs: int = 1, progress: bool = True,
) -> Dict:
    """Score labelled test clips and compute pooled + per-clip metrics."""
    feats, _ = extract_many(clip_dirs, model.cfg, model.frame_size, n_jobs, progress)
    per_clip, ys, ss = [], [], []
    for d, f in zip(clip_dirs, feats):
        pred = model.predict(f)
        labels = load_frame_labels(d, len(pred["scores"]), gt_ranges)
        if labels is None:
            log.warning("No ground truth for %s; skipped", d.name)
            continue
        entry = {"clip": d.name, "n_frames": int(len(labels)),
                 "n_anomalous": int(labels.sum()), "scores": pred["scores"], "labels": labels}
        if 0 < labels.sum() < len(labels):
            entry["auc"] = float(roc_auc_score(labels, pred["scores"]))
        per_clip.append(entry)
        ys.append(labels)
        ss.append(pred["scores"])
    if not per_clip:
        raise RuntimeError("No test clips with ground truth were found")

    y, s = np.concatenate(ys), np.concatenate(ss)
    result: Dict = {"n_clips": len(per_clip), "n_frames": int(len(y)),
                    "n_anomalous_frames": int(y.sum()), "threshold": model.threshold_}
    if 0 < y.sum() < len(y):
        pred = (s > model.threshold_).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
        result.update({
            "auc": float(roc_auc_score(y, s)), "eer": compute_eer(y, s),
            "average_precision": float(average_precision_score(y, s)),
            "precision": float(p), "recall": float(r), "f1": float(f1),
            "false_positive_rate": float(((pred == 1) & (y == 0)).sum() / max((y == 0).sum(), 1)),
        })
        aucs = [c["auc"] for c in per_clip if "auc" in c]
        result["mean_clip_auc"] = float(np.mean(aucs)) if aucs else None
    else:
        log.warning("Ground truth has a single class; AUC/EER undefined")
    result["clips"] = per_clip
    return result


def evaluate_dataset(model: CrowdAnomalyModel, data_root, subset: str, n_jobs: int = 1) -> Dict:
    sub = find_subset_dir(data_root, subset)
    gt_file = find_gt_m_file(sub)
    gt = parse_gt_m_file(gt_file) if gt_file else None
    return evaluate_clips(model, discover_clips(sub, "Test"), gt, n_jobs=n_jobs)


def save_report(result: Dict, out_dir) -> None:
    """Write metrics.json, roc.png and timelines.png into ``out_dir``."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    clips: List[Dict] = result["clips"]
    summary = {k: v for k, v in result.items() if k != "clips"}
    summary["per_clip_auc"] = {c["clip"]: c.get("auc") for c in clips}
    (out / "metrics.json").write_text(json.dumps(summary, indent=2))

    y = np.concatenate([c["labels"] for c in clips])
    s = np.concatenate([c["scores"] for c in clips])
    if 0 < y.sum() < len(y):
        fpr, tpr, _ = roc_curve(y, s)
        fig, ax = plt.subplots(figsize=(4.5, 4.5))
        ax.plot(fpr, tpr, lw=2, label=f"AUC = {result['auc']:.3f}")
        ax.plot([0, 1], [0, 1], "--", color="grey")
        ax.set(xlabel="False positive rate", ylabel="True positive rate", title="Frame-level ROC")
        ax.legend(loc="lower right")
        fig.tight_layout()
        fig.savefig(out / "roc.png", dpi=150)
        plt.close(fig)

    n = min(len(clips), 8)
    fig, axes = plt.subplots(n, 1, figsize=(9, 1.8 * n), sharex=False, squeeze=False)
    for ax, c in zip(axes[:, 0], clips[:n]):
        ax.plot(c["scores"], lw=1.2)
        ax.axhline(result["threshold"], color="r", ls="--", lw=0.9)
        lab = c["labels"].astype(bool)
        ax.fill_between(np.arange(len(lab)), 0, 1, where=lab, transform=ax.get_xaxis_transform(),
                        color="orange", alpha=0.3)
        ax.set_ylabel(c["clip"], rotation=0, ha="right", va="center", fontsize=8)
    axes[-1, 0].set_xlabel("frame")
    fig.suptitle("Anomaly score (orange = ground-truth anomaly, red = threshold)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "timelines.png", dpi=150)
    plt.close(fig)
