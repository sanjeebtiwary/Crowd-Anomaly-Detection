"""Frame-level evaluation: ROC-AUC, EER, and precision/recall at the model threshold."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score, roc_curve

from .data import discover_clips, find_gt_m_file, find_subset_dir, load_frame_labels, parse_gt_m_file
from .features import extract_many
from .model import CrowdAnomalyModel

log = logging.getLogger(__name__)


def compute_eer(y_true: np.ndarray, scores: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(y_true, scores)
    fnr = 1 - tpr
    i = int(np.nanargmin(np.abs(fnr - fpr)))
    return float((fpr[i] + fnr[i]) / 2)


def evaluate_features(
    model: CrowdAnomalyModel, clip_dirs: Sequence[Path], feats: Sequence[np.ndarray], gt_ranges=None,
) -> Dict:
    """Score precomputed clip features and compute pooled + per-clip metrics."""
    per_clip, ys, ss, unlabeled = [], [], [], []
    for d, f in zip(clip_dirs, feats):
        pred = model.predict(f)
        labels = load_frame_labels(d, len(pred["scores"]), gt_ranges)
        if labels is None:
            unlabeled.append(d.name)
            continue
        entry = {"clip": d.name, "n_frames": int(len(labels)),
                 "n_anomalous": int(labels.sum()), "scores": pred["scores"], "labels": labels}
        if 0 < labels.sum() < len(labels):
            entry["auc"] = float(roc_auc_score(labels, pred["scores"]))
        per_clip.append(entry)
        ys.append(labels)
        ss.append(pred["scores"])
    if unlabeled:
        log.warning(
            "No ground truth for %d of %d test clips (evaluating the other %d). Frame-level labels "
            "come from the dataset's UCSDped*.m file, which this download appears to lack; pixel "
            "masks exist only for a subset of clips. Pass --gt-file <path to .m> to use all clips.",
            len(unlabeled), len(clip_dirs), len(per_clip))
    if not per_clip:
        raise RuntimeError("No test clips with ground truth were found")

    y, s = np.concatenate(ys), np.concatenate(ss)
    result: Dict = {"n_clips": len(per_clip), "n_clips_total": len(clip_dirs),
                    "n_frames": int(len(y)), "n_anomalous_frames": int(y.sum()),
                    "anomalous_fraction": float(y.mean()), "threshold": model.threshold_}
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


def evaluate_clips(
    model: CrowdAnomalyModel, clip_dirs: Sequence[Path], gt_ranges=None,
    n_jobs: int = 1, progress: bool = True, cache_dir=None,
) -> Dict:
    """Extract features for ``clip_dirs`` and evaluate (see ``evaluate_features``)."""
    feats, _ = extract_many(clip_dirs, model.cfg, model.frame_size, n_jobs, progress, cache_dir)
    return evaluate_features(model, clip_dirs, feats, gt_ranges)


def load_gt(subset_dir, gt_file=None):
    """Parse the frame-level ground-truth ``.m`` (explicit path, else auto-detected)."""
    f = Path(gt_file) if gt_file else find_gt_m_file(subset_dir)
    if gt_file and not f.exists():
        raise FileNotFoundError(f"--gt-file not found: {f}")
    return parse_gt_m_file(f) if f else None


def evaluate_dataset(model: CrowdAnomalyModel, data_root, subset: str, n_jobs: int = 1,
                     gt_file=None, cache_dir=None) -> Dict:
    sub = find_subset_dir(data_root, subset)
    return evaluate_clips(model, discover_clips(sub, "Test"), load_gt(sub, gt_file),
                          n_jobs=n_jobs, cache_dir=cache_dir)


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
