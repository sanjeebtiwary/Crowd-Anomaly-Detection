from __future__ import annotations

import argparse
import csv
import itertools
import logging
import sys
from pathlib import Path

from .config import Config
from .data import discover_clips, find_subset_dir
from .download import KAGGLE_HANDLE, download_dataset, resolve_data_root
from .evaluate import evaluate_dataset, evaluate_features, load_gt, save_report
from .features import extract_many
from .inference import predict_source
from .model import CrowdAnomalyModel

log = logging.getLogger("crowd_anomaly")


def _add_common(p):
    p.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    p.add_argument("--jobs", type=int, default=1, help="parallel workers for flow extraction (-1 = all cores)")


def _add_data_args(p):
    p.add_argument("--data-root", default=None,
                   help="local dataset folder (default: download from Kaggle via kagglehub)")
    p.add_argument("--subset", choices=["ped1", "ped2"], default="ped1")
    p.add_argument("--gt-file", default=None,
                   help="frame-level ground-truth .m file (UCSDped1.m); default: auto-detect")
    p.add_argument("--cache-dir", default=".feature_cache", help="where extracted flow features are cached")
    p.add_argument("--no-cache", action="store_true", help="disable the feature cache")


def _add_model_args(p):
    p.add_argument("--components", type=int, default=None)
    p.add_argument("--auto-components", action="store_true", help="choose components by BIC")
    p.add_argument("--grid", type=int, nargs=2, metavar=("ROWS", "COLS"), default=None)
    p.add_argument("--feature-set", choices=["basic", "rich"], default=None)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--smooth-sigma", type=float, default=None)
    p.add_argument("--val-fraction", type=float, default=None)
    p.add_argument("--threshold-percentile", type=float, default=None)
    p.add_argument("--max-samples", type=int, default=None, help="cap on GMM training samples")
    p.add_argument("--frame-size", type=int, nargs=2, metavar=("W", "H"), default=None)
    p.add_argument("--seed", type=int, default=None)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="crowd_anomaly", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="fit the normality model on UCSD training clips")
    _add_data_args(t)
    _add_model_args(t)
    t.add_argument("--model-out", default="models/model.joblib")
    t.add_argument("--max-clips", type=int, default=None, help="use only the first N training clips")
    t.add_argument("--evaluate", action="store_true", help="evaluate on the test split afterwards")
    t.add_argument("--report-dir", default="reports")
    _add_common(t)

    e = sub.add_parser("evaluate", help="frame-level evaluation on the test split")
    _add_data_args(e)
    e.add_argument("--model", required=True)
    e.add_argument("--report-dir", default="reports")
    _add_common(e)

    w = sub.add_parser("sweep", help="compare settings and rank them by test AUC (see warning in README)")
    _add_data_args(w)
    w.add_argument("--grids", default="6x8,12x16", help="comma list, e.g. 6x8,12x16")
    w.add_argument("--feature-sets", default="basic,rich")
    w.add_argument("--components", default="8", help="comma list, e.g. 4,8,16")
    w.add_argument("--top-ks", default="1,3,5")
    w.add_argument("--smooths", default="2")
    w.add_argument("--max-samples", type=int, default=150_000)
    w.add_argument("--max-clips", type=int, default=None)
    w.add_argument("--out", default="reports/sweep.csv")
    _add_common(w)

    p = sub.add_parser("predict", help="score a video file or a folder of frames")
    p.add_argument("--source", required=True, help="video file or folder of image frames")
    p.add_argument("--model", required=True)
    p.add_argument("--out", default="outputs")
    p.add_argument("--no-video", action="store_true")
    p.add_argument("--min-segment", type=int, default=3, help="min frames for a reported segment")
    _add_common(p)

    dl = sub.add_parser("download", help=f"download {KAGGLE_HANDLE} from Kaggle and print its path")
    dl.add_argument("--force", action="store_true", help="re-download even if cached")
    _add_common(dl)
    return ap


def _cache(a):
    return None if a.no_cache else Path(a.cache_dir)


def _config_from_args(a) -> Config:
    cfg = Config()
    if a.components is not None:
        cfg.n_components = a.components
    cfg.auto_components = a.auto_components
    if a.grid:
        cfg.grid_rows, cfg.grid_cols = a.grid
    if a.feature_set:
        cfg.feature_set = a.feature_set
    if a.top_k is not None:
        cfg.top_k = a.top_k
    if a.smooth_sigma is not None:
        cfg.smooth_sigma = a.smooth_sigma
    if a.val_fraction is not None:
        cfg.val_fraction = a.val_fraction
    if a.threshold_percentile is not None:
        cfg.threshold_percentile = a.threshold_percentile
    if a.max_samples is not None:
        cfg.max_train_samples = a.max_samples
    if a.frame_size:
        cfg.frame_size = tuple(a.frame_size)
    if a.seed is not None:
        cfg.seed = a.seed
    return cfg.validate()


def cmd_train(a) -> int:
    cfg = _config_from_args(a)
    data_root = resolve_data_root(a.data_root)
    sub = find_subset_dir(data_root, a.subset)
    clips = discover_clips(sub, "Train")[: a.max_clips]
    log.info("Training on %d clips from %s", len(clips), sub)
    feats, size = extract_many(clips, cfg, n_jobs=a.jobs, cache_dir=_cache(a))
    model = CrowdAnomalyModel(cfg).fit(feats, size)
    path = model.save(a.model_out)
    log.info("Saved model -> %s", path)
    if a.evaluate:
        return _run_eval(model, data_root, a)
    return 0


def _run_eval(model, data_root, a) -> int:
    res = evaluate_dataset(model, data_root, a.subset, n_jobs=a.jobs,
                           gt_file=a.gt_file, cache_dir=_cache(a))
    save_report(res, a.report_dir)
    log.info("Evaluated %d of %d test clips: %d frames, %.0f%% anomalous",
             res["n_clips"], res["n_clips_total"], res["n_frames"], 100 * res["anomalous_fraction"])
    if "auc" in res:
        log.info("Frame-level AUC=%.4f  EER=%.4f  AP=%.4f  P=%.3f R=%.3f F1=%.3f",
                 res["auc"], res["eer"], res["average_precision"], res["precision"],
                 res["recall"], res["f1"])
    log.info("Report written to %s", a.report_dir)
    return 0


def cmd_evaluate(a) -> int:
    return _run_eval(CrowdAnomalyModel.load(a.model), resolve_data_root(a.data_root), a)


def _csv_list(text, cast):
    return [cast(x) for x in text.split(",") if x.strip()]


def cmd_sweep(a) -> int:
    """Grid over feature/model settings. Flow features are cached, so only the first
    combination per (grid, feature set) pays the extraction cost."""
    sub = find_subset_dir(resolve_data_root(a.data_root), a.subset)
    train_clips = discover_clips(sub, "Train")[: a.max_clips]
    test_clips = discover_clips(sub, "Test")
    gt = load_gt(sub, a.gt_file)
    rows = []
    for grid, fset in itertools.product(_csv_list(a.grids, str), _csv_list(a.feature_sets, str)):
        r, c = (int(x) for x in grid.lower().split("x"))
        base = Config(grid_rows=r, grid_cols=c, feature_set=fset, max_train_samples=a.max_samples).validate()
        log.info("Features: grid %dx%d, %s", r, c, fset)
        tr, size = extract_many(train_clips, base, n_jobs=a.jobs, cache_dir=_cache(a))
        te, _ = extract_many(test_clips, base, size, n_jobs=a.jobs, cache_dir=_cache(a))
        for k, top_k, sm in itertools.product(_csv_list(a.components, int), _csv_list(a.top_ks, int),
                                              _csv_list(a.smooths, float)):
            try:
                cfg = Config(**{**base.to_dict(), "n_components": k, "top_k": top_k, "smooth_sigma": sm}).validate()
            except ValueError as exc:
                log.warning("Skipping invalid combination (%s)", exc)
                continue
            model = CrowdAnomalyModel(cfg).fit(tr, size)
            res = evaluate_features(model, test_clips, te, gt)
            row = {"grid": grid, "features": fset, "components": k, "top_k": top_k, "smooth": sm,
                   "auc": res.get("auc"), "eer": res.get("eer"), "mean_clip_auc": res.get("mean_clip_auc"),
                   "clips": res["n_clips"]}
            rows.append(row)
            log.info("  k=%-2d top_k=%-2d smooth=%.1f -> AUC=%.4f EER=%.4f", k, top_k, sm,
                     row["auc"] or float("nan"), row["eer"] or float("nan"))
    rows.sort(key=lambda r: -(r["auc"] or 0))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n")
        wr.writeheader()
        wr.writerows(rows)
    log.info("Best: %s", rows[0])
    log.warning("Settings were ranked on the TEST labels, so the best AUC is optimistic. "
                "Report it as tuned on test, or hold out some labelled clips.")
    log.info("Wrote %s", out)
    return 0


def cmd_predict(a) -> int:
    model = CrowdAnomalyModel.load(a.model)
    res = predict_source(model, a.source, a.out, save_video=not a.no_video, min_segment_len=a.min_segment)
    n = len(res["segments"])
    log.info("%d frames, %d anomalous segment(s) (threshold %.3f)", res["n_frames"], n, res["threshold"])
    for s, e in res["segments"]:
        log.info("  frames %d-%d", s, e)
    for k in ("csv", "plot", "video"):
        if k in res:
            log.info("Wrote %s", res[k])
    return 0


def cmd_download(a) -> int:
    print(download_dataset(force=a.force))
    return 0


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    try:
        return {"train": cmd_train, "evaluate": cmd_evaluate, "sweep": cmd_sweep,
                "predict": cmd_predict, "download": cmd_download}[args.cmd](args)
    except (FileNotFoundError, ValueError, IOError, RuntimeError) as exc:
        log.error("%s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
