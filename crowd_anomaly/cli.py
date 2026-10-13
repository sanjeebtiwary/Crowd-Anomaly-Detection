"""Command line interface:  python -m crowd_anomaly {train,evaluate,predict,demo}"""
from __future__ import annotations

import argparse
import logging
import sys
import tempfile
from pathlib import Path

from .config import Config
from .data import discover_clips, find_subset_dir
from .evaluate import evaluate_dataset, save_report
from .features import extract_many
from .inference import predict_source
from .model import CrowdAnomalyModel
from .synthetic import make_synthetic_ucsd

log = logging.getLogger("crowd_anomaly")


def _add_common(p):
    p.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    p.add_argument("--jobs", type=int, default=1, help="parallel workers for flow extraction (-1 = all cores)")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="crowd_anomaly", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="fit the normality model on UCSD training clips")
    t.add_argument("--data-root", required=True, help="folder containing ped1 / ped2")
    t.add_argument("--subset", choices=["ped1", "ped2"], default="ped1")
    t.add_argument("--model-out", default="models/model.joblib")
    t.add_argument("--max-clips", type=int, default=None, help="use only the first N training clips")
    t.add_argument("--components", type=int, default=None)
    t.add_argument("--auto-components", action="store_true", help="choose components by BIC")
    t.add_argument("--grid", type=int, nargs=2, metavar=("ROWS", "COLS"), default=None)
    t.add_argument("--top-k", type=int, default=None)
    t.add_argument("--val-fraction", type=float, default=None)
    t.add_argument("--frame-size", type=int, nargs=2, metavar=("W", "H"), default=None)
    t.add_argument("--seed", type=int, default=None)
    t.add_argument("--evaluate", action="store_true", help="evaluate on the test split afterwards")
    t.add_argument("--report-dir", default="reports")
    _add_common(t)

    e = sub.add_parser("evaluate", help="frame-level evaluation on the test split")
    e.add_argument("--data-root", required=True)
    e.add_argument("--subset", choices=["ped1", "ped2"], default="ped1")
    e.add_argument("--model", required=True)
    e.add_argument("--report-dir", default="reports")
    _add_common(e)

    p = sub.add_parser("predict", help="score a video file or a folder of frames")
    p.add_argument("--source", required=True, help="video file or folder of image frames")
    p.add_argument("--model", required=True)
    p.add_argument("--out", default="outputs")
    p.add_argument("--no-video", action="store_true")
    p.add_argument("--min-segment", type=int, default=3, help="min frames for a reported segment")
    _add_common(p)

    d = sub.add_parser("demo", help="end-to-end smoke test on a tiny synthetic dataset")
    d.add_argument("--workdir", default=None)
    _add_common(d)
    return ap


def _config_from_args(a) -> Config:
    cfg = Config()
    if a.components is not None:
        cfg.n_components = a.components
    cfg.auto_components = a.auto_components
    if a.grid:
        cfg.grid_rows, cfg.grid_cols = a.grid
    if a.top_k is not None:
        cfg.top_k = a.top_k
    if a.val_fraction is not None:
        cfg.val_fraction = a.val_fraction
    if a.frame_size:
        cfg.frame_size = tuple(a.frame_size)
    if a.seed is not None:
        cfg.seed = a.seed
    return cfg.validate()


def cmd_train(a) -> int:
    cfg = _config_from_args(a)
    sub = find_subset_dir(a.data_root, a.subset)
    clips = discover_clips(sub, "Train")[: a.max_clips]
    log.info("Training on %d clips from %s", len(clips), sub)
    feats, size = extract_many(clips, cfg, n_jobs=a.jobs)
    model = CrowdAnomalyModel(cfg).fit(feats, size)
    path = model.save(a.model_out)
    log.info("Saved model -> %s", path)
    if a.evaluate:
        return _run_eval(model, a.data_root, a.subset, a.report_dir, a.jobs)
    return 0


def _run_eval(model, data_root, subset, report_dir, jobs) -> int:
    res = evaluate_dataset(model, data_root, subset, n_jobs=jobs)
    save_report(res, report_dir)
    if "auc" in res:
        log.info("Frame-level AUC=%.4f  EER=%.4f  AP=%.4f  P=%.3f R=%.3f F1=%.3f  (%d clips, %d frames)",
                 res["auc"], res["eer"], res["average_precision"], res["precision"],
                 res["recall"], res["f1"], res["n_clips"], res["n_frames"])
    log.info("Report written to %s", report_dir)
    return 0


def cmd_evaluate(a) -> int:
    return _run_eval(CrowdAnomalyModel.load(a.model), a.data_root, a.subset, a.report_dir, a.jobs)


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


def cmd_demo(a) -> int:
    work = Path(a.workdir) if a.workdir else Path(tempfile.mkdtemp(prefix="crowd_anomaly_demo_"))
    log.info("Demo workspace: %s  (synthetic data - results say nothing about UCSD)", work)
    sub = make_synthetic_ucsd(work / "data")
    cfg = Config(frame_size=None, n_components=4, top_k=2, smooth_sigma=1.0)
    clips = discover_clips(sub, "Train")
    feats, size = extract_many(clips, cfg, n_jobs=a.jobs)
    model = CrowdAnomalyModel(cfg).fit(feats, size)
    model.save(work / "model.joblib")
    _run_eval(model, work / "data", "ped1", work / "reports", a.jobs)
    test_clip = discover_clips(sub, "Test")[0]
    res = predict_source(model, test_clip, work / "outputs")
    log.info("Predicted segments on %s: %s", test_clip.name, res["segments"])
    log.info("All outputs in %s", work)
    return 0


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    try:
        return {"train": cmd_train, "evaluate": cmd_evaluate,
                "predict": cmd_predict, "demo": cmd_demo}[args.cmd](args)
    except (FileNotFoundError, ValueError, IOError, RuntimeError) as exc:
        log.error("%s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
