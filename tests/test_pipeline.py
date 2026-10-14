import numpy as np
import pytest

from src.crowd_anomaly import Config, CrowdAnomalyModel
from src.crowd_anomaly.data import (discover_clips, find_subset_dir, load_frame_labels,
                                parse_gt_m_file, find_gt_m_file, prepare_frame)
from src.crowd_anomaly.evaluate import compute_eer, evaluate_dataset
from src.crowd_anomaly.features import extract_many, pool_flow
from src.crowd_anomaly.inference import predict_source
from src.crowd_anomaly.model import align_to_frames, flags_to_segments
from synthetic import make_synthetic_ucsd


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    root = tmp_path_factory.mktemp("data")
    make_synthetic_ucsd(root)
    return root


@pytest.fixture(scope="module")
def trained(dataset):
    cfg = Config(n_components=4, top_k=2, smooth_sigma=1.0)
    sub = find_subset_dir(dataset, "ped1")
    feats, size = extract_many(discover_clips(sub, "Train"), cfg, progress=False)
    return CrowdAnomalyModel(cfg).fit(feats, size)


def test_discovery_and_ordering(dataset):
    sub = find_subset_dir(dataset, "ped1")
    names = [c.name for c in discover_clips(sub, "Test")]
    assert names == ["Test001", "Test002", "Test003", "Test004"]  # no *_gt folders
    assert find_subset_dir(sub, "ped1") == sub  # root may be the subset itself
    with pytest.raises(FileNotFoundError):
        find_subset_dir(dataset, "ped2")


def test_parse_gt_m_file(tmp_path):
    f = tmp_path / "x.m"
    f.write_text("TestVideoFile{1}.gt_frame = [60:152];\n"
                 "TestVideoFile{2}.gt_frame = [5:90, 140:150];\n"
                 "TestVideoFile{3}.gt_frame = [];\n")
    gt = parse_gt_m_file(f)
    assert gt == {1: [(60, 152)], 2: [(5, 90), (140, 150)], 3: []}


def test_labels_from_ranges_and_masks(dataset):
    sub = find_subset_dir(dataset, "ped1")
    gt = parse_gt_m_file(find_gt_m_file(sub))
    clips = discover_clips(sub, "Test")
    a = load_frame_labels(clips[0], 50, gt)
    b = load_frame_labels(clips[0], 50, None)  # falls back to pixel masks
    assert a.sum() == 20 and np.array_equal(a, b)
    assert load_frame_labels(clips[3], 50, None) is None  # no mask, no ranges


def test_pool_flow_shape():
    out = pool_flow(np.ones((60, 80, 2), np.float32), 6, 8)
    assert out.shape == (6, 8, 3)
    assert np.allclose(out[..., 2], np.sqrt(2), atol=1e-5)


def test_prepare_frame_handles_color_and_resize():
    out = prepare_frame(np.zeros((50, 40, 3), np.uint8), (80, 60), blur_ksize=3)
    assert out.shape == (60, 80) and out.dtype == np.uint8


def test_helpers():
    assert flags_to_segments(np.array([0, 1, 1, 0, 1, 0, 1, 1, 1]), 2) == [(1, 2), (6, 8)]
    assert list(align_to_frames(np.array([3, 4]))) == [3, 3, 4]
    y = np.array([0, 0, 1, 1])
    assert compute_eer(y, np.array([0.1, 0.2, 0.8, 0.9])) == 0.0


def test_anomalies_score_higher(dataset, trained):
    sub = find_subset_dir(dataset, "ped1")
    res = evaluate_dataset(trained, dataset, "ped1")
    assert res["n_clips"] == 4
    assert res["auc"] > 0.9
    assert res["recall"] > 0.8


def test_save_load_roundtrip(trained, tmp_path, dataset):
    path = trained.save(tmp_path / "m.joblib")
    loaded = CrowdAnomalyModel.load(path)
    clip = discover_clips(find_subset_dir(dataset, "ped1"), "Test")[0]
    feats, _ = extract_many([clip], trained.cfg, trained.frame_size, progress=False)
    assert np.allclose(trained.frame_scores(feats[0]), loaded.frame_scores(feats[0]))
    assert loaded.threshold_ == trained.threshold_ and loaded.frame_size == trained.frame_size


def test_predict_outputs(trained, dataset, tmp_path):
    clip = discover_clips(find_subset_dir(dataset, "ped1"), "Test")[0]
    res = predict_source(trained, clip, tmp_path, min_segment_len=3)
    assert len(res["scores"]) == 50 and res["segments"]
    for name in ("scores.csv", "scores.png"):
        assert (tmp_path / name).exists()
    assert res["video"].exists()


def test_unfitted_and_bad_config():
    with pytest.raises(RuntimeError):
        CrowdAnomalyModel().frame_scores(np.zeros((2, 6, 8, 3)))
    with pytest.raises(ValueError):
        Config(top_k=0).validate()
    with pytest.raises(ValueError):
        Config(blur_ksize=4).validate()


def test_single_frame_raises(tmp_path):
    import cv2
    cv2.imwrite(str(tmp_path / "001.png"), np.zeros((20, 20), np.uint8))
    with pytest.raises(ValueError):
        extract_many([tmp_path], Config(), (20, 20), progress=False)


def test_auto_components(dataset):
    cfg = Config(auto_components=True, component_candidates=(2, 4), top_k=2)
    sub = find_subset_dir(dataset, "ped1")
    feats, size = extract_many(discover_clips(sub, "Train")[:4], cfg, progress=False)
    m = CrowdAnomalyModel(cfg).fit(feats, size)
    assert m.n_components_ in (2, 4)


def test_video_file_at_different_resolution(trained, dataset, tmp_path):
    """A .avi at 2x the training resolution must be resized and scored."""
    import cv2
    from src.crowd_anomaly.data import iter_frames
    clip = discover_clips(find_subset_dir(dataset, "ped1"), "Test")[0]
    video = tmp_path / "test.avi"
    w, h = trained.frame_size
    wr = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10, (w * 2, h * 2))
    assert wr.isOpened()
    for f in iter_frames(clip):
        wr.write(cv2.resize(cv2.cvtColor(f, cv2.COLOR_GRAY2BGR), (w * 2, h * 2)))
    wr.release()
    res = predict_source(trained, video, tmp_path / "out", save_video=False)
    assert res["n_frames"] == 50
    assert res["scores"][25:38].mean() > res["scores"][:10].mean()


def test_cli_errors_cleanly(tmp_path):
    from crowd_anomaly.cli import main
    assert main(["train", "--data-root", str(tmp_path / "missing")]) == 1
    assert main(["predict", "--source", "nope.avi", "--model", str(tmp_path / "nope.joblib")]) == 1


def test_subset_folder_named_just_ped1(tmp_path):
    """Kaggle-style layout: dataset/ped1/{Train,Test} (no 'UCSD' prefix)."""
    import shutil
    src = make_synthetic_ucsd(tmp_path / "src")
    dst = tmp_path / "dataset" / "ped1"
    shutil.copytree(src, dst)
    assert find_subset_dir(tmp_path / "dataset", "ped1") == dst
    assert len(discover_clips(dst, "Train")) == 8
    assert find_gt_m_file(dst) is not None
    with pytest.raises(FileNotFoundError, match="Folders found"):
        find_subset_dir(tmp_path / "dataset", "ped2")


def test_resolve_data_root_uses_kagglehub(tmp_path, monkeypatch):
    """No --data-root -> kagglehub.dataset_download('orvile/ucsd-anomaly-dataset')."""
    import sys, types
    from crowd_anomaly.download import resolve_data_root, KAGGLE_HANDLE
    calls = []
    fake = types.ModuleType("kagglehub")
    fake.dataset_download = lambda handle, force_download=False: calls.append(handle) or str(tmp_path)
    monkeypatch.setitem(sys.modules, "kagglehub", fake)
    assert resolve_data_root(None) == tmp_path and calls == [KAGGLE_HANDLE]
    assert resolve_data_root("local") .name == "local" and len(calls) == 1  # explicit path wins


def test_cli_train_downloads_from_kaggle(tmp_path, monkeypatch):
    """End to end: train + evaluate with the dataset 'downloaded' by a stubbed kagglehub,
    nested like a real Kaggle cache (versions/1/dataset/ped1/...)."""
    import shutil, sys, types
    from crowd_anomaly.cli import main
    cache = tmp_path / "cache" / "versions" / "1"
    shutil.copytree(make_synthetic_ucsd(tmp_path / "src"), cache / "dataset" / "ped1")
    fake = types.ModuleType("kagglehub")
    fake.dataset_download = lambda handle, force_download=False: str(cache)
    monkeypatch.setitem(sys.modules, "kagglehub", fake)
    rc = main(["train", "--subset", "ped1", "--components", "4", "--top-k", "2",
               "--model-out", str(tmp_path / "m.joblib"), "--evaluate",
               "--report-dir", str(tmp_path / "rep")])
    assert rc == 0 and (tmp_path / "m.joblib").exists()
    assert (tmp_path / "rep" / "metrics.json").exists()
    assert main(["download"]) == 0


# --------------------------------------------------------------------------- #
# Rich features, caching, ground-truth fallbacks, sweep
# --------------------------------------------------------------------------- #
def test_rich_features_capture_small_fast_objects():
    flow = np.zeros((60, 80, 2), np.float32)
    flow[10:13, 10:13, 0] = 8.0                       # tiny fast patch inside one 10x10 cell
    basic = pool_flow(flow, 6, 8, "basic")
    rich = pool_flow(flow, 6, 8, "rich")
    assert basic.shape == (6, 8, 3) and rich.shape == (6, 8, 5)
    assert np.allclose(rich[..., :3], basic)
    assert rich[1, 1, 3] == pytest.approx(8.0)        # max magnitude survives pooling
    assert rich[1, 1, 3] > 10 * basic[1, 1, 2]        # ...while the mean dilutes it
    assert rich[1, 1, 4] > 0 and rich[3, 5, 4] == 0   # std only where there is variation


def test_rich_model_trains_and_detects(dataset):
    cfg = Config(feature_set="rich", n_components=4, top_k=2, smooth_sigma=1.0)
    sub = find_subset_dir(dataset, "ped1")
    feats, size = extract_many(discover_clips(sub, "Train"), cfg, progress=False)
    assert feats[0].shape[-1] == 5
    model = CrowdAnomalyModel(cfg).fit(feats, size)
    assert evaluate_dataset(model, dataset, "ped1")["auc"] > 0.9


def test_feature_cache_hits_and_invalidates(dataset, tmp_path, monkeypatch):
    import crowd_anomaly.features as F
    sub = find_subset_dir(dataset, "ped1")
    clip = discover_clips(sub, "Train")[0]
    calls = []
    real = F.extract_features
    monkeypatch.setattr(F, "extract_features", lambda *a, **k: calls.append(1) or real(*a, **k))
    cfg = Config()
    a = F.clip_features(clip, cfg, (160, 120), tmp_path)
    b = F.clip_features(clip, cfg, (160, 120), tmp_path)
    assert len(calls) == 1 and np.array_equal(a, b)                    # second call came from cache
    F.clip_features(clip, Config(grid_rows=4, grid_cols=4), (160, 120), tmp_path)
    assert len(calls) == 2                                              # different grid -> recompute
    F.clip_features(clip, cfg, (160, 120), None)
    assert len(calls) == 3                                              # cache disabled


def test_missing_gt_file_uses_masks_then_gt_file_covers_all(tmp_path, caplog):
    """Kaggle-like copy: no .m file, masks for only 2 clips; --gt-file restores the rest."""
    import shutil
    src = make_synthetic_ucsd(tmp_path / "s")
    root = tmp_path / "dataset"
    shutil.copytree(src, root / "ped1")
    m_file = next((root / "ped1").rglob("*.m"))
    saved = tmp_path / "UCSDped1.m"
    shutil.move(str(m_file), saved)
    assert find_gt_m_file(root / "ped1") is None

    cfg = Config(n_components=4, top_k=2, smooth_sigma=1.0)
    feats, size = extract_many(discover_clips(root / "ped1", "Train"), cfg, progress=False)
    model = CrowdAnomalyModel(cfg).fit(feats, size)
    with caplog.at_level("WARNING"):
        partial = evaluate_dataset(model, root, "ped1")
    assert partial["n_clips"] == 2 and partial["n_clips_total"] == 4
    assert any("--gt-file" in r.message for r in caplog.records)
    full = evaluate_dataset(model, root, "ped1", gt_file=saved)
    assert full["n_clips"] == 4
    with pytest.raises(FileNotFoundError):
        evaluate_dataset(model, root, "ped1", gt_file=tmp_path / "nope.m")


def test_find_gt_m_file_searches_recursively(tmp_path):
    sub = tmp_path / "ped1"
    (sub / "Test" / "deep" / "er").mkdir(parents=True)
    f = sub / "Test" / "deep" / "er" / "UCSDped1.m"
    f.write_text("TestVideoFile{1}.gt_frame = [1:5];")
    (sub / "._junk.m").write_text("x")
    assert find_gt_m_file(sub) == f


def test_cli_sweep_ranks_settings(dataset, tmp_path):
    from crowd_anomaly.cli import main
    out = tmp_path / "sweep.csv"
    rc = main(["sweep", "--data-root", str(dataset), "--subset", "ped1",
               "--grids", "6x8", "--feature-sets", "basic,rich", "--components", "4",
               "--top-ks", "1,2", "--smooths", "1", "--max-samples", "20000",
               "--cache-dir", str(tmp_path / "cache"), "--out", str(out)])
    assert rc == 0
    lines = out.read_text().strip().splitlines()
    assert len(lines) == 1 + 4                                   # header + 2 feature sets x 2 top_k
    aucs = [float(l.split(",")[5]) for l in lines[1:]]
    assert aucs == sorted(aucs, reverse=True) and aucs[0] > 0.9
