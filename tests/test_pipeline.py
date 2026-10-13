import numpy as np
import pytest

from crowd_anomaly import Config, CrowdAnomalyModel
from crowd_anomaly.data import (discover_clips, find_subset_dir, load_frame_labels,
                                parse_gt_m_file, find_gt_m_file, prepare_frame)
from crowd_anomaly.evaluate import compute_eer, evaluate_dataset
from crowd_anomaly.features import extract_many, pool_flow
from crowd_anomaly.inference import predict_source
from crowd_anomaly.model import align_to_frames, flags_to_segments
from crowd_anomaly.synthetic import make_synthetic_ucsd


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
    from crowd_anomaly.data import iter_frames
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
