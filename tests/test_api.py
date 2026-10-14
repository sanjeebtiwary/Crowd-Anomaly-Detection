"""HTTP API tests (offline: the model is trained on synthetic data, then served)."""
import cv2
import pytest
from fastapi.testclient import TestClient
from synthetic import make_synthetic_ucsd

from crowd_anomaly import Config, CrowdAnomalyModel
from crowd_anomaly.data import discover_clips, find_subset_dir, iter_frames
from crowd_anomaly.features import extract_many


@pytest.fixture(scope="module")
def served(tmp_path_factory):
    root = tmp_path_factory.mktemp("api")
    make_synthetic_ucsd(root)
    sub = find_subset_dir(root, "ped1")
    cfg = Config(n_components=4, top_k=2, smooth_sigma=1.0)
    feats, size = extract_many(discover_clips(sub, "Train"), cfg, progress=False)
    model_path = CrowdAnomalyModel(cfg).fit(feats, size).save(root / "model.joblib")

    import os

    os.environ["MODEL_PATH"] = str(model_path)
    from backend.app import main, service

    service.get_model.cache_clear()
    main.settings = main.get_settings()
    with TestClient(main.app) as client:
        yield client, discover_clips(sub, "Test")[0], size
    service.get_model.cache_clear()


def _write_avi(path, clip, size):
    wr = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, size)
    assert wr.isOpened()
    for f in iter_frames(clip):
        wr.write(cv2.cvtColor(f, cv2.COLOR_GRAY2BGR))
    wr.release()


def test_health_and_root(served):
    client, _, _ = served
    assert client.get("/").json()["docs"] == "/docs"
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["model_loaded"] is True


def test_predict_video(served, tmp_path):
    client, clip, size = served
    video = tmp_path / "clip.avi"
    _write_avi(video, clip, size)
    with open(video, "rb") as fh:
        r = client.post("/predict", files={"file": ("clip.avi", fh, "video/x-msvideo")})
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["frames_processed"] == 50 == len(data["scores"]) == len(data["flags"])
    assert data["anomaly_detected"] and data["segments"]


def test_predict_rejects_bad_input(served):
    client, _, _ = served
    assert client.post("/predict", files={"file": ("x.txt", b"hi", "text/plain")}).status_code == 415
    assert client.post("/predict", files={"file": ("x.mp4", b"", "video/mp4")}).status_code == 400
    assert client.post("/predict", files={"file": ("x.mp4", b"not a video", "video/mp4")}).status_code in (422, 500)
