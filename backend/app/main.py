from __future__ import annotations

import logging
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware

from crowd_anomaly import __version__
from crowd_anomaly.features import clip_features
from crowd_anomaly.model import flags_to_segments

from .schemas import PredictionResponse, Segment
from .service import get_model
from .settings import get_settings

log = logging.getLogger("crowd_anomaly.api")
settings = get_settings()

ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".mpeg", ".mpg", ".m4v", ".wmv"}
MAX_UPLOAD_BYTES = settings.max_upload_mb * 1024 * 1024
CHUNK = 1024 * 1024


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:  # warm the model so the first request is fast; /health reports failures
        get_model()
    except Exception as exc:
        log.warning("Model not loaded at startup: %s", exc)
    yield


app = FastAPI(
    title="Crowd Anomaly Detection API",
    version=__version__,
    description="Upload a video and receive frame-level anomaly scores.",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_origins),
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"name": "Crowd Anomaly Detection API", "docs": "/docs", "health": "/health", "predict": "POST /predict"}


@app.get("/health")
def health():
    try:
        model = get_model()
        return {"status": "ok", "model_loaded": True, "threshold": float(model.threshold_)}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Model unavailable: {exc}") from exc


def _score_video(path: Path):
    model = get_model()
    prediction = model.predict(clip_features(path, model.cfg, model.frame_size))
    return model, prediction


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    suffix = Path(file.filename or "video.mp4").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=415,
                            detail=f"Unsupported video extension. Use: {', '.join(sorted(ALLOWED_EXTENSIONS))}")
    tmp_path = None
    try:
        size = 0
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = Path(tmp.name)
            while chunk := await file.read(CHUNK):  # stream to disk, never hold the whole upload in RAM
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail=f"Video exceeds {settings.max_upload_mb} MB limit")
                tmp.write(chunk)
        if size == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")

        model, prediction = await run_in_threadpool(_score_video, tmp_path)  # CPU-bound: keep the event loop free
        scores = [float(x) for x in prediction["scores"]]
        flags = [bool(x) for x in prediction["flags"]]
        segments = [Segment(start_frame=int(a), end_frame=int(b))
                    for a, b in flags_to_segments(prediction["flags"], min_len=settings.min_segment_len)]
        return PredictionResponse(
            filename=Path(file.filename or "video").name, frames_processed=len(scores),
            threshold=float(model.threshold_), anomaly_frames=sum(flags), anomaly_detected=any(flags),
            segments=segments, scores=scores, flags=flags,
        )
    except HTTPException:
        raise
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=f"Model unavailable: {exc}") from exc
    except (ValueError, RuntimeError, IOError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        log.exception("Video processing failed")
        raise HTTPException(status_code=500,
                            detail="Video processing failed. Verify that the video is valid and try again.") from exc
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
        await file.close()
