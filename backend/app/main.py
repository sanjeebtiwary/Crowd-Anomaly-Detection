from pathlib import Path
import os
import tempfile
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from src.features import clip_features
from src.model import flags_to_segments
from .schemas import PredictionResponse, Segment
from .service import get_model

app = FastAPI(title="Crowd Anomaly Detection API", version="1.0.0", description="Upload a video and receive frame-level anomaly scores.")
origins = [x.strip() for x in os.getenv("CORS_ORIGINS", "*").split(",") if x.strip()]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=False, allow_methods=["GET", "POST"], allow_headers=["*"])
ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".mpeg", ".mpg", ".m4v", ".wmv"}
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_MB", "100")) * 1024 * 1024

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

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    suffix = Path(file.filename or "video.mp4").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=415, detail=f"Unsupported video extension. Use: {', '.join(sorted(ALLOWED_EXTENSIONS))}")
    tmp_path = None
    try:
        content = await file.read(MAX_UPLOAD_BYTES + 1)
        if not content:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")
        if len(content) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail=f"Video exceeds {MAX_UPLOAD_BYTES // (1024*1024)} MB limit")
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(content)
            tmp_path = Path(tmp.name)
        model = get_model()
        features = clip_features(tmp_path, model.cfg, model.frame_size)
        prediction = model.predict(features)
        scores = [float(x) for x in prediction["scores"]]
        flags = [bool(x) for x in prediction["flags"]]
        segments = [Segment(start_frame=int(a), end_frame=int(b)) for a, b in flags_to_segments(prediction["flags"], min_len=3)]
        return PredictionResponse(filename=Path(file.filename or "video").name, frames_processed=len(scores), threshold=float(model.threshold_), anomaly_frames=sum(flags), anomaly_detected=any(flags), segments=segments, scores=scores, flags=flags)
    except HTTPException:
        raise
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Video processing failed. Verify that the video is valid and try again.") from exc
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
        await file.close()
