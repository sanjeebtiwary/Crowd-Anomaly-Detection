from pydantic import BaseModel, Field
from typing import List

class Segment(BaseModel):
    start_frame: int
    end_frame: int

class PredictionResponse(BaseModel):
    filename: str
    frames_processed: int
    threshold: float
    anomaly_frames: int
    anomaly_detected: bool
    segments: List[Segment]
    scores: List[float] = Field(description="One anomaly score per frame")
    flags: List[bool] = Field(description="Whether each frame exceeds the model threshold")
