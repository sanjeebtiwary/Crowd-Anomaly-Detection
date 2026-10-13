"""Crowd anomaly detection on the UCSD pedestrian dataset.

Pipeline: video frames -> dense optical flow -> per-cell motion statistics ->
Gaussian-mixture model of *normal* motion -> anomaly score (negative log-likelihood).
"""
from .config import Config
from .model import CrowdAnomalyModel

__all__ = ["Config", "CrowdAnomalyModel"]
__version__ = "1.0.0"
