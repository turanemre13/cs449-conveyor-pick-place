"""OpenCV RGB-D detections and calibrated world-frame observations. Owner: Heysem.

Exposes :class:`RGBDObjectDetector`, :class:`PerceptionConfig`,
:class:`DetectedObject`, and synthetic RGB-D helpers for offline testing.
"""

from .detector import RGBDObjectDetector
from .models import DetectedObject, PerceptionConfig
from .synthetic import (
    advance_specs_along_belt,
    default_overhead_calibration,
    render_synthetic_rgbd,
)

__all__ = [
    "DetectedObject",
    "PerceptionConfig",
    "RGBDObjectDetector",
    "advance_specs_along_belt",
    "default_overhead_calibration",
    "render_synthetic_rgbd",
]
