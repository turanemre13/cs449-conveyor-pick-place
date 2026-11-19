"""Records and configuration for RGB-D perception.

All output positions and dimensions are in the scene's fixed world frame,
in metres; angles use radians in ``(-pi, pi]``. These records are pure Python
and do not depend on RAI or OpenCV.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from numbers import Real
from typing import ClassVar

from conveyor.types import ObjectObservation, Vector3

GEOMETRY_TYPES = ("box", "cylinder")


def _finite(name: str, value: float, *, minimum: float | None = None, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
        raise ValueError(f"{name} must be a finite real number")
    result = float(value)
    if positive and result <= 0.0:
        raise ValueError(f"{name} must be positive")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return result


def _vector3(name: str, value: Vector3, *, positive: bool = False) -> Vector3:
    if not isinstance(value, tuple) or len(value) != 3:
        raise ValueError(f"{name} must be a tuple of three finite real numbers")
    return tuple(_finite(name, x, positive=positive) for x in value)  # type: ignore[return-value]


@dataclass(frozen=True)
class PerceptionConfig:
    """Tunable parameters for overhead RGB-D conveyor object detection.

    Defaults match the dual-arm conveyor scene (belt top at z = 0.75 m,
    2.4 m x 0.4 m belt centered at x = 0, y = 0).
    """

    belt_top_z_m: float = 0.75
    belt_x_limits_m: tuple[float, float] = (-1.20, 1.20)
    belt_y_limits_m: tuple[float, float] = (-0.20, 0.20)
    roi_margin_m: float = 0.002
    min_object_height_m: float = 0.015
    max_object_height_m: float = 0.25
    depth_layer_tolerance_m: float = 0.010
    color_split_distance_rgb: float = 0.14
    min_contour_pixels: int = 12
    circularity_threshold: float = 0.855
    circle_vs_box_iou_margin: float = 0.02

    def __post_init__(self) -> None:
        _finite("belt_top_z_m", self.belt_top_z_m)
        if len(self.belt_x_limits_m) != 2 or self.belt_x_limits_m[0] >= self.belt_x_limits_m[1]:
            raise ValueError("belt_x_limits_m must be (low, high) with low < high")
        if len(self.belt_y_limits_m) != 2 or self.belt_y_limits_m[0] >= self.belt_y_limits_m[1]:
            raise ValueError("belt_y_limits_m must be (low, high) with low < high")
        for x in (*self.belt_x_limits_m, *self.belt_y_limits_m):
            _finite("belt_limits", x)
        _finite("roi_margin_m", self.roi_margin_m)
        _finite("min_object_height_m", self.min_object_height_m, positive=True)
        _finite("max_object_height_m", self.max_object_height_m, positive=True)
        if self.min_object_height_m >= self.max_object_height_m:
            raise ValueError("min_object_height_m must be smaller than max_object_height_m")
        _finite("depth_layer_tolerance_m", self.depth_layer_tolerance_m, positive=True)
        _finite("color_split_distance_rgb", self.color_split_distance_rgb, positive=True)
        if isinstance(self.min_contour_pixels, bool) or not isinstance(self.min_contour_pixels, int) or self.min_contour_pixels < 3:
            raise ValueError("min_contour_pixels must be an integer >= 3")
        _finite("circularity_threshold", self.circularity_threshold, positive=True)
        _finite("circle_vs_box_iou_margin", self.circle_vs_box_iou_margin)


@dataclass(frozen=True)
class DetectedObject:
    """Rich 3-D detection from a single RGB-D frame, in the world frame.

    Supplies the shared :class:`conveyor.types.ObjectObservation` plus estimated
    3-D dimensions, planar yaw, geometry classification (``"box"`` or
    ``"cylinder"``), mean colour and pixel footprint diagnostics.
    """

    detection_id: str
    simulation_time_s: float
    position_m: Vector3
    dimensions_m: Vector3
    yaw_rad: float
    geometry_type: str
    color_rgb: Vector3 = (0.5, 0.5, 0.5)
    pixel_centroid_uv: tuple[float, float] = (0.0, 0.0)
    pixel_area: int = 0
    confidence: float = 1.0
    frame: ClassVar[str] = "world"

    def __post_init__(self) -> None:
        if not isinstance(self.detection_id, str) or not self.detection_id.strip():
            raise ValueError("detection_id must be a non-empty string")
        _finite("simulation_time_s", self.simulation_time_s, minimum=0.0)
        _vector3("position_m", self.position_m)
        _vector3("dimensions_m", self.dimensions_m, positive=True)
        _finite("yaw_rad", self.yaw_rad)
        if self.geometry_type not in GEOMETRY_TYPES:
            raise ValueError(f"geometry_type must be one of {GEOMETRY_TYPES}")
        if self.geometry_type == "cylinder" and abs(self.dimensions_m[0] - self.dimensions_m[1]) > 1e-6:
            raise ValueError("cylinder dimensions_m must have equal x/y diameters")
        _vector3("color_rgb", self.color_rgb)
        if any(not (0.0 <= c <= 1.0) for c in self.color_rgb):
            raise ValueError("color_rgb components must lie in [0, 1]")
        if not isinstance(self.pixel_centroid_uv, tuple) or len(self.pixel_centroid_uv) != 2:
            raise ValueError("pixel_centroid_uv must be a (u, v) tuple")
        _finite("pixel_centroid_u", self.pixel_centroid_uv[0])
        _finite("pixel_centroid_v", self.pixel_centroid_uv[1])
        if isinstance(self.pixel_area, bool) or not isinstance(self.pixel_area, int) or self.pixel_area < 0:
            raise ValueError("pixel_area must be a non-negative integer")
        _finite("confidence", self.confidence, minimum=0.0)
        if self.confidence > 1.0:
            raise ValueError("confidence must lie in [0, 1]")

    @property
    def object_id(self) -> str:
        """Alias matching ``ObjectObservation.object_id``."""
        return self.detection_id

    def to_observation(self) -> ObjectObservation:
        """Convert to the shared centroid-only :class:`ObjectObservation`."""
        return ObjectObservation(
            object_id=self.detection_id,
            simulation_time_s=self.simulation_time_s,
            position_m=self.position_m,
        )
