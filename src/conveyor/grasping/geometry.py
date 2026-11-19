"""Object geometry records and planar footprint distance helpers.

``TrackedState`` carries only a centroid and velocity. Grasp selection also
needs shape, full dimensions and yaw, so they are provided explicitly through
:class:`ObjectGeometry` instead of extending the shared tracking record.
Adapters build it from simulation ``ObjectSpec`` / ``ObjectGroundTruth``
records by duck typing (no simulation import). A perception module can later
construct the same record from segmented footprints.

Conventions follow the repository: world frame, metres, radians; boxes use full
local extents ``(dx, dy, dz)`` with ``yaw_rad`` about world +z; cylinders stand
upright with dimensions ``(diameter, diameter, height)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, cos, hypot, isfinite, pi, sin, sqrt
from numbers import Real
from typing import Any

Point2 = tuple[float, float]
GEOMETRY_TYPES = ("box", "cylinder")


@dataclass(frozen=True)
class ObjectGeometry:
    """Shape information needed by grasp selection, independent of time."""

    object_id: str
    geometry_type: str
    dimensions_m: tuple[float, float, float]
    yaw_rad: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.object_id, str) or not self.object_id.strip():
            raise ValueError("object_id must be a non-empty string")
        if self.geometry_type not in GEOMETRY_TYPES:
            raise ValueError(f"geometry_type must be one of {GEOMETRY_TYPES}")
        if not isinstance(self.dimensions_m, tuple) or len(self.dimensions_m) != 3:
            raise ValueError("dimensions_m must be a tuple of three numbers")
        for value in (*self.dimensions_m, self.yaw_rad):
            if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
                raise ValueError("dimensions_m and yaw_rad must be finite real numbers")
        if any(value <= 0 for value in self.dimensions_m):
            raise ValueError("dimensions_m must be positive")
        if self.geometry_type == "cylinder" and abs(self.dimensions_m[0] - self.dimensions_m[1]) > 1e-9:
            raise ValueError("cylinder dimensions must be (diameter, diameter, height)")

    @property
    def height_m(self) -> float:
        return self.dimensions_m[2]


def yaw_from_quaternion_wxyz(quaternion: tuple[float, float, float, float]) -> float:
    """Heading about world +z of a ``(w, x, y, z)`` quaternion, in ``(-pi, pi]``."""
    w, x, y, z = (float(v) for v in quaternion)
    return atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def tilt_from_quaternion_wxyz(quaternion: tuple[float, float, float, float]) -> float:
    """Angle between the object's local +z axis and world +z, in radians."""
    w, x, y, z = (float(v) for v in quaternion)
    norm = sqrt(w * w + x * x + y * y + z * z)
    if norm == 0:
        raise ValueError("quaternion must be non-zero")
    w, x, y, z = (v / norm for v in (w, x, y, z))
    local_z_world_z = 1.0 - 2.0 * (x * x + y * y)
    return float(atan2(sqrt(max(0.0, 1.0 - local_z_world_z ** 2)), local_z_world_z))


def geometry_from_spec(spec: Any) -> ObjectGeometry:
    """Build geometry from a simulation ``ObjectSpec``-like record."""
    return ObjectGeometry(
        object_id=spec.object_id, geometry_type=spec.geometry_type,
        dimensions_m=tuple(float(x) for x in spec.dimensions_m), yaw_rad=float(spec.yaw_rad),
    )


def geometry_from_ground_truth(record: Any, *, max_tilt_rad: float = 0.17) -> ObjectGeometry:
    """Build geometry from an ``ObjectGroundTruth``-like record.

    Top-down grasps assume an upright object; records tilted more than
    ``max_tilt_rad`` (about 10 degrees by default) raise ``ValueError``.
    """
    quaternion = tuple(record.quaternion_wxyz)
    tilt = tilt_from_quaternion_wxyz(quaternion)
    if tilt > max_tilt_rad:
        raise ValueError(f"{record.object_id!r} is tilted by {tilt:.2f} rad; top-down geometry is not valid")
    return ObjectGeometry(
        object_id=record.object_id, geometry_type=getattr(record, "geometry_type", "box"),
        dimensions_m=tuple(float(x) for x in record.dimensions_m),
        yaw_rad=yaw_from_quaternion_wxyz(quaternion),
    )


def wrap_angle(angle: float) -> float:
    """Wrap an angle to ``(-pi, pi]``."""
    wrapped = (angle + pi) % (2 * pi) - pi
    return pi if wrapped == -pi else wrapped


def wrap_half_turn(angle: float) -> float:
    """Wrap an angle to ``(-pi/2, pi/2]``; a parallel-jaw gripper is 180° symmetric."""
    wrapped = (angle + pi / 2) % pi - pi / 2
    return pi / 2 if wrapped == -pi / 2 else wrapped


# ---------------------------------------------------------------- 2-D footprints
def rectangle_corners(center: Point2, half_x: float, half_y: float, yaw: float) -> list[Point2]:
    """Counter-clockwise corners of an oriented rectangle."""
    c, s = cos(yaw), sin(yaw)
    corners = []
    for sx, sy in ((1, 1), (-1, 1), (-1, -1), (1, -1)):
        lx, ly = sx * half_x, sy * half_y
        corners.append((center[0] + c * lx - s * ly, center[1] + s * lx + c * ly))
    return corners


def footprint(geometry: ObjectGeometry, center: Point2) -> tuple[str, Any]:
    """``("polygon", corners)`` for boxes or ``("circle", (center, radius))`` for cylinders."""
    if geometry.geometry_type == "cylinder":
        return "circle", (center, geometry.dimensions_m[0] / 2)
    return "polygon", rectangle_corners(
        center, geometry.dimensions_m[0] / 2, geometry.dimensions_m[1] / 2, geometry.yaw_rad,
    )


def _point_segment_distance(p: Point2, a: Point2, b: Point2) -> float:
    abx, aby = b[0] - a[0], b[1] - a[1]
    length_sq = abx * abx + aby * aby
    if length_sq == 0:
        return hypot(p[0] - a[0], p[1] - a[1])
    t = max(0.0, min(1.0, ((p[0] - a[0]) * abx + (p[1] - a[1]) * aby) / length_sq))
    return hypot(p[0] - a[0] - t * abx, p[1] - a[1] - t * aby)


def _edges(polygon: list[Point2]):
    return zip(polygon, polygon[1:] + polygon[:1])


def _point_in_convex_polygon(p: Point2, polygon: list[Point2]) -> bool:
    sign = 0
    for a, b in _edges(polygon):
        cross = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
        if abs(cross) < 1e-15:
            continue
        current = 1 if cross > 0 else -1
        if sign == 0:
            sign = current
        elif current != sign:
            return False
    return True


def _convex_polygons_intersect(first: list[Point2], second: list[Point2]) -> bool:
    for polygon in (first, second):
        for a, b in _edges(polygon):
            axis = (-(b[1] - a[1]), b[0] - a[0])
            p1 = [axis[0] * x + axis[1] * y for x, y in first]
            p2 = [axis[0] * x + axis[1] * y for x, y in second]
            if max(p1) < min(p2) or max(p2) < min(p1):
                return False
    return True


def polygon_distance(first: list[Point2], second: list[Point2]) -> float:
    """Euclidean distance between two convex polygons; zero when they overlap."""
    if _convex_polygons_intersect(first, second):
        return 0.0
    distances = [_point_segment_distance(p, a, b) for p in first for a, b in _edges(second)]
    distances += [_point_segment_distance(p, a, b) for p in second for a, b in _edges(first)]
    return min(distances)


def polygon_circle_distance(polygon: list[Point2], center: Point2, radius: float) -> float:
    """Distance between a convex polygon and a disc; zero when they overlap."""
    if _point_in_convex_polygon(center, polygon):
        return 0.0
    edge_distance = min(_point_segment_distance(center, a, b) for a, b in _edges(polygon))
    return max(0.0, edge_distance - radius)


def polygon_footprint_distance(polygon: list[Point2], other: tuple[str, Any]) -> float:
    """Distance from a convex polygon to a ``footprint(...)`` result."""
    kind, data = other
    if kind == "circle":
        center, radius = data
        return polygon_circle_distance(polygon, center, radius)
    return polygon_distance(polygon, data)
