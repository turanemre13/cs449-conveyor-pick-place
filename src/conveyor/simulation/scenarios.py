"""Seeded object layouts, independent of RAI and perception algorithms."""

from dataclasses import dataclass
from math import cos, hypot, isfinite, pi, sin
from numbers import Real
import random


BELT_LENGTH_M = 2.4
BELT_WIDTH_M = .4
BELT_TOP_M = .75
EDGE_MARGIN_M = .005
OBJECT_CLEARANCE_M = .008
ORDERED_MIN_GAP_M = .10
MAX_OBJECT_COUNT = 24
FEED_SPACING_M = 1.18
SHOWCASE_FEED = (
    (0.00, "box"), (1.18, "cylinder"), (2.36, "box"), (3.54, "cylinder"),
    (4.12, "sphere"),
    (4.64, "box"), (4.82, "box"), (5.18, "box"),
    (5.72, "sphere"),
)

_COLORS = (
    (.90, .32, .12), (.16, .57, .88), (.91, .72, .12),
    (.31, .73, .37), (.66, .30, .80), (.91, .37, .62),
)


def _finite_tuple(name: str, value: tuple, count: int) -> None:
    if not isinstance(value, tuple) or len(value) != count:
        raise ValueError(f"{name} must be an immutable tuple of length {count}")
    if any(isinstance(x, bool) or not isinstance(x, Real) or not isfinite(x) for x in value):
        raise ValueError(f"{name} must contain finite real numbers")


@dataclass(frozen=True)
class ObjectSpec:
    """Initial world pose and local full extents; cylinders stand on world +z."""

    object_id: str
    geometry_type: str
    dimensions_m: tuple[float, float, float]
    position_m: tuple[float, float, float]
    yaw_rad: float
    color_rgb: tuple[float, float, float]

    def __post_init__(self) -> None:
        if not isinstance(self.object_id, str) or not self.object_id.strip():
            raise ValueError("object_id must be a nonblank string")
        if self.geometry_type not in ("box", "cylinder", "sphere"):
            raise ValueError("geometry_type must be box, cylinder, or sphere")
        _finite_tuple("dimensions_m", self.dimensions_m, 3)
        _finite_tuple("position_m", self.position_m, 3)
        _finite_tuple("color_rgb", self.color_rgb, 3)
        if any(x <= 0 for x in self.dimensions_m):
            raise ValueError("dimensions_m must be positive")
        if any(not 0 <= x <= 1 for x in self.color_rgb):
            raise ValueError("color_rgb components must lie in [0, 1]")
        if isinstance(self.yaw_rad, bool) or not isinstance(self.yaw_rad, Real) or not isfinite(self.yaw_rad):
            raise ValueError("yaw_rad must be a finite real number")
        if self.geometry_type == "cylinder" and self.dimensions_m[0] != self.dimensions_m[1]:
            raise ValueError("Cylinder x/y dimensions must both equal its diameter")
        if self.geometry_type == "sphere" and len(set(self.dimensions_m)) != 1:
            raise ValueError("Sphere dimensions must all equal its diameter")


def planar_half_extents(spec: ObjectSpec) -> tuple[float, float]:
    """World-axis half extents of the rotated footprint, in metres."""
    dx, dy, _ = spec.dimensions_m
    if spec.geometry_type != "box":
        return dx / 2, dx / 2
    c, s = abs(cos(spec.yaw_rad)), abs(sin(spec.yaw_rad))
    return (c * dx + s * dy) / 2, (s * dx + c * dy) / 2


def footprints_overlap(first: ObjectSpec, second: ObjectSpec, clearance_m: float = 0.0) -> bool:
    """Whether footprints violate a clearance; box/box uses conservative SAT.

    A separating projection with the requested gap guarantees separation. Near
    box corners this can reject a physically valid placement, never admit an
    intersection. Circle/circle and circle/box distances are exact.
    """
    if isinstance(clearance_m, bool) or not isinstance(clearance_m, Real) or not isfinite(clearance_m) or clearance_m < 0:
        raise ValueError("clearance_m must be finite and non-negative")
    delta = tuple(second.position_m[i] - first.position_m[i] for i in (0, 1))
    first_round = first.geometry_type != "box"
    second_round = second.geometry_type != "box"
    if first_round and second_round:
        radii = (first.dimensions_m[0] + second.dimensions_m[0]) / 2
        return hypot(*delta) < radii + clearance_m - 1e-12
    if first_round != second_round:
        box, circle = (first, second) if first.geometry_type == "box" else (second, first)
        dx, dy = (circle.position_m[i] - box.position_m[i] for i in (0, 1))
        c, s = cos(box.yaw_rad), sin(box.yaw_rad)
        local_x, local_y = c * dx + s * dy, -s * dx + c * dy
        gap_x = max(abs(local_x) - box.dimensions_m[0] / 2, 0.0)
        gap_y = max(abs(local_y) - box.dimensions_m[1] / 2, 0.0)
        return hypot(gap_x, gap_y) < circle.dimensions_m[0] / 2 + clearance_m - 1e-12

    axes_first = ((cos(first.yaw_rad), sin(first.yaw_rad)), (-sin(first.yaw_rad), cos(first.yaw_rad)))
    axes_second = ((cos(second.yaw_rad), sin(second.yaw_rad)), (-sin(second.yaw_rad), cos(second.yaw_rad)))
    for axis in (*axes_first, *axes_second):
        projection = abs(delta[0] * axis[0] + delta[1] * axis[1])
        radius_first = sum(first.dimensions_m[i] / 2 * abs(sum(axis[j] * axes_first[i][j] for j in (0, 1))) for i in (0, 1))
        radius_second = sum(second.dimensions_m[i] / 2 * abs(sum(axis[j] * axes_second[i][j] for j in (0, 1))) for i in (0, 1))
        if projection >= radius_first + radius_second + clearance_m - 1e-12:
            return False
    return True


def generate_scenario(name: str = "clutter", seed: int = 7, object_count: int = 8) -> tuple[ObjectSpec, ...]:
    """Generate seeded layouts with stable IDs, including an upstream feed.

    ``single`` always returns the original one-box baseline. Other presets honor
    counts from 1 to 24, subject to available space. ``ordered`` places a single
    file in ascending ID/x order with at least 10 cm between longitudinal
    footprints. Layout generation never changes global random state.
    """
    if name not in ("single", "ordered", "sparse", "clutter", "feed", "showcase"):
        raise ValueError("scenario must be single, ordered, sparse, clutter, feed, or showcase")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    if isinstance(object_count, bool) or not isinstance(object_count, int) or not 1 <= object_count <= MAX_OBJECT_COUNT:
        raise ValueError(f"object_count must be an integer from 1 to {MAX_OBJECT_COUNT}")
    if name == "single":
        return (ObjectSpec("object_001", "box", (.06, .045, .08), (-1.05, 0.0, .79), 0.0, _COLORS[0]),)

    rng = random.Random(seed)
    if name in ("feed", "showcase"):
        return _feed_layout(rng, object_count, showcase=name == "showcase")
    if name == "ordered":
        return _ordered_layout(rng, object_count)
    placed: list[ObjectSpec] = []
    for index in range(object_count):
        geometry = "box" if index % 2 == 0 else "cylinder"
        if geometry == "box":
            dims = (rng.uniform(.045, .10), rng.uniform(.035, .065), rng.uniform(.04, .09))
        else:
            diameter = rng.uniform(.04, .065)
            dims = (diameter, diameter, rng.uniform(.04, .09))
        yaw = rng.uniform(-pi, pi)
        color = _COLORS[index % len(_COLORS)]
        identifier = f"object_{index + 1:03d}"
        prototype = ObjectSpec(identifier, geometry, dims, (0.0, 0.0, BELT_TOP_M + dims[2] / 2), yaw, color)
        hx, hy = planar_half_extents(prototype)
        low_y, high_y = -BELT_WIDTH_M / 2 + EDGE_MARGIN_M + hy, BELT_WIDTH_M / 2 - EDGE_MARGIN_M - hy
        low_x, high_x = (-1.08, -.12) if name == "clutter" else (-1.10, 1.05)
        low_x = max(low_x, -BELT_LENGTH_M / 2 + EDGE_MARGIN_M + hx)
        high_x = min(high_x, BELT_LENGTH_M / 2 - EDGE_MARGIN_M - hx)
        candidates = []
        for _ in range(512):
            if not placed and name == "clutter":
                x, y = -.85, rng.uniform(-.05, .05)
            else:
                x, y = rng.uniform(low_x, high_x), rng.uniform(low_y, high_y)
            candidate = ObjectSpec(identifier, geometry, dims, (x, y, prototype.position_m[2]), yaw, color)
            if any(footprints_overlap(candidate, old, OBJECT_CLEARANCE_M) for old in placed):
                continue
            nearest = min((hypot(x - old.position_m[0], y - old.position_m[1]) for old in placed), default=0.0)
            candidates.append((nearest, candidate))
            if not placed:
                break
        if not candidates:
            raise ValueError(f"Could not place {object_count} objects in {name!r}; reduce --objects or choose another seed")
        selected = min(candidates, key=lambda item: item[0]) if name == "clutter" else max(candidates, key=lambda item: item[0])
        placed.append(selected[1])
    return tuple(placed)


def _feed_layout(rng: random.Random, object_count: int, *, showcase: bool) -> tuple[ObjectSpec, ...]:
    """Queue mixed products on the upstream deck before any enter the main belt."""
    products = []
    for index in range(object_count):
        if showcase:
            cycle, phase = divmod(index, len(SHOWCASE_FEED))
            distance, geometry = SHOWCASE_FEED[phase]
            distance += cycle * (SHOWCASE_FEED[-1][0] + FEED_SPACING_M)
        else:
            distance, geometry = index * FEED_SPACING_M, "box" if index % 2 == 0 else "cylinder"
        if geometry == "sphere":
            dimensions, yaw = (.070, .070, .070), 0.0
        elif index < 2:
            width = rng.uniform(.049, .054)
            depth = rng.uniform(.039, .044) if geometry == "box" else width
            height = rng.uniform(.064, .077)
            yaw = rng.uniform(-.15, .15) if geometry == "box" else 0.0
            dimensions = (width, depth, height)
        else:
            template = products[0 if geometry == "box" else 1]
            dimensions, yaw = template.dimensions_m, template.yaw_rad
        x = -BELT_LENGTH_M / 2 - .12 - distance
        color = (.65, .22, .79) if geometry == "sphere" else _COLORS[index % len(_COLORS)]
        products.append(ObjectSpec(
            f"object_{index + 1:03d}", geometry, dimensions,
            (x, -.025, BELT_TOP_M + dimensions[2] / 2), yaw, color,
        ))
    return tuple(products)


def _ordered_layout(rng: random.Random, object_count: int) -> tuple[ObjectSpec, ...]:
    """Spread a single file across the belt, keeping the edge gaps readable."""
    prototypes = []
    for index in range(object_count):
        geometry = "box" if index % 2 == 0 else "cylinder"
        if geometry == "box":
            dims = (rng.uniform(.045, .10), rng.uniform(.035, .065), rng.uniform(.04, .09))
        else:
            diameter = rng.uniform(.04, .065)
            dims = (diameter, diameter, rng.uniform(.04, .09))
        prototypes.append(ObjectSpec(
            f"object_{index + 1:03d}", geometry, dims,
            (0.0, rng.uniform(-.035, .035), BELT_TOP_M + dims[2] / 2),
            rng.uniform(-pi, pi), _COLORS[index % len(_COLORS)],
        ))

    # Reserve room at the ends; object widths include their seeded yaw.
    left_edge, right_edge = -1.10, 1.08
    widths = [2 * planar_half_extents(spec)[0] for spec in prototypes]
    spare = right_edge - left_edge - sum(widths)
    if spare < ORDERED_MIN_GAP_M * (object_count - 1):
        raise ValueError(
            f"Cannot fit {object_count} ordered objects with at least "
            f"{ORDERED_MIN_GAP_M:.2f} m longitudinal gaps; reduce --objects "
            "or choose sparse/clutter"
        )
    gap = spare / (object_count - 1) if object_count > 1 else 0.0
    cursor = left_edge
    result = []
    for spec, width in zip(prototypes, widths):
        result.append(ObjectSpec(
            spec.object_id, spec.geometry_type, spec.dimensions_m,
            (cursor + width / 2, spec.position_m[1], spec.position_m[2]),
            spec.yaw_rad, spec.color_rgb,
        ))
        cursor += width + gap
    return tuple(result)
