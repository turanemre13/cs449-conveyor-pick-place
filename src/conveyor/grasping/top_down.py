"""Geometry-based top-down parallel-jaw grasp candidates for boxes and cylinders.

The selector proposes grasps from object shape, size, yaw and (predicted)
centroid, checks configurable gripper limits and neighbour clearance, and ranks
the survivors. It never calls IK/KOMO and never claims physical success: a
geometrically feasible candidate still needs planning (Doğa) and execution.

Frames (all world frame; quaternions in ``(w, x, y, z)`` order):

* **TCP** - ``position_m`` is the tool centre point midway between the finger
  pads. It coincides with the origin of the RAI Panda ``gripper`` frame
  (``a_gripper``/``b_gripper``), which RAI places 0.1035 m from
  ``panda_hand_joint`` along the hand axis, at the finger-pad centres.
* **RAI gripper frame** - ``quaternion_wxyz``. This is the orientation a
  planner should request for ``a_gripper``/``b_gripper``. RAI's
  ``panda.g`` defines ``gripper(panda_joint7): d(180 0 1 0) d(135 0 0 1)
  t(0 0 -.2105)``: its **+z axis points from the TCP back towards the wrist**
  (world **up** in a top-down grasp) and its **+x axis is the finger opening /
  jaw closing direction**. Hence a top-down grasp with closing heading ``psi``
  is simply ``Rz(psi)``: x = ``closing_axis_world``, z = world +z.
* **Approach frame** - ``approach_frame_quaternion_wxyz``: x = closing axis,
  z = approach direction (world **down**). This is the convention used before
  the RAI check; it is kept for readability and is converted explicitly:
  ``q_rai_gripper = q_approach * Rx(pi)`` (see :func:`approach_to_rai_gripper`).
  Do **not** send it to RAI as a gripper target.
* ``gripper_yaw_rad`` is the closing-axis heading, wrapped to ``(-pi/2, pi/2]``
  because a parallel-jaw gripper is symmetric under a half turn.

Motion awareness: when the object's velocity is given, candidates whose jaw
closing axis is perpendicular to the motion are preferred (configurable weight).
With the current interception model the gripper descends over the predicted
interception point while the object still moves; a closing axis parallel to the
motion puts the upstream finger in the object's path, and along-track timing
errors (the dominant prediction error) shift the object *into* a finger instead
of *along* the jaws. This preference is a ranking heuristic only: it does not
replace velocity-matched interception (planning/control) nor physical grasp
validation, and every candidate keeps ``physically_verified=False``.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from math import cos, hypot, inf, isfinite, pi, sin, sqrt

from conveyor.types import Vector3

from .geometry import (
    ObjectGeometry, footprint, polygon_footprint_distance, rectangle_corners, wrap_half_turn,
)

Neighbor = tuple[ObjectGeometry, Vector3]


@dataclass(frozen=True)
class GripperConfig:
    """Parallel-jaw gripper limits. Defaults approximate the Franka Hand.

    All values are metres. They are assumptions for geometric screening and
    should be checked against the RAI Panda model during integration.
    """

    max_opening_m: float = 0.08
    min_object_width_m: float = 0.01
    opening_clearance_m: float = 0.005
    finger_pad_width_m: float = 0.02
    finger_thickness_m: float = 0.012
    finger_length_m: float = 0.045
    grasp_depth_m: float = 0.025
    min_contact_depth_m: float = 0.01
    fingertip_below_tcp_m: float = 0.01
    support_surface_z_m: float = 0.75
    min_support_clearance_m: float = 0.003
    min_neighbor_clearance_m: float = 0.005
    pregrasp_height_m: float = 0.10
    cylinder_yaw_samples: int = 8
    score_weights: tuple[float, float, float] = (0.5, 0.3, 0.2)
    clearance_saturation_m: float = 0.05
    #: Extra score for a jaw closing axis perpendicular to the object's planar
    #: motion (``|sin|`` of the angle, in [0, 1]). 0 disables the preference.
    #: Applied only when ``object_velocity_m_s`` is given and the planar speed
    #: exceeds ``motion_speed_threshold_m_s``. It never overrides hard checks.
    #: Default 0.25 (study on 960 generated objects at 0.12 m/s): best grasps
    #: with |sin| >= 0.9 rise from 119 (weight 0) to 672; weight 0.5 adds only
    #: 2 more but raises near-limit openings (> 75 mm) from 13 to 19, i.e. it
    #: trades jaw-opening margin for alignment too aggressively.
    motion_alignment_weight: float = 0.25
    motion_speed_threshold_m_s: float = 1e-3

    def __post_init__(self) -> None:
        for name in (
            "max_opening_m", "min_object_width_m", "opening_clearance_m", "finger_pad_width_m",
            "finger_thickness_m", "finger_length_m", "grasp_depth_m", "min_contact_depth_m",
            "fingertip_below_tcp_m", "support_surface_z_m", "min_support_clearance_m",
            "min_neighbor_clearance_m", "pregrasp_height_m", "clearance_saturation_m",
            "motion_alignment_weight", "motion_speed_threshold_m_s",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
                raise ValueError(f"{name} must be a finite number")
            if value < 0 and name != "support_surface_z_m":
                raise ValueError(f"{name} must be non-negative")
        if self.max_opening_m <= 0 or self.finger_pad_width_m <= 0 or self.clearance_saturation_m <= 0:
            raise ValueError("max_opening_m, finger_pad_width_m and clearance_saturation_m must be positive")
        if isinstance(self.cylinder_yaw_samples, bool) or not isinstance(self.cylinder_yaw_samples, int) \
                or self.cylinder_yaw_samples < 1:
            raise ValueError("cylinder_yaw_samples must be a positive integer")
        if len(self.score_weights) != 3 or any(w < 0 for w in self.score_weights):
            raise ValueError("score_weights must be three non-negative numbers")


#: Name of the orientation convention used by ``GraspCandidate.quaternion_wxyz``.
RAI_GRIPPER_FRAME_CONVENTION = "rai_panda_gripper: x=closing axis, z=towards wrist (world up), origin=TCP"


@dataclass(frozen=True)
class GraspCandidate:
    """One top-down grasp hypothesis. ``feasible`` is a geometric verdict only.

    ``quaternion_wxyz`` is the RAI Panda ``gripper`` frame orientation (see the
    module docstring); ``approach_frame_quaternion_wxyz`` is the same grasp with
    +z along the approach direction. ``motion_alignment`` is ``|sin|`` of the
    angle between closing axis and planar object motion (1 = perpendicular), or
    ``None`` when no motion was given.
    """

    object_id: str
    candidate_id: str
    grasp_type: str
    position_m: Vector3
    pregrasp_position_m: Vector3
    quaternion_wxyz: tuple[float, float, float, float]
    gripper_yaw_rad: float
    closing_axis_world: Vector3
    approach_axis_world: Vector3
    object_width_m: float
    required_opening_m: float
    contact_length_m: float
    neighbor_clearance_m: float
    score: float
    feasible: bool
    rejection_reasons: tuple[str, ...] = ()
    physically_verified: bool = False
    approach_frame_quaternion_wxyz: tuple[float, float, float, float] = (0.0, 1.0, 0.0, 0.0)
    motion_alignment: float | None = None
    frame_convention: str = RAI_GRIPPER_FRAME_CONVENTION


@dataclass(frozen=True)
class GraspSelectionResult:
    """Ranked feasible candidates (best first) and rejected ones with reasons."""

    object_id: str
    feasible: tuple[GraspCandidate, ...]
    rejected: tuple[GraspCandidate, ...] = field(default_factory=tuple)

    @property
    def best(self) -> GraspCandidate | None:
        return self.feasible[0] if self.feasible else None

    @property
    def has_feasible(self) -> bool:
        return bool(self.feasible)


Quaternion = tuple[float, float, float, float]


def quaternion_multiply(first: Quaternion, second: Quaternion) -> Quaternion:
    """Hamilton product ``first * second`` of ``(w, x, y, z)`` quaternions."""
    w1, x1, y1, z1 = first
    w2, x2, y2, z2 = second
    return (
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    )


def _canonical(q: Quaternion) -> Quaternion:
    """Unit quaternion with a deterministic sign (``q`` and ``-q`` are the same rotation)."""
    norm = sqrt(sum(c * c for c in q))
    q = tuple(c / norm + 0.0 for c in q)  # "+ 0.0" turns -0.0 into 0.0
    for component in q:
        if abs(component) > 1e-12:
            return q if component > 0 else tuple(-c + 0.0 for c in q)
    return q


#: Rotation by pi about the local x axis: flips local y and z, keeps the closing axis.
ROT_X_PI: Quaternion = (0.0, 1.0, 0.0, 0.0)


def approach_frame_quaternion(yaw: float) -> Quaternion:
    """``Rz(yaw) * Rx(pi)``: +x along the closing axis, +z along the approach (world down)."""
    return _canonical(quaternion_multiply((cos(yaw / 2), 0.0, 0.0, sin(yaw / 2)), ROT_X_PI))


def approach_to_rai_gripper(approach_quaternion: Quaternion) -> Quaternion:
    """Convert an approach-frame orientation to the RAI Panda ``gripper`` frame.

    Both frames share +x (closing axis); RAI's +z points back to the wrist, i.e.
    opposite to the approach direction, so ``q_rai = q_approach * Rx(pi)``.
    """
    return _canonical(quaternion_multiply(approach_quaternion, ROT_X_PI))


def rai_gripper_quaternion(yaw: float) -> Quaternion:
    """RAI ``gripper`` orientation of a top-down grasp with closing heading ``yaw``."""
    return approach_to_rai_gripper(approach_frame_quaternion(yaw))


def _axis_options(geometry: ObjectGeometry, config: GripperConfig) -> list[tuple[str, float, float, float]]:
    """``(grasp_type, closing_yaw, width_along_closing, contact_length)`` options."""
    dx, dy, _ = geometry.dimensions_m
    if geometry.geometry_type == "box":
        return [
            ("top_down_box_close_x", geometry.yaw_rad, dx, dy),
            ("top_down_box_close_y", geometry.yaw_rad + pi / 2, dy, dx),
        ]
    samples = config.cylinder_yaw_samples
    return [("top_down_cylinder", k * pi / samples, dx, dx) for k in range(samples)]


def select_top_down_grasps(
    geometry: ObjectGeometry,
    position_m: Vector3,
    *,
    neighbors: Iterable[Neighbor] = (),
    config: GripperConfig | None = None,
    object_velocity_m_s: Vector3 | None = None,
) -> GraspSelectionResult:
    """Generate, screen and rank top-down grasps for one object.

    Parameters
    ----------
    geometry:
        Shape, full dimensions and yaw of the target.
    position_m:
        World centroid where the grasp will happen, normally the predicted
        interception position from :mod:`conveyor.prediction`.
    neighbors:
        Other objects ``(geometry, centroid)`` at the same time. Their planar
        footprints are compared with the open finger pads.
    config:
        Gripper limits; defaults approximate the Franka Hand.
    object_velocity_m_s:
        World velocity of the target (e.g. the tracked belt velocity). When its
        planar speed exceeds ``config.motion_speed_threshold_m_s`` the ranking
        prefers jaw closing axes perpendicular to the motion.

    Ranking score (higher is better) is a weighted sum of terms in [0, 1]:
    remaining jaw-opening margin, finger-pad contact length, neighbour
    clearance (saturating at ``clearance_saturation_m``) and, for moving
    objects, ``motion_alignment_weight * |sin(closing axis, motion)|``. Scores
    are only meaningful for comparing candidates of the same call. The motion
    term changes the *ranking* only; width, height and clearance checks decide
    geometric feasibility exactly as for a static object.
    """
    config = config or GripperConfig()
    if not isinstance(geometry, ObjectGeometry):
        raise TypeError("geometry must be an ObjectGeometry")
    if not isinstance(position_m, tuple) or len(position_m) != 3 or not all(
        isinstance(v, (int, float)) and not isinstance(v, bool) and isfinite(v) for v in position_m
    ):
        raise ValueError("position_m must be a tuple of three finite numbers")
    cx, cy, cz = (float(v) for v in position_m)
    motion_direction = None
    if object_velocity_m_s is not None:
        if not isinstance(object_velocity_m_s, tuple) or len(object_velocity_m_s) != 3 or not all(
            isinstance(v, (int, float)) and not isinstance(v, bool) and isfinite(v) for v in object_velocity_m_s
        ):
            raise ValueError("object_velocity_m_s must be a tuple of three finite numbers")
        planar_speed = hypot(object_velocity_m_s[0], object_velocity_m_s[1])
        if planar_speed > config.motion_speed_threshold_m_s:
            motion_direction = (object_velocity_m_s[0] / planar_speed, object_velocity_m_s[1] / planar_speed)
    # Cheap bounding-circle prefilter: neighbours farther than the open pads
    # plus the clearance saturation distance cannot affect the result.
    pad_reach = hypot(config.max_opening_m / 2 + config.finger_thickness_m, config.finger_pad_width_m / 2)
    neighbor_footprints = []
    for g, p in neighbors:
        if g.object_id == geometry.object_id:
            continue
        radius = hypot(g.dimensions_m[0], g.dimensions_m[1]) / 2
        if hypot(p[0] - cx, p[1] - cy) > pad_reach + radius + config.clearance_saturation_m:
            continue
        neighbor_footprints.append(footprint(g, (p[0], p[1])))

    height = geometry.height_m
    top_z = cz + height / 2
    depth = min(config.grasp_depth_m, height / 2, config.finger_length_m)
    tcp_z = top_z - depth
    lowest_tcp_z = config.support_surface_z_m + config.min_support_clearance_m + config.fingertip_below_tcp_m
    height_reasons: list[str] = []
    if tcp_z < lowest_tcp_z:
        tcp_z = lowest_tcp_z  # Raise the TCP so the fingertips stay above the belt.
        if top_z - tcp_z < config.min_contact_depth_m:
            height_reasons.append("object_too_short_for_finger_contact")

    feasible: list[GraspCandidate] = []
    rejected: list[GraspCandidate] = []
    seen_yaws: set[float] = set()
    for index, (grasp_type, raw_yaw, width, contact) in enumerate(_axis_options(geometry, config)):
        yaw = wrap_half_turn(raw_yaw)
        key = round(yaw, 9)
        if key in seen_yaws:
            continue
        seen_yaws.add(key)
        reasons = list(height_reasons)
        required = width + 2 * config.opening_clearance_m
        if required > config.max_opening_m:
            reasons.append("object_too_wide_for_gripper")
        if width < config.min_object_width_m:
            reasons.append("object_too_narrow_for_gripper")

        axis = (cos(yaw), sin(yaw))
        perpendicular_half = config.finger_pad_width_m / 2
        pad_offset = min(required, config.max_opening_m) / 2 + config.finger_thickness_m / 2
        clearance = inf
        for sign in (-1.0, 1.0):
            pad_center = (cx + sign * pad_offset * axis[0], cy + sign * pad_offset * axis[1])
            pad = rectangle_corners(pad_center, config.finger_thickness_m / 2, perpendicular_half, yaw)
            for other in neighbor_footprints:
                clearance = min(clearance, polygon_footprint_distance(pad, other))
        if clearance < config.min_neighbor_clearance_m:
            reasons.append("insufficient_neighbor_clearance")

        w_margin, w_contact, w_clear = config.score_weights
        margin_term = max(0.0, config.max_opening_m - required) / config.max_opening_m
        contact_term = min(contact, config.finger_pad_width_m) / config.finger_pad_width_m
        clear_term = 1.0 if clearance == inf else min(clearance, config.clearance_saturation_m) / config.clearance_saturation_m
        score = w_margin * margin_term + w_contact * contact_term + w_clear * clear_term
        alignment = None
        if motion_direction is not None:
            # |sin| of the angle between closing axis and motion: 1 = perpendicular.
            alignment = abs(axis[0] * motion_direction[1] - axis[1] * motion_direction[0])
            score += config.motion_alignment_weight * alignment

        position = (cx, cy, tcp_z)
        candidate = GraspCandidate(
            object_id=geometry.object_id,
            candidate_id=f"{geometry.object_id}/{grasp_type}/{index}",
            grasp_type=grasp_type,
            position_m=position,
            pregrasp_position_m=(cx, cy, tcp_z + config.pregrasp_height_m),
            quaternion_wxyz=rai_gripper_quaternion(yaw),
            approach_frame_quaternion_wxyz=approach_frame_quaternion(yaw),
            gripper_yaw_rad=yaw,
            closing_axis_world=(axis[0], axis[1], 0.0),
            approach_axis_world=(0.0, 0.0, -1.0),
            object_width_m=width,
            required_opening_m=required,
            contact_length_m=contact,
            neighbor_clearance_m=clearance,
            score=score if not reasons else 0.0,
            feasible=not reasons,
            rejection_reasons=tuple(reasons),
            motion_alignment=alignment,
        )
        (rejected if reasons else feasible).append(candidate)

    # Deterministic ranking: score, then smaller |yaw|, then candidate id.
    feasible.sort(key=lambda c: (-round(c.score, 12), abs(c.gripper_yaw_rad), c.candidate_id))
    return GraspSelectionResult(geometry.object_id, tuple(feasible), tuple(rejected))
