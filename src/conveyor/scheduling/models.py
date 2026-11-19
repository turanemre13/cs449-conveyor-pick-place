"""Configuration and result records for dual-arm interception scheduling.

All times are absolute simulation seconds (the same clock as ``TrackedState``),
positions are world-frame metres. Every robot, timing and reachability value
here is an explicit, configurable *approximation* used before Doğa's IK/KOMO
planner is integrated; none of them is a verified physical capability.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from numbers import Real

from conveyor.grasping.geometry import ObjectGeometry
from conveyor.grasping.top_down import GraspCandidate, GripperConfig
from conveyor.types import TrackedState, Vector3


def _finite(name: str, value: float, *, minimum: float | None = None, positive: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
        raise ValueError(f"{name} must be a finite real number")
    if positive and value <= 0:
        raise ValueError(f"{name} must be positive")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")


def _vector(name: str, value: Vector3) -> None:
    if not isinstance(value, tuple) or len(value) != 3:
        raise ValueError(f"{name} must be a tuple of three numbers")
    for component in value:
        _finite(name, component)


@dataclass(frozen=True)
class ReachModel:
    """Approximate spherical reach model of a Panda arm (not an IK check).

    A tool-centre point ``p`` counts as reachable when the wrist flange point
    ``p + (0, 0, tool_length_m)`` (top-down grasp) is within
    ``max_reach_m - safety_margin_m`` of the shoulder ``base + (0, 0,
    shoulder_height_m)``, its horizontal distance from the base is at least
    ``min_horizontal_reach_m``, and (optionally) its x lies in ``x_limits_m``.
    Defaults use the nominal Franka Panda figures (855 mm reach, 333 mm
    shoulder height, ~103 mm hand TCP offset).

    ``safety_margin_m`` defaults to 0.03 m. Calibration (18 Nov 2026, RAI
    ``panda.g`` in the project scene, robot A, 251 belt points at TCP heights
    0.775/0.81 m, KOMO position + top-down ``vectorZ`` (+ optional closing-axis
    ``vectorX``) with joint limits, *without* collision objectives): with no
    margin 38-47 sphere-"reachable" points had no IK solution (all within
    0.031 m of the sphere boundary); with 0.03 m no optimistic point remained
    at a 5 mm IK tolerance (5 borderline points at 0.030-0.031 m remained at a
    1 mm tolerance) and only 3-5 IK-solvable points were excluded.
    This is still an approximation: collisions, the other arm, joint-limit
    margins and trajectory feasibility are only checked by the IK/KOMO planner.
    """

    shoulder_height_m: float = 0.333
    tool_length_m: float = 0.1034
    max_reach_m: float = 0.855
    min_horizontal_reach_m: float = 0.25
    safety_margin_m: float = 0.03
    x_limits_m: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        _finite("shoulder_height_m", self.shoulder_height_m)
        _finite("tool_length_m", self.tool_length_m, minimum=0)
        _finite("max_reach_m", self.max_reach_m, positive=True)
        _finite("min_horizontal_reach_m", self.min_horizontal_reach_m, minimum=0)
        _finite("safety_margin_m", self.safety_margin_m, minimum=0)
        if self.safety_margin_m >= self.max_reach_m:
            raise ValueError("safety_margin_m must be smaller than max_reach_m")
        if self.x_limits_m is not None:
            if len(self.x_limits_m) != 2 or self.x_limits_m[0] >= self.x_limits_m[1]:
                raise ValueError("x_limits_m must be (low, high) with low < high")


@dataclass(frozen=True)
class RobotConfig:
    """One arm as seen by the scheduler.

    ``tool_position_m`` is where the tool is when the robot becomes available at
    ``available_at_s``. ``place_position_m`` is the bin approach point the robot
    travels to after a pick; it becomes the next start position.
    """

    robot_id: str
    base_position_m: Vector3
    tool_position_m: Vector3
    place_position_m: Vector3
    available_at_s: float = 0.0
    reach: ReachModel = field(default_factory=ReachModel)

    def __post_init__(self) -> None:
        if not isinstance(self.robot_id, str) or not self.robot_id.strip():
            raise ValueError("robot_id must be a non-empty string")
        _vector("base_position_m", self.base_position_m)
        _vector("tool_position_m", self.tool_position_m)
        _vector("place_position_m", self.place_position_m)
        _finite("available_at_s", self.available_at_s, minimum=0)


@dataclass(frozen=True)
class TimingModel:
    """Approximate pick cycle used until KOMO trajectories provide real durations.

    travel  = motion_overhead_s + |pregrasp - tool| / cartesian_speed_m_s + descend_s
    busy    = gripper_close_s + lift_s + |place - grasp| / cartesian_speed_m_s + release_s
    """

    cartesian_speed_m_s: float = 0.4
    motion_overhead_s: float = 0.25
    descend_s: float = 0.4
    gripper_close_s: float = 0.3
    lift_s: float = 0.3
    release_s: float = 0.3

    def __post_init__(self) -> None:
        _finite("cartesian_speed_m_s", self.cartesian_speed_m_s, positive=True)
        for name in ("motion_overhead_s", "descend_s", "gripper_close_s", "lift_s", "release_s"):
            _finite(name, getattr(self, name), minimum=0)


@dataclass(frozen=True)
class WorkspaceConfig:
    """Region of the belt where an object centroid may be intercepted.

    Defaults follow ``scenes/dual_arm.g``: a 2.4 m belt centred at x=0 and
    0.4 m wide. The downstream limit keeps a margin before the belt end, where
    objects start to fall.
    """

    pick_x_limits_m: tuple[float, float] = (-1.2, 1.15)
    pick_y_limits_m: tuple[float, float] = (-0.2, 0.2)

    def __post_init__(self) -> None:
        for name in ("pick_x_limits_m", "pick_y_limits_m"):
            low, high = getattr(self, name)
            _finite(name, low)
            _finite(name, high)
            if low >= high:
                raise ValueError(f"{name} must be (low, high) with low < high")


@dataclass(frozen=True)
class SchedulerConfig:
    """Shared configuration of the proposed scheduler and the baseline."""

    time_step_s: float = 0.02
    horizon_s: float = 60.0
    max_state_age_s: float = 2.0
    min_robot_separation_m: float = 0.15
    robust_slack_s: float = 0.5
    timing: TimingModel = field(default_factory=TimingModel)
    workspace: WorkspaceConfig = field(default_factory=WorkspaceConfig)
    gripper: GripperConfig = field(default_factory=GripperConfig)

    def __post_init__(self) -> None:
        _finite("time_step_s", self.time_step_s, positive=True)
        _finite("horizon_s", self.horizon_s, positive=True)
        _finite("max_state_age_s", self.max_state_age_s, minimum=0)
        _finite("min_robot_separation_m", self.min_robot_separation_m, minimum=0)
        _finite("robust_slack_s", self.robust_slack_s, minimum=0)
        if self.horizon_s / self.time_step_s > 200_000:
            raise ValueError("horizon_s / time_step_s is too large")


@dataclass(frozen=True)
class SchedulableObject:
    """A tracked object plus the geometry needed for grasping."""

    state: TrackedState
    geometry: ObjectGeometry

    def __post_init__(self) -> None:
        if not isinstance(self.state, TrackedState):
            raise TypeError("state must be a conveyor.types.TrackedState")
        if not isinstance(self.geometry, ObjectGeometry):
            raise TypeError("geometry must be an ObjectGeometry")
        if self.state.object_id != self.geometry.object_id:
            raise ValueError("state and geometry must have the same object_id")

    @property
    def object_id(self) -> str:
        return self.state.object_id


ASSIGNED = "assigned"
REJECTED = "rejected"


@dataclass(frozen=True)
class Assignment:
    """Scheduler decision for one object.

    For ``status == "assigned"``: ``robot_id`` picks the object, closing the
    gripper at ``intercept_time_s`` at ``intercept_position_m`` (predicted
    centroid) using ``grasp``. ``travel_time_s`` is the approach time from the
    robot's position at ``robot_available_at_s`` to that grasp.
    ``deadline_s`` is the last sample of the robot's contiguous reach window
    containing the interception. ``latest_start_s = max over window samples t
    of (t - travel time to the object's grasp pose at t)``: the latest moment
    the robot can leave and still reach the object in that window, with travel
    time and target time always taken at the same sample. ``slack_s =
    latest_start_s - robot_available_at_s`` (>= 0). The window is geometric
    (belt and reach); later grasp/conflict checks are not re-evaluated, so the
    slack is an optimistic timing margin. ``completion_time_s`` is when the
    robot is free again (after placing).

    For ``status == "rejected"`` (= not assigned in this plan) only
    ``object_id``, ``reason`` and ``object_exit_time_s`` are meaningful. The
    reason distinguishes infeasibility (e.g. ``insufficient_time_before_deadline``)
    from ``feasible_but_not_selected``: a robot could still intercept the object,
    but the scheduling policy did not choose it. Feasibility is an estimate from
    the approximate reach/timing models, not a verified motion plan.
    """

    object_id: str
    status: str
    method: str
    reason: str = ""
    robot_id: str | None = None
    intercept_time_s: float | None = None
    intercept_position_m: Vector3 | None = None
    grasp: GraspCandidate | None = None
    robot_available_at_s: float | None = None
    travel_time_s: float | None = None
    deadline_s: float | None = None
    slack_s: float | None = None
    completion_time_s: float | None = None
    object_exit_time_s: float | None = None
    latest_start_s: float | None = None

    @property
    def is_assigned(self) -> bool:
        return self.status == ASSIGNED


@dataclass(frozen=True)
class ScheduleResult:
    """All decisions of one planning call, assigned first in interception order.

    ``excluded_committed`` lists ``(object_id, reason)`` for committed objects
    whose track could not be used for prediction at ``planning_time_s`` (e.g. a
    future-dated timestamp). Their commitment is still honoured (other robots
    stay clear of the pick, the object counts as picked at its interception
    time), but they are not modelled as neighbours in this call, so grasp
    clearance next to them is unverified until the next re-plan.
    """

    method: str
    planning_time_s: float
    robot_ids: tuple[str, ...]
    assignments: tuple[Assignment, ...]
    excluded_committed: tuple[tuple[str, str], ...] = ()

    @property
    def assigned(self) -> tuple[Assignment, ...]:
        return tuple(a for a in self.assignments if a.is_assigned)

    @property
    def rejected(self) -> tuple[Assignment, ...]:
        return tuple(a for a in self.assignments if not a.is_assigned)

    def for_robot(self, robot_id: str) -> tuple[Assignment, ...]:
        return tuple(a for a in self.assigned if a.robot_id == robot_id)

    def by_object(self) -> dict[str, Assignment]:
        return {a.object_id: a for a in self.assignments}
