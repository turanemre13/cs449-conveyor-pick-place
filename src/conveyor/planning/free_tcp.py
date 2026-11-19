"""Speed-limited Cartesian interception for a freely moving tool centre point."""

from dataclasses import dataclass
from itertools import pairwise
from math import dist, isfinite

from conveyor.prediction import ConstantVelocityPredictor
from conveyor.scheduling import Assignment
from conveyor.types import TrackedState, Vector3


@dataclass(frozen=True)
class TcpSample:
    simulation_time_s: float
    position_m: Vector3

    def __post_init__(self) -> None:
        if not isfinite(self.simulation_time_s) or self.simulation_time_s < 0:
            raise ValueError("The TCP sample time must be finite and non-negative")
        if len(self.position_m) != 3 or not all(isfinite(value) for value in self.position_m):
            raise ValueError("The TCP sample position must contain three finite coordinates")


@dataclass(frozen=True)
class FreeTcpPlan:
    object_id: str
    quaternion_wxyz: tuple[float, float, float, float]
    grasp_offset_m: Vector3
    waypoints: tuple[TcpSample, ...]

    @property
    def intercept_time_s(self) -> float:
        return self.waypoints[-1].simulation_time_s

    @property
    def tracking_start_s(self) -> float:
        return self.waypoints[-2].simulation_time_s

    def position_at(self, simulation_time_s: float) -> Vector3:
        """Interpolate on the simulation clock and hold the endpoints outside the plan."""
        for start, end in pairwise(self.waypoints):
            if simulation_time_s <= end.simulation_time_s:
                duration = end.simulation_time_s - start.simulation_time_s
                fraction = max(0.0, (simulation_time_s - start.simulation_time_s) / duration)
                return tuple(a + fraction * (b - a) for a, b in zip(start.position_m, end.position_m))
        return self.waypoints[-1].position_m


def plan_free_tcp(
    assignment: Assignment,
    state: TrackedState,
    start: TcpSample,
    *,
    max_speed_m_s: float = 0.4,
    descend_s: float = 0.4,
    tracking_s: float = 0.1,
) -> FreeTcpPlan:
    """Approach, descend with belt motion, then match object velocity before interception.

    Orientation is held at the assigned grasp orientation throughout the plan.
    """
    if any(not isfinite(value) or value <= 0 for value in (max_speed_m_s, descend_s, tracking_s)):
        raise ValueError("Speed limit, descent time and tracking time must be finite and positive")
    grasp = assignment.grasp
    intercept_s = assignment.intercept_time_s
    centroid = assignment.intercept_position_m
    if not assignment.is_assigned or grasp is None or centroid is None or intercept_s is None:
        raise ValueError(f"No assigned interception: {assignment.reason or 'incomplete assignment'}")
    if not isfinite(intercept_s):
        raise ValueError("The interception time must be finite")
    if not grasp.feasible or state.object_id != assignment.object_id or grasp.object_id != state.object_id:
        raise ValueError("The tracked object and feasible grasp must match the assignment")
    if state.simulation_time_s > start.simulation_time_s:
        raise ValueError("The object state is newer than the planning time")
    if assignment.object_exit_time_s is not None and intercept_s >= assignment.object_exit_time_s:
        raise ValueError("The object exits before interception can be completed")
    if assignment.deadline_s is not None and intercept_s > assignment.deadline_s:
        raise ValueError("The interception is after the assignment deadline")

    departure_s = start.simulation_time_s
    if assignment.robot_available_at_s is not None:
        departure_s = max(departure_s, assignment.robot_available_at_s)
    tracking_start_s = intercept_s - tracking_s
    pregrasp_s = tracking_start_s - descend_s
    if pregrasp_s <= departure_s:
        raise ValueError("Too late to reach pre-grasp before descent")

    velocity, _ = ConstantVelocityPredictor(planar_motion=True).effective_velocity(state)
    pregrasp_position = tuple(
        p + v * (pregrasp_s - intercept_s) for p, v in zip(grasp.pregrasp_position_m, velocity)
    )
    tracking_position = tuple(p - v * tracking_s for p, v in zip(grasp.position_m, velocity))
    waypoints = (
        TcpSample(departure_s, start.position_m),
        TcpSample(pregrasp_s, pregrasp_position),
        TcpSample(tracking_start_s, tracking_position),
        TcpSample(intercept_s, grasp.position_m),
    )
    for first, second in pairwise(waypoints):
        duration = second.simulation_time_s - first.simulation_time_s
        if dist(first.position_m, second.position_m) > (max_speed_m_s + 1e-9) * duration:
            raise ValueError("The assigned interception exceeds the TCP speed limit")

    grasp_offset = tuple(tcp - center for tcp, center in zip(grasp.position_m, centroid))
    return FreeTcpPlan(assignment.object_id, grasp.quaternion_wxyz, grasp_offset, waypoints)
