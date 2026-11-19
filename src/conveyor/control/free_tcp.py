"""Evaluate interception against the observed object and tool motion."""

from dataclasses import dataclass
from math import dist
from typing import TYPE_CHECKING, Protocol

from conveyor.planning.free_tcp import TcpSample
from conveyor.types import Vector3

if TYPE_CHECKING:
    from conveyor.simulation import ObjectGroundTruth


class InterceptionTarget(Protocol):
    @property
    def object_id(self) -> str: ...

    @property
    def grasp_offset_m(self) -> Vector3: ...

    @property
    def tracking_start_s(self) -> float: ...

    @property
    def intercept_time_s(self) -> float: ...


@dataclass(frozen=True)
class InterceptionResult:
    success: bool
    reason: str
    time_error_s: float
    position_error_m: float | None = None
    velocity_error_m_s: float | None = None


def evaluate_interception(
    plan: InterceptionTarget,
    previous: TcpSample,
    current: TcpSample,
    object_state: "ObjectGroundTruth | None",
    *,
    time_tolerance_s: float = 1e-6,
    position_tolerance_m: float = 0.005,
    velocity_tolerance_m_s: float = 0.01,
) -> InterceptionResult:
    """Check the event using two measured TCP samples from the tracking segment."""
    sample_duration = current.simulation_time_s - previous.simulation_time_s
    if sample_duration <= 0:
        raise ValueError("TCP samples must have increasing simulation timestamps")
    time_error = abs(current.simulation_time_s - plan.intercept_time_s)
    if (object_state is None or object_state.object_id != plan.object_id
            or object_state.motion_state != "on_belt" or object_state.velocity_m_s is None):
        return InterceptionResult(False, "object_unavailable", time_error)
    if (time_error > time_tolerance_s
            or abs(object_state.simulation_time_s - current.simulation_time_s) > time_tolerance_s
            or previous.simulation_time_s < plan.tracking_start_s - time_tolerance_s):
        return InterceptionResult(False, "incorrect_sample_time", time_error)

    target_position = tuple(p + offset for p, offset in zip(object_state.position_m, plan.grasp_offset_m))
    tcp_velocity = tuple((b - a) / sample_duration for a, b in zip(previous.position_m, current.position_m))
    position_error = dist(current.position_m, target_position)
    velocity_error = dist(tcp_velocity, object_state.velocity_m_s)
    if position_error > position_tolerance_m:
        reason = "position_error"
    elif velocity_error > velocity_tolerance_m_s:
        reason = "velocity_error"
    else:
        reason = "intercepted"
    return InterceptionResult(reason == "intercepted", reason, time_error, position_error, velocity_error)
