"""Planning-level metrics for comparing schedules.

These metrics describe the *plan* under the approximate reach/timing model.
They are estimates and are not equivalent to physically executed grasps.

Two rates are reported and must not be confused:

* ``throughput_per_min`` - objects intercepted per minute of a **fixed
  observation window** that is derived from the *inputs* (e.g. until the last
  object leaves the pickable belt) and is therefore identical for every method
  compared on the same inputs. On fixed inputs it is proportional to the number
  of scheduled objects.
* ``processing_rate_per_min`` - scheduled objects per minute of the method's own
  makespan (planning time to last completion). A method that schedules fewer
  objects but finishes earlier can score higher, so it is a descriptive
  per-method rate, **not** a comparison metric. (Before 18 Nov 2026 this value
  was reported under the name ``throughput_per_min``.)
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass

from conveyor.prediction.constant_velocity import ConstantVelocityPredictor

from .models import SchedulableObject, ScheduleResult, WorkspaceConfig


@dataclass(frozen=True)
class ScheduleMetrics:
    method: str
    objects: int
    scheduled: int
    missed: int
    per_robot: dict[str, int]
    makespan_s: float
    processing_rate_per_min: float
    mean_slack_s: float
    min_slack_s: float
    robust_scheduled: int
    miss_reasons: dict[str, int]
    observation_window_s: float | None = None
    throughput_per_min: float | None = None

    @property
    def scheduled_ratio(self) -> float:
        return self.scheduled / self.objects if self.objects else 0.0


def input_observation_window_s(
    objects: Iterable[SchedulableObject], now_s: float, workspace: WorkspaceConfig | None = None,
) -> float | None:
    """Input-only observation window: time from ``now_s`` until the last moving
    object's centroid leaves the pickable x range. ``None`` if no object moves
    (then no fixed conveyor window exists)."""
    workspace = workspace or WorkspaceConfig()
    predictor = ConstantVelocityPredictor(planar_motion=True)
    exits = []
    for obj in objects:
        velocity, stationary = predictor.effective_velocity(obj.state)
        if stationary or velocity[0] == 0.0:
            continue
        limit = workspace.pick_x_limits_m[1] if velocity[0] > 0 else workspace.pick_x_limits_m[0]
        exits.append(predictor.time_to_reach_coordinate(obj.state, 0, limit))
    if not exits:
        return None
    return max(max(exits) - now_s, 0.0)


def summarize(
    result: ScheduleResult, *, robust_slack_s: float = 0.5, observation_window_s: float | None = None,
) -> ScheduleMetrics:
    """Counts, robot load balance, rates and slack statistics.

    ``makespan_s`` runs from the planning time to the last robot completion.
    ``processing_rate_per_min`` is scheduled objects per minute of that makespan.
    ``throughput_per_min`` (only when ``observation_window_s`` is given) counts
    assignments intercepted within ``[planning_time, planning_time + window]``
    per minute of the window; pass the same input-derived window for every
    compared method (see :func:`input_observation_window_s`).
    ``robust_scheduled`` counts assignments whose slack is at least
    ``robust_slack_s`` - a proxy for interceptions that tolerate timing errors.
    """
    assigned = result.assigned
    slacks = [a.slack_s for a in assigned]
    makespan = max((a.completion_time_s for a in assigned), default=result.planning_time_s) - result.planning_time_s
    per_robot = {robot_id: 0 for robot_id in result.robot_ids}
    for a in assigned:
        per_robot[a.robot_id] += 1
    throughput = None
    if observation_window_s is not None:
        if observation_window_s <= 0:
            raise ValueError("observation_window_s must be positive")
        end = result.planning_time_s + observation_window_s
        within = sum(1 for a in assigned if a.intercept_time_s <= end + 1e-9)
        throughput = 60.0 * within / observation_window_s
    return ScheduleMetrics(
        method=result.method,
        objects=len(result.assignments),
        scheduled=len(assigned),
        missed=len(result.rejected),
        per_robot=per_robot,
        makespan_s=makespan,
        processing_rate_per_min=60.0 * len(assigned) / makespan if makespan > 0 else 0.0,
        mean_slack_s=sum(slacks) / len(slacks) if slacks else 0.0,
        min_slack_s=min(slacks) if slacks else 0.0,
        robust_scheduled=sum(1 for s in slacks if s >= robust_slack_s),
        miss_reasons=dict(sorted(Counter(a.reason for a in result.rejected).items())),
        observation_window_s=observation_window_s,
        throughput_per_min=throughput,
    )
