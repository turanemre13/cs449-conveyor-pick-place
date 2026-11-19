"""Rolling-horizon (online) dispatch on top of a one-shot scheduler.

Objects become visible only once they enter the belt. Every
``replan_period_s`` the scheduler re-plans the visible, not yet committed
objects; for each robot, the first planned pick is *committed* only when the
robot must start moving before the next re-plan. Later picks stay flexible and
can change when new objects arrive. This mirrors how the integrated pipeline
would call the scheduler as tracking updates arrive.

The object motion is the same constant-velocity model used for prediction, so
this is a kinematic planning-level evaluation: it does not execute motions,
check IK, or simulate grasp physics.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

from conveyor.prediction.constant_velocity import ConstantVelocityPredictor

from .base import SchedulerBase
from .models import Assignment, RobotConfig, SchedulableObject


@dataclass(frozen=True)
class OnlineResult:
    method: str
    objects: int
    committed: tuple[Assignment, ...]
    missed_object_ids: tuple[str, ...]
    replans: int
    robot_ids: tuple[str, ...]
    #: Simulated duration [0, end_time_s]; derived from the inputs only, so it is
    #: identical for every method run on the same objects.
    observation_window_s: float = 0.0

    @property
    def scheduled(self) -> int:
        return len(self.committed)

    @property
    def missed(self) -> int:
        return len(self.missed_object_ids)

    def per_robot(self) -> dict[str, int]:
        counts = {r: 0 for r in self.robot_ids}
        for a in self.committed:
            counts[a.robot_id] += 1
        return counts

    @property
    def makespan_s(self) -> float:
        return max((a.completion_time_s for a in self.committed), default=0.0)

    @property
    def throughput_per_min(self) -> float:
        """Committed picks per minute of the fixed observation window (comparable across methods)."""
        return 60.0 * self.scheduled / self.observation_window_s if self.observation_window_s > 0 else 0.0

    @property
    def processing_rate_per_min(self) -> float:
        """Committed picks per minute of this method's own makespan (descriptive only)."""
        return 60.0 * self.scheduled / self.makespan_s if self.makespan_s > 0 else 0.0

    @property
    def mean_slack_s(self) -> float:
        return sum(a.slack_s for a in self.committed) / self.scheduled if self.committed else 0.0


def run_online(
    scheduler: SchedulerBase,
    objects: Sequence[SchedulableObject],
    robots: Sequence[RobotConfig],
    *,
    replan_period_s: float = 0.25,
    visible_from_x_m: float | None = None,
    end_time_s: float | None = None,
) -> OnlineResult:
    """Simulate rolling re-planning; ``objects`` hold their states at time 0.

    ``visible_from_x_m`` defaults to the lower pick limit (belt start).
    """
    if replan_period_s <= 0:
        raise ValueError("replan_period_s must be positive")
    config = scheduler.config
    x_low, x_high = config.workspace.pick_x_limits_m
    visible_x = x_low if visible_from_x_m is None else visible_from_x_m
    predictor = ConstantVelocityPredictor(max_horizon_s=1e6, planar_motion=True)

    def state_at(obj: SchedulableObject, t: float) -> SchedulableObject:
        p = predictor.predict(obj.state, t)
        return SchedulableObject(replace(obj.state, simulation_time_s=t, position_m=p.position_m), obj.geometry)

    if end_time_s is None:
        exits = [predictor.time_to_reach_coordinate(o.state, 0, x_high) for o in objects]
        end_time_s = max((e for e in exits if e is not None), default=config.horizon_s) + replan_period_s

    available = {r.robot_id: (r.available_at_s, r.tool_position_m) for r in robots}
    committed: list[Assignment] = []
    committed_ids: set[str] = set()
    replans = 0
    t = 0.0
    while t <= end_time_s + 1e-9:
        current = {o.object_id: state_at(o, t) for o in objects}
        visible = [
            s for oid, s in current.items()
            if visible_x <= s.state.position_m[0] <= x_high
        ]
        open_ids = [s.object_id for s in visible if s.object_id not in committed_ids]
        if open_ids:
            active = [a for a in committed if a.intercept_time_s > t]
            robot_configs = [
                replace(r, available_at_s=max(t, available[r.robot_id][0]),
                        tool_position_m=available[r.robot_id][1])
                for r in robots
            ]
            plan_objects = [s for s in visible if s.object_id not in committed_ids]
            plan_objects += [current[a.object_id] for a in active if a.object_id in current
                             and visible_x <= current[a.object_id].state.position_m[0] <= x_high]
            result = scheduler.schedule(plan_objects, robot_configs, t, committed=active)
            replans += 1
            for robot in robots:
                plan = sorted(result.for_robot(robot.robot_id), key=lambda a: a.intercept_time_s)
                if not plan:
                    continue
                first = plan[0]
                start = first.intercept_time_s - first.travel_time_s
                if start <= t + replan_period_s:
                    committed.append(first)
                    committed_ids.add(first.object_id)
                    place = next(r.place_position_m for r in robots if r.robot_id == robot.robot_id)
                    available[robot.robot_id] = (first.completion_time_s, place)
        t = round(t + replan_period_s, 9)

    missed = tuple(sorted(o.object_id for o in objects if o.object_id not in committed_ids))
    return OnlineResult(
        scheduler.name, len(objects), tuple(sorted(committed, key=lambda a: a.intercept_time_s)),
        missed, replans, tuple(sorted(r.robot_id for r in robots)), float(end_time_s),
    )
