"""Common input validation and result assembly for scheduling methods."""

from __future__ import annotations

from collections.abc import Sequence
from math import isfinite

from .feasibility import (
    FUTURE_TOLERANCE_S, REASON_FUTURE, REASON_NOT_SELECTED, REASON_STALE, REASON_UNREACHABLE, Commitment,
    InterceptOption, InterceptionModel, RobotRuntime,
)
from .models import ASSIGNED, REJECTED, Assignment, RobotConfig, SchedulableObject, ScheduleResult, SchedulerConfig


class SchedulerBase:
    """Validation, bookkeeping and output for one-shot interception planning.

    Subclasses implement :meth:`_decide`, which repeatedly calls
    :meth:`_commit` and leaves undecided objects to be reported as rejected.
    Planning is deterministic: ties are broken by robot and object IDs.
    """

    name = "base"

    def __init__(self, config: SchedulerConfig | None = None) -> None:
        self.config = config or SchedulerConfig()

    def schedule(
        self, objects: Sequence[SchedulableObject], robots: Sequence[RobotConfig], now_s: float,
        *, committed: Sequence[Assignment] = (),
    ) -> ScheduleResult:
        """Plan which robot intercepts which object, where and when.

        ``now_s`` is the absolute simulation time of the decision. Track states
        may be slightly older (up to ``max_state_age_s``) and are extrapolated
        forward. Track states *newer* than ``now_s`` (beyond
        ``FUTURE_TOLERANCE_S``) are never extrapolated backwards: an uncommitted
        one is rejected with ``future_dated_track``; a committed one keeps its
        commitment but is reported in ``ScheduleResult.excluded_committed``.
        Callers should plan at a ``now_s`` no earlier than the newest track time.
        Works with one or more robots; robots and objects need unique IDs.

        ``committed`` lists earlier assignments that are already being executed
        (rolling re-planning). Their objects are not re-assigned or reported,
        but if present in ``objects`` they still count as neighbours until their
        interception time, and their picks block the other robots nearby. The
        caller must reflect them in each robot's ``available_at_s`` and
        ``tool_position_m``.
        """
        if isinstance(now_s, bool) or not isinstance(now_s, (int, float)) or not isfinite(now_s) or now_s < 0:
            raise ValueError("now_s must be a finite non-negative simulation time")
        robots = list(robots)
        objects = list(objects)
        if not robots:
            raise ValueError("at least one robot is required")
        robot_ids = [r.robot_id for r in robots]
        if len(set(robot_ids)) != len(robot_ids):
            raise ValueError("robot IDs must be unique")
        object_ids = [o.object_id for o in objects]
        if len(set(object_ids)) != len(object_ids):
            raise ValueError("object IDs must be unique")

        committed = [a for a in committed if a.is_assigned]
        committed_ids = {a.object_id for a in committed}
        if len(committed_ids) != len(committed):
            raise ValueError("committed assignments must have unique object IDs")

        self._decisions: dict[str, Assignment] = {}
        self.pending: set[str] = set()
        valid = []
        excluded_committed: list[tuple[str, str]] = []
        for obj in objects:
            age = now_s - obj.state.simulation_time_s
            future_dated = age < -FUTURE_TOLERANCE_S
            if obj.object_id in committed_ids:
                # Already being executed: never re-decided. A future-dated state
                # cannot be predicted at now_s without backward extrapolation, so
                # it is left out of the neighbour model and reported explicitly.
                # Old committed states are kept: forward extrapolation is valid
                # and keeping a neighbour is the conservative choice.
                if future_dated:
                    excluded_committed.append((obj.object_id, REASON_FUTURE))
                else:
                    valid.append(obj)
                continue
            if future_dated:
                self._reject(obj.object_id, REASON_FUTURE, None)
            elif age > self.config.max_state_age_s:
                self._reject(obj.object_id, REASON_STALE, None)
            else:
                valid.append(obj)

        self.model = InterceptionModel(self.config, valid, robots, now_s)
        self.runtime = {
            r.robot_id: RobotRuntime(r, max(float(now_s), r.available_at_s), r.tool_position_m)
            for r in sorted(robots, key=lambda r: r.robot_id)
        }
        timing = self.config.timing
        self.commitments: list[Commitment] = [
            Commitment(
                robot_id=a.robot_id, object_id=a.object_id, intercept_time_s=a.intercept_time_s,
                position_m=a.intercept_position_m, busy_from_s=a.intercept_time_s - timing.descend_s,
                busy_until_s=a.intercept_time_s + timing.gripper_close_s + timing.lift_s,
            )
            for a in committed
        ]
        self.pending = {o.object_id for o in valid if o.object_id not in committed_ids}
        self._decide()
        for object_id in sorted(self.pending):
            self._reject(object_id, self._final_reason(object_id), self.model.exit_time.get(object_id))

        assigned = sorted(
            (a for a in self._decisions.values() if a.is_assigned),
            key=lambda a: (a.intercept_time_s, a.robot_id, a.object_id),
        )
        rejected = sorted((a for a in self._decisions.values() if not a.is_assigned), key=lambda a: a.object_id)
        return ScheduleResult(
            self.name, float(now_s), tuple(sorted(robot_ids)), tuple(assigned + rejected),
            tuple(sorted(excluded_committed)),
        )

    # ------------------------------------------------------------- subclass API
    def _decide(self) -> None:  # pragma: no cover - abstract
        raise NotImplementedError

    def _options(self, robot_id: str, object_id: str) -> tuple[InterceptOption | None, str]:
        return self.model.find_interception(self.runtime[robot_id], object_id, self.commitments)

    def _commit(self, option: InterceptOption) -> None:
        robot = self.runtime[option.robot_id]
        robot.available_at_s = option.completion_time_s
        robot.tool_position_m = option.end_tool_position_m
        self.commitments.append(self.model.commitment(option))
        self.pending.discard(option.object_id)
        self._decisions[option.object_id] = Assignment(
            object_id=option.object_id, status=ASSIGNED, method=self.name, reason="",
            robot_id=option.robot_id, intercept_time_s=option.intercept_time_s,
            intercept_position_m=option.intercept_position_m, grasp=option.grasp,
            robot_available_at_s=option.available_at_s, travel_time_s=option.travel_time_s,
            deadline_s=option.deadline_s, slack_s=option.slack_s,
            completion_time_s=option.completion_time_s,
            object_exit_time_s=self.model.exit_time.get(option.object_id),
            latest_start_s=option.latest_start_s,
        )

    def _reject(self, object_id: str, reason: str, exit_time: float | None) -> None:
        self.pending.discard(object_id)
        self._decisions[object_id] = Assignment(
            object_id=object_id, status=REJECTED, method=self.name, reason=reason,
            object_exit_time_s=exit_time,
        )

    def _final_reason(self, object_id: str) -> str:
        """Why ``object_id`` was left unassigned, evaluated with the final robot states."""
        results = [self._options(robot_id, object_id) for robot_id in self.runtime]
        if any(option is not None for option, _ in results):
            # Still interceptable: the policy chose not to take it (not infeasible).
            return REASON_NOT_SELECTED
        if not self.model.has_any_window(object_id) and not self.model.geometry_reject[object_id]:
            return REASON_UNREACHABLE
        reasons = [reason for _, reason in results]
        # Prefer the most informative reason over "unreachable for this robot".
        for reason in reasons:
            if reason != REASON_UNREACHABLE:
                return reason
        return reasons[0]
