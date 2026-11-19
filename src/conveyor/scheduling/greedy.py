"""Deadline-aware, reachability-aware greedy dual-arm scheduler (proposed method).

Decision loop (event driven, deterministic):

1. For every robot ``i`` and unassigned object ``j`` compute the earliest
   feasible interception (see :class:`InterceptionModel`) and its slack
   ``S_ij = t_latest_start,ij - t_available,i``, where ``t_latest_start,ij`` is
   the latest departure that still reaches ``j`` inside robot ``i``'s reach
   window (travel time evaluated at the same sample as the target time).
2. Dispatch the robot that becomes available first (ties: robot ID) among
   those with at least one feasible option.
3. That robot ranks its options by an **urgency** key (smaller = more urgent),
   then by earliest interception time, then by object ID. ``urgency`` selects
   the key:

   * ``"system"`` (default): ``max_k S_kj`` over all robots ``k`` that can
     still intercept ``j`` - how long the *whole system* can postpone ``j``.
     An object that only this robot can still catch keeps its own (small)
     slack and becomes urgent; an object the other arm can also catch later
     is postponed. There is no separate "critical first" rule.
   * ``"critical"``: lexicographic ``(0 if no other robot can catch j else 1,
     S_ij)`` - objects only this robot can reach first, then least slack.
   * ``"robot"``: ``S_ij`` of the dispatched robot only (plain least slack).

4. **Gap filling** (``gap_filling=True``): if a lower-ranked object can be
   intercepted *earlier* and the robot can still catch the chosen object
   afterwards, the earlier object is taken first, using otherwise idle time.
5. Commit, update the robot's availability and position (the bin), repeat until
   no robot has a feasible option. Remaining objects are rejected with a reason.

Both arms compete for every object over the object's whole predicted path,
including the shared region; no object is permanently bound to a robot. This is
a heuristic: it is not guaranteed to maximise the number of picks (e.g. taking
the least-slack object can block two later ones that a different order would
have caught).
"""

from __future__ import annotations

from .base import SchedulerBase
from .feasibility import InterceptOption, RobotRuntime


class DeadlineAwareGreedyScheduler(SchedulerBase):
    """Proposed method; see the module docstring for the decision rules."""

    name = "deadline_aware_greedy"

    def __init__(self, config=None, *, gap_filling: bool = True, urgency: str = "system") -> None:
        super().__init__(config)
        if urgency not in ("system", "critical", "robot"):
            raise ValueError("urgency must be 'system', 'critical' or 'robot'")
        self.gap_filling = gap_filling
        self.urgency = urgency

    def _decide(self) -> None:
        while self.pending:
            options: dict[str, dict[str, InterceptOption]] = {robot_id: {} for robot_id in self.runtime}
            for robot_id in self.runtime:
                for object_id in sorted(self.pending):
                    option, _ = self._options(robot_id, object_id)
                    if option is not None:
                        options[robot_id][object_id] = option
            active = [robot_id for robot_id in self.runtime if options[robot_id]]
            if not active:
                return
            robot_id = min(active, key=lambda r: (self.runtime[r].available_at_s, r))
            candidates = options[robot_id]

            def priority(option: InterceptOption) -> tuple:
                others = [options[o][option.object_id] for o in self.runtime
                          if o != robot_id and option.object_id in options[o]]
                if self.urgency == "system":
                    # Latest moment *any* robot could still start for this object.
                    urgency = (max([option.slack_s, *(o.slack_s for o in others)]),)
                elif self.urgency == "critical":
                    urgency = (1 if others else 0, option.slack_s)
                else:
                    urgency = (option.slack_s,)
                return (*(round(u, 9) for u in urgency), option.intercept_time_s, option.object_id)

            ranked = sorted(candidates.values(), key=priority)
            chosen = ranked[0]
            if self.gap_filling:
                chosen = self._fill_gap(robot_id, chosen, ranked[1:])
            self._commit(chosen)

    def _fill_gap(self, robot_id: str, chosen: InterceptOption, others: list[InterceptOption]) -> InterceptOption:
        """Take an earlier object first if the chosen one is still catchable afterwards."""
        robot = self.runtime[robot_id]
        for option in others:
            if option.intercept_time_s >= chosen.intercept_time_s:
                continue
            trial = RobotRuntime(robot.config, option.completion_time_s, option.end_tool_position_m)
            commitments = [*self.commitments, self.model.commitment(option)]
            follow_up, _ = self.model.find_interception(trial, chosen.object_id, commitments)
            if follow_up is not None:
                return option
        return chosen
