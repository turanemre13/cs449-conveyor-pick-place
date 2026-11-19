"""Shared interception feasibility model used by every scheduling method.

Using one model for the proposed scheduler and the baseline means that a
comparison isolates the *decision logic* (which object, which robot, when),
not differences in the physical assumptions.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from math import ceil, dist, hypot, sqrt

from conveyor.grasping.top_down import GraspCandidate, select_top_down_grasps
from conveyor.prediction.constant_velocity import ConstantVelocityPredictor
from conveyor.types import Vector3

from .models import RobotConfig, SchedulableObject, SchedulerConfig

#: Rejection reasons, from most to least specific.
REASON_INVALID_GRASP = "no_geometrically_feasible_grasp"
REASON_UNREACHABLE = "never_in_reach_of_any_robot_before_exit"
REASON_TIME = "insufficient_time_before_deadline"
REASON_CLEARANCE = "neighbor_clearance_or_robot_conflict"
REASON_STALE = "stale_track"
#: Track timestamp later than the planning time ``now_s``: predicting the object
#: at ``now_s`` would need backward extrapolation, which is not performed.
REASON_FUTURE = "future_dated_track"
#: Some robot still has a feasible interception, but the scheduling policy did
#: not select the object (e.g. a single-pass FIFO order). Not an infeasibility.
REASON_NOT_SELECTED = "feasible_but_not_selected"
#: Tolerance for clock jitter between track timestamps and the planning time.
FUTURE_TOLERANCE_S = 1e-9


@dataclass
class RobotRuntime:
    """Mutable planning-time state of one robot."""

    config: RobotConfig
    available_at_s: float
    tool_position_m: Vector3

    @property
    def robot_id(self) -> str:
        return self.config.robot_id


@dataclass(frozen=True)
class Commitment:
    """An already scheduled pick, used for neighbour and conflict checks."""

    robot_id: str
    object_id: str
    intercept_time_s: float
    position_m: Vector3
    busy_from_s: float
    busy_until_s: float


@dataclass(frozen=True)
class InterceptOption:
    """A feasible (robot, object, time) triple with its timing terms."""

    robot_id: str
    object_id: str
    intercept_time_s: float
    intercept_position_m: Vector3
    grasp: GraspCandidate
    available_at_s: float
    travel_time_s: float
    deadline_s: float
    slack_s: float
    completion_time_s: float
    end_tool_position_m: Vector3
    latest_start_s: float


class InterceptionModel:
    """Reachability windows, travel times and earliest feasible interception."""

    def __init__(self, config: SchedulerConfig, objects: Sequence[SchedulableObject],
                 robots: Sequence[RobotConfig], now_s: float) -> None:
        self.config = config
        self.now_s = float(now_s)
        self.objects = {o.object_id: o for o in objects}
        oldest_age = max((self.now_s - o.state.simulation_time_s for o in objects), default=0.0)
        self.predictor = ConstantVelocityPredictor(
            max_horizon_s=config.horizon_s + max(oldest_age, config.max_state_age_s) + 1.0,
            planar_motion=True,
        )
        # Sample until the horizon, or until every object has left the belt.
        self.exit_time = {o.object_id: self._exit_time(o) for o in objects}
        horizon = config.horizon_s
        if objects and all(e is not None for e in self.exit_time.values()):
            horizon = min(horizon, max(max(self.exit_time.values()) - self.now_s, 0.0) + config.time_step_s)
        self._steps = int(ceil(horizon / config.time_step_s))
        self.times = [self.now_s + k * config.time_step_s for k in range(self._steps + 1)]
        # Grasp TCP offset from the centroid, time invariant under translation.
        self._tcp_offset: dict[str, Vector3] = {}
        self.geometry_reject: dict[str, bool] = {}
        for obj in objects:
            alone = select_top_down_grasps(
                obj.geometry, obj.state.position_m, config=config.gripper,
                object_velocity_m_s=self.predictor.effective_velocity(obj.state)[0],
            )
            reference = alone.best or (alone.rejected[0] if alone.rejected else None)
            self.geometry_reject[obj.object_id] = not alone.has_feasible
            if reference is None:
                self._tcp_offset[obj.object_id] = (0.0, 0.0, 0.0)
            else:
                self._tcp_offset[obj.object_id] = tuple(
                    r - p for r, p in zip(reference.position_m, obj.state.position_m)
                )
        velocities = [self.predictor.effective_velocity(o.state)[0] for o in objects]
        self._uniform_velocity = all(
            max(abs(a - b) for a, b in zip(v, velocities[0])) < 1e-12 for v in velocities
        )
        self._grasp_cache: dict = {}
        self._positions = {
            o.object_id: [self.predict(o.object_id, t) for t in self.times] for o in objects
        }
        self.windows = {
            (r.robot_id, o.object_id): self._reach_mask(r, o.object_id) for r in robots for o in objects
        }

    # ----------------------------------------------------------- basic geometry
    def predict(self, object_id: str, time_s: float) -> Vector3:
        return self.predictor.predict(self.objects[object_id].state, time_s).position_m

    def tcp_position(self, object_id: str, centroid: Vector3) -> Vector3:
        offset = self._tcp_offset[object_id]
        return (centroid[0] + offset[0], centroid[1] + offset[1], centroid[2] + offset[2])

    def on_belt(self, position: Vector3) -> bool:
        (x0, x1), (y0, y1) = self.config.workspace.pick_x_limits_m, self.config.workspace.pick_y_limits_m
        return x0 <= position[0] <= x1 and y0 <= position[1] <= y1

    @staticmethod
    def reachable(robot: RobotConfig, tcp: Vector3) -> bool:
        """Approximate reach test (see :class:`ReachModel`); not an IK solution."""
        reach = robot.reach
        bx, by, bz = robot.base_position_m
        horizontal = hypot(tcp[0] - bx, tcp[1] - by)
        if horizontal < reach.min_horizontal_reach_m:
            return False
        if reach.x_limits_m is not None and not reach.x_limits_m[0] <= tcp[0] <= reach.x_limits_m[1]:
            return False
        dz = tcp[2] + reach.tool_length_m - (bz + reach.shoulder_height_m)
        return sqrt(horizontal ** 2 + dz ** 2) <= reach.max_reach_m - reach.safety_margin_m

    def _exit_time(self, obj: SchedulableObject) -> float | None:
        """Time the centroid leaves the pickable belt region (``None``: never)."""
        x0, x1 = self.config.workspace.pick_x_limits_m
        vx = self.predictor.effective_velocity(obj.state)[0][0]
        if vx > 0:
            return self.predictor.time_to_reach_coordinate(obj.state, 0, x1)
        if vx < 0:
            return self.predictor.time_to_reach_coordinate(obj.state, 0, x0)
        return None

    def _reach_mask(self, robot: RobotConfig, object_id: str) -> list[bool]:
        return [
            self.on_belt(p) and self.reachable(robot, self.tcp_position(object_id, p))
            for p in self._positions[object_id]
        ]

    # ----------------------------------------------------------- timing model
    def travel_time(self, start: Vector3, grasp: Vector3) -> float:
        """Approach time to the grasp pose, including the final descent."""
        timing = self.config.timing
        pregrasp = (grasp[0], grasp[1], grasp[2] + self.config.gripper.pregrasp_height_m)
        return timing.motion_overhead_s + dist(start, pregrasp) / timing.cartesian_speed_m_s + timing.descend_s

    def busy_after_intercept(self, robot: RobotConfig, grasp: Vector3) -> float:
        """Close, lift, transfer to the place point and release."""
        timing = self.config.timing
        return (timing.gripper_close_s + timing.lift_s
                + dist(grasp, robot.place_position_m) / timing.cartesian_speed_m_s + timing.release_s)

    def latest_start(self, robot: RobotRuntime, object_id: str, first: int, last: int) -> float:
        """Latest departure time from ``robot.tool_position_m`` that still reaches
        the object somewhere in samples ``first..last`` of one reach window:
        ``max_k (t_k - travel(tool -> grasp pose at t_k))``.

        Each term uses the object's position at the same ``t_k``. When the
        object is slower than ``cartesian_speed_m_s`` the expression strictly
        increases with ``t`` (the travel time changes by less than the elapsed
        time), so the maximum is at ``last``; otherwise every sample is checked.
        """
        positions = self._positions[object_id]
        planar_speed = hypot(*self.predictor.effective_velocity(self.objects[object_id].state)[0][:2])
        indices = (last,) if planar_speed < self.config.timing.cartesian_speed_m_s else range(first, last + 1)
        return max(
            self.times[k] - self.travel_time(robot.tool_position_m, self.tcp_position(object_id, positions[k]))
            for k in indices
        )

    def has_any_window(self, object_id: str) -> bool:
        return any(any(mask) for (r, o), mask in self.windows.items() if o == object_id)

    # ----------------------------------------------------------- interception search
    def _present_neighbors(self, object_id: str, time_s: float, picked: dict[str, float]) -> tuple[str, ...]:
        result = []
        for other_id in self.objects:
            if other_id == object_id:
                continue
            picked_at = picked.get(other_id)
            if picked_at is not None and picked_at <= time_s:
                continue  # Already removed from the belt by an earlier pick.
            exit_time = self.exit_time[other_id]
            if exit_time is not None and exit_time < time_s:
                continue
            result.append(other_id)
        return tuple(sorted(result))

    def _best_grasp(self, object_id: str, index: int, time_s: float,
                    picked: dict[str, float]) -> GraspCandidate | None:
        """Best top-down grasp at sample ``index`` given the neighbours still present.

        When all objects share one velocity (a rigid belt), their relative layout
        is time invariant, so a result for the same neighbour set is reused and
        only translated. Otherwise the grasp is recomputed for every sample.
        """
        neighbor_ids = self._present_neighbors(object_id, time_s, picked)
        centroid = self._positions[object_id][index]
        key = (object_id, neighbor_ids)
        if self._uniform_velocity and key in self._grasp_cache:
            cached = self._grasp_cache[key]
            if cached is None:
                return None
            reference, grasp = cached
            shift = tuple(c - r for c, r in zip(centroid, reference))
            return replace(
                grasp,
                position_m=tuple(p + s for p, s in zip(grasp.position_m, shift)),
                pregrasp_position_m=tuple(p + s for p, s in zip(grasp.pregrasp_position_m, shift)),
            )
        neighbors = [(self.objects[n].geometry, self._positions[n][index]) for n in neighbor_ids]
        best = select_top_down_grasps(
            self.objects[object_id].geometry, centroid, neighbors=neighbors, config=self.config.gripper,
            object_velocity_m_s=self.predictor.effective_velocity(self.objects[object_id].state)[0],
        ).best
        if self._uniform_velocity:
            self._grasp_cache[key] = None if best is None else (centroid, best)
        return best

    def _conflicts(self, robot_id: str, position: Vector3, start: float, end: float,
                   commitments: Sequence[Commitment]) -> bool:
        separation = self.config.min_robot_separation_m
        for c in commitments:
            if c.robot_id == robot_id:
                continue
            if start < c.busy_until_s and c.busy_from_s < end and \
                    hypot(position[0] - c.position_m[0], position[1] - c.position_m[1]) < separation:
                return True
        return False

    def find_interception(
        self, robot: RobotRuntime, object_id: str, commitments: Sequence[Commitment],
    ) -> tuple[InterceptOption | None, str]:
        """Earliest feasible interception of ``object_id`` by ``robot``.

        A sample time ``t`` is feasible when the predicted object is on the
        pickable belt and within the robot's approximate reach, the robot can
        arrive in time (``available + travel <= t``), the pick does not overlap
        in time and space with another robot's pick, and a top-down grasp passes
        the geometric checks against the neighbours present at ``t``.
        Returns ``(option, "")`` or ``(None, reason)``.
        """
        if self.geometry_reject[object_id]:
            return None, REASON_INVALID_GRASP
        mask = self.windows[(robot.robot_id, object_id)]
        if not any(mask):
            return None, REASON_UNREACHABLE
        timing = self.config.timing
        picked = {c.object_id: c.intercept_time_s for c in commitments}
        positions = self._positions[object_id]
        start_index = max(0, int(ceil((robot.available_at_s - self.now_s) / self.config.time_step_s - 1e-9)))
        reason = REASON_TIME
        for index in range(start_index, len(self.times)):
            if not mask[index]:
                continue
            t = self.times[index]
            centroid = positions[index]
            tcp = self.tcp_position(object_id, centroid)
            travel = self.travel_time(robot.tool_position_m, tcp)
            if robot.available_at_s + travel > t + 1e-9:
                continue
            busy_end = t + self.busy_after_intercept(robot.config, tcp)
            if self._conflicts(robot.robot_id, centroid, t - timing.descend_s, t + timing.gripper_close_s + timing.lift_s, commitments):
                reason = REASON_CLEARANCE
                continue
            grasp = self._best_grasp(object_id, index, t, picked)
            if grasp is None:
                reason = REASON_CLEARANCE
                continue
            end = index
            while end + 1 < len(mask) and mask[end + 1]:
                end += 1
            deadline = self.times[end]
            # Slack: how long the robot's departure can still be postponed and
            # reach the object within this reach window. Travel time and target
            # time refer to the same sample (see latest_start). The window is
            # geometric (belt + reach); grasp/conflict checks are not repeated
            # for later samples, so slack is an optimistic timing margin.
            latest_start = self.latest_start(robot, object_id, index, end)
            return InterceptOption(
                robot_id=robot.robot_id, object_id=object_id, intercept_time_s=t,
                intercept_position_m=centroid, grasp=grasp,
                available_at_s=robot.available_at_s, travel_time_s=travel, deadline_s=deadline,
                slack_s=latest_start - robot.available_at_s,
                completion_time_s=busy_end, end_tool_position_m=robot.config.place_position_m,
                latest_start_s=latest_start,
            ), ""
        return None, reason

    def commitment(self, option: InterceptOption) -> Commitment:
        timing = self.config.timing
        return Commitment(
            robot_id=option.robot_id, object_id=option.object_id,
            intercept_time_s=option.intercept_time_s, position_m=option.intercept_position_m,
            busy_from_s=option.intercept_time_s - timing.descend_s,
            busy_until_s=option.intercept_time_s + timing.gripper_close_s + timing.lift_s,
        )
