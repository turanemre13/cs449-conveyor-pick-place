"""Dispatch both Pandas on one running conveyor with serialized arm motion."""

from dataclasses import dataclass
from math import inf
from typing import TYPE_CHECKING, Callable

from conveyor.planning.moving_panda import plan_moving_panda
from conveyor.planning.panda import PandaPlanningError
from conveyor.planning.panda_gripper import FingerClosure
from conveyor.scheduling import (
    DeadlineAwareGreedyScheduler, robot_configs_from_snapshots, schedulable_objects_from_ground_truth,
)

from .panda_pickup import execute_panda_pickup
from .panda_placement import PandaPlacementResult, execute_panda_placement

if TYPE_CHECKING:
    from conveyor.simulation import ConveyorSimulation


@dataclass(frozen=True)
class PandaJobResult:
    robot_id: str
    object_id: str
    planning_time_s: float
    intercepted_at_s: float
    finished_at_s: float
    placement: PandaPlacementResult
    pickup_time_s: float
    closure: FingerClosure
    track_id: str | None = None


SerialPlacement = PandaJobResult


@dataclass(frozen=True)
class SerialRunResult:
    placed: tuple[PandaJobResult, ...]
    unplaced: tuple[tuple[str, str], ...]
    finished_at_s: float


def execute_serial_pandas(
    simulation: "ConveyorSimulation",
    *,
    after_step: Callable[[str, str, float], bool] | None = None,
) -> SerialRunResult | None:
    """Reschedule after each retreat. Return None if the viewer cancels playback."""
    if simulation.exit_mode != "remove":
        raise ValueError("Serial Panda execution requires remove-at-exit mode")
    initial_ids = simulation.on_belt_object_ids
    scheduler = DeadlineAwareGreedyScheduler()
    completed = []
    reasons: dict[str, str] = {}

    while simulation.on_belt_object_ids:
        now = simulation.simulation_time_s
        robots = simulation.get_robot_states()
        truth = simulation.get_ground_truth(active_only=True)
        objects = schedulable_objects_from_ground_truth(truth)
        tracks = {item.object_id: item.state for item in objects}
        candidates = robot_configs_from_snapshots(robots, simulation.get_targets())
        schedule = scheduler.schedule(objects, candidates, now)
        reasons = {assignment.object_id: assignment.reason for assignment in schedule.rejected}
        # With one execution slot, dispatch the assignment whose window closes first.
        ordered = sorted(
            schedule.assigned,
            key=lambda item: (item.deadline_s if item.deadline_s is not None else inf, item.intercept_time_s),
        )
        for assignment in ordered:
            object_id = assignment.object_id
            neighbors = tuple(track for key, track in tracks.items() if key != object_id)
            try:
                plan = plan_moving_panda(
                    simulation.config, robots[assignment.robot_id], assignment, tracks[object_id],
                    close_duration_s=scheduler.config.timing.gripper_close_s, neighbors=neighbors,
                )
            except (PandaPlanningError, ValueError) as exc:
                reasons[object_id] = str(exc)
                continue
            break
        else:
            break

        pick_step = place_step = None
        if after_step is not None:
            def pick_step(time_s):
                return after_step(plan.robot_id, "Intercept and lift", time_s)

            def place_step(time_s):
                return after_step(plan.robot_id, "Place and retreat", time_s)

        try:
            pickup = execute_panda_pickup(simulation, plan, after_step=pick_step)
            if pickup is None:
                return None
            placement = execute_panda_placement(simulation, plan, after_step=place_step)
            if placement is None:
                return None
        except (PandaPlanningError, ValueError) as exc:
            reasons[plan.object_id] = str(exc)
            break

        completed.append(PandaJobResult(
            plan.robot_id, plan.object_id, now, plan.intercept_time_s, simulation.simulation_time_s, placement,
            pickup.pickup_time_s, plan.closure,
        ))

    placed = {item.object_id for item in completed}
    final = simulation.get_ground_truth()
    unplaced = tuple(
        (key, "exited_before_placement" if final[key].motion_state == "exited" else reasons.get(key, "not_dispatched"))
        for key in initial_ids if key not in placed
    )
    return SerialRunResult(tuple(completed), unplaced, simulation.simulation_time_s)
