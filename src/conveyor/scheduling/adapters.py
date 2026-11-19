"""Build scheduler inputs from the repository's scene and simulation records.

All adapters duck-type the simulation dataclasses (``ObjectSpec``,
``ObjectGroundTruth``, ``RobotState``, ``TargetBin``) so this package does not
import the simulation or RAI.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from conveyor.grasping.geometry import geometry_from_ground_truth, geometry_from_spec
from conveyor.prediction.ground_truth import tracked_state_from_ground_truth
from conveyor.types import TrackedState

from .models import ReachModel, RobotConfig, SchedulableObject

#: Values copied from ``scenes/dual_arm.g`` (robot mounts and bin approach frames).
SCENE_ROBOT_BASES = {"robot_a": (-0.55, -0.60, 0.75), "robot_b": (0.55, -0.60, 0.75)}
SCENE_PLACE_APPROACH = {"robot_a": (-1.0, -0.85, 1.05), "robot_b": (1.0, -0.85, 1.05)}
#: Assumed ready pose of the tool (not read from RAI): above the near belt edge.
DEFAULT_READY_OFFSET = (0.0, 0.25, 0.35)


def default_robot_configs(
    robot_ids: Iterable[str] = ("robot_a", "robot_b"),
    *,
    reach: ReachModel | None = None,
    available_at_s: float = 0.0,
) -> list[RobotConfig]:
    """Robots placed as in ``scenes/dual_arm.g`` with an assumed ready tool pose.

    Pass ``("robot_a",)`` or ``("robot_b",)`` for a one-robot configuration.
    """
    reach = reach or ReachModel()
    result = []
    for robot_id in robot_ids:
        if robot_id not in SCENE_ROBOT_BASES:
            raise ValueError(f"unknown scene robot {robot_id!r}")
        base = SCENE_ROBOT_BASES[robot_id]
        ready = tuple(b + o for b, o in zip(base, DEFAULT_READY_OFFSET))
        result.append(RobotConfig(robot_id, base, ready, SCENE_PLACE_APPROACH[robot_id], available_at_s, reach))
    return result


def robot_configs_from_snapshots(
    robot_states: Mapping[str, Any],
    targets: Mapping[str, Any] | None = None,
    *,
    available_at_s: Mapping[str, float] | None = None,
    reach: ReachModel | None = None,
) -> list[RobotConfig]:
    """Use live ``RobotState``/``TargetBin`` snapshots from ``ConveyorSimulation``.

    The tool starts at the actual gripper position; the place point is the
    approach position of the bin whose ``preferred_robot_id`` matches the robot
    (falling back to the scene default). Availability defaults to the snapshot
    time because the simulation does not yet report controller status.
    """
    reach = reach or ReachModel()
    places = dict(SCENE_PLACE_APPROACH)
    for target in (targets or {}).values():
        places[target.preferred_robot_id] = tuple(float(x) for x in target.approach_position_m)
    result = []
    for robot_id, state in sorted(robot_states.items()):
        available = (available_at_s or {}).get(robot_id, float(state.simulation_time_s))
        result.append(RobotConfig(
            robot_id=robot_id,
            base_position_m=tuple(float(x) for x in state.base_position_m),
            tool_position_m=tuple(float(x) for x in state.gripper_position_m),
            place_position_m=places.get(robot_id, tuple(float(x) for x in state.base_position_m)),
            available_at_s=available, reach=reach,
        ))
    return result


def schedulable_objects_from_ground_truth(records: Mapping[str, Any] | Iterable[Any]) -> list[SchedulableObject]:
    """On-belt ground-truth objects as scheduler input (tracker stand-in)."""
    items = records.values() if isinstance(records, Mapping) else records
    result = []
    for record in items:
        if getattr(record, "motion_state", "on_belt") != "on_belt":
            continue
        result.append(SchedulableObject(tracked_state_from_ground_truth(record), geometry_from_ground_truth(record)))
    return sorted(result, key=lambda o: o.object_id)


def schedulable_objects_from_specs(
    specs: Iterable[Any], belt_speed_m_s: float, *, time_s: float = 0.0,
    x_offset_m: float = 0.0,
) -> list[SchedulableObject]:
    """Initial scenario specs (``generate_scenario``) moving at the belt speed.

    Equivalent to ground truth at ``time_s`` for objects still on the belt.
    ``x_offset_m`` shifts the layout along the belt, e.g. to model a batch that
    has not entered the belt yet (negative values).
    """
    result = []
    for spec in specs:
        x, y, z = spec.position_m
        state = TrackedState(spec.object_id, float(time_s), (float(x) + x_offset_m, float(y), float(z)),
                             (float(belt_speed_m_s), 0.0, 0.0))
        result.append(SchedulableObject(state, geometry_from_spec(spec)))
    return result


def schedulable_objects_from_tracker(
    tracker_or_tracks: Any,
    *,
    confirmed_only: bool = True,
) -> list[SchedulableObject]:
    """Convert :class:`conveyor.tracking.MultiObjectTracker` or :class:`TrackedObject` records to scheduler inputs."""
    if hasattr(tracker_or_tracks, "schedulable_objects"):
        items = tracker_or_tracks.schedulable_objects(confirmed_only=confirmed_only)
    elif hasattr(tracker_or_tracks, "get_tracked_objects"):
        items = [t.to_schedulable_object() for t in tracker_or_tracks.get_tracked_objects(confirmed_only=confirmed_only)]
    else:
        items = []
        for item in tracker_or_tracks:
            if isinstance(item, SchedulableObject):
                items.append(item)
            elif hasattr(item, "to_schedulable_object"):
                if confirmed_only and hasattr(item, "is_confirmed") and not item.is_confirmed:
                    continue
                items.append(item.to_schedulable_object())
            else:
                raise TypeError(f"Cannot convert {type(item).__name__!r} to SchedulableObject")
    return sorted(items, key=lambda o: o.object_id)
