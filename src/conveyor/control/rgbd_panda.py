"""Feed RAI RGB-D tracking into scheduled Panda pickups."""

from dataclasses import dataclass, replace
from math import dist
from typing import TYPE_CHECKING

from conveyor.perception import PerceptionConfig, RGBDObjectDetector
from conveyor.planning.moving_panda import PandaInterception, plan_moving_panda
from conveyor.planning.panda import PandaPlanningError
from conveyor.prediction import tracked_states_from_ground_truth
from conveyor.scheduling import (
    DeadlineAwareGreedyScheduler, SchedulableObject, robot_configs_from_snapshots, schedulable_objects_from_tracker,
)
from conveyor.tracking import MultiObjectTracker, TrackerConfig

if TYPE_CHECKING:
    from conveyor.simulation import CameraObservation, ConveyorSimulation


@dataclass(frozen=True)
class CameraPickupPlan:
    track_id: str
    motion: PandaInterception


class ConveyorCameraTracker:
    """Keep the RGB-D detector and Kalman track identities across captures."""

    def __init__(self, initial_belt_speed_m_s: float):
        self.initial_belt_speed_m_s = initial_belt_speed_m_s
        # The separated demo objects do not need splitting along their lit faces.
        self.detector = RGBDObjectDetector(PerceptionConfig(split_clutter_components=False))
        self.tracker = MultiObjectTracker(TrackerConfig(
            min_hits_to_confirm=3, position_measurement_std_m=0.003,
            initial_velocity_std_m_s=0.004, acceleration_process_std_m_s2=0.006,
            initial_belt_velocity_m_s=(initial_belt_speed_m_s, 0.0, 0.0),
        ))

    def observe(self, simulation: "ConveyorSimulation", *, belt_offset_m: float = 0.0) -> list[SchedulableObject]:
        observation = _pixel_center_observation(simulation.get_observation())
        detections = self.detector.detect(observation)
        # Remove travel caused by changes from the tracker’s initial belt speed.
        # Object motion relative to the belt is still measured by RGB-D.
        compensated = [replace(item, position_m=(item.position_m[0] - belt_offset_m, *item.position_m[1:]))
                       for item in detections]
        tracks = self.tracker.update(compensated, observation.simulation_time_s)
        velocity_offset = simulation.belt_speed_m_s - self.initial_belt_speed_m_s
        objects = []
        for item in schedulable_objects_from_tracker(track for track in tracks if track.missed_frames == 0):
            state = item.state
            objects.append(replace(item, state=replace(
                state, position_m=(state.position_m[0] + belt_offset_m, *state.position_m[1:]),
                velocity_m_s=(state.velocity_m_s[0] + velocity_offset, *state.velocity_m_s[1:]),
            )))
        return objects


def _pixel_center_observation(observation: "CameraObservation") -> "CameraObservation":
    calibration = observation.calibration
    fx, skew_x, cx = calibration.intrinsics[0]
    skew_y, fy, cy = calibration.intrinsics[1]
    # RAI reports viewport coordinates; detector indices start at pixel centers.
    intrinsics = ((fx, skew_x, cx - 0.5), (skew_y, fy, cy - 0.5), calibration.intrinsics[2])
    return replace(observation, calibration=replace(calibration, intrinsics=intrinsics))


def plan_camera_pickup(simulation: "ConveyorSimulation", *, close_jaws: bool = True) -> CameraPickupPlan:
    """Collect real RGB-D frames, then schedule with measured geometry and velocity."""
    if len(simulation.on_belt_object_ids) != 1:
        raise ValueError("The camera pickup demo requires exactly one on-belt object")
    return plan_camera_pickups(simulation, close_jaws=close_jaws)[0]


def _bind_scene_objects(objects, truth):
    bindings = {}
    bound_frames = set()
    for tracked in objects:
        choices = sorted(
            (dist(record.position_m, tracked.state.position_m), record.object_id)
            for record in truth.values() if record.geometry_type == tracked.geometry.geometry_type
        )
        if (not choices or choices[0][0] > 0.02 or choices[0][1] in bound_frames
                or (len(choices) > 1 and choices[1][0] - choices[0][0] < 0.005)):
            raise PandaPlanningError(f"Camera track {tracked.object_id} has no unambiguous scene object")
        object_id = choices[0][1]
        bindings[tracked.object_id] = object_id
        bound_frames.add(object_id)
    return bindings


def plan_camera_pickups(simulation: "ConveyorSimulation", *, close_jaws: bool = True) -> tuple[CameraPickupPlan, ...]:
    """Capture and schedule a separated batch from the live overhead camera."""
    expected = set(simulation.on_belt_object_ids)
    if not expected or len(expected) > 2:
        raise ValueError("Camera pickup supports one or two separated on-belt objects")
    camera = ConveyorCameraTracker(simulation.belt_speed_m_s)
    for frame in range(11):
        if frame:
            simulation.step(0.05)
        objects = camera.observe(simulation)
    if set(simulation.on_belt_object_ids) != expected:
        raise PandaPlanningError("The on-belt objects changed during camera acquisition")
    return plan_tracked_pickups(simulation, objects, close_jaws=close_jaws)


def plan_tracked_pickups(
    simulation: "ConveyorSimulation", objects: list[SchedulableObject], *,
    close_jaws: bool = True, expected_bindings: dict[str, str] | None = None,
) -> tuple[CameraPickupPlan, ...]:
    """Schedule fresh camera tracks from the current robot and belt state."""
    expected = set(simulation.on_belt_object_ids)
    if len(objects) != len(expected):
        raise PandaPlanningError(f"Expected {len(expected)} current camera tracks, got {len(objects)}")
    robot_states = simulation.get_robot_states()
    robots = robot_configs_from_snapshots(robot_states, simulation.get_targets())
    scheduler = DeadlineAwareGreedyScheduler()
    schedule = scheduler.schedule(objects, robots, simulation.simulation_time_s)
    if len(schedule.assigned) != len(objects):
        reason = schedule.rejected[0].reason if schedule.rejected else "incomplete assignment"
        raise PandaPlanningError(f"Camera tracks were not fully assigned: {reason}")
    if len(expected) == 2 and len({item.robot_id for item in schedule.assigned}) != 2:
        raise PandaPlanningError("Concurrent camera pickup needs one assignment per Panda")

    # The frame name is needed for simulation collision checks and attachment.
    # The target position, dimensions, yaw and interception still come from the tracker.
    truth = {key: record for key, record in simulation.get_ground_truth(active_only=True).items() if key in expected}
    bindings = _bind_scene_objects(objects, truth)
    if expected_bindings is not None and bindings != expected_bindings:
        raise PandaPlanningError("Camera track identities changed before replanning")
    collision_tracks = tracked_states_from_ground_truth(truth)
    estimated = {item.object_id: item.state for item in objects}
    plans = []
    for assignment in schedule.assigned:
        track_id = assignment.object_id
        object_id = bindings[track_id]
        bound = replace(assignment, object_id=object_id, grasp=replace(assignment.grasp, object_id=object_id))
        motion = plan_moving_panda(
            simulation.config, robot_states[bound.robot_id], bound, replace(estimated[track_id], object_id=object_id),
            close_duration_s=scheduler.config.timing.gripper_close_s if close_jaws else 0.0,
            collision_track=collision_tracks[object_id],
            neighbors=tuple(track for key, track in collision_tracks.items() if key != object_id),
        )
        plans.append(CameraPickupPlan(track_id, motion))
    return tuple(plans)
