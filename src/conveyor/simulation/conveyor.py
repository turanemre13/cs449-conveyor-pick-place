"""Seeded kinematic transport with optional native dynamics after the exit."""

from copy import copy
from dataclasses import dataclass
from math import cos, isfinite, pi, sin
from numbers import Real
from typing import ClassVar

from .scene import build_dual_arm_scene
from .scenarios import ObjectSpec, generate_scenario, planar_half_extents
from .robots import RobotState, get_robot_snapshots
from .targets import TargetBin, get_target_snapshots


@dataclass(frozen=True)
class ObjectGroundTruth:
    """Simulation state for debugging/evaluation, never a camera measurement."""

    object_id: str
    simulation_time_s: float
    position_m: tuple[float, float, float]
    quaternion_wxyz: tuple[float, float, float, float]
    dimensions_m: tuple[float, float, float]
    velocity_m_s: tuple[float, float, float] | None
    motion_state: str
    geometry_type: str = "box"
    exit_time_s: float | None = None
    frame: ClassVar[str] = "world"


def _nonnegative(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite non-negative number")


class ConveyorSimulation:
    """Two stationary arms and seeded objects carried along world +x.

    Surface seams follow a closed top/roller/underside loop as visual cues only.
    The ``remove`` baseline archives objects at the last fully supported pose.
    In ``fall`` mode, objects pass the edge and enter native rigid-body physics.
    ``release_object`` stops prescribed motion for a later controller handoff.
    """

    object_id = "object_001"

    def __init__(
        self, belt_speed_m_s: float = 0.12, *, scenario: str = "single",
        seed: int = 7, object_count: int = 8, exit_mode: str = "remove",
    ):
        _nonnegative("belt_speed_m_s", belt_speed_m_s)
        if exit_mode not in ("remove", "fall"):
            raise ValueError("exit_mode must be remove or fall")
        specs = generate_scenario(scenario, seed, object_count)
        self.scenario, self.seed, self.object_count = scenario, seed, object_count
        self.exit_mode = exit_mode
        self.belt_speed_m_s = float(belt_speed_m_s)
        self.config = build_dual_arm_scene()
        belt = self.config.getFrame("conveyor")
        self._belt_center = tuple(float(x) for x in belt.getPosition())
        self._length, self._width, self._height = (float(x) for x in belt.getSize()[:3])
        del belt
        # Retract the waiting grippers outside the belt and overhead sensor view.
        self.config.setJointState([-1.0, -1.0], ["a_panda_joint2", "b_panda_joint2"])
        self._initial_joints = self.config.getJointState().copy()
        self._initial_finger_states = {
            name: self.config.getFrame(name).getJointState().copy()
            for name in ("a_panda_finger_joint1", "b_panda_finger_joint1")
        }
        self.object_specs = specs
        self.object_ids = tuple(spec.object_id for spec in specs)
        self._camera = None
        self._falling = None
        self._viewer = None
        self._closed = False
        self._closed_ground_truth: dict[str, ObjectGroundTruth] = {}
        self._closed_robot_states: dict[str, RobotState] = {}
        self._closed_targets: dict[str, TargetBin] = {}
        self._seams = []
        # Paint-like visual strips: no collision or force generation.
        import robotic as ry

        if scenario == "feed":
            self._add_infeed()

        for index in range(32):
            seam = self.config.addFrame(f"belt_seam_{index:02d}")
            seam.setShape(ry.ST.box, [.006, self._width - .01, .0002])
            seam.setColor([.34, .43, .48]).setContact(0)
            self._seams.append(seam)
        del seam
        self._restore_objects(specs)

    def _add_infeed(self) -> None:
        import robotic as ry

        entry = self._belt_center[0] - self._length / 2
        upstream = entry - .56
        self._infeed_release_x = upstream + .06
        center = (entry + upstream) / 2
        y = self._belt_center[1]
        top = self._belt_center[2] + self._height / 2
        self.config.addFrame("conveyor_infeed", "world").setShape(
            ry.ST.ssBox, [entry - upstream, self._width, self._height, .005],
        ).setPosition([center, y, self._belt_center[2]]).setColor([.13, .18, .20]).setContact(1)
        self.config.addFrame("conveyor_infeed_support", "world").setShape(
            ry.ST.ssBox, [.07, .36, .67, .004],
        ).setPosition([center, y, .335]).setColor([.37, .42, .46]).setContact(0)

        # Products enter under an opaque cover and travel out of its open end.
        hood_length, hood_height = .28, .14
        hood_x = upstream + hood_length / 2
        hood_color = [.26, .34, .38]
        panels = (
            ("cover", [hood_length, self._width + .03, .025], [hood_x, y, top + hood_height]),
            ("rear", [.02, self._width + .03, hood_height], [upstream, y, top + hood_height / 2]),
            ("near", [hood_length, .02, hood_height], [hood_x, y - self._width / 2, top + hood_height / 2]),
            ("far", [hood_length, .02, hood_height], [hood_x, y + self._width / 2, top + hood_height / 2]),
        )
        for name, size, position in panels:
            self.config.addFrame(f"conveyor_infeed_{name}", "world").setShape(
                ry.ST.ssBox, [*size, .004],
            ).setPosition(position).setColor(hood_color).setContact(0)

        # The stream needs room for more than one product per destination.
        for name in ("target_bin_a", "target_bin_b"):
            root = self.config.getFrame(name)
            root.setAttributes({**root.getAttributes(), "inner_size_m": [.46, .30, .14]})
            self.config.getFrame(f"{name}_bottom").setShape(ry.ST.ssBox, [.484, .324, .015, .003])
            for side, x in (("left", -.236), ("right", .236)):
                self.config.getFrame(f"{name}_wall_{side}").setShape(
                    ry.ST.ssBox, [.012, .324, .14, .003],
                ).setRelativePosition([x, 0.0, .07])
            for side, y in (("near", -.156), ("far", .156)):
                self.config.getFrame(f"{name}_wall_{side}").setShape(
                    ry.ST.ssBox, [.46, .012, .14, .003],
                ).setRelativePosition([0.0, y, .07])

    def copy_for_planning(self) -> "ConveyorSimulation":
        """Copy a kinematic scene at its current time, including attachments."""
        self._require_open()
        if self.exit_mode != "remove":
            raise ValueError("Planning copies require remove-at-exit mode")
        import robotic as ry

        result = copy(self)
        result.config = ry.Config()
        anchor = "_conveyor_copy_anchor"
        while anchor in self.config.getFrameNames():
            anchor += "_"
        result.config.addFrame(anchor)
        result.config.addConfigurationCopy(self.config)
        result.config.delFrame(anchor)
        result._seams = [result.config.getFrame(frame.name) for frame in self._seams]
        result._motion_states = self._motion_states.copy()
        result._exit_records = self._exit_records.copy()
        result._exit_times = self._exit_times.copy()
        result._camera = None
        result._viewer = None
        return result

    @property
    def object_at_exit(self) -> bool:
        """Legacy single-object flag; batches should use all_objects_finished."""
        return self._motion_states[self.object_id] in ("exited", "falling", "settled")

    @property
    def all_objects_finished(self) -> bool:
        return all(state in ("exited", "settled") for state in self._motion_states.values())

    @property
    def active_object_ids(self) -> tuple[str, ...]:
        return tuple(key for key in self.object_ids if self._motion_states[key] not in ("queued", "exited"))

    @property
    def queued_object_ids(self) -> tuple[str, ...]:
        return tuple(key for key in self.object_ids if self._motion_states[key] == "queued")

    @property
    def on_belt_object_ids(self) -> tuple[str, ...]:
        return tuple(key for key in self.object_ids if self._motion_states[key] == "on_belt")

    @property
    def fallen_object_ids(self) -> tuple[str, ...]:
        return tuple(key for key in self.object_ids if self._motion_states[key] in ("falling", "settled"))

    def reset(self, seed: int | None = None) -> None:
        """Restore a reproducible batch and reset its clock, keeping the Config."""
        selected_seed = self.seed if seed is None else seed
        specs = generate_scenario(self.scenario, selected_seed, self.object_count)
        self.seed = selected_seed
        self._closed = False
        self._closed_ground_truth = {}
        self._closed_robot_states = {}
        self._closed_targets = {}
        self._restore_objects(specs)

    def _restore_objects(self, specs: tuple[ObjectSpec, ...]) -> None:
        if self._camera is not None:
            self._camera.close()
        if self._falling is not None:
            self._falling.reset()
        names = set(self.config.getFrameNames())
        for key in self.object_ids:
            if key in names:
                self.config.delFrame(key)
        self.object_specs = specs
        self.object_ids = tuple(spec.object_id for spec in specs)
        self.simulation_time_s = 0.0
        self._motion_states = {key: "on_belt" for key in self.object_ids}
        self._exit_records: dict[str, ObjectGroundTruth] = {}
        self._exit_times: dict[str, float] = {}
        self.config.setJointState(self._initial_joints)
        for name, coordinates in self._initial_finger_states.items():
            self.config.getFrame(name).setJointState(coordinates)
        for spec in specs:
            frame = self.config.addFrame(spec.object_id)
            if self.scenario == "feed" and spec.position_m[0] < self._infeed_release_x:
                self._motion_states[spec.object_id] = "queued"
            else:
                self._set_product_shape(frame, spec)
            frame.setPosition(list(spec.position_m))
            frame.setQuaternion([cos(spec.yaw_rad / 2), 0, 0, sin(spec.yaw_rad / 2)])
        del frame
        self._rebuild_native_caches()
        self._update_seams()
        self._refresh_collision_proxies()

    @staticmethod
    def _set_product_shape(frame, spec: ObjectSpec) -> None:
        import robotic as ry

        if spec.geometry_type == "box":
            frame.setShape(ry.ST.ssBox, [*spec.dimensions_m, .005])
        else:
            frame.setShape(ry.ST.cylinder, [spec.dimensions_m[2], spec.dimensions_m[0] / 2])
        frame.setColor(list(spec.color_rgb)).setContact(1).setMass(.12)

    def _activate_queued_product(self, frame, spec: ObjectSpec, x: float) -> bool:
        if self._motion_states[spec.object_id] != "queued" or x < self._infeed_release_x:
            return False
        self._set_product_shape(frame, spec)
        self._motion_states[spec.object_id] = "on_belt"
        return True

    def step(self, dt: float) -> None:
        """Advance simulation time by dt seconds; independent of wall-clock time."""
        self._require_open()
        _nonnegative("dt", dt)
        if dt == 0:
            raise ValueError("dt must be positive")
        if self.exit_mode == "fall":
            self._step_with_falls(float(dt))
            self._update_seams()
            self._refresh_collision_proxies()
            return
        previous_time = self.simulation_time_s
        self.simulation_time_s += float(dt)
        topology_changed = False
        for spec in self.object_specs:
            key = spec.object_id
            if self._motion_states[key] not in ("on_belt", "queued"):
                continue
            frame = self.config.getFrame(key)
            position = frame.getPosition()
            next_x = float(position[0]) + self.belt_speed_m_s * float(dt)
            topology_changed |= self._activate_queued_product(frame, spec, next_x)
            exit_x = self._belt_center[0] + self._length / 2 - planar_half_extents(spec)[0]
            if next_x >= exit_x and self.belt_speed_m_s > 0:
                exit_time = previous_time + max(0.0, exit_x - float(position[0])) / self.belt_speed_m_s
                frame.setPosition([exit_x, float(position[1]), float(position[2])])
                self._motion_states[key] = "exited"
                self._exit_records[key] = self._snapshot(spec, exit_time, exit_time)
                del frame
                self.config.delFrame(key)
                topology_changed = True
            else:
                frame.setPosition([next_x, float(position[1]), float(position[2])])
        if topology_changed:
            frame = None  # Drop any borrowed wrapper before native frame rebuild.
            self._rebuild_native_caches()
        self._update_seams()
        self._refresh_collision_proxies()

    def _step_with_falls(self, dt: float) -> None:
        """Split at handoff events and at 5 ms to resolve edge/floor contacts."""
        target_time = self.simulation_time_s + dt
        edge_x = self._belt_center[0] + self._length / 2 + .004
        while True:
            # Hand off at COM-over-edge; the body may still contact/tip on deck.
            if self.belt_speed_m_s > 0:
                for spec in self.object_specs:
                    key = spec.object_id
                    if self._motion_states[key] != "on_belt":
                        continue
                    frame = self.config.getFrame(key)
                    if float(frame.getPosition()[0]) >= edge_x - 1e-10:
                        if self._falling is None:
                            from .falling import FallingObjects

                            self._falling = FallingObjects(self.config)
                        self._falling.add_object(spec, frame.getPose(), (self.belt_speed_m_s, 0.0, 0.0))
                        self._motion_states[key] = "falling"
                        self._exit_times[key] = self.simulation_time_s

            if self.simulation_time_s >= target_time:
                break
            step_s = min(.005, target_time - self.simulation_time_s)
            if self.belt_speed_m_s > 0:
                for key in self.on_belt_object_ids:
                    distance = edge_x - float(self.config.getFrame(key).getPosition()[0])
                    if distance > 1e-10:
                        step_s = min(step_s, distance / self.belt_speed_m_s)
            if self._falling is not None:
                self._falling.step(step_s)
            topology_changed = False
            for spec in self.object_specs:
                if self._motion_states[spec.object_id] not in ("on_belt", "queued"):
                    continue
                frame = self.config.getFrame(spec.object_id)
                position = frame.getPosition()
                x = float(position[0]) + self.belt_speed_m_s * step_s
                frame.setPosition([x, float(position[1]), float(position[2])])
                topology_changed |= self._activate_queued_product(frame, spec, x)
            if topology_changed:
                frame = None
                self._rebuild_native_caches()
            self.simulation_time_s = min(target_time, self.simulation_time_s + step_s)
            if self._falling is not None:
                for key in self.fallen_object_ids:
                    self._motion_states[key] = "settled" if self._falling.is_settled(key) else "falling"

    def release_object(self, object_id: str) -> None:
        """Stop conveyor pose updates; this neither grasps nor attaches the box."""
        self._require_open()
        if object_id not in self._motion_states:
            raise KeyError(object_id)
        if self._motion_states[object_id] == "exited":
            raise ValueError(f"Object has already exited: {object_id}")
        if self._motion_states[object_id] == "queued":
            raise ValueError(f"Object is still queued in the feeder: {object_id}")
        if self._motion_states[object_id] in ("falling", "settled"):
            self._falling.remove_object(object_id)
        self._motion_states[object_id] = "released"

    def get_ground_truth(self, *, active_only: bool = False) -> dict[str, ObjectGroundTruth]:
        """Current states plus optional historical terminal records for exits.

        Exited records retain their actual exit timestamp and terminal pose.
        Queued positions describe when products will reach the covered feeder.
        Use active_only=True for the objects with visible geometry in the scene.
        """
        if self._closed:
            return {
                key: state for key, state in self._closed_ground_truth.items()
                if not active_only or state.motion_state not in ("queued", "exited")
            }
        result = {}
        for spec in self.object_specs:
            if self._motion_states[spec.object_id] == "exited":
                if not active_only:
                    result[spec.object_id] = self._exit_records[spec.object_id]
            elif not active_only or self._motion_states[spec.object_id] != "queued":
                result[spec.object_id] = self._snapshot(spec, self.simulation_time_s)
        return result

    def _snapshot(self, spec: ObjectSpec, time_s: float, exit_time: float | None = None) -> ObjectGroundTruth:
        frame = self.config.getFrame(spec.object_id)
        state = self._motion_states[spec.object_id]
        velocity = self.belt_speed_m_s if state in ("on_belt", "queued") else 0.0
        linear_velocity = None if state == "released" else (velocity, 0.0, 0.0)
        if state in ("falling", "settled"):
            linear_velocity = self._falling.get_velocity(spec.object_id)
        return ObjectGroundTruth(
            object_id=spec.object_id, simulation_time_s=time_s,
            position_m=tuple(float(x) for x in frame.getPosition()),
            quaternion_wxyz=tuple(float(x) for x in frame.getQuaternion()),
            dimensions_m=spec.dimensions_m,
            velocity_m_s=linear_velocity,
            motion_state=state, geometry_type=spec.geometry_type,
            exit_time_s=exit_time if exit_time is not None else self._exit_times.get(spec.object_id),
        )

    def get_observation(self):
        """Capture actual RGB-D; requires OpenGL/WSLg, even without a GUI view."""
        self._require_open()
        if self._camera is None:
            from .camera import RGBDCamera

            self._camera = RGBDCamera(self.config)
        return self._camera.capture(self.simulation_time_s)

    def get_robot_states(self) -> dict[str, RobotState]:
        """Read named arm/finger joints and world poses at the current time."""
        if self._closed:
            return dict(self._closed_robot_states)
        return get_robot_snapshots(self.config, self.simulation_time_s)

    def get_targets(self) -> dict[str, TargetBin]:
        """Read bin geometry and nominal points; no IK feasibility is implied."""
        if self._closed:
            return dict(self._closed_targets)
        return get_target_snapshots(self.config, self.simulation_time_s)

    def view(self, message: str = "") -> int:
        """Open/update the optional GUI and preserve it across scene resets."""
        self._require_open()
        key = self.config.view(False, message)
        self._viewer = self.config.get_viewer()
        return key

    def close(self) -> None:
        """Release resources while preserving immutable final ground truth.

        Call reset() before stepping, capturing, or releasing objects again.
        """
        if self._closed:
            return
        try:
            self._closed_ground_truth = self.get_ground_truth()
            self._closed_robot_states = self.get_robot_states()
            self._closed_targets = self.get_targets()
        finally:
            if self._camera is not None:
                self._camera.close()
                self._camera = None
            if self._falling is not None:
                self._falling.close()
                self._falling = None
            if self._viewer is not None:
                self.config.view_close()
                self._viewer = None
            self._closed = True

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("Simulation is closed; call reset() before continuing")

    def _refresh_collision_proxies(self) -> None:
        # RAI 0.4.1 can retain broadphase candidates after root-frame moves.
        # Reapply current joints to invalidate them without commanding motion.
        self.config.setJointState(self.config.getJointState())

    def _rebuild_native_caches(self) -> None:
        """Reset RAI 0.4.1's structural collision cache, retaining Config identity.

        FCL does not rebuild its geometry table after delFrame. clear/copy is
        the bound reset path. A temporary destination frame keeps the native
        frame array one-dimensional during addConfigurationCopy; remove it
        before collision queries. Reacquire all internally borrowed Frames.
        """
        import robotic as ry

        seam_names = tuple(frame.name for frame in self._seams)
        self._seams = []  # Release pybind Frame wrappers before native clear().
        if self._camera is not None:
            self._camera.close()
        snapshot = ry.Config()
        snapshot.addConfigurationCopy(self.config)
        anchor = "_conveyor_cache_anchor"
        while anchor in snapshot.getFrameNames():
            anchor += "_"
        self.config.clear()
        self.config.addFrame(anchor)
        self.config.addConfigurationCopy(snapshot)
        self.config.delFrame(anchor)
        if self._viewer is not None:
            self.config.set_viewer(self._viewer)
        self._seams = [self.config.getFrame(name) for name in seam_names]

    def _update_seams(self) -> None:
        # A capsule-shaped cross-section: straight top, right semicircle,
        # straight underside, left semicircle. Positive speed runs the top +x.
        radius = self._height / 2 + .0001
        arc = pi * radius
        perimeter = 2 * self._length + 2 * arc
        travel = (self.belt_speed_m_s * self.simulation_time_s) % perimeter
        for index, seam in enumerate(self._seams):
            distance = (index * perimeter / len(self._seams) + travel) % perimeter
            if distance < self._length:
                x, z, angle = -self._length / 2 + distance, radius, 0.0
            elif distance < self._length + arc:
                angle = (distance - self._length) / radius
                x = self._length / 2 + radius * sin(angle)
                z = radius * cos(angle)
            elif distance < 2 * self._length + arc:
                x = self._length / 2 - (distance - self._length - arc)
                z, angle = -radius, pi
            else:
                angle = pi + (distance - 2 * self._length - arc) / radius
                x = -self._length / 2 + radius * sin(angle)
                z = radius * cos(angle)
            seam.setPosition([
                self._belt_center[0] + x, self._belt_center[1], self._belt_center[2] + z,
            ])
            seam.setQuaternion([cos(angle / 2), 0.0, sin(angle / 2), 0.0])
