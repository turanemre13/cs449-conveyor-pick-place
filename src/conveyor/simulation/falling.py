"""CPU PhysX dynamics for missed objects after the kinematic belt handoff.

The lightweight physics world contains only fixed environment geometry and the
objects handed to it. Robot joints and carried objects remain outside this
world, so the simulator never acts as a robot controller.
"""

from math import floor, isfinite, sqrt
from numbers import Real
from typing import TYPE_CHECKING

from .scenarios import ObjectSpec

if TYPE_CHECKING:
    import robotic as ry


def _finite_vector(name: str, value, length: int) -> tuple[float, ...]:
    try:
        result = tuple(value)
    except TypeError as exc:
        raise ValueError(f"{name} must contain {length} finite numbers") from exc
    if len(result) != length or any(
        isinstance(x, bool) or not isinstance(x, Real) or not isfinite(x) for x in result
    ):
        raise ValueError(f"{name} must contain {length} finite numbers")
    return tuple(float(x) for x in result)


class FallingObjects:
    """Own real rigid-body motion, floor impacts, and object/object collisions.

    ``add_object`` takes a world pose in ``(x,y,z,w,qx,qy,qz)`` order and the
    inherited world linear velocity. Initial angular velocity is zero; tipping
    and scattering emerge from edge and impact contacts, without random kicks.

    The installed RAI binding cannot register new actors dynamically. A small
    PhysX world is therefore rebuilt on add/remove events, retaining every
    existing body's pose and linear/angular velocity. It is not rebuilt during
    ordinary steps. Contact solver caches are renewed at those actor events.
    """

    MAX_SUBSTEP_S = .005
    SETTLE_LINEAR_M_S = .03
    SETTLE_ANGULAR_RAD_S = .20
    SETTLE_DURATION_S = .50

    def __init__(self, main_config: "ry.Config"):
        import robotic as ry

        self._ry = ry
        self._main = main_config
        self._physics = None
        self._engine = None
        self._ids: list[str] = []
        self._quiet_s: dict[str, float] = {}
        self._velocities: dict[str, tuple[tuple[float, ...], tuple[float, ...]]] = {}
        self.time_s = 0.0
        self.reset()

    @property
    def object_ids(self) -> tuple[str, ...]:
        return tuple(self._ids)

    def reset(self) -> None:
        """Discard dynamic actors and copy current fixed geometry from the scene."""
        self._engine = None
        self._physics = self._ry.Config()
        self._ids.clear()
        self._quiet_s.clear()
        self._velocities.clear()
        self.time_s = 0.0
        self._pending_s = 0.0
        for name in self._main.getFrameNames():
            if name not in ("floor", "conveyor") and not name.startswith((
                "conveyor_leg_", "robot_a_pedestal", "robot_b_pedestal", "target_bin_", "reject_bin_",
            )):
                continue
            source = self._main.getFrame(name)
            if source.getShapeType() == self._ry.ST.none:
                continue
            frame = self._physics.addFrame(name)
            frame.setShape(source.getShapeType(), source.getSize())
            frame.setPose(source.getPose()).setContact(1)
            frame.setAttributes({"friction": .45, "restitution": .18})
        if "floor" not in self._physics.getFrameNames():
            raise ValueError("Falling-object physics requires a floor frame")

    def add_object(self, spec: ObjectSpec, pose, velocity_m_s) -> None:
        """Transfer a still-visible main-scene object to rigid-body ownership."""
        world_pose = _finite_vector("pose", pose, 7)
        velocity = _finite_vector("velocity_m_s", velocity_m_s, 3)
        if self._physics is None:
            raise RuntimeError("FallingObjects is closed; call reset before reusing it")
        if spec.object_id in self._ids:
            raise ValueError(f"Object already belongs to physics: {spec.object_id}")
        if spec.object_id not in self._main.getFrameNames():
            raise KeyError(spec.object_id)
        quaternion_length = sqrt(sum(x * x for x in world_pose[3:]))
        if abs(quaternion_length - 1.0) > 1e-5:
            raise ValueError("pose quaternion must have unit length")

        velocities = self._read_velocities()
        source = self._main.getFrame(spec.object_id)
        frame = self._physics.addFrame(spec.object_id)
        frame.setShape(source.getShapeType(), source.getSize()).setPose(list(world_pose)).setMass(.12).setContact(1)
        frame.setAttributes({"friction": .45, "restitution": .18})
        self._ids.append(spec.object_id)
        self._quiet_s[spec.object_id] = 0.0
        velocities[spec.object_id] = (velocity, (0.0, 0.0, 0.0))
        self._rebuild_engine(velocities)
        self._main.getFrame(spec.object_id).setPose(list(world_pose))

    def remove_object(self, object_id: str) -> None:
        """Relinquish an actor without changing its last main-scene pose."""
        if object_id not in self._ids:
            raise KeyError(object_id)
        velocities = self._read_velocities()
        velocities.pop(object_id)
        self._physics.delFrame(object_id)
        self._ids.remove(object_id)
        self._quiet_s.pop(object_id)
        self._velocities.pop(object_id, None)
        self._rebuild_engine(velocities)

    def step(self, dt: float) -> None:
        """Advance PhysX in fixed 5 ms steps and copy poses back."""
        if isinstance(dt, bool) or not isinstance(dt, Real) or not isfinite(dt) or dt <= 0:
            raise ValueError("dt must be a finite positive number")
        if self._physics is None:
            raise RuntimeError("FallingObjects is closed; call reset before reusing it")
        # Retain fractional steps across camera frames and belt handoffs.
        # Near-zero remainder steps can destabilize PhysX contacts.
        h = self.MAX_SUBSTEP_S
        total = self._pending_s + float(dt)
        count = floor((total + 1e-12) / h)
        self._pending_s = max(0.0, total - count * h)
        for _ in range(count):
            if self._engine is not None:
                self._engine.step([], h, self._ry.ControlMode.none)
                self._velocities = self._read_velocities()
                for key, (linear, angular) in self._velocities.items():
                    quiet = (
                        sum(v * v for v in linear) <= self.SETTLE_LINEAR_M_S ** 2
                        and sum(v * v for v in angular) <= self.SETTLE_ANGULAR_RAD_S ** 2
                    )
                    self._quiet_s[key] = self._quiet_s[key] + h if quiet else 0.0
        self.time_s += count * h
        for key in self._ids:
            self._main.getFrame(key).setPose(self._physics.getFrame(key).getPose())

    def get_velocity(self, object_id: str) -> tuple[float, float, float]:
        return self._velocities[object_id][0]

    def get_angular_velocity(self, object_id: str) -> tuple[float, float, float]:
        return self._velocities[object_id][1]

    def is_settled(self, object_id: str) -> bool:
        """Whether native velocities have stayed small for at least half a second.

        This reports motion; it neither freezes the body nor disables contacts.
        A later impact can make the same body unsettled again.
        """
        return self._quiet_s[object_id] >= self.SETTLE_DURATION_S

    def close(self) -> None:
        """Release the native engine; leave current main-scene object poses alone."""
        self._engine = None
        self._physics = None
        self._ids.clear()
        self._quiet_s.clear()
        self._velocities.clear()

    def _read_velocities(self):
        if self._engine is None:
            return {}
        state = self._engine.getState()
        return {
            frame.name: tuple(tuple(float(x) for x in row) for row in state.freeVel[index])
            for index, frame in enumerate(self._engine.getFreeFrames())
        }

    def _rebuild_engine(self, velocities) -> None:
        import numpy as np

        self._engine = None
        if not self._ids:
            self._velocities = {}
            return
        try:
            self._engine = self._ry.Simulation(self._physics, self._ry.SimulationEngine.physx, 0)
        except RuntimeError as exc:
            raise RuntimeError(
                "CPU PhysX could not initialize. Use the Linux robotic==0.4.1 "
                "backend in requirements-rai.txt or select --exit-mode remove."
            ) from exc
        state = self._engine.getState()
        state.freeVel = np.array([
            velocities[frame.name] for frame in self._engine.getFreeFrames()
        ], dtype=float)
        state.time = self.time_s
        self._engine.setState(state)
        self._velocities = self._read_velocities()
