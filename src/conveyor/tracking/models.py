"""Records and configuration for multi-object Kalman tracking.

All coordinates and velocities use the scene's fixed world frame, in metres
and metres/second; timestamps are simulation seconds since the latest reset.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from numbers import Real
from typing import ClassVar

from conveyor.grasping.geometry import GEOMETRY_TYPES, ObjectGeometry
from conveyor.scheduling.models import SchedulableObject
from conveyor.types import TrackedState, Vector3

TRACK_TENTATIVE = "tentative"
TRACK_CONFIRMED = "confirmed"


def _finite(name: str, value: float, *, minimum: float | None = None, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
        raise ValueError(f"{name} must be a finite real number")
    result = float(value)
    if positive and result <= 0.0:
        raise ValueError(f"{name} must be positive")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return result


def _vector3(name: str, value: Vector3, *, positive: bool = False) -> Vector3:
    if not isinstance(value, tuple) or len(value) != 3:
        raise ValueError(f"{name} must be a tuple of three finite real numbers")
    return tuple(_finite(name, x, positive=positive) for x in value)  # type: ignore[return-value]


@dataclass(frozen=True)
class TrackerConfig:
    """Parameters for constant-velocity Kalman filtering and data association.

    Parameters
    ----------
    max_association_distance_m:
        Gating radius in metres between a track's predicted world centroid
        and a new measurement centroid.
    color_distance_weight_m:
        Additional penalty (in equivalent metres) per unit RGB Euclidean
        difference when associating Rich :class:`DetectedObject` records.
    min_hits_to_confirm:
        Number of associated measurements required before a track's status
        becomes ``"confirmed"``. Set to 1 to emit tracks immediately on the
        first frame, or 2+ to filter transient false positives.
    max_missed_frames:
        Maximum consecutive frames a track may go unassociated before deletion.
    max_time_without_update_s:
        Maximum simulation time a track is kept alive without measurements.
    position_measurement_std_m:
        1-sigma world position measurement noise (metres) for Kalman matrix R.
    acceleration_process_std_m_s2:
        Continuous-time white-noise acceleration spectral density (m/s^2) for Q.
    initial_position_std_m:
        Initial position standard deviation (metres) for P_0.
    initial_velocity_std_m_s:
        Initial velocity standard deviation (m/s) for P_0.
    initial_belt_velocity_m_s:
        Prior velocity assigned to a newly spawned track (defaults to 0).
    planar_belt_constraint:
        If true, damps spurious vertical velocity ``vz`` toward zero since
        objects on the horizontal belt translate along ``(x, y)``.
    geometry_smoothing_alpha:
        Exponential moving average weight in ``(0, 1]`` for dimensions and yaw.
    belt_exit_x_m:
        Tracks whose predicted ``x`` exceeds this boundary are pruned.
    """

    max_association_distance_m: float = 0.09
    color_distance_weight_m: float = 0.04
    min_hits_to_confirm: int = 1
    max_missed_frames: int = 4
    max_time_without_update_s: float = 0.60
    position_measurement_std_m: float = 0.0025
    acceleration_process_std_m_s2: float = 0.08
    initial_position_std_m: float = 0.005
    initial_velocity_std_m_s: float = 0.25
    initial_belt_velocity_m_s: Vector3 = (0.0, 0.0, 0.0)
    planar_belt_constraint: bool = True
    geometry_smoothing_alpha: float = 0.35
    belt_exit_x_m: float = 1.25

    def __post_init__(self) -> None:
        _finite("max_association_distance_m", self.max_association_distance_m, positive=True)
        _finite("color_distance_weight_m", self.color_distance_weight_m, minimum=0.0)
        if isinstance(self.min_hits_to_confirm, bool) or not isinstance(self.min_hits_to_confirm, int) or self.min_hits_to_confirm < 1:
            raise ValueError("min_hits_to_confirm must be an integer >= 1")
        if isinstance(self.max_missed_frames, bool) or not isinstance(self.max_missed_frames, int) or self.max_missed_frames < 0:
            raise ValueError("max_missed_frames must be a non-negative integer")
        _finite("max_time_without_update_s", self.max_time_without_update_s, positive=True)
        _finite("position_measurement_std_m", self.position_measurement_std_m, positive=True)
        _finite("acceleration_process_std_m_s2", self.acceleration_process_std_m_s2, positive=True)
        _finite("initial_position_std_m", self.initial_position_std_m, positive=True)
        _finite("initial_velocity_std_m_s", self.initial_velocity_std_m_s, positive=True)
        _vector3("initial_belt_velocity_m_s", self.initial_belt_velocity_m_s)
        _finite("geometry_smoothing_alpha", self.geometry_smoothing_alpha, positive=True)
        if self.geometry_smoothing_alpha > 1.0:
            raise ValueError("geometry_smoothing_alpha must lie in (0, 1]")
        _finite("belt_exit_x_m", self.belt_exit_x_m)


@dataclass(frozen=True)
class TrackedObject:
    """Complete tracked object record combining kinematic state and 3-D geometry.

    ``state`` is the shared :class:`conveyor.types.TrackedState` consumed by
    ``ConstantVelocityPredictor``. ``dimensions_m``, ``yaw_rad`` and
    ``geometry_type`` supply the smoothed 3-D shape required by Hüseyin's
    grasp selection and dual-arm scheduling modules.
    """

    state: TrackedState
    geometry_type: str = "box"
    dimensions_m: Vector3 = (0.06, 0.06, 0.06)
    yaw_rad: float = 0.0
    color_rgb: Vector3 = (0.5, 0.5, 0.5)
    hits: int = 1
    age_frames: int = 1
    time_since_update_s: float = 0.0
    missed_frames: int = 0
    status: str = TRACK_CONFIRMED
    position_std_m: Vector3 = (0.0, 0.0, 0.0)
    velocity_std_m_s: Vector3 = (0.0, 0.0, 0.0)
    frame: ClassVar[str] = "world"

    def __post_init__(self) -> None:
        if not isinstance(self.state, TrackedState):
            raise TypeError("state must be a conveyor.types.TrackedState")
        if self.geometry_type not in GEOMETRY_TYPES:
            raise ValueError(f"geometry_type must be one of {GEOMETRY_TYPES}")
        _vector3("dimensions_m", self.dimensions_m, positive=True)
        _finite("yaw_rad", self.yaw_rad)
        _vector3("color_rgb", self.color_rgb)
        if self.status not in (TRACK_TENTATIVE, TRACK_CONFIRMED):
            raise ValueError(f"status must be {TRACK_TENTATIVE!r} or {TRACK_CONFIRMED!r}")
        _finite("time_since_update_s", self.time_since_update_s, minimum=0.0)
        _vector3("position_std_m", self.position_std_m)
        _vector3("velocity_std_m_s", self.velocity_std_m_s)

    @property
    def object_id(self) -> str:
        return self.state.object_id

    @property
    def simulation_time_s(self) -> float:
        return self.state.simulation_time_s

    @property
    def position_m(self) -> Vector3:
        return self.state.position_m

    @property
    def velocity_m_s(self) -> Vector3:
        return self.state.velocity_m_s

    @property
    def is_confirmed(self) -> bool:
        return self.status == TRACK_CONFIRMED

    def to_object_geometry(self) -> ObjectGeometry:
        """Build a :class:`conveyor.grasping.geometry.ObjectGeometry` from this track."""
        dims = (
            (self.dimensions_m[0], self.dimensions_m[0], self.dimensions_m[2])
            if self.geometry_type == "cylinder"
            else self.dimensions_m
        )
        return ObjectGeometry(
            object_id=self.object_id,
            geometry_type=self.geometry_type,
            dimensions_m=dims,
            yaw_rad=0.0 if self.geometry_type == "cylinder" else self.yaw_rad,
        )

    def to_schedulable_object(self) -> SchedulableObject:
        """Build a :class:`conveyor.scheduling.models.SchedulableObject` from this track."""
        return SchedulableObject(state=self.state, geometry=self.to_object_geometry())
