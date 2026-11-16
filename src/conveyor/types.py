"""Shared world-frame interfaces, independent of RAI and OpenCV.

Positions use metres, linear velocities metres/second, and timestamps seconds
since simulation start. Future angular quantities must use radians.
"""

from dataclasses import dataclass
from math import isfinite
from numbers import Real
from typing import ClassVar

Vector3 = tuple[float, float, float]


def _finite_number(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
        raise ValueError(f"{name} must be a finite real number")


def _vector3(name: str, value: Vector3) -> None:
    if not isinstance(value, tuple) or len(value) != 3:
        raise ValueError(f"{name} must be a tuple of three finite real numbers")
    for component in value:
        _finite_number(name, component)


def _identity_and_time(object_id: str, simulation_time_s: float) -> None:
    if not isinstance(object_id, str) or not object_id.strip():
        raise ValueError("object_id must be a non-empty string")
    _finite_number("simulation_time_s", simulation_time_s)
    if simulation_time_s < 0:
        raise ValueError("simulation_time_s must be non-negative")


@dataclass(frozen=True)
class ObjectObservation:
    """Measured object centroid at acquisition time, already in the world frame.

    object_id is a detection identifier; tracking owns cross-frame association.
    """

    object_id: str
    simulation_time_s: float
    position_m: Vector3
    frame: ClassVar[str] = "world"

    def __post_init__(self) -> None:
        _identity_and_time(self.object_id, self.simulation_time_s)
        _vector3("position_m", self.position_m)


@dataclass(frozen=True)
class TrackedState:
    """Estimated centroid and velocity at the filter's simulation timestamp.

    object_id is a stable track identifier assigned by tracking.
    """

    object_id: str
    simulation_time_s: float
    position_m: Vector3
    velocity_m_s: Vector3
    frame: ClassVar[str] = "world"

    def __post_init__(self) -> None:
        _identity_and_time(self.object_id, self.simulation_time_s)
        _vector3("position_m", self.position_m)
        _vector3("velocity_m_s", self.velocity_m_s)
