"""Constant-velocity motion prediction for tracked conveyor objects.

Model (world frame, metres, seconds since simulation start)::

    p_hat(t + tau) = p_hat(t) + v_hat(t) * tau,   tau = t_target - t >= 0

The predictor consumes the shared :class:`conveyor.types.TrackedState` record
(from Heysem's tracker, or from simulation ground truth through
:mod:`conveyor.prediction.ground_truth`). It is pure Python, independent of RAI,
the viewer and the simulation package.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from math import isfinite, sqrt
from numbers import Real
from typing import ClassVar

from conveyor.types import TrackedState, Vector3


def _check_finite(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
        raise ValueError(f"{name} must be a finite real number")
    return float(value)


@dataclass(frozen=True)
class PredictedState:
    """Predicted world-frame object centroid at ``prediction_time_s``.

    ``reference_time_s`` is the timestamp of the tracked state used as input;
    ``horizon_s`` is their difference. ``is_stationary`` reports that the input
    speed was below the predictor's stationary threshold and was treated as zero.
    """

    object_id: str
    reference_time_s: float
    prediction_time_s: float
    horizon_s: float
    position_m: Vector3
    velocity_m_s: Vector3
    is_stationary: bool
    frame: ClassVar[str] = "world"


class ConstantVelocityPredictor:
    """Predict future positions of tracked objects with a constant-velocity model.

    Parameters
    ----------
    max_horizon_s:
        Largest allowed look-ahead ``tau``. Long horizons amplify velocity
        errors linearly, so requests beyond it raise ``ValueError``.
    stationary_speed_m_s:
        Speeds at or below this threshold are treated as zero, so tracker noise
        on a stopped belt does not make a stationary object drift.
    planar_motion:
        If true, the vertical velocity component is ignored (objects carried
        on a horizontal belt keep their height). Default false keeps the plain
        3-D constant-velocity equation.
    past_tolerance_s:
        Target times earlier than the state timestamp by at most this amount
        are clamped to the state time (handles floating-point jitter). Larger
        backward requests are rejected: this module predicts, it does not smooth.
    """

    def __init__(
        self,
        *,
        max_horizon_s: float = 30.0,
        stationary_speed_m_s: float = 1e-3,
        planar_motion: bool = False,
        past_tolerance_s: float = 1e-9,
    ) -> None:
        self.max_horizon_s = _check_finite("max_horizon_s", max_horizon_s)
        self.stationary_speed_m_s = _check_finite("stationary_speed_m_s", stationary_speed_m_s)
        self.past_tolerance_s = _check_finite("past_tolerance_s", past_tolerance_s)
        if self.max_horizon_s <= 0:
            raise ValueError("max_horizon_s must be positive")
        if self.stationary_speed_m_s < 0 or self.past_tolerance_s < 0:
            raise ValueError("stationary_speed_m_s and past_tolerance_s must be non-negative")
        self.planar_motion = bool(planar_motion)

    # ------------------------------------------------------------------ helpers
    def effective_velocity(self, state: TrackedState) -> tuple[Vector3, bool]:
        """Velocity actually used for prediction and whether it was zeroed."""
        _require_state(state)
        vx, vy, vz = state.velocity_m_s
        if self.planar_motion:
            vz = 0.0
        speed = sqrt(vx * vx + vy * vy + vz * vz)
        if speed <= self.stationary_speed_m_s:
            return (0.0, 0.0, 0.0), True
        return (float(vx), float(vy), float(vz)), False

    def _horizon(self, state: TrackedState, target_time_s: float) -> float:
        target = _check_finite("target_time_s", target_time_s)
        if target < 0:
            raise ValueError("target_time_s must be non-negative simulation time")
        tau = target - state.simulation_time_s
        if tau < 0:
            if -tau <= self.past_tolerance_s:
                return 0.0
            raise ValueError(
                f"target_time_s={target} is before the state time "
                f"{state.simulation_time_s} of {state.object_id!r}; backward prediction is not supported"
            )
        if tau > self.max_horizon_s:
            raise ValueError(f"prediction horizon {tau:.3f} s exceeds max_horizon_s={self.max_horizon_s}")
        return tau

    # ------------------------------------------------------------------ public API
    def predict(self, state: TrackedState, target_time_s: float) -> PredictedState:
        """Predict one object's centroid at absolute simulation time ``target_time_s``."""
        _require_state(state)
        tau = self._horizon(state, target_time_s)
        velocity, stationary = self.effective_velocity(state)
        px, py, pz = state.position_m
        position = (px + velocity[0] * tau, py + velocity[1] * tau, pz + velocity[2] * tau)
        return PredictedState(
            object_id=state.object_id,
            reference_time_s=float(state.simulation_time_s),
            prediction_time_s=float(state.simulation_time_s + tau),
            horizon_s=tau,
            position_m=tuple(float(x) for x in position),
            velocity_m_s=velocity,
            is_stationary=stationary,
        )

    def predict_many(
        self, states: Iterable[TrackedState] | Mapping[str, TrackedState], target_time_s: float,
    ) -> dict[str, PredictedState]:
        """Predict several objects at one common absolute time, keyed by object ID.

        Duplicate object IDs are rejected because they would be silently overwritten.
        Each state may carry its own timestamp (asynchronous tracks are allowed).
        """
        items = states.values() if isinstance(states, Mapping) else states
        result: dict[str, PredictedState] = {}
        for state in items:
            _require_state(state)
            if state.object_id in result:
                raise ValueError(f"duplicate object_id {state.object_id!r}")
            result[state.object_id] = self.predict(state, target_time_s)
        return result

    def predict_trajectory(self, state: TrackedState, times_s: Iterable[float]) -> list[PredictedState]:
        """Predict one object at several absolute times (e.g. candidate interceptions)."""
        return [self.predict(state, t) for t in times_s]

    def time_to_reach_coordinate(
        self, state: TrackedState, axis: int, value: float,
    ) -> float | None:
        """Absolute time at which the centroid's ``axis`` coordinate reaches ``value``.

        Returns ``None`` when the velocity along ``axis`` is zero (the object
        never gets there), the state time when the coordinate is already
        reached or lies behind the object's direction of motion, otherwise
        ``t + (value - p) / v``. Ignores ``max_horizon_s`` so that it can be used
        for deadlines such as the belt exit.
        """
        _require_state(state)
        if axis not in (0, 1, 2):
            raise ValueError("axis must be 0, 1 or 2")
        target = _check_finite("value", value)
        velocity, stationary = self.effective_velocity(state)
        position = state.position_m[axis]
        speed = velocity[axis]
        if stationary or speed == 0.0:
            return None
        delta = target - position
        if delta * speed <= 0:
            # Already at, or already past, the coordinate along the motion direction.
            return float(state.simulation_time_s)
        return float(state.simulation_time_s + delta / speed)


def _require_state(state: object) -> None:
    if not isinstance(state, TrackedState):
        raise TypeError(f"expected conveyor.types.TrackedState, got {type(state).__name__}")
