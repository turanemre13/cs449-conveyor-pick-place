"""3-D constant-velocity Kalman filter for a single conveyor object track."""

from __future__ import annotations

from math import atan2, cos, isfinite, pi, sin, sqrt
from typing import Any

from conveyor.grasping.geometry import wrap_half_turn
from conveyor.types import TrackedState, Vector3

from .models import TRACK_CONFIRMED, TRACK_TENTATIVE, TrackedObject, TrackerConfig


def _blend_half_turn_angles(old_angle: float, new_angle: float, alpha: float) -> float:
    """Smooth two pi-periodic orientation angles (parallel-jaw box symmetry)."""
    delta = wrap_half_turn(new_angle - old_angle)
    return wrap_half_turn(old_angle + alpha * delta)


class ConstantVelocityKalmanFilter3D:
    """Linear Kalman filter with state ``[x, y, z, vx, vy, vz]^T`` and measurement ``[x, y, z]^T``.

    Because x, y and z decouple under isotropic process/measurement models,
    the 6x6 matrix updates are maintained explicitly in NumPy with Joseph-form
    covariance stabilization so numerical symmetry and positive-definiteness
    are guaranteed across arbitrarily long simulation runs.
    """

    def __init__(
        self,
        track_id: str,
        initial_time_s: float,
        initial_position_m: Vector3,
        config: TrackerConfig | None = None,
        *,
        geometry_type: str = "box",
        dimensions_m: Vector3 = (0.06, 0.06, 0.06),
        yaw_rad: float = 0.0,
        color_rgb: Vector3 = (0.5, 0.5, 0.5),
    ) -> None:
        import numpy as np

        if not isinstance(track_id, str) or not track_id.strip():
            raise ValueError("track_id must be a non-empty string")
        if not isfinite(initial_time_s) or initial_time_s < 0.0:
            raise ValueError("initial_time_s must be finite and non-negative")

        self.config = config or TrackerConfig()
        self.track_id = track_id
        self.last_timestamp_s = float(initial_time_s)
        self.last_update_time_s = float(initial_time_s)

        px, py, pz = (float(v) for v in initial_position_m)
        vx0, vy0, vz0 = (float(v) for v in self.config.initial_belt_velocity_m_s)
        if self.config.planar_belt_constraint:
            vz0 = 0.0
        self.x = np.array([px, py, pz, vx0, vy0, vz0], dtype=np.float64)

        p_var = self.config.initial_position_std_m ** 2
        v_var = self.config.initial_velocity_std_m_s ** 2
        self.P = np.diag([p_var, p_var, p_var, v_var, v_var, v_var]).astype(np.float64)

        self.H = np.zeros((3, 6), dtype=np.float64)
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.H[2, 2] = 1.0

        r_var = self.config.position_measurement_std_m ** 2
        self.R = np.diag([r_var, r_var, r_var]).astype(np.float64)

        # Geometry and lifecycle state
        self.dimensions_m: Vector3 = tuple(float(d) for d in dimensions_m)  # type: ignore[assignment]
        self.yaw_rad: float = 0.0 if geometry_type == "cylinder" else wrap_half_turn(float(yaw_rad))
        self.color_rgb: Vector3 = tuple(float(c) for c in color_rgb)  # type: ignore[assignment]
        self._shape_votes: dict[str, int] = {"box": 0, "cylinder": 0}
        self._shape_votes[geometry_type if geometry_type in self._shape_votes else "box"] += 1
        self.hits: int = 1
        self.age_frames: int = 1
        self.missed_frames: int = 0

    @property
    def geometry_type(self) -> str:
        if self._shape_votes["cylinder"] > self._shape_votes["box"]:
            return "cylinder"
        return "box"

    @property
    def status(self) -> str:
        return TRACK_CONFIRMED if self.hits >= self.config.min_hits_to_confirm else TRACK_TENTATIVE

    @property
    def position_m(self) -> Vector3:
        return (float(self.x[0]), float(self.x[1]), float(self.x[2]))

    @property
    def velocity_m_s(self) -> Vector3:
        vz = 0.0 if self.config.planar_belt_constraint else float(self.x[5])
        return (float(self.x[3]), float(self.x[4]), vz)

    @property
    def position_std_m(self) -> Vector3:
        return (
            sqrt(max(0.0, float(self.P[0, 0]))),
            sqrt(max(0.0, float(self.P[1, 1]))),
            sqrt(max(0.0, float(self.P[2, 2]))),
        )

    @property
    def velocity_std_m_s(self) -> Vector3:
        return (
            sqrt(max(0.0, float(self.P[3, 3]))),
            sqrt(max(0.0, float(self.P[4, 4]))),
            sqrt(max(0.0, float(self.P[5, 5]))),
        )

    def predict_to(self, target_time_s: float) -> Vector3:
        """Advance state and covariance to ``target_time_s >= last_timestamp_s``."""
        import numpy as np

        if not isfinite(target_time_s) or target_time_s < 0.0:
            raise ValueError("target_time_s must be finite and non-negative")
        dt = float(target_time_s) - self.last_timestamp_s
        if dt < -1e-9:
            raise ValueError(
                f"target_time_s={target_time_s} is earlier than track timestamp {self.last_timestamp_s}"
            )
        if dt <= 0.0:
            return self.position_m

        F = np.eye(6, dtype=np.float64)
        F[0, 3] = dt
        F[1, 4] = dt
        F[2, 5] = dt

        q = self.config.acceleration_process_std_m_s2 ** 2
        dt2 = dt * dt
        dt3 = 0.5 * dt2 * dt
        dt4 = 0.25 * dt2 * dt2
        Q = np.zeros((6, 6), dtype=np.float64)
        for axis in range(3):
            scale = 0.05 if (axis == 2 and self.config.planar_belt_constraint) else 1.0
            Q[axis, axis] = q * scale * dt4
            Q[axis, axis + 3] = q * scale * dt3
            Q[axis + 3, axis] = q * scale * dt3
            Q[axis + 3, axis + 3] = q * scale * dt2

        self.x = F @ self.x
        if self.config.planar_belt_constraint:
            self.x[5] = 0.0
        self.P = F @ self.P @ F.T + Q
        self.P = 0.5 * (self.P + self.P.T)
        self.last_timestamp_s = float(target_time_s)
        return self.position_m

    def update(self, measurement_position_m: Vector3, measurement_time_s: float, detection: Any | None = None) -> None:
        """Apply a 3-D world position measurement at ``measurement_time_s``."""
        import numpy as np

        if measurement_time_s > self.last_timestamp_s:
            self.predict_to(measurement_time_s)

        z = np.array([float(v) for v in measurement_position_m], dtype=np.float64)
        y = z - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.x = self.x + (K @ y)
        if self.config.planar_belt_constraint:
            self.x[5] = 0.0

        I = np.eye(6, dtype=np.float64)
        I_KH = I - (K @ self.H)
        # Joseph form covariance update
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T
        self.P = 0.5 * (self.P + self.P.T)

        self.last_update_time_s = float(self.last_timestamp_s)
        self.hits += 1
        self.age_frames += 1
        self.missed_frames = 0

        if detection is not None:
            self._update_geometry(detection)

    def mark_missed(self) -> None:
        """Record that this track received no measurement in the current frame."""
        self.age_frames += 1
        self.missed_frames += 1

    def _update_geometry(self, detection: Any) -> None:
        alpha = self.config.geometry_smoothing_alpha
        geom = getattr(detection, "geometry_type", None)
        if geom in self._shape_votes:
            self._shape_votes[geom] += 1

        det_dims = getattr(detection, "dimensions_m", None)
        if det_dims is not None and len(det_dims) == 3:
            old_dx, old_dy, old_dz = self.dimensions_m
            new_dx, new_dy, new_dz = (float(v) for v in det_dims)
            if self.geometry_type == "cylinder":
                diam = (1.0 - alpha) * 0.5 * (old_dx + old_dy) + alpha * 0.5 * (new_dx + new_dy)
                self.dimensions_m = (diam, diam, (1.0 - alpha) * old_dz + alpha * new_dz)
                self.yaw_rad = 0.0
            else:
                dx = (1.0 - alpha) * old_dx + alpha * new_dx
                dy = (1.0 - alpha) * old_dy + alpha * new_dy
                dz = (1.0 - alpha) * old_dz + alpha * new_dz
                self.dimensions_m = (dx, dy, dz)
                det_yaw = getattr(detection, "yaw_rad", None)
                if det_yaw is not None:
                    self.yaw_rad = _blend_half_turn_angles(self.yaw_rad, float(det_yaw), alpha)

        det_color = getattr(detection, "color_rgb", None)
        if det_color is not None and len(det_color) == 3:
            self.color_rgb = tuple(
                (1.0 - alpha) * o + alpha * float(n) for o, n in zip(self.color_rgb, det_color)
            )  # type: ignore[assignment]

    def to_tracked_state(self) -> TrackedState:
        """Build the shared :class:`conveyor.types.TrackedState`."""
        return TrackedState(
            object_id=self.track_id,
            simulation_time_s=self.last_timestamp_s,
            position_m=self.position_m,
            velocity_m_s=self.velocity_m_s,
        )

    def to_tracked_object(self) -> TrackedObject:
        """Build the rich :class:`TrackedObject` snapshot."""
        dims = (
            (self.dimensions_m[0], self.dimensions_m[0], self.dimensions_m[2])
            if self.geometry_type == "cylinder"
            else self.dimensions_m
        )
        return TrackedObject(
            state=self.to_tracked_state(),
            geometry_type=self.geometry_type,
            dimensions_m=dims,
            yaw_rad=0.0 if self.geometry_type == "cylinder" else self.yaw_rad,
            color_rgb=self.color_rgb,
            hits=self.hits,
            age_frames=self.age_frames,
            time_since_update_s=max(0.0, self.last_timestamp_s - self.last_update_time_s),
            missed_frames=self.missed_frames,
            status=self.status,
            position_std_m=self.position_std_m,
            velocity_std_m_s=self.velocity_std_m_s,
        )
