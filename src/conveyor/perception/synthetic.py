"""Synthetic overhead RGB-D renderer for Mac/headless testing without RAI.

Produces :class:`conveyor.simulation.camera.CameraObservation`-compatible
snapshots from :class:`conveyor.simulation.scenarios.ObjectSpec` records (or
any duck-typed object with ``position_m``, ``dimensions_m``, ``geometry_type``,
``yaw_rad`` and ``color_rgb``). It uses the exact camera intrinsics and
transform of the dual-arm RAI scene so perception and tracking can be tested
offline on macOS without an OpenGL/RAI binary.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import cos, isfinite, sin
from typing import Any

from conveyor.simulation.camera import CameraCalibration, CameraObservation


DEFAULT_INTRINSICS = (
    (288.0, 0.0, 320.0),
    (0.0, 288.0, 180.0),
    (0.0, 0.0, 1.0),
)
DEFAULT_CAMERA_TO_WORLD = (
    (1.0, 0.0, 0.0, 0.0),
    (0.0, -1.0, 0.0, 0.0),
    (0.0, 0.0, -1.0, 2.5),
    (0.0, 0.0, 0.0, 1.0),
)


def default_overhead_calibration(width: int = 640, height: int = 360) -> CameraCalibration:
    """Return the calibrated 640x360 overhead camera matching ``RGBDCamera``."""
    return CameraCalibration(
        width=width,
        height=height,
        intrinsics=DEFAULT_INTRINSICS,
        camera_to_world=DEFAULT_CAMERA_TO_WORLD,
    )


def advance_specs_along_belt(
    specs: tuple[Any, ...] | list[Any],
    belt_speed_m_s: float,
    simulation_time_s: float,
    *,
    belt_exit_x_m: float = 1.20,
) -> list[Any]:
    """Translate initial ``ObjectSpec`` records along +x at ``belt_speed_m_s``."""
    active: list[Any] = []
    for spec in specs:
        x0, y0, z0 = spec.position_m
        x = float(x0) + float(belt_speed_m_s) * float(simulation_time_s)
        if x <= belt_exit_x_m:
            active.append(replace(spec, position_m=(x, float(y0), float(z0))))
    return active


def render_synthetic_rgbd(
    objects: tuple[Any, ...] | list[Any],
    simulation_time_s: float = 0.0,
    *,
    calibration: CameraCalibration | None = None,
    belt_top_z_m: float = 0.75,
    belt_length_m: float = 2.4,
    belt_width_m: float = 0.4,
    floor_z_m: float = 0.0,
    depth_noise_std_m: float = 0.0,
    rgb_noise_std: float = 0.0,
    rng_seed: int | None = None,
) -> CameraObservation:
    """Render an overhead RGB-D observation of conveyor objects in pure NumPy."""
    import numpy as np

    if not isfinite(simulation_time_s) or simulation_time_s < 0.0:
        raise ValueError("simulation_time_s must be finite and non-negative")

    cal = calibration or default_overhead_calibration()
    width, height = cal.width, cal.height
    fx, fy = cal.intrinsics[0][0], cal.intrinsics[1][1]
    cx, cy = cal.intrinsics[0][2], cal.intrinsics[1][2]
    cam_z = cal.camera_to_world[2][3]

    # Floor background: color [.66, .68, .70], depth = cam_z - floor_z_m = 2.50 m
    rgb = np.empty((height, width, 3), dtype=np.uint8)
    rgb[:, :, 0] = int(round(0.66 * 255))
    rgb[:, :, 1] = int(round(0.68 * 255))
    rgb[:, :, 2] = int(round(0.70 * 255))
    depth_m = np.full((height, width), float(cam_z - floor_z_m), dtype=np.float32)

    # Pixel grid coordinates
    u_grid = np.arange(width, dtype=np.float64)[None, :]   # (1, W)
    v_grid = np.arange(height, dtype=np.float64)[:, None]  # (H, 1)

    # Conveyor deck at z = belt_top_z_m (depth = 1.75 m), color [.10, .14, .17]
    belt_depth = float(cam_z - belt_top_z_m)
    belt_wx = (u_grid - cx) * belt_depth / fx
    belt_wy = -(v_grid - cy) * belt_depth / fy
    on_belt = (np.abs(belt_wx) <= 0.5 * belt_length_m) & (np.abs(belt_wy) <= 0.5 * belt_width_m)
    depth_m[on_belt] = belt_depth
    rgb[on_belt] = (int(round(0.10 * 255)), int(round(0.14 * 255)), int(round(0.17 * 255)))

    # Sort objects from lowest top surface to highest so taller objects occlude shorter ones.
    sorted_objects = sorted(objects, key=lambda o: float(o.position_m[2]) + 0.5 * float(o.dimensions_m[2]))

    for obj in sorted_objects:
        ox, oy, oz = (float(v) for v in obj.position_m)
        dx, dy, dz = (float(v) for v in obj.dimensions_m)
        z_top = oz + 0.5 * dz
        obj_depth = float(cam_z - z_top)
        if obj_depth <= 0.1:
            continue

        # At the object's top plane z = z_top, each pixel (u, v) corresponds to world (wx, wy):
        wx = (u_grid - cx) * obj_depth / fx
        wy = -(v_grid - cy) * obj_depth / fy
        rel_x = wx - ox
        rel_y = wy - oy

        geom = getattr(obj, "geometry_type", "box")
        if geom == "cylinder":
            radius = 0.5 * dx
            hit = (rel_x * rel_x + rel_y * rel_y) <= (radius * radius)
        else:
            yaw = float(getattr(obj, "yaw_rad", 0.0))
            c, s = cos(yaw), sin(yaw)
            local_x = c * rel_x + s * rel_y
            local_y = -s * rel_x + c * rel_y
            hit = (np.abs(local_x) <= 0.5 * dx) & (np.abs(local_y) <= 0.5 * dy)

        closer = hit & (obj_depth < depth_m)
        if np.any(closer):
            depth_m[closer] = obj_depth
            color = getattr(obj, "color_rgb", (0.9, 0.32, 0.12))
            rgb[closer] = tuple(int(round(float(c) * 255)) for c in color)

    if depth_noise_std_m > 0.0 or rgb_noise_std > 0.0:
        rng = np.random.default_rng(rng_seed)
        if depth_noise_std_m > 0.0:
            depth_m = (depth_m + rng.normal(0.0, depth_noise_std_m, size=depth_m.shape)).astype(np.float32)
        if rgb_noise_std > 0.0:
            noisy_rgb = rgb.astype(np.float32) + rng.normal(0.0, rgb_noise_std, size=rgb.shape)
            rgb = np.clip(np.round(noisy_rgb), 0, 255).astype(np.uint8)

    rgb.setflags(write=False)
    depth_m.setflags(write=False)
    return CameraObservation(
        simulation_time_s=float(simulation_time_s),
        rgb=rgb,
        depth_m=depth_m,
        calibration=cal,
    )
