"""Seeded synthetic arrival streams for scheduling experiments.

Objects are loaded onto the belt one after another with random centre-to-centre
spacing, so a faster belt delivers more objects per second. Each object starts
upstream of the belt (x < belt start) at time 0 and becomes visible when it
enters. Dimensions follow the ranges of ``conveyor.simulation.scenarios``; an
optional fraction of oversized boxes exceeds the gripper opening on purpose.
The generator uses a local ``random.Random`` and never touches global state.
"""

from __future__ import annotations

import random
from math import pi

from conveyor.grasping.geometry import ObjectGeometry
from conveyor.types import TrackedState

from .models import SchedulableObject

BELT_START_X_M = -1.2
BELT_TOP_Z_M = 0.75


def generate_arrival_stream(
    *,
    seed: int,
    count: int,
    belt_speed_m_s: float,
    spacing_range_m: tuple[float, float] = (0.14, 0.40),
    lateral_range_m: float = 0.12,
    oversized_fraction: float = 0.0,
) -> list[SchedulableObject]:
    """``count`` objects at time 0, the first one entering the belt immediately."""
    if count < 1:
        raise ValueError("count must be positive")
    if belt_speed_m_s <= 0:
        raise ValueError("arrival streams need a moving belt")
    rng = random.Random(seed)
    result = []
    x = BELT_START_X_M
    for index in range(count):
        object_id = f"stream_{index + 1:03d}"
        if rng.random() < oversized_fraction:
            geometry_type, dims = "box", (rng.uniform(0.09, 0.12), rng.uniform(0.09, 0.12), rng.uniform(0.04, 0.09))
        elif index % 2 == 0:
            geometry_type, dims = "box", (rng.uniform(.045, .10), rng.uniform(.035, .065), rng.uniform(.04, .09))
        else:
            diameter = rng.uniform(.04, .065)
            geometry_type, dims = "cylinder", (diameter, diameter, rng.uniform(.04, .09))
        yaw = rng.uniform(-pi, pi) if geometry_type == "box" else 0.0
        y = rng.uniform(-lateral_range_m, lateral_range_m)
        state = TrackedState(object_id, 0.0, (x, y, BELT_TOP_Z_M + dims[2] / 2), (belt_speed_m_s, 0.0, 0.0))
        result.append(SchedulableObject(state, ObjectGeometry(object_id, geometry_type, dims, yaw)))
        x -= rng.uniform(*spacing_range_m)
    return result
