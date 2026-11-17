"""Destination-bin geometry and reference points, without reachability claims."""

from dataclasses import dataclass
from math import isfinite
from numbers import Real
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    import robotic as ry


@dataclass(frozen=True)
class TargetBin:
    """World geometry for a nominal destination; placement still needs planning.

    position_m is the centre of the interior floor. place/approach positions
    are reference points, not gripper poses or object-independent release poses.
    """

    target_id: str
    preferred_robot_id: str
    simulation_time_s: float
    bin_frame: str
    position_m: tuple[float, float, float]
    quaternion_wxyz: tuple[float, float, float, float]
    inner_dimensions_m: tuple[float, float, float]
    place_frame: str
    place_position_m: tuple[float, float, float]
    approach_frame: str
    approach_position_m: tuple[float, float, float]
    frame: ClassVar[str] = "world"


def _finite_tuple(name: str, values, count: int) -> tuple[float, ...]:
    result = tuple(values)
    if len(result) != count or any(
        isinstance(x, bool) or not isinstance(x, Real) or not isfinite(x) for x in result
    ):
        raise ValueError(f"{name} must contain {count} finite real numbers")
    return tuple(float(x) for x in result)


def get_target_snapshots(config: "ry.Config", simulation_time_s: float) -> dict[str, TargetBin]:
    """Read the authored bins and their parent-transformed world references."""
    if (
        isinstance(simulation_time_s, bool) or not isinstance(simulation_time_s, Real)
        or not isfinite(simulation_time_s) or simulation_time_s < 0
    ):
        raise ValueError("simulation_time_s must be finite and non-negative")
    names = ("target_bin_a", "target_bin_b")
    required = {f"{name}{suffix}" for name in names for suffix in ("", "_place", "_approach")}
    missing = required - set(config.getFrameNames())
    if missing:
        raise ValueError(f"Target snapshots require missing frames: {', '.join(sorted(missing))}")
    result = {}
    for name in names:
        root = config.getFrame(name)
        attrs = root.getAttributes()
        dimensions = _finite_tuple(f"{name}.inner_size_m", attrs.get("inner_size_m", ()), 3)
        if any(x <= 0 for x in dimensions):
            raise ValueError(f"{name}.inner_size_m must be positive")
        preferred = attrs.get("preferred_robot_id")
        if preferred not in ("robot_a", "robot_b"):
            raise ValueError(f"{name}.preferred_robot_id must name robot_a or robot_b")
        place_name, approach_name = f"{name}_place", f"{name}_approach"
        result[name] = TargetBin(
            target_id=name, preferred_robot_id=preferred, simulation_time_s=float(simulation_time_s),
            bin_frame=name, position_m=_finite_tuple(name, root.getPosition(), 3),
            quaternion_wxyz=_finite_tuple(name, root.getQuaternion(), 4),
            inner_dimensions_m=dimensions,
            place_frame=place_name,
            place_position_m=_finite_tuple(place_name, config.getFrame(place_name).getPosition(), 3),
            approach_frame=approach_name,
            approach_position_m=_finite_tuple(approach_name, config.getFrame(approach_name).getPosition(), 3),
        )
    return result
