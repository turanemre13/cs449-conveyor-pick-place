"""Advance predicted belt geometry in RAI planning copies."""

from typing import Callable

from conveyor.types import TrackedState, Vector3


def belt_position(track: TrackedState, simulation_time_s: float) -> Vector3:
    elapsed = simulation_time_s - track.simulation_time_s
    return tuple(p + v * elapsed for p, v in zip(track.position_m, track.velocity_m_s))


def position_belt_frames(get_frame: Callable, tracks: tuple[TrackedState, ...], simulation_time_s: float) -> None:
    """Set translating frames at a predicted instant. Shapes persist in the preview after a live exit."""
    for track in tracks:
        get_frame(track.object_id).setPosition(belt_position(track, simulation_time_s))


def position_belt_slices(komo, tracks: tuple[TrackedState, ...], start_s: float, duration_s: float, slices: int) -> None:
    if not tracks:
        return
    for index in range(1, slices + 1):
        phase = index / slices
        position_belt_frames(lambda name: komo.getFrame(name, phase), tracks, start_s + duration_s * phase)
