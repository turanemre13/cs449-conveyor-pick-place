"""Import and boundary-validation smoke test; no robotics backend required."""

from importlib import import_module

import pytest

from conveyor.types import ObjectObservation, TrackedState


def test_package_imports_and_interface_validation():
    for module in (
        "simulation", "perception", "tracking", "prediction", "grasping",
        "planning", "control", "pipeline",
    ):
        import_module(f"conveyor.{module}")

    observation = ObjectObservation("detection-1", 0.0, (0.4, -0.1, 0.1))
    state = TrackedState("track-1", 0.0, observation.position_m, (0.1, 0.0, 0.0))
    assert observation.frame == state.frame == "world"
    assert state.position_m == observation.position_m
    assert state.velocity_m_s == (0.1, 0.0, 0.0)

    for invalid in (
        {"object_id": " "},
        {"simulation_time_s": -1.0},
        {"simulation_time_s": float("inf")},
        {"simulation_time_s": True},
        {"position_m": (0.0, 0.0)},
        {"position_m": (0.0, float("nan"), 0.0)},
        {"position_m": [0.0, 0.0, 0.0]},
    ):
        fields = dict(object_id="detection-1", simulation_time_s=0.0,
                      position_m=(0.0, 0.0, 0.0))
        with pytest.raises(ValueError):
            ObjectObservation(**(fields | invalid))

    with pytest.raises(ValueError):
        TrackedState("track-1", 0.0, (0.0, 0.0, 0.0), (float("inf"), 0.0, 0.0))
