"""Constant-velocity prediction; pure Python, no RAI required."""

from dataclasses import dataclass

import pytest

from conveyor.prediction import (
    ConstantVelocityPredictor, tracked_state_from_ground_truth, tracked_states_from_ground_truth,
)
from conveyor.types import TrackedState


def state(object_id="obj", t=1.0, p=(0.0, 0.1, 0.79), v=(0.12, 0.0, 0.0)):
    return TrackedState(object_id, t, p, v)


def test_single_object_follows_the_constant_velocity_equation():
    predictor = ConstantVelocityPredictor()
    result = predictor.predict(state(), 3.5)
    assert result.horizon_s == pytest.approx(2.5)
    assert result.position_m == pytest.approx((0.3, 0.1, 0.79))
    assert result.prediction_time_s == pytest.approx(3.5)
    assert result.reference_time_s == 1.0
    assert result.velocity_m_s == (0.12, 0.0, 0.0)
    assert not result.is_stationary
    assert result.frame == "world"


def test_zero_horizon_returns_the_input_position():
    result = ConstantVelocityPredictor().predict(state(), 1.0)
    assert result.position_m == (0.0, 0.1, 0.79)
    assert result.horizon_s == 0.0


@pytest.mark.parametrize("speed", [0.0, 0.05, 0.12, 0.25, 0.5])
def test_different_conveyor_speeds(speed):
    result = ConstantVelocityPredictor().predict(state(v=(speed, 0.0, 0.0), t=0.0, p=(-1.0, 0.0, 0.8)), 4.0)
    assert result.position_m[0] == pytest.approx(-1.0 + 4.0 * speed)
    assert result.position_m[1:] == (0.0, 0.8)


def test_stationary_objects_and_noise_threshold():
    predictor = ConstantVelocityPredictor(stationary_speed_m_s=1e-3)
    still = predictor.predict(state(v=(0.0, 0.0, 0.0)), 10.0)
    assert still.position_m == (0.0, 0.1, 0.79) and still.is_stationary
    noisy = predictor.predict(state(v=(5e-4, -3e-4, 0.0)), 10.0)
    assert noisy.position_m == (0.0, 0.1, 0.79) and noisy.is_stationary
    assert noisy.velocity_m_s == (0.0, 0.0, 0.0)


def test_planar_motion_ignores_vertical_velocity():
    s = state(v=(0.1, 0.0, 0.02))
    assert ConstantVelocityPredictor().predict(s, 2.0).position_m[2] == pytest.approx(0.81)
    assert ConstantVelocityPredictor(planar_motion=True).predict(s, 2.0).position_m[2] == pytest.approx(0.79)


def test_multiple_objects_with_asynchronous_timestamps():
    predictor = ConstantVelocityPredictor()
    states = [state("a", 1.0, (0.0, 0.0, 0.8)), state("b", 1.5, (0.2, -0.1, 0.8), (0.0, 0.1, 0.0))]
    result = predictor.predict_many(states, 2.0)
    assert set(result) == {"a", "b"}
    assert result["a"].position_m == pytest.approx((0.12, 0.0, 0.8))
    assert result["b"].position_m == pytest.approx((0.2, -0.05, 0.8))
    assert predictor.predict_many({s.object_id: s for s in states}, 2.0) == result
    with pytest.raises(ValueError, match="duplicate"):
        predictor.predict_many([states[0], states[0]], 2.0)


def test_trajectory_prediction():
    predictions = ConstantVelocityPredictor().predict_trajectory(state(t=0.0, p=(0, 0, 0)), [0.0, 1.0, 2.0])
    assert [p.position_m[0] for p in predictions] == pytest.approx([0.0, 0.12, 0.24])


def test_invalid_timestamps_and_inputs_are_rejected():
    predictor = ConstantVelocityPredictor(max_horizon_s=5.0)
    with pytest.raises(ValueError, match="before the state time"):
        predictor.predict(state(t=2.0), 1.0)
    for bad in (float("nan"), float("inf"), -1.0, True, "3"):
        with pytest.raises(ValueError):
            predictor.predict(state(), bad)
    with pytest.raises(ValueError, match="max_horizon_s"):
        predictor.predict(state(t=0.0), 5.1)
    with pytest.raises(TypeError):
        predictor.predict((0.0, 0.0, 0.0), 1.0)
    for kwargs in ({"max_horizon_s": 0.0}, {"stationary_speed_m_s": -1.0}, {"max_horizon_s": float("nan")}):
        with pytest.raises(ValueError):
            ConstantVelocityPredictor(**kwargs)


def test_tiny_backward_jitter_is_clamped():
    predictor = ConstantVelocityPredictor(past_tolerance_s=1e-6)
    assert predictor.predict(state(t=1.0), 1.0 - 1e-7).position_m == (0.0, 0.1, 0.79)


def test_time_to_reach_coordinate_for_belt_exit_deadlines():
    predictor = ConstantVelocityPredictor()
    s = state(t=2.0, p=(-1.0, 0.0, 0.8), v=(0.2, 0.0, 0.0))
    assert predictor.time_to_reach_coordinate(s, 0, 1.2) == pytest.approx(13.0)
    assert predictor.time_to_reach_coordinate(s, 0, -1.5) == 2.0  # already passed
    assert predictor.time_to_reach_coordinate(state(v=(0, 0, 0)), 0, 1.2) is None
    with pytest.raises(ValueError):
        predictor.time_to_reach_coordinate(s, 3, 0.0)


@dataclass(frozen=True)
class FakeGroundTruth:
    object_id: str
    simulation_time_s: float
    position_m: tuple
    velocity_m_s: tuple | None
    motion_state: str = "on_belt"


def test_ground_truth_adapter_keeps_only_belt_objects():
    records = {
        "a": FakeGroundTruth("a", 0.5, (0.0, 0.0, 0.79), (0.12, 0.0, 0.0)),
        "b": FakeGroundTruth("b", 0.5, (1.3, 0.0, 0.3), (0.1, 0.0, -1.0), "falling"),
        "c": FakeGroundTruth("c", 0.5, (0.2, 0.0, 0.79), None, "released"),
    }
    states = tracked_states_from_ground_truth(records)
    assert list(states) == ["a"]
    assert states["a"] == TrackedState("a", 0.5, (0.0, 0.0, 0.79), (0.12, 0.0, 0.0))
    with pytest.raises(ValueError, match="no known velocity"):
        tracked_state_from_ground_truth(records["c"])


def test_prediction_matches_kinematic_belt_ground_truth():
    # The simulated belt moves objects at exactly belt_speed along +x.
    speed, dt, steps = 0.16, 0.02, 250
    s = state(t=0.0, p=(-1.05, 0.0, 0.79), v=(speed, 0.0, 0.0))
    x = -1.05
    for _ in range(steps):
        x += speed * dt
    assert ConstantVelocityPredictor().predict(s, steps * dt).position_m[0] == pytest.approx(x, abs=1e-9)
