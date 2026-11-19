"""Check free-tool timing, speed limits and interception outcome."""

from dataclasses import replace
from itertools import pairwise
from math import dist
from pathlib import Path
import subprocess
import sys

import pytest

from conveyor.control.free_tcp import evaluate_interception
from conveyor.grasping import ObjectGeometry, select_top_down_grasps
from conveyor.planning.free_tcp import TcpSample, plan_free_tcp
from conveyor.scheduling import Assignment
from conveyor.simulation import ObjectGroundTruth
from conveyor.types import TrackedState


@pytest.fixture
def interception():
    track = TrackedState("box", 1.0, (-0.86, 0.0, 0.79), (0.12, 0.0, 0.0))
    centroid = (-0.5, 0.0, 0.79)
    geometry = ObjectGeometry(track.object_id, "box", (0.06, 0.045, 0.08))
    grasp = select_top_down_grasps(geometry, centroid, object_velocity_m_s=track.velocity_m_s).best
    assignment = Assignment(
        track.object_id, "assigned", "test", robot_id="robot_a", intercept_time_s=4.0,
        intercept_position_m=centroid, grasp=grasp, robot_available_at_s=1.2, deadline_s=5.0,
        object_exit_time_s=7.0,
    )
    return assignment, track, TcpSample(1.0, (-0.7, -0.3, 1.1))


def test_plan_waits_until_available_and_matches_object_velocity(interception):
    assignment, track, start = interception
    plan = plan_free_tcp(*interception)
    assert plan.position_at(1.1) == start.position_m
    assert plan.position_at(plan.intercept_time_s) == pytest.approx(assignment.grasp.position_m)
    assert plan.quaternion_wxyz == assignment.grasp.quaternion_wxyz
    for first, last in pairwise(plan.waypoints):
        assert dist(first.position_m, last.position_m) / (last.simulation_time_s - first.simulation_time_s) <= 0.4
    first, last = plan.waypoints[-2:]
    duration = last.simulation_time_s - first.simulation_time_s
    assert tuple((b - a) / duration for a, b in zip(first.position_m, last.position_m)) == pytest.approx(track.velocity_m_s)


def test_plan_rejects_late_exit_and_speed_limit(interception):
    assignment, track, start = interception
    with pytest.raises(ValueError, match="Too late"):
        plan_free_tcp(assignment, track, replace(start, simulation_time_s=3.75))
    with pytest.raises(ValueError, match="exits before"):
        plan_free_tcp(replace(assignment, object_exit_time_s=4.0), track, start)
    with pytest.raises(ValueError, match="speed limit"):
        plan_free_tcp(assignment, track, start, max_speed_m_s=0.10)


def test_evaluation_uses_observed_pose_velocity_and_availability(interception):
    assignment, track, _ = interception
    plan = plan_free_tcp(*interception)
    first, last = plan.waypoints[-2:]
    truth = ObjectGroundTruth(
        track.object_id, plan.intercept_time_s, assignment.intercept_position_m,
        (1.0, 0.0, 0.0, 0.0), (0.06, 0.045, 0.08), track.velocity_m_s, "on_belt",
    )
    assert evaluate_interception(plan, first, last, truth).success
    assert evaluate_interception(plan, first, last, None).reason == "object_unavailable"
    assert evaluate_interception(plan, first, replace(last, simulation_time_s=4.02), truth).reason == "incorrect_sample_time"
    shifted = replace(truth, position_m=(-0.49, 0.0, 0.79))
    assert evaluate_interception(plan, first, last, shifted).reason == "position_error"
    stopped = replace(truth, velocity_m_s=(0.0, 0.0, 0.0))
    assert evaluate_interception(plan, first, last, stopped).reason == "velocity_error"


def test_headless_demo_runs_with_real_simulation_and_scheduler():
    pytest.importorskip("robotic")
    demo = Path(__file__).resolve().parents[1] / "scripts" / "free_tcp_demo.py"
    run = subprocess.run([sys.executable, "-E", str(demo), "--no-view"], capture_output=True, text=True, timeout=15)
    assert run.returncode == 0, run.stdout + run.stderr
    assert "robot_a / object_001 / scheduled" in run.stdout
    assert "intercepted | time error 0.000 ms | TCP error 0.000 mm | relative speed 0.000 mm/s" in run.stdout
