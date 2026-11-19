"""Prediction/grasping/scheduling on live ConveyorSimulation data (needs RAI).

Ground truth stands in for the tracker. No viewer, camera or robot motion is
used; this checks interfaces and prediction accuracy, not physical grasping.
"""

from math import pi

import pytest

pytest.importorskip("robotic")

from conveyor.grasping import geometry_from_ground_truth  # noqa: E402
from conveyor.prediction import ConstantVelocityPredictor, tracked_states_from_ground_truth  # noqa: E402
from conveyor.scheduling import (  # noqa: E402
    DeadlineAwareGreedyScheduler, SchedulerConfig, robot_configs_from_snapshots,
    schedulable_objects_from_ground_truth,
)
from conveyor.scheduling.adapters import SCENE_PLACE_APPROACH, SCENE_ROBOT_BASES  # noqa: E402
from conveyor.simulation import ConveyorSimulation  # noqa: E402


def test_prediction_matches_simulated_belt_motion():
    simulation = ConveyorSimulation(0.15, scenario="ordered", seed=7, object_count=6, exit_mode="remove")
    try:
        simulation.step(0.5)
        states = tracked_states_from_ground_truth(simulation.get_ground_truth(active_only=True))
        assert states
        predicted = ConstantVelocityPredictor(planar_motion=True).predict_many(states, 2.5)
        for _ in range(100):
            simulation.step(0.02)
        truth = simulation.get_ground_truth(active_only=True)
        compared = 0
        for object_id, prediction in predicted.items():
            if object_id in truth and truth[object_id].motion_state == "on_belt":
                assert prediction.position_m == pytest.approx(truth[object_id].position_m, abs=1e-6)
                compared += 1
        assert compared >= 3
    finally:
        simulation.close()


def test_scheduler_runs_on_simulation_snapshots():
    simulation = ConveyorSimulation(0.12, scenario="ordered", seed=7, object_count=8, exit_mode="remove")
    try:
        simulation.step(0.2)
        robot_states, targets = simulation.get_robot_states(), simulation.get_targets()
        for robot_id, state in robot_states.items():
            assert state.base_position_m == pytest.approx(SCENE_ROBOT_BASES[robot_id], abs=1e-6)
        for target in targets.values():
            assert target.approach_position_m == pytest.approx(SCENE_PLACE_APPROACH[target.preferred_robot_id], abs=1e-6)

        truth = simulation.get_ground_truth(active_only=True)
        for spec in simulation.object_specs:
            geometry = geometry_from_ground_truth(truth[spec.object_id])
            difference = (geometry.yaw_rad - spec.yaw_rad) % (2 * pi)
            assert min(difference, 2 * pi - difference) == pytest.approx(0.0, abs=1e-5)

        robots = robot_configs_from_snapshots(robot_states, targets)
        objects = schedulable_objects_from_ground_truth(truth)
        result = DeadlineAwareGreedyScheduler(SchedulerConfig(time_step_s=0.05)).schedule(
            objects, robots, simulation.simulation_time_s,
        )
        assert len(result.assignments) == len(objects)
        assert result.assigned
        assert all(a.intercept_time_s >= simulation.simulation_time_s for a in result.assigned)
    finally:
        simulation.close()


# =============================================================== RAI frame / KOMO checks (review of 18 Nov 2026)
import numpy as np  # noqa: E402
import robotic as ry  # noqa: E402

from conveyor.grasping import ObjectGeometry, select_top_down_grasps  # noqa: E402
from conveyor.scheduling import ReachModel, default_robot_configs  # noqa: E402
from conveyor.scheduling.feasibility import InterceptionModel  # noqa: E402

IK_TOLERANCE = 0.05  # summed |eq| at objective scale 10, i.e. about 5 mm / 0.005 rad in total


def rotation_columns(q):
    w, x, y, z = q
    return (
        (1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)),  # local x in world
        (2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x)),  # local y in world
        (2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)),  # local z in world
    )


def solve_gripper_pose(config, gripper, position, x_axis, z_axis):
    """Minimal collision-free KOMO IK for one gripper pose; best of three deterministic starts."""
    q0 = config.getJointState().copy()
    rng = np.random.default_rng(0)
    best = float("inf")
    try:
        for scale in (0.0, 0.5, 1.0):
            config.setJointState(q0 + scale * rng.standard_normal(q0.shape))
            komo = ry.KOMO(config, 1, 1, 0, False)
            komo.addControlObjective([], 0, 1e-2)
            komo.addObjective([], ry.FS.jointLimits, [], ry.OT.ineq, [1e1])
            komo.addObjective([], ry.FS.position, [gripper], ry.OT.eq, [1e1], list(position))
            komo.addObjective([], ry.FS.vectorX, [gripper], ry.OT.eq, [1e1], list(x_axis))
            komo.addObjective([], ry.FS.vectorZ, [gripper], ry.OT.eq, [1e1], list(z_axis))
            ret = ry.NLP_Solver(komo.nlp(), verbose=0).solve()
            best = min(best, ret.eq + max(ret.ineq, 0.0))
            if best < 1e-3:
                break
    finally:
        config.setJointState(q0)
    return best


def test_rai_panda_gripper_frame_convention():
    """+z of ``a_gripper`` points to the wrist and +x is the finger opening direction."""
    simulation = ConveyorSimulation(0.12, scenario="single", seed=7)
    try:
        config = simulation.config
        gripper = config.getFrame("a_gripper")
        rotation = np.array(gripper.getRotationMatrix())
        x_axis, z_axis = rotation[:, 0], rotation[:, 2]
        hand = np.array(config.getFrame("a_panda_hand_joint").getPosition())
        offset = hand - np.array(gripper.getPosition())
        assert np.linalg.norm(offset) == pytest.approx(0.1035, abs=2e-3)
        assert offset / np.linalg.norm(offset) == pytest.approx(z_axis, abs=1e-3)  # wrist along +z
        fingers = np.array(config.getFrame("a_finger1").getPosition()) - np.array(config.getFrame("a_finger2").getPosition())
        fingers /= np.linalg.norm(fingers)
        assert abs(float(fingers @ x_axis)) > 0.85  # finger separation mostly along +x (pads offset in y/z)
        assert abs(float(fingers @ z_axis)) < 0.05
    finally:
        simulation.close()


@pytest.mark.parametrize("geometry", [
    ObjectGeometry("box", "box", (0.06, 0.045, 0.08), 0.6),
    ObjectGeometry("cyl", "cylinder", (0.05, 0.05, 0.06)),
])
def test_komo_accepts_the_grasp_quaternion_but_not_the_old_z_down_frame(geometry):
    simulation = ConveyorSimulation(0.12, scenario="single", seed=7)
    try:
        centroid = (-0.55, 0.0, 0.75 + geometry.dimensions_m[2] / 2)
        candidate = select_top_down_grasps(geometry, centroid, object_velocity_m_s=(0.12, 0.0, 0.0)).best
        x_axis, _, z_axis = rotation_columns(candidate.quaternion_wxyz)
        assert x_axis == pytest.approx(candidate.closing_axis_world, abs=1e-9)
        good = solve_gripper_pose(simulation.config, "a_gripper", candidate.position_m, x_axis, z_axis)
        assert good < IK_TOLERANCE
        old_x, _, old_z = rotation_columns(candidate.approach_frame_quaternion_wxyz)  # pre-fix convention
        bad = solve_gripper_pose(simulation.config, "a_gripper", candidate.position_m, old_x, old_z)
        assert bad > 10 * IK_TOLERANCE
    finally:
        simulation.close()


def test_reach_model_boundary_points_are_komo_feasible():
    """Points the default (0.03 m margin) model accepts right at its boundary are IK-solvable."""
    robot = default_robot_configs(("robot_a",))[0]
    assert robot.reach == ReachModel()
    simulation = ConveyorSimulation(0.12, scenario="single", seed=7)
    try:
        points = []
        for y in (-0.2, 0.0, 0.15):
            for z in (0.775, 0.81):
                xs = [x / 1000 for x in range(-1200, 400, 5)
                      if InterceptionModel.reachable(robot, (x / 1000, y, z))]
                points += [(xs[0], y, z), (xs[-1], y, z)]  # extreme accepted points on both sides
        for point in points:
            error = solve_gripper_pose(simulation.config, "a_gripper", point, (1.0, 0.0, 0.0), (0.0, 0.0, 1.0))
            assert error < IK_TOLERANCE, point
    finally:
        simulation.close()
