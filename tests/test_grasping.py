"""Top-down grasp candidates for boxes and cylinders; no RAI required."""

from dataclasses import dataclass
from math import cos, pi, sin, sqrt

import pytest

from conveyor.grasping import (
    RAI_GRIPPER_FRAME_CONVENTION, GripperConfig, ObjectGeometry, approach_frame_quaternion, approach_to_rai_gripper,
    geometry_from_ground_truth, geometry_from_spec, rai_gripper_quaternion, select_top_down_grasps,
    yaw_from_quaternion_wxyz,
)
from conveyor.grasping.geometry import polygon_circle_distance, polygon_distance, rectangle_corners


def box(object_id="box", dims=(0.06, 0.045, 0.08), yaw=0.0):
    return ObjectGeometry(object_id, "box", dims, yaw)


def cylinder(object_id="cyl", diameter=0.05, height=0.06):
    return ObjectGeometry(object_id, "cylinder", (diameter, diameter, height))


def rotate(vector, quaternion):
    w, x, y, z = quaternion
    matrix = (
        (1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)),
        (2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)),
        (2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)),
    )
    return tuple(sum(row[i] * vector[i] for i in range(3)) for row in matrix)


def test_box_produces_two_axis_aligned_candidates_and_prefers_the_narrow_side():
    result = select_top_down_grasps(box(), (0.0, 0.0, 0.79))
    assert len(result.feasible) == 2 and not result.rejected
    best = result.best
    assert best.grasp_type == "top_down_box_close_y"
    assert best.object_width_m == pytest.approx(0.045)
    assert best.required_opening_m == pytest.approx(0.055)
    assert best.gripper_yaw_rad == pytest.approx(pi / 2)
    assert best.position_m == pytest.approx((0.0, 0.0, 0.83 - 0.025))
    assert best.pregrasp_position_m[2] == pytest.approx(best.position_m[2] + 0.10)
    assert not best.physically_verified


@pytest.mark.parametrize("yaw", [0.0, 0.3, pi / 4, 1.2, -2.5, pi])
def test_box_grasp_axes_follow_object_yaw(yaw):
    result = select_top_down_grasps(box(yaw=yaw), (0.2, -0.1, 0.79))
    for candidate in result.feasible:
        axis = candidate.closing_axis_world
        local_x = (cos(yaw), sin(yaw))
        alignment = abs(axis[0] * local_x[0] + axis[1] * local_x[1])
        expected = 1.0 if candidate.grasp_type.endswith("close_x") else 0.0
        assert alignment == pytest.approx(expected, abs=1e-9)
        assert -pi / 2 < candidate.gripper_yaw_rad <= pi / 2


@pytest.mark.parametrize("yaw", [0.0, 0.7, -1.3, pi / 2, 2.4])
@pytest.mark.parametrize("make", [box, cylinder])
def test_grasp_quaternion_matches_rai_panda_gripper_frame(yaw, make):
    """Regression for the z-down orientation bug: ``quaternion_wxyz`` is the RAI
    Panda ``gripper`` frame, whose +z points to the wrist (up for top-down) and
    whose +x is the finger closing direction (``panda.g``: gripper(panda_joint7)
    ``d(180 0 1 0) d(135 0 0 1) t(0 0 -.2105)``)."""
    geometry = make(yaw=yaw) if make is box else make()
    for candidate in select_top_down_grasps(geometry, (0.0, 0.0, 0.79)).feasible:
        q = candidate.quaternion_wxyz
        assert sqrt(sum(c * c for c in q)) == pytest.approx(1.0)
        z_axis, x_axis, y_axis = rotate((0, 0, 1), q), rotate((1, 0, 0), q), rotate((0, 1, 0), q)
        assert z_axis == pytest.approx((0, 0, 1), abs=1e-9)  # towards the wrist: world up
        assert z_axis != pytest.approx((0, 0, -1), abs=1e-3)  # the original bug
        assert x_axis == pytest.approx(candidate.closing_axis_world, abs=1e-9)
        cross = (z_axis[1] * x_axis[2] - z_axis[2] * x_axis[1], z_axis[2] * x_axis[0] - z_axis[0] * x_axis[2],
                 z_axis[0] * x_axis[1] - z_axis[1] * x_axis[0])
        assert y_axis == pytest.approx(cross, abs=1e-9)  # right-handed frame
        assert q[0] >= 0  # canonical sign, (w, x, y, z) order: pure yaw has x = y = 0
        assert q[1] == pytest.approx(0.0, abs=1e-12) and q[2] == pytest.approx(0.0, abs=1e-12)
        # The approach frame is kept separately: same closing axis, +z along the approach (down).
        a = candidate.approach_frame_quaternion_wxyz
        assert rotate((0, 0, 1), a) == pytest.approx(candidate.approach_axis_world, abs=1e-9)
        assert rotate((1, 0, 0), a) == pytest.approx(candidate.closing_axis_world, abs=1e-9)
        assert approach_to_rai_gripper(a) == pytest.approx(q, abs=1e-12)
        assert candidate.frame_convention == RAI_GRIPPER_FRAME_CONVENTION


def test_rai_gripper_quaternion_is_pure_yaw_in_wxyz_order():
    assert rai_gripper_quaternion(0.0) == pytest.approx((1.0, 0.0, 0.0, 0.0))
    assert rai_gripper_quaternion(pi / 2) == pytest.approx((cos(pi / 4), 0.0, 0.0, sin(pi / 4)))
    assert approach_frame_quaternion(0.0) == pytest.approx((0.0, 1.0, 0.0, 0.0))  # Rx(pi)
    for yaw in (-1.2, 0.3, 1.5):
        assert yaw_from_quaternion_wxyz(rai_gripper_quaternion(yaw)) == pytest.approx(yaw)


def test_cylinder_has_rotationally_symmetric_candidates():
    result = select_top_down_grasps(cylinder(), (0.0, 0.0, 0.78), config=GripperConfig(cylinder_yaw_samples=6))
    assert len(result.feasible) == 6
    assert {c.object_width_m for c in result.feasible} == {0.05}
    yaws = sorted(round(c.gripper_yaw_rad, 6) for c in result.feasible)
    assert len(set(yaws)) == 6


def test_width_limits_reject_oversized_and_too_thin_objects():
    wide = select_top_down_grasps(box(dims=(0.11, 0.10, 0.06)), (0, 0, 0.78))
    assert not wide.has_feasible and wide.best is None
    assert all("object_too_wide_for_gripper" in c.rejection_reasons for c in wide.rejected)
    # One side too wide, the other still graspable.
    long_box = select_top_down_grasps(box(dims=(0.12, 0.05, 0.06)), (0, 0, 0.78))
    assert [c.grasp_type for c in long_box.feasible] == ["top_down_box_close_y"]
    assert not select_top_down_grasps(cylinder(diameter=0.09), (0, 0, 0.78)).has_feasible
    thin = select_top_down_grasps(box(dims=(0.005, 0.2, 0.05)), (0, 0, 0.775))
    assert all(
        "object_too_narrow_for_gripper" in c.rejection_reasons or "object_too_wide_for_gripper" in c.rejection_reasons
        for c in thin.rejected
    ) and not thin.has_feasible


def test_configurable_gripper_width_changes_feasibility():
    geometry = cylinder(diameter=0.07)
    assert select_top_down_grasps(geometry, (0, 0, 0.78)).has_feasible
    narrow = GripperConfig(max_opening_m=0.06)
    assert not select_top_down_grasps(geometry, (0, 0, 0.78), config=narrow).has_feasible


def test_short_objects_raise_the_tcp_above_the_belt():
    config = GripperConfig()
    # 2.4 cm box: the nominal TCP (top - 1.2 cm) would violate the belt clearance, so it is raised.
    low = select_top_down_grasps(box(dims=(0.05, 0.04, 0.024)), (0, 0, 0.762), config=config).best
    assert low.position_m[2] - config.fingertip_below_tcp_m == pytest.approx(
        config.support_surface_z_m + config.min_support_clearance_m)
    assert 0.774 - low.position_m[2] >= config.min_contact_depth_m
    flat = select_top_down_grasps(box(dims=(0.05, 0.04, 0.012)), (0, 0, 0.756))
    assert not flat.has_feasible
    assert "object_too_short_for_finger_contact" in flat.rejected[0].rejection_reasons


def test_neighbor_clearance_rejects_blocked_jaw_directions():
    target = box("t", dims=(0.06, 0.04, 0.06))
    # Neighbour directly beside the target along world y blocks closing along y.
    neighbor = (box("n", dims=(0.06, 0.04, 0.06)), (0.0, 0.05, 0.78))
    result = select_top_down_grasps(target, (0.0, 0.0, 0.78), neighbors=[neighbor])
    assert [c.grasp_type for c in result.feasible] == ["top_down_box_close_x"]
    assert "insufficient_neighbor_clearance" in result.rejected[0].rejection_reasons
    # A distant neighbour does not matter, and the target itself is ignored.
    far = (box("n", dims=(0.06, 0.04, 0.06)), (0.5, 0.5, 0.78))
    assert len(select_top_down_grasps(target, (0, 0, 0.78), neighbors=[far, (target, (0, 0, 0.78))]).feasible) == 2


def test_cylinder_picks_the_jaw_direction_away_from_neighbors():
    target = cylinder("t", diameter=0.05)
    neighbors = [(cylinder("n1"), (0.065, 0.0, 0.78)), (cylinder("n2"), (-0.065, 0.0, 0.78))]
    result = select_top_down_grasps(target, (0.0, 0.0, 0.78), neighbors=neighbors)
    assert result.has_feasible
    assert abs(result.best.closing_axis_world[1]) == pytest.approx(1.0, abs=1e-9)
    assert all(abs(c.closing_axis_world[0]) > 0.5 for c in result.rejected)



# --------------------------------------------------------------- motion-aware ranking
BELT_VELOCITY = (0.12, 0.0, 0.0)


def test_moving_cylinder_closes_perpendicular_to_belt_motion():
    static = select_top_down_grasps(cylinder(), (0.0, 0.0, 0.78))
    assert static.best.closing_axis_world == pytest.approx((1.0, 0.0, 0.0), abs=1e-9)  # old tie-break
    assert static.best.motion_alignment is None
    moving = select_top_down_grasps(cylinder(), (0.0, 0.0, 0.78), object_velocity_m_s=BELT_VELOCITY)
    assert abs(moving.best.closing_axis_world[1]) == pytest.approx(1.0, abs=1e-9)
    assert moving.best.motion_alignment == pytest.approx(1.0)
    # Same candidate set and the same feasibility verdicts: only the ranking changed.
    assert sorted(c.candidate_id for c in moving.feasible) == sorted(c.candidate_id for c in static.feasible)
    assert all(not c.physically_verified for c in moving.feasible)


def test_moving_box_prefers_perpendicular_closing_over_narrower_parallel_side():
    geometry = box(dims=(0.04, 0.06, 0.06), yaw=0.0)  # narrow side along the belt (x)
    static = select_top_down_grasps(geometry, (0.0, 0.0, 0.78))
    assert static.best.grasp_type == "top_down_box_close_x"
    moving = select_top_down_grasps(geometry, (0.0, 0.0, 0.78), object_velocity_m_s=BELT_VELOCITY)
    assert moving.best.grasp_type == "top_down_box_close_y"
    assert moving.best.closing_axis_world == pytest.approx((0.0, 1.0, 0.0), abs=1e-9)
    assert len(moving.feasible) == 2  # the parallel grasp is still offered, ranked lower


@pytest.mark.parametrize("yaw", [0.2, -0.5, 1.1, 2.9])
def test_moving_box_best_candidate_is_the_most_perpendicular_one(yaw):
    geometry = box(dims=(0.05, 0.05, 0.06), yaw=yaw)  # square: width terms tie
    moving = select_top_down_grasps(geometry, (0.0, 0.0, 0.78), object_velocity_m_s=BELT_VELOCITY)
    assert moving.best.motion_alignment == pytest.approx(max(c.motion_alignment for c in moving.feasible))
    assert moving.best.motion_alignment == pytest.approx(max(abs(cos(yaw)), abs(sin(yaw))))


def test_motion_direction_follows_the_velocity_vector():
    moving_y = select_top_down_grasps(cylinder(), (0.0, 0.0, 0.78), object_velocity_m_s=(0.0, -0.1, 0.0))
    assert abs(moving_y.best.closing_axis_world[0]) == pytest.approx(1.0, abs=1e-9)
    diagonal = select_top_down_grasps(cylinder(), (0.0, 0.0, 0.78), object_velocity_m_s=(0.1, 0.1, 0.0),
                                      config=GripperConfig(cylinder_yaw_samples=4))
    assert diagonal.best.gripper_yaw_rad == pytest.approx(-pi / 4)  # perpendicular to (1, 1)


def test_motion_preference_never_overrides_hard_geometric_checks():
    # The perpendicular side (0.075 m + clearance) exceeds the 80 mm opening.
    geometry = box(dims=(0.04, 0.075, 0.06))
    moving = select_top_down_grasps(geometry, (0.0, 0.0, 0.78), object_velocity_m_s=BELT_VELOCITY)
    assert [c.grasp_type for c in moving.feasible] == ["top_down_box_close_x"]
    assert moving.best.motion_alignment == pytest.approx(0.0, abs=1e-12)  # parallel, still geometric only
    assert "object_too_wide_for_gripper" in moving.rejected[0].rejection_reasons
    # Neighbours across the belt block perpendicular jaws: the best remaining direction is used.
    neighbors = [(cylinder("n1"), (0.0, 0.065, 0.78)), (cylinder("n2"), (0.0, -0.065, 0.78))]
    blocked = select_top_down_grasps(cylinder(), (0.0, 0.0, 0.78), neighbors=neighbors,
                                     object_velocity_m_s=BELT_VELOCITY)
    assert blocked.has_feasible
    assert all(abs(c.closing_axis_world[1]) > 0.5 for c in blocked.rejected)
    assert blocked.best.motion_alignment == pytest.approx(max(c.motion_alignment for c in blocked.feasible))


def test_motion_preference_is_configurable_and_ignores_stationary_objects():
    off = GripperConfig(motion_alignment_weight=0.0)
    geometry = box(dims=(0.04, 0.06, 0.06))
    assert select_top_down_grasps(geometry, (0, 0, 0.78), object_velocity_m_s=BELT_VELOCITY,
                                  config=off).best.grasp_type == "top_down_box_close_x"
    slow = select_top_down_grasps(geometry, (0, 0, 0.78), object_velocity_m_s=(0.0005, 0.0, 0.0))
    assert slow.best.grasp_type == "top_down_box_close_x" and slow.best.motion_alignment is None
    with pytest.raises(ValueError):
        select_top_down_grasps(geometry, (0, 0, 0.78), object_velocity_m_s=(float("nan"), 0.0, 0.0))
    with pytest.raises(ValueError):
        GripperConfig(motion_alignment_weight=-1.0)

def test_ranking_is_deterministic():
    a = select_top_down_grasps(cylinder(), (0.1, 0.0, 0.78))
    b = select_top_down_grasps(cylinder(), (0.1, 0.0, 0.78))
    assert a == b


def test_invalid_geometry_and_positions():
    for args in (("x", "sphere", (0.1, 0.1, 0.1)), ("x", "box", (0.0, 0.1, 0.1)),
                 ("x", "cylinder", (0.05, 0.06, 0.1)), (" ", "box", (0.1, 0.1, 0.1))):
        with pytest.raises(ValueError):
            ObjectGeometry(*args)
    with pytest.raises(ValueError):
        select_top_down_grasps(box(), (0.0, float("nan"), 0.8))
    with pytest.raises(ValueError):
        GripperConfig(max_opening_m=-0.1)


def test_planar_distance_helpers():
    a = rectangle_corners((0, 0), 0.5, 0.5, 0.0)
    b = rectangle_corners((2, 0), 0.5, 0.5, 0.0)
    assert polygon_distance(a, b) == pytest.approx(1.0)
    assert polygon_distance(a, rectangle_corners((0.9, 0), 0.5, 0.5, pi / 4)) == 0.0
    assert polygon_circle_distance(a, (0, 2), 0.5) == pytest.approx(1.0)
    assert polygon_circle_distance(a, (0, 0), 0.1) == 0.0


@dataclass(frozen=True)
class FakeSpec:
    object_id: str
    geometry_type: str
    dimensions_m: tuple
    position_m: tuple
    yaw_rad: float


@dataclass(frozen=True)
class FakeTruth:
    object_id: str
    quaternion_wxyz: tuple
    dimensions_m: tuple
    geometry_type: str


def test_geometry_adapters_from_simulation_records():
    spec = FakeSpec("object_001", "box", (0.06, 0.045, 0.08), (-1.05, 0, 0.79), 0.4)
    assert geometry_from_spec(spec) == ObjectGeometry("object_001", "box", (0.06, 0.045, 0.08), 0.4)
    yaw = 0.9
    q = (cos(yaw / 2), 0.0, 0.0, sin(yaw / 2))
    assert yaw_from_quaternion_wxyz(q) == pytest.approx(yaw)
    geometry = geometry_from_ground_truth(FakeTruth("o", q, (0.05, 0.05, 0.06), "cylinder"))
    assert geometry.yaw_rad == pytest.approx(yaw) and geometry.geometry_type == "cylinder"
    tipped = (cos(0.5), sin(0.5), 0.0, 0.0)  # 1 rad about x
    with pytest.raises(ValueError, match="tilted"):
        geometry_from_ground_truth(FakeTruth("o", tipped, (0.05, 0.05, 0.06), "box"))
