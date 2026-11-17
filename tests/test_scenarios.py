"""Scenario properties without requiring a robotics backend."""

from dataclasses import replace
from math import cos, hypot, pi, sin
import random

import pytest

from conveyor.simulation.scenarios import (
    BELT_LENGTH_M, BELT_TOP_M, BELT_WIDTH_M, EDGE_MARGIN_M,
    OBJECT_CLEARANCE_M, ORDERED_MIN_GAP_M, ObjectSpec, footprints_overlap, generate_scenario,
    planar_half_extents,
)


def test_single_layout_preserves_the_original_box():
    (spec,) = generate_scenario("single")
    assert spec.object_id == "object_001"
    assert spec.geometry_type == "box"
    assert spec.position_m == (-1.05, 0.0, .79)
    assert spec.dimensions_m == (.06, .045, .08)
    assert spec.yaw_rad == 0


@pytest.mark.parametrize("name", ["sparse", "clutter"])
@pytest.mark.parametrize("count", [1, 8, 24])
def test_seeded_layouts_have_supported_nonintersecting_varied_objects(name, count):
    layout = generate_scenario(name, seed=7, object_count=count)
    assert layout == generate_scenario(name, seed=7, object_count=count)
    assert layout != generate_scenario(name, seed=8, object_count=count)
    assert len(layout) == count
    assert [item.object_id for item in layout] == [f"object_{index + 1:03d}" for index in range(count)]
    if count >= 2:
        assert {item.geometry_type for item in layout} == {"box", "cylinder"}
        assert len({item.yaw_rad for item in layout}) > 1
        assert len({item.dimensions_m for item in layout}) > 1
    for index, spec in enumerate(layout):
        assert spec.position_m[2] - spec.dimensions_m[2] / 2 == pytest.approx(BELT_TOP_M)
        # Inspect all rotated corners independently of planar_half_extents.
        if spec.geometry_type == "box":
            c, s = cos(spec.yaw_rad), sin(spec.yaw_rad)
            for sign_x in (-1, 1):
                for sign_y in (-1, 1):
                    x = spec.position_m[0] + c * sign_x * spec.dimensions_m[0] / 2 - s * sign_y * spec.dimensions_m[1] / 2
                    y = spec.position_m[1] + s * sign_x * spec.dimensions_m[0] / 2 + c * sign_y * spec.dimensions_m[1] / 2
                    assert abs(x) <= BELT_LENGTH_M / 2 - EDGE_MARGIN_M
                    assert abs(y) <= BELT_WIDTH_M / 2 - EDGE_MARGIN_M
        else:
            assert abs(spec.position_m[0]) + spec.dimensions_m[0] / 2 <= BELT_LENGTH_M / 2 - EDGE_MARGIN_M
            assert abs(spec.position_m[1]) + spec.dimensions_m[1] / 2 <= BELT_WIDTH_M / 2 - EDGE_MARGIN_M
        for old in layout[:index]:
            assert not footprints_overlap(spec, old, OBJECT_CLEARANCE_M)


def test_clutter_is_more_compact_than_the_sparse_scenario():
    def nearest_distances(layout):
        return [min(hypot(a.position_m[0] - b.position_m[0], a.position_m[1] - b.position_m[1]) for b in layout if a != b) for a in layout]

    clutter = generate_scenario("clutter", seed=7, object_count=8)
    sparse = generate_scenario("sparse", seed=7, object_count=8)
    assert max(item.position_m[0] for item in clutter) < -.12
    assert sum(nearest_distances(clutter)) < .55 * sum(nearest_distances(sparse))
    assert max(nearest_distances(clutter)) < .12


def test_footprint_checks_handle_rotation_and_circle_box_gaps():
    box = ObjectSpec("box", "box", (.10, .04, .08), (0.0, 0.0, .79), pi / 2, (.9, .3, .1))
    assert planar_half_extents(box) == pytest.approx((.02, .05))
    circle = ObjectSpec("circle", "cylinder", (.04, .04, .06), (.05, 0.0, .78), .9, (.1, .4, .8))
    assert not footprints_overlap(box, circle, .008)
    assert footprints_overlap(box, replace(circle, position_m=(.045, 0.0, .78)), .008)
    assert footprints_overlap(box, replace(circle, position_m=(.03, 0.0, .78)))
    assert not footprints_overlap(box, replace(box, object_id="other", position_m=(.06, 0, .79)))


def test_generator_does_not_change_global_random_state_and_rejects_invalid_options():
    state = random.getstate()
    generate_scenario("clutter", seed=7)
    assert random.getstate() == state
    for options in ({"name": "missing"}, {"seed": True}, {"object_count": 0}, {"object_count": 25}, {"object_count": True}):
        with pytest.raises(ValueError):
            generate_scenario(**options)


@pytest.mark.parametrize("count", [1, 8, 12])
def test_ordered_arrivals_form_a_supported_spaced_single_file(count):
    state = random.getstate()
    layout = generate_scenario("ordered", seed=7, object_count=count)
    assert random.getstate() == state
    assert layout == generate_scenario("ordered", seed=7, object_count=count)
    assert layout != generate_scenario("ordered", seed=8, object_count=count)
    assert len(layout) == count
    assert [item.object_id for item in layout] == [f"object_{index + 1:03d}" for index in range(count)]
    assert [item.position_m[0] for item in layout] == sorted(item.position_m[0] for item in layout)
    if count > 1:
        assert {item.geometry_type for item in layout} == {"box", "cylinder"}
    for index, spec in enumerate(layout):
        hx, hy = planar_half_extents(spec)
        assert abs(spec.position_m[0]) + hx < BELT_LENGTH_M / 2 - EDGE_MARGIN_M
        assert abs(spec.position_m[1]) <= .035
        assert abs(spec.position_m[1]) + hy < BELT_WIDTH_M / 2 - EDGE_MARGIN_M
        assert spec.position_m[2] - spec.dimensions_m[2] / 2 == pytest.approx(BELT_TOP_M)
        if index:
            previous = layout[index - 1]
            previous_right = previous.position_m[0] + planar_half_extents(previous)[0]
            edge_gap = spec.position_m[0] - hx - previous_right
            assert edge_gap >= ORDERED_MIN_GAP_M - 1e-12
            assert not footprints_overlap(previous, spec, OBJECT_CLEARANCE_M)


def test_ordered_rejects_counts_that_would_remove_visible_arrival_gaps():
    with pytest.raises(ValueError, match="reduce --objects"):
        generate_scenario("ordered", object_count=24)
