"""Deadline-aware dual-arm scheduling, FIFO baseline and online dispatch.

Pure Python: scenarios come from the repository's RAI-free generator.
Results are planning-level estimates under approximate reach/timing models.
"""

from dataclasses import dataclass, replace
from math import cos, sin

import pytest

from conveyor.grasping import ObjectGeometry
from conveyor.prediction import ConstantVelocityPredictor
from conveyor.scheduling import (
    DeadlineAwareGreedyScheduler, FifoNearestRobotScheduler, ReachModel, RobotConfig, SchedulableObject,
    SchedulerConfig, default_robot_configs, robot_configs_from_snapshots,
    schedulable_objects_from_ground_truth, schedulable_objects_from_specs, summarize,
)
from conveyor.scheduling.feasibility import (
    REASON_INVALID_GRASP, REASON_STALE, REASON_TIME, REASON_UNREACHABLE, InterceptionModel,
)
from conveyor.scheduling.online import run_online
from conveyor.scheduling.streams import generate_arrival_stream
from conveyor.simulation.scenarios import generate_scenario
from conveyor.types import TrackedState

CONFIG = SchedulerConfig(time_step_s=0.05)
METHODS = (DeadlineAwareGreedyScheduler, FifoNearestRobotScheduler)


def obj(object_id, x, y=0.0, speed=0.12, t=0.0, dims=(0.06, 0.045, 0.08), kind="box", yaw=0.0):
    if kind == "cylinder":
        dims = (dims[0], dims[0], dims[2])
    state = TrackedState(object_id, t, (x, y, 0.75 + dims[2] / 2), (speed, 0.0, 0.0))
    return SchedulableObject(state, ObjectGeometry(object_id, kind, dims, yaw))


def check_schedule_consistency(result, objects, robots, config=CONFIG):
    """Invariants every valid schedule must satisfy."""
    by_id = {o.object_id: o for o in objects}
    robot_by_id = {r.robot_id: r for r in robots}
    ids = [a.object_id for a in result.assignments]
    assert len(ids) == len(set(ids)) == len(objects)  # every object decided exactly once
    predictor = ConstantVelocityPredictor(max_horizon_s=1e6, planar_motion=True)
    for robot_id in result.robot_ids:
        timeline = sorted(result.for_robot(robot_id), key=lambda a: a.intercept_time_s)
        previous_completion = max(result.planning_time_s, robot_by_id[robot_id].available_at_s)
        tool = robot_by_id[robot_id].tool_position_m
        for a in timeline:
            # Robot cannot start before it is free; it arrives before the interception.
            assert a.robot_available_at_s >= previous_completion - 1e-9
            assert a.robot_available_at_s + a.travel_time_s <= a.intercept_time_s + 1e-9
            assert a.intercept_time_s <= a.deadline_s + 1e-9
            # Slack = latest departure - availability, where the latest departure uses the
            # travel time to the object's grasp pose AT THE DEADLINE (same time reference).
            centroid_at_deadline = predictor.predict(by_id[a.object_id].state, a.deadline_s).position_m
            offset = [g - c for g, c in zip(a.grasp.position_m, a.intercept_position_m)]
            pregrasp = (centroid_at_deadline[0] + offset[0], centroid_at_deadline[1] + offset[1],
                        centroid_at_deadline[2] + offset[2] + config.gripper.pregrasp_height_m)
            timing = config.timing
            travel_at_deadline = (timing.motion_overhead_s + timing.descend_s
                                  + sum((p - q) ** 2 for p, q in zip(pregrasp, tool)) ** 0.5 / timing.cartesian_speed_m_s)
            assert a.latest_start_s == pytest.approx(a.deadline_s - travel_at_deadline, abs=1e-9)
            assert a.slack_s == pytest.approx(a.latest_start_s - a.robot_available_at_s, abs=1e-9)
            assert a.latest_start_s >= a.intercept_time_s - a.travel_time_s - 1e-9
            assert a.slack_s >= -1e-9
            tool = robot_by_id[robot_id].place_position_m
            assert a.completion_time_s > a.intercept_time_s
            expected = predictor.predict(by_id[a.object_id].state, a.intercept_time_s).position_m
            assert a.intercept_position_m == pytest.approx(expected)
            assert a.grasp is not None and a.grasp.feasible and a.grasp.object_id == a.object_id
            assert a.grasp.position_m[:2] == pytest.approx(a.intercept_position_m[:2])
            assert InterceptionModel.reachable(robot_by_id[robot_id], a.grasp.position_m)
            x0, x1 = config.workspace.pick_x_limits_m
            assert x0 <= a.intercept_position_m[0] <= x1
            previous_completion = a.completion_time_s
    for a in result.rejected:
        assert a.robot_id is None and a.intercept_time_s is None and a.reason


# --------------------------------------------------------------- progression
@pytest.mark.parametrize("method", METHODS)
def test_stage1_single_object_single_robot(method):
    (spec,) = generate_scenario("single")
    objects = schedulable_objects_from_specs([spec], 0.12)
    robots = default_robot_configs(("robot_a",))
    result = method(CONFIG).schedule(objects, robots, 0.0)
    check_schedule_consistency(result, objects, robots)
    (a,) = result.assignments
    assert a.is_assigned and a.robot_id == "robot_a" and a.object_id == "object_001"
    assert a.object_exit_time_s == pytest.approx((1.15 + 1.05) / 0.12)


@pytest.mark.parametrize("method", METHODS)
def test_stage2_sequential_objects_two_robots(method):
    objects = schedulable_objects_from_specs(generate_scenario("ordered", seed=7, object_count=8), 0.12)
    robots = default_robot_configs()
    result = method(CONFIG).schedule(objects, robots, 0.0)
    check_schedule_consistency(result, objects, robots)
    assert len(result.assigned) >= 3
    assert {a.robot_id for a in result.assigned} == {"robot_a", "robot_b"}


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("seed", [1, 4, 7])
def test_stage3_cluttered_objects(method, seed):
    objects = schedulable_objects_from_specs(generate_scenario("clutter", seed=seed, object_count=10), 0.08)
    robots = default_robot_configs()
    result = method(CONFIG).schedule(objects, robots, 0.0)
    check_schedule_consistency(result, objects, robots)
    assert result.assigned


def test_stage4_two_robots_never_worse_than_one_on_ordered_layouts():
    for seed in range(4):
        objects = schedulable_objects_from_specs(generate_scenario("ordered", seed=seed, object_count=8), 0.12)
        one = DeadlineAwareGreedyScheduler(CONFIG).schedule(objects, default_robot_configs(("robot_a",)), 0.0)
        two = DeadlineAwareGreedyScheduler(CONFIG).schedule(objects, default_robot_configs(), 0.0)
        assert len(two.assigned) > len(one.assigned)
        assert {a.robot_id for a in one.assigned} <= {"robot_a"}


# --------------------------------------------------------------- decision logic
def test_deadline_awareness_takes_the_urgent_object_before_a_waiting_one():
    # One robot. A stationary object can wait; a fast object soon leaves reach.
    robots = default_robot_configs(("robot_a",))
    still = obj("still", -0.40, speed=0.0)
    fast = obj("fast", -0.90, speed=0.25)
    greedy = DeadlineAwareGreedyScheduler(CONFIG).schedule([still, fast], robots, 0.0)
    check_schedule_consistency(greedy, [still, fast], robots)
    assert [a.object_id for a in greedy.assigned] == ["fast", "still"]
    fifo = FifoNearestRobotScheduler(CONFIG).schedule([still, fast], robots, 0.0)
    assert fifo.by_object()["fast"].reason == REASON_TIME  # FIFO serves "still" first and loses "fast"


def test_robots_compete_in_a_shared_region_without_duplicate_assignments():
    reach = ReachModel(max_reach_m=1.3)  # Artificially large overlap for this test.
    robots = default_robot_configs(reach=reach)
    objects = [obj(f"o{i}", -0.3 + 0.12 * i, y=0.05 * (-1) ** i, speed=0.05) for i in range(6)]
    result = DeadlineAwareGreedyScheduler(CONFIG).schedule(objects, robots, 0.0)
    check_schedule_consistency(result, objects, robots)
    assert len(result.assigned) == 6
    assert summarize(result).per_robot["robot_a"] >= 1 and summarize(result).per_robot["robot_b"] >= 1


def test_object_only_reachable_by_downstream_robot_goes_to_it():
    objects = [obj("late", 0.6, speed=0.05)]
    result = DeadlineAwareGreedyScheduler(CONFIG).schedule(objects, default_robot_configs(), 0.0)
    assert result.assigned[0].robot_id == "robot_b"
    only_a = DeadlineAwareGreedyScheduler(CONFIG).schedule(objects, default_robot_configs(("robot_a",)), 0.0)
    assert only_a.rejected[0].reason == REASON_UNREACHABLE


def test_impossible_assignments_are_rejected_with_reasons():
    robots = default_robot_configs()
    objects = [
        obj("too_wide", -0.8, dims=(0.12, 0.10, 0.06)),
        obj("too_fast", -0.5, speed=1.5),
        obj("off_belt", -0.5, y=0.6, speed=0.1),
        obj("stale", -0.8, t=0.0),
    ]
    result = DeadlineAwareGreedyScheduler(CONFIG).schedule(objects[:3], robots, 0.0)
    reasons = {a.object_id: a.reason for a in result.rejected}
    assert reasons == {"too_wide": REASON_INVALID_GRASP, "too_fast": REASON_TIME, "off_belt": REASON_UNREACHABLE}
    stale = DeadlineAwareGreedyScheduler(CONFIG).schedule([objects[3]], robots, 10.0)
    assert stale.rejected[0].reason == REASON_STALE


def test_robot_busy_until_late_cannot_take_objects():
    robots = [replace(r, available_at_s=100.0) for r in default_robot_configs()]
    result = DeadlineAwareGreedyScheduler(CONFIG).schedule([obj("a", -0.9)], robots, 0.0)
    assert result.rejected[0].reason == REASON_TIME


def test_stationary_belt_objects_are_scheduled():
    objects = [obj("s1", -0.7, speed=0.0), obj("s2", 0.6, speed=0.0)]
    robots = default_robot_configs()
    result = DeadlineAwareGreedyScheduler(CONFIG).schedule(objects, robots, 0.0)
    check_schedule_consistency(result, objects, robots)
    assert {a.object_id: a.robot_id for a in result.assigned} == {"s1": "robot_a", "s2": "robot_b"}
    assert all(a.object_exit_time_s is None for a in result.assigned)


def test_scheduling_uses_absolute_time_with_slightly_old_tracks():
    robots = default_robot_configs(("robot_a",), available_at_s=5.0)
    o = obj("a", -1.0, speed=0.1, t=4.5)
    result = DeadlineAwareGreedyScheduler(CONFIG).schedule([o], robots, 5.0)
    check_schedule_consistency(result, [o], robots)
    assert result.assigned[0].intercept_time_s >= 5.0 + result.assigned[0].travel_time_s - 1e-9


def test_deterministic_and_independent_of_input_order():
    objects = schedulable_objects_from_specs(generate_scenario("clutter", seed=3, object_count=10), 0.1)
    robots = default_robot_configs()
    for method in METHODS:
        first = method(CONFIG).schedule(objects, robots, 0.0)
        assert first == method(CONFIG).schedule(list(reversed(objects)), list(reversed(robots)), 0.0)


def test_committed_assignments_are_respected():
    objects = schedulable_objects_from_specs(generate_scenario("ordered", seed=7, object_count=8), 0.12)
    robots = default_robot_configs()
    full = DeadlineAwareGreedyScheduler(CONFIG).schedule(objects, robots, 0.0)
    fixed = full.assigned[0]
    robots_after = [
        replace(r, available_at_s=fixed.completion_time_s, tool_position_m=r.place_position_m)
        if r.robot_id == fixed.robot_id else r for r in robots
    ]
    again = DeadlineAwareGreedyScheduler(CONFIG).schedule(objects, robots_after, 0.0, committed=[fixed])
    assert fixed.object_id not in again.by_object()
    assert len(again.assignments) == len(objects) - 1


def test_invalid_scheduler_inputs():
    robots = default_robot_configs()
    scheduler = DeadlineAwareGreedyScheduler(CONFIG)
    with pytest.raises(ValueError):
        scheduler.schedule([obj("a", 0.0)], [], 0.0)
    with pytest.raises(ValueError):
        scheduler.schedule([obj("a", 0.0)], robots + robots[:1], 0.0)
    with pytest.raises(ValueError):
        scheduler.schedule([obj("a", 0.0), obj("a", 0.1)], robots, 0.0)
    with pytest.raises(ValueError):
        scheduler.schedule([obj("a", 0.0)], robots, -1.0)
    with pytest.raises(ValueError):
        DeadlineAwareGreedyScheduler(CONFIG, urgency="random")
    with pytest.raises(ValueError):
        SchedulerConfig(time_step_s=0.0)
    with pytest.raises(ValueError):
        ReachModel(max_reach_m=0.5, safety_margin_m=0.6)
    with pytest.raises(ValueError):
        SchedulableObject(TrackedState("a", 0.0, (0, 0, 0), (0, 0, 0)), ObjectGeometry("b", "box", (1, 1, 1)))
    assert DeadlineAwareGreedyScheduler(CONFIG).schedule([], robots, 0.0).assignments == ()


def test_reach_model_is_explicit_and_configurable():
    robot = default_robot_configs(("robot_a",))[0]
    assert InterceptionModel.reachable(robot, (-0.55, 0.0, 0.8))
    assert not InterceptionModel.reachable(robot, (0.6, 0.0, 0.8))
    assert not InterceptionModel.reachable(robot, (-0.55, -0.55, 0.8))  # too close to the base
    limited = replace(robot, reach=ReachModel(x_limits_m=(-1.2, -0.6)))
    assert not InterceptionModel.reachable(limited, (-0.55, 0.0, 0.8))


# --------------------------------------------------------------- metrics, baseline, online
def test_metrics_summary():
    objects = schedulable_objects_from_specs(generate_scenario("ordered", seed=7, object_count=8), 0.12)
    m = summarize(DeadlineAwareGreedyScheduler(CONFIG).schedule(objects, default_robot_configs(), 0.0))
    assert m.objects == 8 and m.scheduled + m.missed == 8
    assert sum(m.per_robot.values()) == m.scheduled
    assert m.processing_rate_per_min > 0 and m.makespan_s > 0
    assert m.throughput_per_min is None  # no fixed observation window given
    assert sum(m.miss_reasons.values()) == m.missed
    assert 0 <= m.robust_scheduled <= m.scheduled


def test_proposed_method_is_not_worse_than_fifo_on_clutter_in_aggregate():
    totals = {method.name: 0 for method in METHODS}
    for seed in range(5):
        objects = schedulable_objects_from_specs(generate_scenario("clutter", seed=seed, object_count=10), 0.05)
        for method in METHODS:
            totals[method.name] += len(method(CONFIG).schedule(objects, default_robot_configs(), 0.0).assigned)
    assert totals["deadline_aware_greedy"] >= totals["fifo_nearest_robot"]


def test_arrival_stream_is_seeded_and_does_not_touch_global_random():
    import random
    state = random.getstate()
    a = generate_arrival_stream(seed=3, count=6, belt_speed_m_s=0.1, oversized_fraction=0.3)
    assert random.getstate() == state
    assert a == generate_arrival_stream(seed=3, count=6, belt_speed_m_s=0.1, oversized_fraction=0.3)
    assert a != generate_arrival_stream(seed=4, count=6, belt_speed_m_s=0.1, oversized_fraction=0.3)
    xs = [o.state.position_m[0] for o in a]
    assert xs[0] == -1.2 and xs == sorted(xs, reverse=True)


@pytest.mark.parametrize("method", METHODS)
def test_online_rolling_dispatch(method):
    objects = generate_arrival_stream(seed=2, count=8, belt_speed_m_s=0.15)
    robots = default_robot_configs()
    result = run_online(method(CONFIG), objects, robots, replan_period_s=0.5)
    ids = [a.object_id for a in result.committed]
    assert len(ids) == len(set(ids))
    assert result.scheduled + result.missed == len(objects)
    for robot_id in result.robot_ids:
        timeline = [a for a in result.committed if a.robot_id == robot_id]
        for previous, current in zip(timeline, timeline[1:]):
            assert current.robot_available_at_s >= previous.completion_time_s - 1e-9
    assert result == run_online(method(CONFIG), objects, robots, replan_period_s=0.5)


# --------------------------------------------------------------- adapters
@dataclass(frozen=True)
class FakeRobotState:
    robot_id: str
    simulation_time_s: float
    base_position_m: tuple
    gripper_position_m: tuple


@dataclass(frozen=True)
class FakeTarget:
    preferred_robot_id: str
    approach_position_m: tuple


@dataclass(frozen=True)
class FakeTruth:
    object_id: str
    simulation_time_s: float
    position_m: tuple
    quaternion_wxyz: tuple
    dimensions_m: tuple
    velocity_m_s: tuple | None
    motion_state: str
    geometry_type: str


def test_adapters_from_simulation_snapshots():
    states = {
        "robot_b": FakeRobotState("robot_b", 2.0, (0.55, -0.6, 0.75), (0.5, -0.8, 1.3)),
        "robot_a": FakeRobotState("robot_a", 2.0, (-0.55, -0.6, 0.75), (-0.5, -0.8, 1.3)),
    }
    targets = {"bin_a": FakeTarget("robot_a", (-1.0, -0.85, 1.05)), "bin_b": FakeTarget("robot_b", (1.0, -0.85, 1.05))}
    robots = robot_configs_from_snapshots(states, targets)
    assert [r.robot_id for r in robots] == ["robot_a", "robot_b"]
    assert robots[0].tool_position_m == (-0.5, -0.8, 1.3) and robots[0].available_at_s == 2.0
    assert robots[1].place_position_m == (1.0, -0.85, 1.05)

    yaw = 0.5
    q = (cos(yaw / 2), 0.0, 0.0, sin(yaw / 2))
    records = {
        "object_001": FakeTruth("object_001", 2.0, (-0.8, 0.0, 0.79), q, (0.06, 0.045, 0.08), (0.12, 0, 0), "on_belt", "box"),
        "object_002": FakeTruth("object_002", 2.0, (1.3, 0.0, 0.2), q, (0.05, 0.05, 0.06), (0.1, 0, -1), "falling", "cylinder"),
    }
    objects = schedulable_objects_from_ground_truth(records)
    assert [o.object_id for o in objects] == ["object_001"]
    assert objects[0].geometry.yaw_rad == pytest.approx(yaw)
    result = DeadlineAwareGreedyScheduler(CONFIG).schedule(objects, robots, 2.0)
    check_schedule_consistency(result, objects, robots)


# =============================================================== regression tests (review of 18 Nov 2026)
from math import hypot, pi, sqrt  # noqa: E402

from conveyor.scheduling import Assignment, ScheduleResult, input_observation_window_s  # noqa: E402
from conveyor.scheduling.feasibility import (  # noqa: E402
    REASON_FUTURE, REASON_NOT_SELECTED, RobotRuntime,
)


# --------------------------------------------------------------- S1: future-dated committed track
def committed_pick(object_id, robot_id="robot_a", t=6.0, position=(-0.78, 0.0, 0.78)):
    return Assignment(object_id, "assigned", "test", robot_id=robot_id, intercept_time_s=t,
                      intercept_position_m=position)


def test_future_dated_committed_track_does_not_crash_and_is_reported():
    # Original crash input: track at t=5.0, planning at now=4.9, object already committed.
    objects = [obj("o1", -0.9, t=5.0), obj("o2", -1.0, t=4.9)]
    robots = default_robot_configs()
    result = DeadlineAwareGreedyScheduler(SchedulerConfig()).schedule(objects, robots, 4.9,
                                                                     committed=[committed_pick("o1")])
    assert result.excluded_committed == (("o1", REASON_FUTURE),)
    assert set(result.by_object()) == {"o2"}  # committed objects are never re-decided
    assert all(a.intercept_time_s >= 4.9 for a in result.assigned)


def test_committed_pick_still_blocks_the_other_robot_when_its_track_is_excluded():
    robots = default_robot_configs(reach=ReachModel(max_reach_m=1.3))  # both robots reach x = -0.2
    blocked_at = (-0.2, 0.0, 0.78)
    target = obj("o2", -0.2 - 0.12 * 2.0, t=4.0)  # passes x = -0.2 around t = 6
    plan = DeadlineAwareGreedyScheduler(CONFIG).schedule(
        [obj("o1", -0.3, t=5.0), target], [robots[1]], 4.0,
        committed=[committed_pick("o1", robot_id="robot_a", t=6.0, position=blocked_at)],
    )
    assert plan.excluded_committed == (("o1", REASON_FUTURE),)
    timing = CONFIG.timing
    for a in plan.assigned:
        overlaps = (a.intercept_time_s - timing.descend_s < 6.0 + timing.gripper_close_s + timing.lift_s
                    and 6.0 - timing.descend_s < a.intercept_time_s + timing.gripper_close_s + timing.lift_s)
        if overlaps:
            distance = hypot(a.intercept_position_m[0] - blocked_at[0], a.intercept_position_m[1] - blocked_at[1])
            assert distance >= CONFIG.min_robot_separation_m


def test_future_and_stale_uncommitted_tracks_have_explicit_reasons():
    result = DeadlineAwareGreedyScheduler(CONFIG).schedule(
        [obj("future", -0.9, t=5.0), obj("stale", -0.9, t=1.0), obj("fresh", -1.0, t=4.9)],
        default_robot_configs(), 4.9)
    reasons = {a.object_id: a.reason for a in result.rejected}
    assert reasons["future"] == REASON_FUTURE and reasons["stale"] == REASON_STALE
    assert reasons.get("fresh") not in (REASON_FUTURE, REASON_STALE)


def test_valid_committed_tracks_are_still_used():
    objects = [obj("o1", -1.0, t=4.8), obj("o2", -1.1, t=4.9)]
    result = DeadlineAwareGreedyScheduler(CONFIG).schedule(objects, default_robot_configs(), 4.9,
                                                          committed=[committed_pick("o1", t=7.0)])
    assert result.excluded_committed == ()
    assert set(result.by_object()) == {"o2"}


# --------------------------------------------------------------- S2: meaningful rejection reasons
def test_unassigned_but_feasible_objects_are_reported_as_not_selected():
    # Original case: FIFO, clutter-8, seed 6, 0.05 m/s, dt 0.05 -> object_007 had reason "".
    objects = schedulable_objects_from_specs(generate_scenario("clutter", seed=6, object_count=8), 0.05)
    scheduler = FifoNearestRobotScheduler(CONFIG)
    result = scheduler.schedule(objects, default_robot_configs(), 0.0)
    reasons = {a.object_id: a.reason for a in result.rejected}
    assert reasons["object_007"] == REASON_NOT_SELECTED
    # Not artificially "infeasible": the label is used exactly when an option still exists.
    for object_id, reason in reasons.items():
        has_option = any(scheduler._options(r, object_id)[0] is not None for r in scheduler.runtime)
        assert (reason == REASON_NOT_SELECTED) == has_option


@pytest.mark.parametrize("seed", range(10))
def test_rejection_reasons_are_never_empty(seed):
    objects = schedulable_objects_from_specs(generate_scenario("clutter", seed=seed, object_count=8), 0.05)
    for method in METHODS:
        result = method(CONFIG).schedule(objects, default_robot_configs(), 0.0)
        assert all(a.reason for a in result.rejected)
        if method is DeadlineAwareGreedyScheduler:  # greedy stops only when nothing is feasible
            assert all(a.reason != REASON_NOT_SELECTED for a in result.rejected)


# --------------------------------------------------------------- S4: consistent slack
def brute_force_latest_start(model, robot, object_id, intercept_time):
    """Independent max over the reach window of (t - travel(t)) with same-time terms."""
    mask = model.windows[(robot.robot_id, object_id)]
    index = model.times.index(intercept_time)
    end = index
    while end + 1 < len(mask) and mask[end + 1]:
        end += 1
    best = max(
        model.times[k] - model.travel_time(
            robot.tool_position_m, model.tcp_position(object_id, model.predict(object_id, model.times[k])))
        for k in range(index, end + 1)
    )
    return best, model.times[end]


def test_slack_uses_travel_time_at_the_deadline_not_at_the_interception():
    # Review example (margin-free reach model as in the review): reported 5.393 s, consistent ~5.299 s.
    robot = default_robot_configs(("robot_a",), reach=ReachModel(safety_margin_m=0.0))[0]
    model = InterceptionModel(SchedulerConfig(), [obj("o1", -1.1, y=0.15, dims=(0.05, 0.04, 0.06))], [robot], 0.0)
    runtime = RobotRuntime(robot, 0.0, robot.tool_position_m)
    option, _ = model.find_interception(runtime, "o1", [])
    latest, deadline = brute_force_latest_start(model, runtime, "o1", option.intercept_time_s)
    assert option.deadline_s == deadline
    assert option.latest_start_s == pytest.approx(latest, abs=1e-12)
    assert option.slack_s == pytest.approx(latest - runtime.available_at_s, abs=1e-12)
    assert option.slack_s == pytest.approx(5.299, abs=2e-3)
    old_definition = option.deadline_s - option.available_at_s - option.travel_time_s
    assert old_definition == pytest.approx(5.393, abs=2e-3)


@pytest.mark.parametrize("available", [0.0, 1.0, 2.5, 4.0])
@pytest.mark.parametrize("speed", [0.05, 0.12, 0.6])  # 0.6 m/s > cartesian speed: full window scan
def test_slack_matches_brute_force_for_several_availabilities_and_speeds(available, speed):
    robot = default_robot_configs(("robot_a",))[0]
    model = InterceptionModel(SchedulerConfig(), [obj("o1", -1.2, speed=speed)], [robot], 0.0)
    runtime = RobotRuntime(robot, available, robot.tool_position_m)
    option, reason = model.find_interception(runtime, "o1", [])
    if option is None:
        assert reason == REASON_TIME
        return
    latest, _ = brute_force_latest_start(model, runtime, "o1", option.intercept_time_s)
    assert option.latest_start_s == pytest.approx(latest, abs=1e-12)
    assert option.slack_s == pytest.approx(latest - available, abs=1e-12)
    assert option.intercept_time_s - option.travel_time_s <= option.latest_start_s + 1e-12


def test_latest_start_is_independent_of_availability_within_one_window():
    robot = default_robot_configs(("robot_a",))[0]
    model = InterceptionModel(SchedulerConfig(), [obj("o1", -1.15, speed=0.08)], [robot], 0.0)
    options = [model.find_interception(RobotRuntime(robot, t, robot.tool_position_m), "o1", [])[0]
               for t in (0.0, 1.0, 2.0)]
    assert all(o is not None for o in options)
    assert len({round(o.latest_start_s, 12) for o in options}) == 1
    assert [o.slack_s for o in options] == pytest.approx([options[0].slack_s - d for d in (0.0, 1.0, 2.0)])


# --------------------------------------------------------------- reach safety margin
def sphere_slack(robot, tcp):
    reach = robot.reach
    bx, by, bz = robot.base_position_m
    horizontal = hypot(tcp[0] - bx, tcp[1] - by)
    dz = tcp[2] + reach.tool_length_m - (bz + reach.shoulder_height_m)
    return reach.max_reach_m - sqrt(horizontal ** 2 + dz ** 2)


def test_default_reach_model_keeps_a_calibrated_safety_margin():
    assert ReachModel().safety_margin_m == pytest.approx(0.03)
    robot = default_robot_configs(("robot_a",))[0]
    nominal = replace(robot, reach=ReachModel(safety_margin_m=0.0))
    seen_band = seen_inside = False
    for x in [i / 1000 for i in range(-200, 100)]:  # walk along y = 0 across robot A's boundary
        tcp = (x, 0.0, 0.785)
        slack = sphere_slack(nominal, tcp)
        if 0.0 <= slack < 0.03 - 1e-6:
            seen_band = True
            assert InterceptionModel.reachable(nominal, tcp) and not InterceptionModel.reachable(robot, tcp)
        elif slack >= 0.03 + 1e-6:
            seen_inside = True
            assert InterceptionModel.reachable(robot, tcp)
    assert seen_band and seen_inside


def test_overlapping_workspace_survives_the_safety_margin():
    robot_a, robot_b = default_robot_configs()
    shared = [(x / 100, y / 100, 0.785) for x in range(-30, 31) for y in range(-20, 21)
              if InterceptionModel.reachable(robot_a, (x / 100, y / 100, 0.785))
              and InterceptionModel.reachable(robot_b, (x / 100, y / 100, 0.785))]
    assert shared  # both arms still reach the middle of the belt near its robot-side edge
    assert all(abs(p[0]) <= 0.2 and p[1] < 0.05 for p in shared)


# --------------------------------------------------------------- metrics
def fake_assignment(object_id, t, completion):
    return Assignment(object_id, "assigned", "m", robot_id="robot_a", intercept_time_s=t,
                      intercept_position_m=(0, 0, 0.78), completion_time_s=completion, slack_s=1.0)


def test_processing_rate_can_favour_fewer_picks_but_fixed_window_throughput_does_not():
    more = ScheduleResult("more", 0.0, ("robot_a",), (fake_assignment("a", 2, 5), fake_assignment("b", 7, 10)))
    fewer = ScheduleResult("fewer", 0.0, ("robot_a",), (fake_assignment("a", 1, 3),
                                                       Assignment("b", "rejected", "fewer", reason="x")))
    m_more, m_fewer = (summarize(r, observation_window_s=20.0) for r in (more, fewer))
    assert m_fewer.processing_rate_per_min > m_more.processing_rate_per_min  # the misleading old metric
    assert m_more.throughput_per_min == pytest.approx(6.0) and m_fewer.throughput_per_min == pytest.approx(3.0)
    assert m_more.observation_window_s == m_fewer.observation_window_s == 20.0
    with pytest.raises(ValueError):
        summarize(more, observation_window_s=0.0)


def test_observation_window_is_derived_from_inputs_only():
    objects = schedulable_objects_from_specs(generate_scenario("ordered", seed=7, object_count=8), 0.12)
    window = input_observation_window_s(objects, 0.0)
    first_x = min(o.state.position_m[0] for o in objects)
    assert window == pytest.approx((1.15 - first_x) / 0.12)
    for method in METHODS:
        m = summarize(method(CONFIG).schedule(objects, default_robot_configs(), 0.0), observation_window_s=window)
        assert m.throughput_per_min == pytest.approx(60.0 * m.scheduled / window)
    assert input_observation_window_s([obj("s", 0.0, speed=0.0)], 0.0) is None


def test_online_throughput_uses_the_same_window_for_both_methods():
    objects = generate_arrival_stream(seed=2, count=8, belt_speed_m_s=0.15)
    results = [run_online(m(CONFIG), objects, default_robot_configs(), replan_period_s=0.5) for m in METHODS]
    assert results[0].observation_window_s == results[1].observation_window_s > 0
    for r in results:
        assert r.throughput_per_min == pytest.approx(60.0 * r.scheduled / r.observation_window_s)
        assert r.processing_rate_per_min == pytest.approx(60.0 * r.scheduled / r.makespan_s)


# --------------------------------------------------------------- scenario coverage after the fixes
def test_one_busy_robot_while_the_other_is_available():
    robots = default_robot_configs()
    busy_b = [robots[0], replace(robots[1], available_at_s=100.0)]
    objects = [obj("a", -1.0), obj("b", -0.8, y=-0.1)]
    result = DeadlineAwareGreedyScheduler(CONFIG).schedule(objects, busy_b, 0.0)
    check_schedule_consistency(result, objects, busy_b)
    assert result.assigned and all(a.robot_id == "robot_a" for a in result.assigned)


def test_objects_close_to_their_deadline():
    robots = default_robot_configs()
    for x in (0.70, 0.80, 0.85, 0.95):
        o = obj(f"x{x}", x)
        result = DeadlineAwareGreedyScheduler(CONFIG).schedule([o], robots, 0.0)
        check_schedule_consistency(result, [o], robots)
        (a,) = result.assignments
        if a.is_assigned:
            assert a.intercept_time_s <= a.deadline_s and a.slack_s >= 0
        else:
            assert a.reason in (REASON_TIME, REASON_UNREACHABLE)
    assert DeadlineAwareGreedyScheduler(CONFIG).schedule([obj("early", 0.70)], robots, 0.0).assigned
    assert not DeadlineAwareGreedyScheduler(CONFIG).schedule([obj("late", 0.95)], robots, 0.0).assigned


@pytest.mark.parametrize("speed", [0.01, 0.02, 0.25, 0.4])
@pytest.mark.parametrize("method", METHODS)
def test_low_and_high_belt_speeds_produce_consistent_schedules(speed, method):
    objects = schedulable_objects_from_specs(generate_scenario("ordered", seed=3, object_count=6), speed)
    robots = default_robot_configs()
    result = method(CONFIG).schedule(objects, robots, 0.0)
    check_schedule_consistency(result, objects, robots)
    assert all(a.grasp.motion_alignment is not None for a in result.assigned)  # motion-aware ranking used


def test_scheduled_cylinder_grasp_closes_across_the_belt_with_rai_orientation():
    objects = [obj("cyl", -1.0, kind="cylinder", dims=(0.05, 0.05, 0.06))]
    grasp = DeadlineAwareGreedyScheduler(CONFIG).schedule(objects, default_robot_configs(), 0.0).assigned[0].grasp
    assert abs(grasp.closing_axis_world[1]) == pytest.approx(1.0, abs=1e-9)
    assert grasp.motion_alignment == pytest.approx(1.0)
    assert grasp.quaternion_wxyz == pytest.approx((cos(pi / 4), 0.0, 0.0, sin(pi / 4)), abs=1e-9)
