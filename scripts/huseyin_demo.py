"""Demo of Hüseyin's modules: prediction -> grasp selection -> dual-arm scheduling.

Runs from a plain terminal without RAI or a viewer::

    python scripts/huseyin_demo.py
    python scripts/huseyin_demo.py --scenario clutter --objects 10 --speed 0.08 --seed 3
    python scripts/huseyin_demo.py --robots robot_a       # one-robot configuration

Object layouts come from the repository's seeded ``generate_scenario`` (the same
layouts the RAI simulation spawns), used here as ground truth in place of the
tracker. All schedule numbers are planning-level estimates.
"""

from __future__ import annotations

import argparse
from math import degrees

from conveyor.grasping import select_top_down_grasps
from conveyor.prediction import ConstantVelocityPredictor
from conveyor.scheduling import (
    DeadlineAwareGreedyScheduler, FifoNearestRobotScheduler, SchedulerConfig, default_robot_configs,
    input_observation_window_s, schedulable_objects_from_specs, summarize,
)
from conveyor.simulation.scenarios import generate_scenario


def fmt(vector, digits=3):
    return "(" + ", ".join(f"{v:+.{digits}f}" for v in vector) + ")"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--scenario", default="ordered", choices=("single", "ordered", "sparse", "clutter"))
    parser.add_argument("--objects", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--speed", type=float, default=0.12, help="belt speed in m/s")
    parser.add_argument("--robots", nargs="+", default=["robot_a", "robot_b"], choices=("robot_a", "robot_b"))
    args = parser.parse_args()

    specs = generate_scenario(args.scenario, seed=args.seed, object_count=args.objects)
    objects = schedulable_objects_from_specs(specs, args.speed)
    robots = default_robot_configs(args.robots)
    config = SchedulerConfig()

    print(f"=== Scenario {args.scenario!r}, {len(objects)} objects, seed {args.seed}, belt {args.speed} m/s ===")
    print("Input (ground truth standing in for the tracker, t = 0 s):")
    for o in objects:
        g = o.geometry
        print(f"  {o.object_id}  {g.geometry_type:8s} dims={fmt(g.dimensions_m)} m  yaw={degrees(g.yaw_rad):+6.1f} deg"
              f"  p={fmt(o.state.position_m)}  v={fmt(o.state.velocity_m_s)}")

    print("\n--- Task A: constant-velocity prediction p(t+tau) = p(t) + v*tau ---")
    predictor = ConstantVelocityPredictor(planar_motion=True)
    first = objects[0].state
    for tau in (0.0, 1.0, 2.0, 5.0):
        print(f"  {first.object_id} at t={tau:4.1f} s -> {fmt(predictor.predict(first, tau).position_m)}")
    exit_time = predictor.time_to_reach_coordinate(first, 0, config.workspace.pick_x_limits_m[1])
    print(f"  {first.object_id} leaves the pickable belt region at t = "
          f"{'never (stationary)' if exit_time is None else f'{exit_time:.2f} s'}")

    print("\n--- Task B: top-down grasp candidates at t = 0 (with neighbours, belt motion preferred across) ---")
    print("  quaternion = RAI Panda 'gripper' frame (w,x,y,z): x = closing axis, z = towards wrist (up)")
    for o in objects[: min(4, len(objects))]:
        neighbors = [(n.geometry, n.state.position_m) for n in objects if n is not o]
        result = select_top_down_grasps(o.geometry, o.state.position_m, neighbors=neighbors, config=config.gripper,
                                        object_velocity_m_s=o.state.velocity_m_s)
        print(f"  {o.object_id}: {len(result.feasible)} feasible, {len(result.rejected)} rejected")
        if result.best:
            b = result.best
            motion = "-" if b.motion_alignment is None else f"{b.motion_alignment:.2f}"
            print(f"    best {b.grasp_type}: TCP={fmt(b.position_m)} yaw={degrees(b.gripper_yaw_rad):+6.1f} deg "
                  f"opening={1000 * b.required_opening_m:.1f} mm score={b.score:.3f} |sin(closing,motion)|={motion}")
            print(f"      RAI gripper quaternion (w,x,y,z) = {fmt(b.quaternion_wxyz)}")
        for r in result.rejected[:2]:
            print(f"    rejected {r.grasp_type} yaw={degrees(r.gripper_yaw_rad):+6.1f} deg: {', '.join(r.rejection_reasons)}")

    print("\n--- Task C/D: dual-arm scheduling (estimates, not executed) ---")
    window = input_observation_window_s(objects, 0.0, config.workspace)
    for method in (DeadlineAwareGreedyScheduler, FifoNearestRobotScheduler):
        result = method(config).schedule(objects, robots, 0.0)
        print(f"\n[{result.method}]")
        print("  robot    object      intercept t   intercept position        slack   done at   grasp")
        for a in result.assigned:
            print(f"  {a.robot_id}  {a.object_id}  {a.intercept_time_s:8.2f} s   {fmt(a.intercept_position_m)}"
                  f"  {a.slack_s:5.2f} s  {a.completion_time_s:6.2f} s  {a.grasp.grasp_type}")
        for a in result.rejected:
            exit_text = "-" if a.object_exit_time_s is None else f"{a.object_exit_time_s:.2f} s"
            print(f"  REJECTED {a.object_id}: {a.reason} (leaves belt at {exit_text})")
        m = summarize(result, observation_window_s=window)
        rate = "n/a (no moving objects)" if m.throughput_per_min is None else f"{m.throughput_per_min:.1f} obj/min"
        print(f"  summary: scheduled {m.scheduled}/{m.objects}, missed {m.missed}, per robot {m.per_robot}, "
              f"throughput {rate} over a {window or 0:.1f} s window, robust {m.robust_scheduled}")


if __name__ == "__main__":
    main()
