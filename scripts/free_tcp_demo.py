"""Replay a single scheduled conveyor interception with a free Cartesian tool."""

import argparse
from math import isfinite
from time import monotonic, sleep

from conveyor.control.free_tcp import evaluate_interception
from conveyor.planning.free_tcp import TcpSample, plan_free_tcp
from conveyor.scheduling import (
    DeadlineAwareGreedyScheduler, robot_configs_from_snapshots, schedulable_objects_from_ground_truth,
)
from conveyor.simulation import ConveyorSimulation
from conveyor.simulation.viewer import make_close_checker


TCP_FRAME = "free_tcp"


def add_tool(config, opening_m):
    """Add visible jaws in the assigned TCP frame."""
    import robotic as ry

    config.addFrame(TCP_FRAME, "world")
    finger_thickness_m = 0.012
    finger_x = (opening_m + finger_thickness_m) / 2
    for name, position, size in (
        ("palm", (0, 0, 0.055), (opening_m + 0.04, 0.036, 0.025)),
        ("left_jaw", (-finger_x, 0, 0.019), (finger_thickness_m, 0.022, 0.058)),
        ("right_jaw", (finger_x, 0, 0.019), (finger_thickness_m, 0.022, 0.058)),
    ):
        config.addFrame(f"{TCP_FRAME}_{name}", TCP_FRAME).setRelativePosition(position).setShape(
            ry.ST.ssBox, [*size, 0.003],
        ).setColor([0.08, 0.76, 0.72]).setContact(0)


def run_interception(simulation, plan, render=None):
    config = simulation.config
    config.getFrame(TCP_FRAME).setQuaternion(plan.quaternion_wxyz)
    config.getFrame(TCP_FRAME).setPosition(plan.position_at(simulation.simulation_time_s))
    clock_start = monotonic()
    samples = []
    for event_s in (plan.tracking_start_s, plan.intercept_time_s):
        while simulation.simulation_time_s < event_s:
            simulation.step(min(0.02, event_s - simulation.simulation_time_s))
            now_s = simulation.simulation_time_s
            # Removal at the belt edge can rebuild the native frame handles.
            config.getFrame(TCP_FRAME).setPosition(plan.position_at(now_s))
            if render is not None:
                if not render(f"Free TCP | t = {now_s:.2f} s | Intercept at {plan.intercept_time_s:.2f} s"):
                    return None
                sleep(max(0.0, clock_start + now_s - monotonic()))
        position = tuple(config.getFrame(TCP_FRAME).getPosition())
        samples.append(TcpSample(simulation.simulation_time_s, position))
    target = simulation.get_ground_truth(active_only=True).get(plan.object_id)
    return evaluate_interception(plan, samples[0], samples[1], target)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-view", action="store_true", help="run one interception without a viewer")
    parser.add_argument("--speed", type=float, default=0.12, help="belt speed in m/s")
    args = parser.parse_args()
    if not isfinite(args.speed) or args.speed < 0:
        parser.error("--speed must be finite and non-negative")

    simulation = ConveyorSimulation(belt_speed_m_s=args.speed, scenario="single", exit_mode="remove")
    try:
        now_s = simulation.simulation_time_s
        robot_states = simulation.get_robot_states()
        robots = robot_configs_from_snapshots(robot_states, simulation.get_targets())
        objects = schedulable_objects_from_ground_truth(simulation.get_ground_truth(active_only=True))
        scheduler = DeadlineAwareGreedyScheduler()
        schedule = scheduler.schedule(objects, robots, now_s)
        if not schedule.assigned:
            print(f"Interception rejected ({schedule.rejected[0].reason})")
            return 1

        assignment = schedule.assigned[0]
        robot = robot_states[assignment.robot_id]
        timing = scheduler.config.timing
        plan = plan_free_tcp(
            assignment, objects[0].state, TcpSample(now_s, robot.gripper_position_m),
            max_speed_m_s=timing.cartesian_speed_m_s, descend_s=timing.descend_s,
        )
        add_tool(simulation.config, assignment.grasp.required_opening_m)
        simulation.config.getFrame(TCP_FRAME).setPosition(plan.position_at(now_s))
        print(f"{assignment.robot_id} / {plan.object_id} / scheduled at {plan.intercept_time_s:.2f} s")

        render = None
        if not args.no_view:
            simulation.view("Free TCP interception")
            simulation.config.viewer().focus([-0.55, -0.2, 1.0], 1.8)
            simulation.config.viewer().visualsOnly()
            should_close = make_close_checker(simulation.config)

            def render(message):
                return not should_close() and simulation.view(message) not in (27, ord("q"), ord("Q")) and not should_close()

        while True:
            result = run_interception(simulation, plan, render)
            if result is None:
                return 0
            message = f"{result.reason} | time error {result.time_error_s * 1000:.3f} ms"
            if result.position_error_m is not None and result.velocity_error_m_s is not None:
                message += (f" | TCP error {result.position_error_m * 1000:.3f} mm"
                            f" | relative speed {result.velocity_error_m_s * 1000:.3f} mm/s")
            print(message)
            if render is None or not result.success:
                return int(not result.success)
            replay_at = monotonic() + 3.0
            while monotonic() < replay_at:
                if not render(message + " | Replay in a moment"):
                    return 0
                sleep(0.05)
            simulation.reset()
    except ValueError as exc:
        print(f"Interception rejected ({exc})")
        return 1
    except KeyboardInterrupt:
        return 0
    finally:
        simulation.close()


if __name__ == "__main__":
    raise SystemExit(main())
