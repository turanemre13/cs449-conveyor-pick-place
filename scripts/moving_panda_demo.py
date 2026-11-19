"""Move one Panda to intercept a scheduled box on the running conveyor."""

import argparse
from math import degrees, isfinite
from time import monotonic, sleep

from conveyor.control.moving_panda import execute_moving_panda
from conveyor.control.panda_pickup import execute_panda_pickup
from conveyor.control.panda_placement import execute_panda_placement
from conveyor.planning.moving_panda import plan_moving_panda
from conveyor.planning.panda import PandaPlanningError
from conveyor.scheduling import (
    DeadlineAwareGreedyScheduler, robot_configs_from_snapshots, schedulable_objects_from_ground_truth,
)
from conveyor.simulation import ConveyorSimulation
from conveyor.simulation.viewer import make_close_checker


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-view", action="store_true", help="run one interception without a viewer")
    parser.add_argument("--speed", type=float, default=0.12, help="belt speed in m/s")
    parser.add_argument("--lift", action="store_true", help="close the jaws and lift after interception")
    parser.add_argument("--place", action="store_true", help="pick, lift, and place the box in the assigned bin")
    parser.add_argument("--perception", action="store_true", help="schedule from live RGB-D tracking (requires X11)")
    args = parser.parse_args()
    if not isfinite(args.speed) or args.speed <= 0:
        parser.error("--speed must be finite and positive")

    simulation = ConveyorSimulation(belt_speed_m_s=args.speed, scenario="single", exit_mode="remove")
    try:
        pick_box = args.lift or args.place
        if args.perception:
            from conveyor.control.rgbd_panda import plan_camera_pickup

            camera_plan = plan_camera_pickup(simulation, close_jaws=pick_box)
            plan = camera_plan.motion
            print(f"RGB-D track {camera_plan.track_id} bound to {plan.object_id}")
        else:
            robots = simulation.get_robot_states()
            scheduler_robots = robot_configs_from_snapshots(robots, simulation.get_targets())
            objects = schedulable_objects_from_ground_truth(simulation.get_ground_truth(active_only=True))
            scheduler = DeadlineAwareGreedyScheduler()
            schedule = scheduler.schedule(objects, scheduler_robots, simulation.simulation_time_s)
            if not schedule.assigned:
                raise PandaPlanningError(schedule.rejected[0].reason)
            assignment = schedule.assigned[0]
            plan = plan_moving_panda(
                simulation.config, robots[assignment.robot_id], assignment, objects[0].state,
                close_duration_s=scheduler.config.timing.gripper_close_s if pick_box else 0.0,
            )
        print(f"{plan.robot_id} | {plan.object_id} | interception at {plan.intercept_time_s:.2f} s")
        print(f"{len(plan.samples)} validated samples | minimum clearance {plan.min_clearance_m * 1000:.2f} mm")

        after_step = None
        placing = False
        if not args.no_view:
            simulation.view("Panda | moving box interception")
            viewer = simulation.config.viewer()
            viewer.focus([-0.73, -0.38, 0.95] if args.place else [-0.70, -0.12, 0.98], 2.2 if args.place else 1.8)
            viewer.visualsOnly()
            should_close = make_close_checker(simulation.config)

            def render(message):
                return not should_close() and simulation.view(message) not in (27, ord("q"), ord("Q")) and not should_close()

            def after_step(now):
                sleep(max(0.0, start + now - monotonic()))
                phase = "Intercepting"
                if placing:
                    phase = "Placing in bin"
                elif now > plan.end_time_s + 1e-6:
                    phase = "Lifting"
                elif now > plan.intercept_time_s + 1e-6:
                    phase = "Closing jaws"
                return render(f"Panda | {phase} | t = {now:.2f} s")

        while True:
            start = monotonic() - simulation.simulation_time_s
            pickup = None
            placement = None
            placing = False
            if pick_box:
                pickup = execute_panda_pickup(simulation, plan, after_step=after_step)
                if pickup is None:
                    return 0
                result = pickup.interception
                if args.place:
                    placing = True
                    placement = execute_panda_placement(simulation, plan, after_step=after_step)
                    if placement is None:
                        return 0
            else:
                result = execute_moving_panda(simulation, plan, after_step=after_step)
                if result is None:
                    return 0
            measured = result.alignment
            if placement is not None:
                status = f"Placed in {placement.target_id} | released at {placement.release_time_s:.2f} s"
            elif pickup is not None:
                status = f"Lifted box {pickup.lift_height_m * 1000:.1f} mm | grasp at {pickup.pickup_time_s:.2f} s"
            elif result.success:
                status = "Intercepted"
            elif measured.success:
                status = "Interception failed (orientation_error)"
            else:
                status = f"Interception failed ({measured.reason})"
            message = f"{status} | time error {measured.time_error_s * 1000:.3f} ms"
            if measured.position_error_m is not None and measured.velocity_error_m_s is not None:
                message += (f" | TCP error {measured.position_error_m * 1000:.2f} mm"
                            f" | relative speed {measured.velocity_error_m_s * 1000:.2f} mm/s"
                            f" | orientation error {degrees(result.orientation_error_rad):.2f} deg")
            print(message)
            if args.no_view or not result.success:
                return int(not result.success)
            replay_at = monotonic() + 3.0
            while monotonic() < replay_at:
                if not render(message):
                    return 0
                sleep(0.05)
            simulation.reset()
            if plan.planned_at_s > 0.0:
                simulation.step(plan.planned_at_s)
    except (PandaPlanningError, ValueError) as exc:
        print(f"Motion rejected ({exc})")
        return 1
    except KeyboardInterrupt:
        return 0
    finally:
        simulation.close()


if __name__ == "__main__":
    raise SystemExit(main())
