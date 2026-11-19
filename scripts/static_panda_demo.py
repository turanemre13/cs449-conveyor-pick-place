"""Use KOMO to move one Panda to a static TCP pose in the dual-arm scene."""

import argparse
from math import degrees, sqrt
from time import monotonic, sleep

from conveyor.control.static_panda import execute_static_panda
from conveyor.planning.static_panda import StaticPlanningError, TcpPose, plan_static_panda
from conveyor.simulation import ConveyorSimulation
from conveyor.simulation.viewer import make_close_checker


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-view", action="store_true", help="run one validated motion without a viewer")
    args = parser.parse_args()
    target = TcpPose((-0.55, 0.0, 0.90), (sqrt(0.5), 0.0, 0.0, sqrt(0.5)))
    simulation = ConveyorSimulation(belt_speed_m_s=0.0)
    try:
        robot = simulation.get_robot_states()["robot_a"]
        path = plan_static_panda(simulation.config, robot, target)
        print(f"{robot.robot_id} | {len(path.samples)} checked samples | peak joint speed {path.peak_joint_speed_rad_s:.3f} rad/s")
        print(f"Minimum clearance {path.min_clearance_m * 1000:.1f} mm ({' / '.join(path.closest_pair)})")

        after_step = None
        if not args.no_view:
            simulation.view("Panda A | static KOMO target")
            viewer = simulation.config.viewer()
            viewer.focus([-0.55, -0.22, 1.04], 1.8)
            viewer.visualsOnly()
            should_close = make_close_checker(simulation.config)

            def render(message):
                return not should_close() and simulation.view(message) not in (27, ord("q"), ord("Q")) and not should_close()

            def after_step(elapsed):
                sleep(max(0.0, start + elapsed - monotonic()))
                return render(f"Panda A | static KOMO target | t = {elapsed:.2f} s")

        while True:
            start = monotonic()
            error = execute_static_panda(simulation, path, after_step=after_step)
            if error is None:
                return 0
            status = "Reached goal" if error.within_tolerance else "TCP pose outside tolerance"
            message = (f"{status} | TCP error {error.position_m * 1000:.3f} mm"
                       f" | orientation error {degrees(error.orientation_rad):.3f} deg")
            print(message)
            if args.no_view or not error.within_tolerance:
                return int(not error.within_tolerance)
            replay_at = monotonic() + 3.0
            while monotonic() < replay_at:
                if not render(message):
                    return 0
                sleep(0.05)
            simulation.reset()
    except (StaticPlanningError, ValueError) as exc:
        print(f"Static motion rejected ({exc})")
        return 1
    except KeyboardInterrupt:
        return 0
    finally:
        simulation.close()


if __name__ == "__main__":
    raise SystemExit(main())
