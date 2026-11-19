"""Run two Panda pick-and-place assignments on a shared, validated timeline."""

import argparse
from math import isfinite
from time import monotonic, sleep

from conveyor.control.dual_pandas import play_camera_panda_batch, play_panda_batch, prepare_panda_batch
from conveyor.planning.panda import PandaPlanningError
from conveyor.simulation import ConveyorSimulation
from conveyor.simulation.viewer import make_close_checker


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-view", action="store_true", help="run one batch without opening the viewer")
    parser.add_argument("--serial", action="store_true", help="preview and replay one arm at a time")
    parser.add_argument("--perception", action="store_true", help="schedule from live RGB-D tracks (requires X11)")
    parser.add_argument("--speed", type=float, default=0.022, help="belt speed in m/s")
    parser.add_argument("--change-speed-to", type=float, help="change the belt speed (m/s) at 0.75 s in camera mode")
    parser.add_argument("--scenario", choices=("ordered", "sparse", "clutter"), default="sparse")
    parser.add_argument("--seed", type=int, default=21, help="layout seed")
    args = parser.parse_args()
    if not isfinite(args.speed) or args.speed <= 0:
        parser.error("--speed must be finite and positive")
    if args.serial and args.perception:
        parser.error("--perception currently uses concurrent motion")
    if args.change_speed_to is not None:
        if not args.perception:
            parser.error("--change-speed-to requires --perception")
        if not isfinite(args.change_speed_to) or args.change_speed_to <= 0.0:
            parser.error("--change-speed-to must be finite and positive")

    simulation = ConveyorSimulation(args.speed, scenario=args.scenario, seed=args.seed, object_count=2, exit_mode="remove")
    try:
        mode = "RGB-D concurrent" if args.perception else "Serialized" if args.serial else "Concurrent"
        print(f"Preparing {mode.lower()} motions and checking the shared scene...")
        batch = prepare_panda_batch(simulation, serial=args.serial, perception=args.perception)
        print(f"{len(batch.frames)} checked frames | both arms moving {batch.overlap_s:.2f} s")
        if not args.no_view:
            simulation.view(f"Two Pandas | {mode} coordination")
            viewer = simulation.config.viewer()
            viewer.focus([0.0, -0.38, 0.95], 2.8)
            viewer.visualsOnly()
            should_close = make_close_checker(simulation.config)

            def render(message):
                return not should_close() and simulation.view(message) not in (27, ord("q"), ord("Q")) and not should_close()

        def after_step(now):
            nonlocal last_draw, speed_changed
            if args.change_speed_to is not None and not speed_changed and now >= 0.75:
                simulation.belt_speed_m_s = args.change_speed_to
                speed_changed = True
                print(f"Belt speed changed to {args.change_speed_to:.3f} m/s at {now:.2f} s", flush=True)
            if args.no_view:
                return True
            sleep(max(0.0, started + now - monotonic()))
            if now - last_draw < 1 / 60:
                return not should_close()
            last_draw = now
            placed = sum(
                obj.motion_state == "released" and simulation.config.getFrame(obj.object_id).getParent().name == "world"
                for obj in simulation.get_ground_truth(active_only=True).values()
            )
            return render(f"Two Pandas | {mode} | placed {placed}/{len(batch.jobs)} | t = {now:.2f} s")

        def on_replanning(now, active):
            nonlocal started, replan_started
            if active:
                replan_started = monotonic()
                print(f"Updated tracks at {now:.2f} s, checking replacement trajectories...", flush=True)
                if not args.no_view:
                    render(f"Two Pandas | Replanning from RGB-D | t = {now:.2f} s")
            else:
                elapsed = monotonic() - replan_started
                started += elapsed
                print(f"Replacement trajectories validated in {elapsed:.2f} wall-clock s", flush=True)

        while True:
            started = monotonic()
            last_draw = -1.0
            replan_started = 0.0
            speed_changed = False
            if args.perception:
                playback = play_camera_panda_batch(
                    simulation, batch, after_step=after_step, on_replanning=on_replanning,
                )
                if playback is None:
                    return 0
                print(f"RGB-D approach | {playback.camera_frames} frames | {len(playback.replanned_at_s)} replans")
                jobs = playback.jobs
            else:
                if not play_panda_batch(simulation, batch, after_step=after_step):
                    return 0
                jobs = batch.jobs
            for item in jobs:
                item_label = f"{item.track_id} ({item.object_id})" if item.track_id else item.object_id
                print(f"{item.robot_id} placed {item_label} in {item.placement.target_id} at {item.placement.release_time_s:.2f} s")
            message = f"Placed {len(jobs)}/{len(simulation.object_ids)} | {simulation.simulation_time_s:.2f} s"
            print(message)
            if args.no_view:
                return 0
            replay_at = monotonic() + 3.0
            while monotonic() < replay_at:
                if not render(message):
                    return 0
                sleep(0.05)
            simulation.belt_speed_m_s = args.speed
            simulation.reset()
    except (PandaPlanningError, ValueError) as exc:
        print(f"Batch rejected ({exc})")
        return 1
    except KeyboardInterrupt:
        return 0
    finally:
        simulation.close()


if __name__ == "__main__":
    raise SystemExit(main())
