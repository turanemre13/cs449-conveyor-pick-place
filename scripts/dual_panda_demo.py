"""Run two Panda pick-and-place assignments on a shared, validated timeline."""

import argparse
from math import isfinite
from time import monotonic, sleep

from conveyor.control.dual_pandas import play_camera_panda_batch, play_panda_batch, prepare_panda_batch
from conveyor.control.feed_pandas import prepare_feed_run
from conveyor.planning.panda import PandaPlanningError
from conveyor.simulation import ConveyorSimulation
from conveyor.simulation.scenarios import MAX_OBJECT_COUNT
from conveyor.simulation.viewer import make_close_checker


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-view", action="store_true", help="run one batch without opening the viewer")
    parser.add_argument("--serial", action="store_true", help="preview and replay one arm at a time")
    parser.add_argument("--stream", action="store_true", help="sort successive products arriving from an upstream belt")
    parser.add_argument("--objects", type=int, help="number of queued products in stream mode (default 4)")
    parser.add_argument("--perception", action="store_true", help="schedule from live RGB-D tracks (requires X11)")
    parser.add_argument("--speed", type=float, help="belt speed in m/s (default 0.12 for a stream, otherwise 0.022)")
    parser.add_argument("--change-speed-to", type=float, help="change the belt speed (m/s) at 0.75 s in camera mode")
    parser.add_argument("--scenario", choices=("ordered", "sparse", "clutter"), default="sparse")
    parser.add_argument("--seed", type=int, help="layout seed (default 7 for a stream, otherwise 21)")
    args = parser.parse_args()
    if args.speed is None:
        args.speed = 0.12 if args.stream else 0.022
    if args.seed is None:
        args.seed = 7 if args.stream else 21
    if not isfinite(args.speed) or args.speed <= 0:
        parser.error("--speed must be finite and positive")
    if args.serial and args.perception:
        parser.error("--perception currently uses concurrent motion")
    if args.stream and (args.serial or args.change_speed_to is not None):
        parser.error("--stream has its own dispatch loop and uses a fixed belt speed")
    if args.objects is not None and (not args.stream or not 1 <= args.objects <= MAX_OBJECT_COUNT):
        parser.error(f"--objects requires --stream and a count from 1 to {MAX_OBJECT_COUNT}")
    if args.change_speed_to is not None:
        if not args.perception:
            parser.error("--change-speed-to requires --perception")
        if not isfinite(args.change_speed_to) or args.change_speed_to <= 0.0:
            parser.error("--change-speed-to must be finite and positive")

    scenario = "feed" if args.stream else args.scenario
    count = (args.objects or 4) if args.stream else 2
    simulation = ConveyorSimulation(args.speed, scenario=scenario, seed=args.seed, object_count=count, exit_mode="remove")
    try:
        mode = "RGB-D product feed" if args.stream else (
            "RGB-D concurrent" if args.perception else "Serialized" if args.serial else "Concurrent"
        )
        print(f"Preparing {mode.lower()} motions and checking the shared scene...")
        feed = prepare_feed_run(simulation) if args.stream else None
        batch = feed.batch if feed else prepare_panda_batch(simulation, serial=args.serial, perception=args.perception)
        print(f"{len(batch.frames)} checked frames | both arms moving {batch.overlap_s:.2f} s")
        shapes = {spec.object_id: spec.geometry_type for spec in simulation.object_specs}
        if not args.no_view:
            simulation.view(f"Two Pandas | {mode} coordination")
            viewer = simulation.config.viewer()
            if feed:
                viewer.focus([-0.25, -0.38, 0.88], 3.5)
            else:
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
            if feed:
                active = {job.robot_id: shapes[job.object_id] for job in batch.jobs
                          if job.planning_time_s <= now < job.placement.release_time_s}
                return render(f"RGB-D feed | A {active.get('robot_a', 'waiting')} | B {active.get('robot_b', 'waiting')} "
                              f"| placed {placed}/{len(shapes)} | t = {now:.2f} s")
            return render(f"Two Pandas | {mode} | placed {placed}/{len(simulation.object_ids)} | t = {now:.2f} s")

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
            if args.perception and not feed:
                playback = play_camera_panda_batch(
                    simulation, batch, after_step=after_step, on_replanning=on_replanning,
                )
                if playback is None:
                    return 0
                print(f"RGB-D approach | {playback.camera_frames} frames | {len(playback.replanned_at_s)} replans "
                      f"| {playback.candidate_retries} alternative attempts")
                jobs = playback.jobs
            else:
                if not play_panda_batch(simulation, batch, after_step=after_step):
                    return 0
                jobs = batch.jobs
            for item in jobs:
                item_label = f"{item.track_id} ({item.object_id})" if item.track_id else item.object_id
                if feed:
                    item_label += f" {shapes[item.object_id]}"
                print(f"{item.robot_id} placed {item_label} in {item.placement.target_id} at {item.placement.release_time_s:.2f} s")
            message = f"Placed {len(jobs)}/{len(simulation.object_ids)} | {simulation.simulation_time_s:.2f} s"
            print(message)
            if feed:
                for object_id, reason in feed.missed:
                    print(f"Missed {object_id}: {reason}")
                if not args.no_view:
                    while render(message):
                        sleep(0.05)
                return int(bool(feed.missed))
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
