"""Show seeded moving objects and two Panda arms; optionally export actual RGB-D."""

import argparse
from dataclasses import asdict
import json
from math import isfinite
from pathlib import Path
import time

from conveyor.simulation import ConveyorSimulation


def save_capture(simulation: ConveyorSimulation, directory: Path) -> None:
    """Save actual sensor arrays and separate evaluation metadata, without OpenCV."""
    import struct
    import zlib
    import numpy as np

    observation = simulation.get_observation()
    directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        directory / "rgbd.npz", rgb=observation.rgb, depth_m=observation.depth_m,
        simulation_time_s=observation.simulation_time_s,
        intrinsics=np.asarray(observation.calibration.intrinsics),
        camera_to_world=np.asarray(observation.calibration.camera_to_world),
    )

    # Encode the original uint8 RGB rows losslessly; no imaging dependency.
    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack("!I", len(data)) + kind + data + struct.pack("!I", zlib.crc32(kind + data))

    height, width = observation.rgb.shape[:2]
    rows = b"".join(b"\0" + row.tobytes() for row in observation.rgb)
    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(b"IHDR", struct.pack("!IIBBBBB", width, height, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b"")
    (directory / "rgb.png").write_bytes(png)
    metadata = {
        "camera.json": {"simulation_time_s": observation.simulation_time_s, **asdict(observation.calibration)},
        "ground_truth.json": {
            "simulation_time_s": observation.simulation_time_s, "frame": "world",
            "objects": {key: asdict(value) for key, value in simulation.get_ground_truth(active_only=True).items()},
        },
        "scenario.json": {
            "scenario": simulation.scenario, "seed": simulation.seed,
            "object_count": len(simulation.object_ids), "belt_speed_m_s": simulation.belt_speed_m_s,
            "initial_objects": [asdict(spec) for spec in simulation.object_specs],
        },
        "robot_states.json": {
            "frame": "world", "robots": {key: asdict(value) for key, value in simulation.get_robot_states().items()},
        },
        "targets.json": {
            "frame": "world", "targets": {key: asdict(value) for key, value in simulation.get_targets().items()},
        },
    }
    for filename, value in metadata.items():
        (directory / filename).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    print(f"RGB-D and separate ground truth saved at t={observation.simulation_time_s:.3f}s: {directory}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--speed", type=float, default=.12, help="belt speed in m/s (>=0)")
    parser.add_argument("--dt", type=float, default=.02, help="simulation step in seconds")
    parser.add_argument("--duration", type=float, default=0, help="total simulation seconds; 0 runs until closed")
    parser.add_argument("--scenario", choices=("single", "ordered", "sparse", "clutter"), default="ordered")
    parser.add_argument("--objects", type=int, default=8, help="objects per batch, 1..24 (single always uses one)")
    parser.add_argument("--seed", type=int, default=7, help="reproducible scenario seed")
    parser.add_argument("--exit-mode", choices=("fall", "remove"), default="fall", help="native fall/impact physics or removal baseline")
    parser.add_argument("--replay-delay", type=float, default=3.0, help="seconds to show the completed batch before replay")
    parser.add_argument("--capture-dir", type=Path, help="save initial RGB-D/metadata; rendering needs OpenGL/WSLg")
    parser.add_argument("--no-view", action="store_true", help="advance without opening a window")
    parser.add_argument("--no-repeat", action="store_true", help="leave fallen objects on the floor without replaying")
    args = parser.parse_args()
    if not isfinite(args.speed) or args.speed < 0:
        parser.error("--speed must be finite and non-negative")
    if not isfinite(args.dt) or args.dt <= 0:
        parser.error("--dt must be finite and positive")
    if not isfinite(args.duration) or args.duration < 0:
        parser.error("--duration must be finite and non-negative")
    if not isfinite(args.replay_delay) or args.replay_delay < 0:
        parser.error("--replay-delay must be finite and non-negative")
    if args.no_view and args.duration == 0:
        parser.error("--no-view requires a positive --duration")

    try:
        simulation = ConveyorSimulation(
            belt_speed_m_s=args.speed, scenario=args.scenario, seed=args.seed, object_count=args.objects,
            exit_mode=args.exit_mode,
        )
    except (RuntimeError, FileNotFoundError, ValueError) as exc:
        parser.exit(1, f"Simulation setup failed: {exc}\n")
    config = simulation.config
    config.computeCollisions()
    penetrations = config.getCollisions(-1e-5)
    if penetrations:
        parser.exit(1, f"Initial penetrations: {penetrations}\n")

    print(f"Two Pandas; {len(simulation.object_ids)} objects; {args.scenario}, seed {args.seed}; belt {args.speed:.3f} m/s.")
    if args.exit_mode == "fall":
        print("Belt transport is kinematic; missed objects tip, fall and collide through native PhysX.")
    else:
        print("Kinematic transport with the original removal-at-exit baseline.")
    print("Robot arms stay parked; picking and perception are pending.")
    for target in simulation.get_targets().values():
        print(f"{target.preferred_robot_id} -> {target.target_id}: {target.position_m} m (nominal, IK pending).")
    if not args.no_view:
        print("Stop with Q, Esc, window close, or terminal Ctrl+C.")
    if not args.no_repeat:
        print(f"Demo replays {args.replay_delay:g}s after all objects finish; each replay resets simulation time.")

    elapsed = 0.0
    passes = 1
    completed_since = None
    try:
        if args.capture_dir is not None:
            save_capture(simulation, args.capture_dir)
        if not args.no_view:
            from conveyor.simulation.viewer import make_close_checker

            simulation.view("CS449: two arms and moving conveyor")
            config.viewer().focus([.15, -.10, .70], 2.6)
            config.viewer().visualsOnly()
            should_close = make_close_checker(config)

        wall_start = time.monotonic()
        while args.duration == 0 or elapsed < args.duration:
            if not args.no_view:
                if should_close():
                    break
                key = simulation.view(f"CS449 | {args.scenario} | belt={len(simulation.on_belt_object_ids)} fallen={len(simulation.fallen_object_ids)} | t={simulation.simulation_time_s:.2f}s")
                if key in (27, ord("q"), ord("Q")) or should_close():
                    break
            if simulation.all_objects_finished:
                if completed_since is None:
                    completed_since = elapsed
                if not args.no_repeat and elapsed - completed_since >= args.replay_delay:
                    simulation.reset()
                    passes += 1
                    completed_since = None
            else:
                completed_since = None
            dt = args.dt if args.duration == 0 else min(args.dt, args.duration - elapsed)
            simulation.step(dt)
            elapsed += dt
            if not args.no_view:
                # Keep the GUI near real time; scheduling/evaluation use sim time.
                time.sleep(max(0.0, wall_start + elapsed - time.monotonic()))
    except KeyboardInterrupt:
        pass
    except (RuntimeError, OSError) as exc:
        parser.exit(1, f"Simulation failed: {exc}\n")
    finally:
        simulation.close()

    exited = len(simulation.object_ids) - len(simulation.active_object_ids)
    print(f"Finished: {elapsed:.2f}s total; pass {passes}; on belt={len(simulation.on_belt_object_ids)}, fallen={len(simulation.fallen_object_ids)}, removed={exited}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
