"""Validate and optionally display Step 1's static scene from the repository root."""

import argparse
from importlib.metadata import version
from pathlib import Path
import time

from conveyor.simulation import build_minimal_scene
from conveyor.simulation.scene import DEFAULT_SCENE_PATH


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE_PATH)
    parser.add_argument(
        "--no-view", action="store_true",
        help="load geometry and check collisions without opening a viewer",
    )
    parser.add_argument(
        "--view-seconds", type=float,
        help="close the viewer after this many seconds instead of waiting for a key",
    )
    args = parser.parse_args()
    if args.view_seconds is not None and (
        not 0 < args.view_seconds < float("inf") or args.no_view
    ):
        parser.error("--view-seconds must be finite, positive, and used with the viewer")

    try:
        config = build_minimal_scene(args.scene)
    except (RuntimeError, FileNotFoundError, ValueError) as exc:
        parser.exit(1, f"Scene setup failed: {exc}\n")

    print(f"RAI robotic: {version('robotic')}")
    print(f"Scene: {args.scene.resolve()}")
    print(f"Frames: {len(config.getFrameNames())}; active joints: {config.getJointDimension()}")
    print("World: metres, +z up, future conveyor direction +x")
    for name in ("a_panda_base", "a_gripper", "conveyor", "object_001"):
        position = config.getFrame(name).getPosition()
        print(f"{name}: {tuple(round(float(x), 4) for x in position)} m")

    config.computeCollisions()
    penetrations = config.getCollisions(-1e-5)
    if penetrations:
        for first, second, distance in penetrations:
            print(f"Penetration: {first} / {second}: {distance:.6f} m")
        return 1
    print("Initial collision check passed (penetration tolerance: 0.00001 m).")
    print("Static scene only: conveyor motion, physics stepping, and camera are pending.")

    if not args.no_view:
        try:
            message = "CS449 Step 1: static conveyor scene"
            config.view(False, message)
            config.viewer().focus([0.0, -0.15, 0.8], 1.8)
            config.viewer().visualsOnly()
            config.view(args.view_seconds is None, message)
            if args.view_seconds is not None:
                time.sleep(args.view_seconds)
        finally:
            config.view_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
