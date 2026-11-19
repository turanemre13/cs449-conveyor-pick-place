"""Demo of Heysem's modules: RGB-D multi-object detection -> Kalman tracking.

Runs from a plain terminal on macOS/Linux/Windows without RAI or a viewer::

    PYTHONPATH=src python3 scripts/heysem_demo.py
    PYTHONPATH=src python3 scripts/heysem_demo.py --scenario clutter --objects 8 --speed 0.12 --seed 7

Can also inspect an actual RAI capture directory exported by ``run_simulation.py --capture-dir``::

    PYTHONPATH=src python3 scripts/heysem_demo.py --capture-dir outputs/demo
"""

from __future__ import annotations

import argparse
from math import degrees, hypot
from pathlib import Path
from typing import Any

from conveyor.perception import (
    RGBDObjectDetector,
    advance_specs_along_belt,
    render_synthetic_rgbd,
)
from conveyor.simulation.camera import CameraCalibration, CameraObservation
from conveyor.simulation.scenarios import generate_scenario
from conveyor.tracking import MultiObjectTracker


def _fmt(vec: tuple[float, ...], digits: int = 3) -> str:
    return "(" + ", ".join(f"{v:+.{digits}f}" for v in vec) + ")"


def _load_capture_dir(directory: Path) -> CameraObservation:
    import numpy as np

    npz_path = directory / "rgbd.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Capture file not found: {npz_path}")
    data = np.load(npz_path)
    rgb = np.array(data["rgb"], dtype=np.uint8, copy=True)
    depth_m = np.array(data["depth_m"], dtype=np.float32, copy=True)
    sim_time = float(data["simulation_time_s"])
    intrinsics = tuple(tuple(float(x) for x in row) for row in data["intrinsics"])
    cam_to_world = tuple(tuple(float(x) for x in row) for row in data["camera_to_world"])
    cal = CameraCalibration(
        width=int(rgb.shape[1]),
        height=int(rgb.shape[0]),
        intrinsics=intrinsics,
        camera_to_world=cam_to_world,
    )
    return CameraObservation(simulation_time_s=sim_time, rgb=rgb, depth_m=depth_m, calibration=cal)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--scenario", default="clutter", choices=("single", "ordered", "sparse", "clutter"))
    parser.add_argument("--objects", type=int, default=8, help="object count (1..24)")
    parser.add_argument("--seed", type=int, default=7, help="reproducible scenario seed")
    parser.add_argument("--speed", type=float, default=0.12, help="conveyor speed in m/s")
    parser.add_argument("--dt", type=float, default=0.05, help="frame capture period in seconds")
    parser.add_argument("--frames", type=int, default=20, help="number of sequential RGB-D frames to track")
    parser.add_argument("--depth-noise-mm", type=float, default=0.5, help="synthetic depth noise 1-sigma in mm")
    parser.add_argument("--capture-dir", type=Path, help="optional path to an exported rgbd.npz directory")
    args = parser.parse_args()

    detector = RGBDObjectDetector()
    tracker = MultiObjectTracker()

    if args.capture_dir is not None:
        obs = _load_capture_dir(args.capture_dir)
        dets = detector.detect(obs)
        tracks = tracker.update(dets, obs.simulation_time_s)
        print(f"=== Loaded RGB-D capture from {args.capture_dir} at t={obs.simulation_time_s:.3f} s ===")
        print(f"Detected {len(dets)} objects:")
        for d in dets:
            print(
                f"  {d.detection_id}  {d.geometry_type:8s}  pos={_fmt(d.position_m)} m  "
                f"dims={_fmt(d.dimensions_m)} m  yaw={degrees(d.yaw_rad):+6.1f} deg  pixels={d.pixel_area}"
            )
        print(f"Active tracks: {len(tracks)}")
        return 0

    specs = generate_scenario(args.scenario, seed=args.seed, object_count=args.objects)
    print(
        f"=== Heysem Perception & Tracking Demo | scenario={args.scenario!r}, "
        f"objects={len(specs)}, seed={args.seed}, speed={args.speed:.2f} m/s ==="
    )

    gt_to_track: dict[str, str] = {}
    id_switches = 0
    for step in range(args.frames):
        t = step * args.dt
        current_specs = advance_specs_along_belt(specs, args.speed, t)
        obs = render_synthetic_rgbd(
            current_specs,
            simulation_time_s=t,
            depth_noise_std_m=args.depth_noise_mm / 1000.0,
            rng_seed=1000 + step,
        )
        dets = detector.detect(obs)
        tracks = tracker.update(dets, simulation_time_s=t)

        for spec in current_specs:
            if not tracks:
                continue
            nearest = min(
                tracks,
                key=lambda trk: hypot(
                    trk.position_m[0] - spec.position_m[0],
                    trk.position_m[1] - spec.position_m[1],
                ),
            )
            if spec.object_id not in gt_to_track:
                gt_to_track[spec.object_id] = nearest.object_id
            elif gt_to_track[spec.object_id] != nearest.object_id:
                id_switches += 1
                gt_to_track[spec.object_id] = nearest.object_id

        if step in (0, args.frames // 2, args.frames - 1):
            print(f"\n--- Frame {step:02d} (t = {t:.2f} s) | detections={len(dets)} | confirmed tracks={len(tracks)} ---")
            for trk in tracks:
                print(
                    f"  {trk.object_id}  {trk.geometry_type:8s}  p={_fmt(trk.position_m)} m  "
                    f"v={_fmt(trk.velocity_m_s)} m/s  dims={_fmt(trk.dimensions_m)} m  "
                    f"yaw={degrees(trk.yaw_rad):+6.1f} deg  hits={trk.hits}"
                )

    final_t = (args.frames - 1) * args.dt
    final_specs = advance_specs_along_belt(specs, args.speed, final_t)
    final_tracks = {trk.object_id: trk for trk in tracker.get_tracked_objects()}
    pos_errors_mm: list[float] = []
    vel_errors_mm_s: list[float] = []
    shape_correct = 0

    for spec in final_specs:
        tid = gt_to_track.get(spec.object_id)
        if tid and tid in final_tracks:
            trk = final_tracks[tid]
            pos_errors_mm.append(
                1000.0 * hypot(trk.position_m[0] - spec.position_m[0], trk.position_m[1] - spec.position_m[1])
            )
            vel_errors_mm_s.append(
                1000.0 * hypot(trk.velocity_m_s[0] - args.speed, trk.velocity_m_s[1])
            )
            if trk.geometry_type == spec.geometry_type:
                shape_correct += 1

    print("\n=== Final Tracking Accuracy Summary ===")
    print(f"  Frames processed : {args.frames} (dt = {args.dt:.3f} s, duration = {final_t:.2f} s)")
    print(f"  ID switches      : {id_switches}")
    if pos_errors_mm:
        print(
            f"  Position error   : mean = {sum(pos_errors_mm)/len(pos_errors_mm):.2f} mm, "
            f"max = {max(pos_errors_mm):.2f} mm"
        )
        print(
            f"  Velocity error   : mean = {sum(vel_errors_mm_s)/len(vel_errors_mm_s):.2f} mm/s, "
            f"max = {max(vel_errors_mm_s):.2f} mm/s"
        )
        print(f"  Shape accuracy   : {shape_correct}/{len(final_specs)} ({100.0*shape_correct/len(final_specs):.1f}%)")
    schedulables = tracker.schedulable_objects()
    print(f"  Schedulable handoff records ready for Hüseyin's module: {len(schedulables)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
