"""Reproducible evaluation of Heysem's multi-object perception and Kalman tracking.

Evaluates detection recall/precision, 3-D centroid error, velocity error,
dimension/yaw error, shape classification accuracy, and track ID switches
across conveyor speeds, object counts, and clutter levels::

    PYTHONPATH=src python3 experiments/perception_tracking_eval.py
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from math import hypot, pi
from pathlib import Path

from conveyor.perception import (
    RGBDObjectDetector,
    advance_specs_along_belt,
    render_synthetic_rgbd,
)
from conveyor.simulation.scenarios import generate_scenario
from conveyor.tracking import MultiObjectTracker


def _half_turn_diff_deg(a_rad: float, b_rad: float) -> float:
    wrapped = (a_rad - b_rad + pi / 2.0) % pi - pi / 2.0
    return abs(wrapped) * 180.0 / pi


@dataclass(frozen=True)
class EvalMetrics:
    scenario: str
    object_count: int
    belt_speed_m_s: float
    depth_noise_mm: float
    seeds: int
    total_gt_objects: int
    detection_recall_pct: float
    mean_position_err_mm: float
    max_position_err_mm: float
    mean_velocity_err_mm_s: float
    max_velocity_err_mm_s: float
    mean_dimension_err_mm: float
    mean_box_yaw_err_deg: float
    shape_accuracy_pct: float
    total_id_switches: int


def evaluate_configuration(
    scenario: str,
    object_count: int,
    belt_speed_m_s: float,
    *,
    seeds: int = 5,
    frames: int = 20,
    dt: float = 0.05,
    depth_noise_mm: float = 0.5,
) -> EvalMetrics:
    detector = RGBDObjectDetector()
    pos_errs: list[float] = []
    vel_errs: list[float] = []
    dim_errs: list[float] = []
    yaw_errs: list[float] = []
    total_gt = 0
    detected_gt = 0
    shape_correct = 0
    id_switches = 0

    for seed in range(seeds):
        tracker = MultiObjectTracker()
        specs = generate_scenario(scenario, seed=seed, object_count=object_count)
        gt_to_track: dict[str, str] = {}

        for step in range(frames):
            t = step * dt
            current_specs = advance_specs_along_belt(specs, belt_speed_m_s, t)
            obs = render_synthetic_rgbd(
                current_specs,
                simulation_time_s=t,
                depth_noise_std_m=depth_noise_mm / 1000.0,
                rng_seed=seed * 1000 + step,
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
                dist = hypot(
                    nearest.position_m[0] - spec.position_m[0],
                    nearest.position_m[1] - spec.position_m[1],
                )
                if dist > 0.04:
                    continue
                if spec.object_id not in gt_to_track:
                    gt_to_track[spec.object_id] = nearest.object_id
                elif gt_to_track[spec.object_id] != nearest.object_id:
                    id_switches += 1
                    gt_to_track[spec.object_id] = nearest.object_id

        final_t = (frames - 1) * dt
        final_specs = advance_specs_along_belt(specs, belt_speed_m_s, final_t)
        final_tracks = {trk.object_id: trk for trk in tracker.get_tracked_objects()}

        for spec in final_specs:
            total_gt += 1
            tid = gt_to_track.get(spec.object_id)
            if tid is None or tid not in final_tracks:
                continue
            trk = final_tracks[tid]
            detected_gt += 1
            pos_errs.append(
                1000.0 * hypot(trk.position_m[0] - spec.position_m[0], trk.position_m[1] - spec.position_m[1])
            )
            vel_errs.append(
                1000.0 * hypot(trk.velocity_m_s[0] - belt_speed_m_s, trk.velocity_m_s[1])
            )
            gt_dims = sorted(spec.dimensions_m[:2])
            tk_dims = sorted(trk.dimensions_m[:2])
            dim_err = (
                abs(tk_dims[0] - gt_dims[0])
                + abs(tk_dims[1] - gt_dims[1])
                + abs(trk.dimensions_m[2] - spec.dimensions_m[2])
            ) / 3.0
            dim_errs.append(1000.0 * dim_err)

            if trk.geometry_type == spec.geometry_type:
                shape_correct += 1
            if spec.geometry_type == "box":
                # Only compare yaw for non-square boxes (where principal axis is well-defined)
                if max(spec.dimensions_m[:2]) / min(spec.dimensions_m[:2]) >= 1.15:
                    gt_dx, gt_dy, _ = spec.dimensions_m
                    gt_yaw = spec.yaw_rad if gt_dx >= gt_dy else (spec.yaw_rad + pi / 2.0)
                    yaw_errs.append(_half_turn_diff_deg(trk.yaw_rad, gt_yaw))

    return EvalMetrics(
        scenario=scenario,
        object_count=object_count,
        belt_speed_m_s=belt_speed_m_s,
        depth_noise_mm=depth_noise_mm,
        seeds=seeds,
        total_gt_objects=total_gt,
        detection_recall_pct=100.0 * detected_gt / max(total_gt, 1),
        mean_position_err_mm=sum(pos_errs) / max(len(pos_errs), 1),
        max_position_err_mm=max(pos_errs, default=0.0),
        mean_velocity_err_mm_s=sum(vel_errs) / max(len(vel_errs), 1),
        max_velocity_err_mm_s=max(vel_errs, default=0.0),
        mean_dimension_err_mm=sum(dim_errs) / max(len(dim_errs), 1),
        mean_box_yaw_err_deg=sum(yaw_errs) / max(len(yaw_errs), 1),
        shape_accuracy_pct=100.0 * shape_correct / max(detected_gt, 1),
        total_id_switches=id_switches,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("outputs/heysem_perception_tracking"))
    args = parser.parse_args()

    configs = [
        ("ordered", 8, 0.08, 0.5),
        ("ordered", 8, 0.12, 0.5),
        ("sparse", 8, 0.12, 0.5),
        ("clutter", 8, 0.05, 0.5),
        ("clutter", 8, 0.12, 0.5),
        ("clutter", 8, 0.20, 0.5),
        ("clutter", 12, 0.12, 0.5),
        ("clutter", 12, 0.12, 1.5),
    ]

    results: list[EvalMetrics] = []
    print(
        f"{'Scenario':8s} | {'Obj':3s} | {'Speed':6s} | {'Noise':5s} | {'Recall':6s} | "
        f"{'Pos(mm)':7s} | {'Vel(mm/s)':9s} | {'Dim(mm)':7s} | {'Yaw(deg)':8s} | {'Shape%':6s} | {'IDSw':4s}"
    )
    print("-" * 98)
    for scenario, count, speed, noise in configs:
        m = evaluate_configuration(
            scenario,
            count,
            speed,
            seeds=args.seeds,
            depth_noise_mm=noise,
        )
        results.append(m)
        print(
            f"{m.scenario:8s} | {m.object_count:3d} | {m.belt_speed_m_s:6.2f} | {m.depth_noise_mm:5.1f} | "
            f"{m.detection_recall_pct:5.1f}% | {m.mean_position_err_mm:7.2f} | "
            f"{m.mean_velocity_err_mm_s:9.2f} | {m.mean_dimension_err_mm:7.2f} | "
            f"{m.mean_box_yaw_err_deg:8.2f} | {m.shape_accuracy_pct:5.1f}% | {m.total_id_switches:4d}"
        )

    args.output.mkdir(parents=True, exist_ok=True)
    json_path = args.output / "results.json"
    json_path.write_text(json.dumps([asdict(r) for r in results], indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved evaluation report to {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
