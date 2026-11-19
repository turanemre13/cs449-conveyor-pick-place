"""Reproducible comparison: deadline-aware greedy scheduler vs FIFO + nearest robot.

Run from the repository root (no RAI needed)::

    python experiments/scheduling_comparison.py            # full run, a few minutes
    python experiments/scheduling_comparison.py --quick    # fewer seeds/speeds

Three experiments, all with fixed seeds and identical inputs for both methods:

1. Static batch: the repository's ``generate_scenario`` layouts (ordered,
   sparse, clutter) planned once at t=0 for several belt speeds; one and two robots.
2. Online stream: seeded synthetic arrival streams with rolling re-planning.
3. Prediction error of the constant-velocity model under velocity noise.

All numbers are PLANNING-LEVEL ESTIMATES under the approximate reach/timing
model in ``conveyor.scheduling.models``; no motion was executed, no IK/KOMO
was solved and no grasp physics was simulated. Results are written to
``outputs/huseyin_scheduling/`` (ignored by git) and printed.

Rates: ``throughput_per_min`` uses a fixed, input-derived observation window
(identical for both methods on the same inputs); ``processing_rate_per_min``
divides by each method's own makespan and is descriptive only. A paired
per-seed comparison (wins/ties/losses, two-sided sign test) is also reported.
"""

from __future__ import annotations

import argparse
import csv
import random
import statistics
import sys
import time
from math import comb
from pathlib import Path

from conveyor.prediction import ConstantVelocityPredictor
from conveyor.scheduling import (
    DeadlineAwareGreedyScheduler, FifoNearestRobotScheduler, SchedulerConfig, default_robot_configs,
    input_observation_window_s, schedulable_objects_from_specs, summarize,
)
from conveyor.scheduling.online import run_online
from conveyor.scheduling.streams import generate_arrival_stream
from conveyor.simulation.scenarios import generate_scenario
from conveyor.types import TrackedState

METHODS = {"proposed": DeadlineAwareGreedyScheduler, "baseline": FifoNearestRobotScheduler}


def static_batch(config, speeds, seeds, rows):
    layouts = (("ordered", 8), ("sparse", 8), ("clutter", 8), ("clutter", 12))
    robot_sets = {"2 robots": default_robot_configs(), "1 robot (A)": default_robot_configs(("robot_a",))}
    for (layout, count) in layouts:
        for robots_label, robots in robot_sets.items():
            if robots_label.startswith("1") and layout != "ordered":
                continue
            for speed in speeds:
                for seed in seeds:
                    objects = schedulable_objects_from_specs(generate_scenario(layout, seed=seed, object_count=count), speed)
                    # Same input-derived window for both methods (until the last object leaves the belt).
                    window = input_observation_window_s(objects, 0.0, config.workspace)
                    for label, method in METHODS.items():
                        m = summarize(method(config).schedule(objects, robots, 0.0), robust_slack_s=config.robust_slack_s,
                                      observation_window_s=window)
                        rows.append({
                            "experiment": "static", "layout": f"{layout}-{count}", "robots": robots_label,
                            "speed_m_s": speed, "seed": seed, "method": label, "objects": m.objects,
                            "scheduled": m.scheduled, "missed": m.missed,
                            "robot_a": m.per_robot.get("robot_a", 0), "robot_b": m.per_robot.get("robot_b", 0),
                            "observation_window_s": round(window, 3),
                            "throughput_per_min": round(m.throughput_per_min, 3),
                            "processing_rate_per_min": round(m.processing_rate_per_min, 3),
                            "robust_scheduled": m.robust_scheduled, "mean_slack_s": round(m.mean_slack_s, 3),
                        })


def online_stream(config, speeds, seeds, count, rows):
    robots = default_robot_configs()
    for speed in speeds:
        for seed in seeds:
            objects = generate_arrival_stream(seed=seed, count=count, belt_speed_m_s=speed, oversized_fraction=0.1)
            for label, method in METHODS.items():
                r = run_online(method(config), objects, robots, replan_period_s=0.5)
                per_robot = r.per_robot()
                robust = sum(1 for a in r.committed if a.slack_s >= config.robust_slack_s)
                rows.append({
                    "experiment": "online", "layout": f"stream-{count}", "robots": "2 robots",
                    "speed_m_s": speed, "seed": seed, "method": label, "objects": r.objects,
                    "scheduled": r.scheduled, "missed": r.missed,
                    "robot_a": per_robot["robot_a"], "robot_b": per_robot["robot_b"],
                    "observation_window_s": round(r.observation_window_s, 3),
                    "throughput_per_min": round(r.throughput_per_min, 3),
                    "processing_rate_per_min": round(r.processing_rate_per_min, 3),
                    "robust_scheduled": robust, "mean_slack_s": round(r.mean_slack_s, 3),
                })


def prediction_error(horizons, sigmas, trials=2000, seed=0):
    """Mean/95th-percentile position error when the velocity estimate is noisy."""
    rng = random.Random(seed)
    predictor = ConstantVelocityPredictor(planar_motion=True)
    results = []
    for sigma in sigmas:
        for tau in horizons:
            errors = []
            for _ in range(trials):
                true_v = 0.12
                noisy = TrackedState("o", 0.0, (0.0, 0.0, 0.8), (true_v + rng.gauss(0, sigma), rng.gauss(0, sigma), 0.0))
                predicted = predictor.predict(noisy, tau).position_m
                errors.append(((predicted[0] - true_v * tau) ** 2 + predicted[1] ** 2) ** 0.5)
            errors.sort()
            results.append((sigma, tau, statistics.fmean(errors), errors[int(0.95 * len(errors))]))
    return results


def aggregate(rows):
    groups = {}
    for row in rows:
        key = (row["experiment"], row["layout"], row["robots"], row["speed_m_s"], row["method"])
        groups.setdefault(key, []).append(row)
    table = []
    for key, items in sorted(groups.items()):
        n = len(items)
        table.append({
            "experiment": key[0], "layout": key[1], "robots": key[2], "speed_m_s": key[3], "method": key[4],
            "runs": n,
            "objects": sum(r["objects"] for r in items),
            "scheduled": sum(r["scheduled"] for r in items),
            "missed": sum(r["missed"] for r in items),
            "robot_a": sum(r["robot_a"] for r in items),
            "robot_b": sum(r["robot_b"] for r in items),
            "robust": sum(r["robust_scheduled"] for r in items),
            "throughput": statistics.fmean(r["throughput_per_min"] for r in items),
            "processing_rate": statistics.fmean(r["processing_rate_per_min"] for r in items),
        })
    return table


def sign_test_p(wins, losses):
    """Two-sided exact sign test (ties dropped)."""
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def paired(rows):
    """Per-seed paired comparison of scheduled counts, proposed minus baseline."""
    by_run = {}
    for row in rows:
        key = (row["experiment"], row["layout"], row["robots"], row["speed_m_s"], row["seed"])
        by_run.setdefault(key, {})[row["method"]] = row["scheduled"]
    groups = {}
    for key, value in by_run.items():
        groups.setdefault(key[:4], []).append(value["proposed"] - value["baseline"])
    table = []
    for key, diffs in sorted(groups.items()):
        wins, losses = sum(d > 0 for d in diffs), sum(d < 0 for d in diffs)
        table.append((key, wins, len(diffs) - wins - losses, losses, sum(diffs), sign_test_p(wins, losses)))
    return table


def markdown(table, pairs, prediction, elapsed, args):
    lines = [
        "# Scheduling comparison (planning-level estimates)", "",
        f"Generated by `experiments/scheduling_comparison.py{' --quick' if args.quick else ''}` in {elapsed:.0f} s.",
        "Identical seeded inputs for both methods. Not physical execution results.",
        "Throughput = picks per minute of a fixed, input-derived observation window (same for both methods).",
        "Processing rate = picks per minute of the method's own makespan (descriptive, not a comparison metric).", "",
        "| Exp. | Layout | Robots | Speed (m/s) | Method | Runs | Scheduled / objects | Missed | A / B "
        "| Robust (slack>=0.5 s) | Throughput (obj/min) | Processing rate (obj/min) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for t in table:
        lines.append(
            f"| {t['experiment']} | {t['layout']} | {t['robots']} | {t['speed_m_s']:.2f} | {t['method']} | {t['runs']} | "
            f"{t['scheduled']} / {t['objects']} ({100 * t['scheduled'] / t['objects']:.0f}%) | {t['missed']} | "
            f"{t['robot_a']} / {t['robot_b']} | {t['robust']} | {t['throughput']:.2f} | {t['processing_rate']:.2f} |"
        )
    lines += ["", "## Paired per-seed comparison (scheduled count, proposed - baseline)", "",
              "| Exp. | Layout | Robots | Speed (m/s) | Wins / ties / losses | Sum of differences | Sign test p |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
    for (experiment, layout, robots, speed), wins, ties, losses, total, p in pairs:
        lines.append(f"| {experiment} | {layout} | {robots} | {speed:.2f} | {wins} / {ties} / {losses} | {total:+d} | {p:.3f} |")
    lines += ["", "## Prediction error under velocity noise", "",
              "| sigma_v (m/s) | horizon (s) | mean error (mm) | 95th pct (mm) |", "| --- | --- | --- | --- |"]
    for sigma, tau, mean, p95 in prediction:
        lines.append(f"| {sigma:.3f} | {tau:.1f} | {1000 * mean:.1f} | {1000 * p95:.1f} |")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--quick", action="store_true", help="fewer seeds and speeds")
    parser.add_argument("--output", type=Path, default=Path("outputs/huseyin_scheduling"))
    args = parser.parse_args(argv)

    config = SchedulerConfig(time_step_s=0.05)
    speeds = (0.05, 0.12, 0.20) if args.quick else (0.05, 0.08, 0.12, 0.16, 0.20, 0.25)
    seeds = range(3) if args.quick else range(10)
    stream_speeds = (0.10, 0.20) if args.quick else (0.08, 0.12, 0.16, 0.20, 0.25)
    stream_seeds = range(2) if args.quick else range(5)

    start = time.perf_counter()
    rows: list[dict] = []
    print("static batch experiments ...", flush=True)
    static_batch(config, speeds, seeds, rows)
    print("online stream experiments ...", flush=True)
    online_stream(config, stream_speeds, stream_seeds, 20, rows)
    prediction = prediction_error((0.5, 1.0, 2.0, 4.0), (0.005, 0.01))
    elapsed = time.perf_counter() - start

    table = aggregate(rows)
    pairs = paired(rows)
    args.output.mkdir(parents=True, exist_ok=True)
    with open(args.output / "runs.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    text = markdown(table, pairs, prediction, elapsed, args)
    (args.output / "summary.md").write_text(text, encoding="utf-8")
    sys.stdout.write(text)
    print(f"\nWrote {args.output / 'runs.csv'} and {args.output / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
