"""Unit and integration tests for multi-object Kalman tracking."""

from __future__ import annotations

from math import hypot
import unittest

from conveyor.grasping import ObjectGeometry, select_top_down_grasps
from conveyor.perception import (
    DetectedObject,
    RGBDObjectDetector,
    advance_specs_along_belt,
    render_synthetic_rgbd,
)
from conveyor.prediction import ConstantVelocityPredictor
from conveyor.scheduling import (
    DeadlineAwareGreedyScheduler,
    FifoNearestRobotScheduler,
    SchedulableObject,
    SchedulerConfig,
    default_robot_configs,
    schedulable_objects_from_tracker,
)
from conveyor.simulation.scenarios import generate_scenario
from conveyor.tracking import (
    ConstantVelocityKalmanFilter3D,
    MultiObjectTracker,
    TRACK_CONFIRMED,
    TRACK_TENTATIVE,
    TrackedObject,
    TrackerConfig,
)
from conveyor.types import ObjectObservation, TrackedState


class TestKalmanFilter3D(unittest.TestCase):
    def test_single_track_converges_to_true_constant_velocity(self) -> None:
        kf = ConstantVelocityKalmanFilter3D(
            track_id="track_001",
            initial_time_s=0.0,
            initial_position_m=(-0.80, 0.05, 0.79),
        )
        true_vx = 0.15
        dt = 0.05
        for step in range(1, 15):
            t = step * dt
            x = -0.80 + true_vx * t
            kf.update((x, 0.05, 0.79), t)

        state = kf.to_tracked_state()
        self.assertIsInstance(state, TrackedState)
        self.assertEqual(state.object_id, "track_001")
        self.assertAlmostEqual(state.position_m[0], -0.80 + true_vx * 14 * dt, delta=0.002)
        self.assertAlmostEqual(state.velocity_m_s[0], true_vx, delta=0.005)
        self.assertAlmostEqual(state.velocity_m_s[1], 0.0, delta=0.002)
        self.assertAlmostEqual(state.velocity_m_s[2], 0.0, delta=1e-6)

    def test_backward_time_prediction_is_rejected(self) -> None:
        kf = ConstantVelocityKalmanFilter3D("track_001", 1.0, (0.0, 0.0, 0.79))
        with self.assertRaises(ValueError):
            kf.predict_to(0.5)


class TestMultiObjectTracker(unittest.TestCase):
    def test_accepts_plain_object_observations_and_maintains_stable_ids(self) -> None:
        tracker = MultiObjectTracker()
        for step in range(10):
            t = step * 0.05
            obs = [
                ObjectObservation(f"frame_{step}_a", t, (-0.6 + 0.12 * t, -0.08, 0.79)),
                ObjectObservation(f"frame_{step}_b", t, (-0.2 + 0.12 * t, 0.08, 0.78)),
            ]
            states = tracker.update_states(obs)
            self.assertEqual(len(states), 2)
            self.assertEqual([s.object_id for s in states], ["track_001", "track_002"])

        s1, s2 = tracker.get_tracked_states()
        self.assertAlmostEqual(s1.velocity_m_s[0], 0.12, delta=0.006)
        self.assertAlmostEqual(s2.velocity_m_s[0], 0.12, delta=0.006)

    def test_clock_reset_automatically_resets_tracker(self) -> None:
        tracker = MultiObjectTracker()
        tracker.update([ObjectObservation("d1", 1.0, (0.0, 0.0, 0.79))])
        tracker.update([ObjectObservation("d1", 1.5, (0.06, 0.0, 0.79))])
        self.assertEqual(tracker.current_time_s, 1.5)

        # Simulation clock resets to t = 0.0
        reset_tracks = tracker.update([ObjectObservation("d_new", 0.0, (-0.9, 0.0, 0.79))])
        self.assertEqual(len(reset_tracks), 1)
        self.assertEqual(reset_tracks[0].object_id, "track_001")
        self.assertEqual(reset_tracks[0].simulation_time_s, 0.0)
        self.assertAlmostEqual(reset_tracks[0].position_m[0], -0.9, delta=1e-6)

    def test_handles_temporary_missed_detections_and_prunes_expired_tracks(self) -> None:
        config = TrackerConfig(max_missed_frames=2, min_hits_to_confirm=2)
        tracker = MultiObjectTracker(config)

        # Frame 0: tentative
        confirmed = tracker.update([ObjectObservation("d1", 0.0, (-0.5, 0.0, 0.79))], confirmed_only=True)
        self.assertEqual(len(confirmed), 0)
        all_tracks = tracker.get_tracked_objects(confirmed_only=False)
        self.assertEqual(len(all_tracks), 1)
        self.assertEqual(all_tracks[0].status, TRACK_TENTATIVE)

        # Frame 1: confirmed
        confirmed = tracker.update([ObjectObservation("d1", 0.05, (-0.494, 0.0, 0.79))], confirmed_only=True)
        self.assertEqual(len(confirmed), 1)
        self.assertEqual(confirmed[0].status, TRACK_CONFIRMED)

        # Frame 2 & 3: occluded (0 detections), track survives and extrapolates
        confirmed = tracker.update([], simulation_time_s=0.10)
        self.assertEqual(len(confirmed), 1)
        self.assertEqual(confirmed[0].missed_frames, 1)

        confirmed = tracker.update([], simulation_time_s=0.15)
        self.assertEqual(len(confirmed), 1)
        self.assertEqual(confirmed[0].missed_frames, 2)

        # Frame 4: exceeds max_missed_frames=2 -> pruned
        confirmed = tracker.update([], simulation_time_s=0.20)
        self.assertEqual(len(confirmed), 0)

    def test_end_to_end_perception_and_tracking_in_clutter_with_zero_id_switches(self) -> None:
        detector = RGBDObjectDetector()
        tracker = MultiObjectTracker()
        specs = generate_scenario("clutter", seed=7, object_count=8)
        speed = 0.15
        dt = 0.04

        gt_to_track_id: dict[str, str] = {}
        for step in range(20):
            t = step * dt
            current_specs = advance_specs_along_belt(specs, speed, t)
            obs = render_synthetic_rgbd(
                current_specs,
                simulation_time_s=t,
                depth_noise_std_m=0.0004,
                rng_seed=42 + step,
            )
            detections = detector.detect(obs)
            tracks = tracker.update(detections, simulation_time_s=t)
            self.assertEqual(len(tracks), len(current_specs))

            for spec in current_specs:
                nearest = min(
                    tracks,
                    key=lambda trk: hypot(
                        trk.position_m[0] - spec.position_m[0],
                        trk.position_m[1] - spec.position_m[1],
                    ),
                )
                if step == 0:
                    gt_to_track_id[spec.object_id] = nearest.object_id
                else:
                    self.assertEqual(
                        nearest.object_id,
                        gt_to_track_id[spec.object_id],
                        f"ID switch on {spec.object_id} at step {step}",
                    )

        # Verify final position and velocity accuracy and schedulable object handoff
        final_specs = advance_specs_along_belt(specs, speed, 19 * dt)
        final_tracks = {trk.object_id: trk for trk in tracker.get_tracked_objects()}
        for spec in final_specs:
            trk = final_tracks[gt_to_track_id[spec.object_id]]
            pos_err = hypot(trk.position_m[0] - spec.position_m[0], trk.position_m[1] - spec.position_m[1])
            vel_err = hypot(trk.velocity_m_s[0] - speed, trk.velocity_m_s[1])
            self.assertLess(pos_err, 0.004)
            self.assertLess(vel_err, 0.006)
            self.assertEqual(trk.geometry_type, spec.geometry_type)

        schedulables = schedulable_objects_from_tracker(tracker)
        self.assertEqual(len(schedulables), len(final_specs))
        for item in schedulables:
            self.assertIsInstance(item, SchedulableObject)
            self.assertIsInstance(item.state, TrackedState)
            self.assertIsInstance(item.geometry, ObjectGeometry)
            self.assertEqual(item.object_id, item.geometry.object_id)

        # Verify direct integration with Hüseyin's predictor, grasp selector, and dual-arm schedulers
        now_s = float(tracker.current_time_s or 0.0)
        predictor = ConstantVelocityPredictor(planar_motion=True)
        predicted = predictor.predict_many([o.state for o in schedulables], now_s + 1.0)
        self.assertEqual(len(predicted), len(final_specs))

        cfg = SchedulerConfig()
        robots = default_robot_configs(("robot_a", "robot_b"), available_at_s=now_s)
        greedy_res = DeadlineAwareGreedyScheduler(cfg).schedule(schedulables, robots, now_s)
        fifo_res = FifoNearestRobotScheduler(cfg).schedule(schedulables, robots, now_s)
        self.assertEqual(len(greedy_res.assignments), len(final_specs))
        self.assertEqual(len(fifo_res.assignments), len(final_specs))
        self.assertGreater(len(greedy_res.assigned), 0)
        for assignment in greedy_res.assigned:
            self.assertIn(assignment.robot_id, ("robot_a", "robot_b"))
            self.assertIsNotNone(assignment.grasp)
            self.assertGreaterEqual(assignment.intercept_time_s, now_s)


if __name__ == "__main__":
    unittest.main()
