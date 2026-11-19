"""Unit and integration tests for RGB-D multi-object perception."""

from __future__ import annotations

from dataclasses import replace
from math import atan2, cos, hypot, pi, sin
import unittest

from conveyor.perception import (
    DetectedObject,
    PerceptionConfig,
    RGBDObjectDetector,
    advance_specs_along_belt,
    default_overhead_calibration,
    render_synthetic_rgbd,
)
from conveyor.simulation.scenarios import ObjectSpec, generate_scenario
from conveyor.types import ObjectObservation


def _half_turn_diff(a: float, b: float) -> float:
    """Smallest angular difference modulo pi/2 or pi for rectangle symmetry."""
    wrapped = (a - b + pi / 2.0) % pi - pi / 2.0
    return abs(wrapped)


class TestPerceptionModelsAndValidation(unittest.TestCase):
    def test_config_validation_rejects_invalid_bounds(self) -> None:
        config = PerceptionConfig()
        self.assertEqual(config.belt_top_z_m, 0.75)

        for invalid in (
            {"belt_x_limits_m": (1.0, -1.0)},
            {"belt_y_limits_m": (0.2, 0.2)},
            {"min_object_height_m": 0.0},
            {"max_object_height_m": 0.01},
            {"depth_layer_tolerance_m": -0.01},
            {"color_split_distance_rgb": 0.0},
            {"min_contour_pixels": 2},
            {"min_contour_pixels": True},
        ):
            with self.assertRaises(ValueError):
                PerceptionConfig(**invalid)  # type: ignore[arg-type]

    def test_detected_object_validation_and_observation_conversion(self) -> None:
        det = DetectedObject(
            detection_id="det_001",
            simulation_time_s=0.5,
            position_m=(-0.4, 0.05, 0.79),
            dimensions_m=(0.08, 0.05, 0.08),
            yaw_rad=0.25,
            geometry_type="box",
            color_rgb=(0.9, 0.3, 0.1),
            pixel_centroid_uv=(250.0, 170.0),
            pixel_area=85,
            confidence=0.95,
        )
        self.assertEqual(det.frame, "world")
        self.assertEqual(det.object_id, "det_001")
        obs = det.to_observation()
        self.assertIsInstance(obs, ObjectObservation)
        self.assertEqual(obs.object_id, "det_001")
        self.assertEqual(obs.simulation_time_s, 0.5)
        self.assertEqual(obs.position_m, (-0.4, 0.05, 0.79))

        # Cylinder requires equal x/y diameters
        with self.assertRaises(ValueError):
            DetectedObject(
                detection_id="det_002",
                simulation_time_s=0.0,
                position_m=(0.0, 0.0, 0.79),
                dimensions_m=(0.06, 0.05, 0.08),
                yaw_rad=0.0,
                geometry_type="cylinder",
            )


class TestRGBDObjectDetector(unittest.TestCase):
    def setUp(self) -> None:
        self.detector = RGBDObjectDetector()

    def test_empty_belt_returns_no_detections(self) -> None:
        obs = render_synthetic_rgbd((), simulation_time_s=0.0)
        self.assertEqual(self.detector.detect(obs), [])
        self.assertEqual(self.detector.detect_observations(obs), [])

    def test_single_box_pose_dimensions_and_orientation(self) -> None:
        box = ObjectSpec(
            object_id="object_001",
            geometry_type="box",
            dimensions_m=(0.09, 0.045, 0.08),
            position_m=(-0.35, 0.06, 0.79),
            yaw_rad=0.42,
            color_rgb=(0.90, 0.32, 0.12),
        )
        obs = render_synthetic_rgbd((box,), simulation_time_s=1.25)
        detections = self.detector.detect(obs)
        self.assertEqual(len(detections), 1)
        det = detections[0]
        self.assertEqual(det.simulation_time_s, 1.25)
        self.assertEqual(det.geometry_type, "box")
        self.assertAlmostEqual(det.position_m[0], box.position_m[0], delta=0.003)
        self.assertAlmostEqual(det.position_m[1], box.position_m[1], delta=0.003)
        self.assertAlmostEqual(det.position_m[2], box.position_m[2], delta=0.001)
        self.assertAlmostEqual(det.dimensions_m[0], box.dimensions_m[0], delta=0.008)
        self.assertAlmostEqual(det.dimensions_m[1], box.dimensions_m[1], delta=0.008)
        self.assertAlmostEqual(det.dimensions_m[2], box.dimensions_m[2], delta=0.001)
        self.assertLess(_half_turn_diff(det.yaw_rad, box.yaw_rad), 0.10)

    def test_single_cylinder_pose_and_diameter(self) -> None:
        cyl = ObjectSpec(
            object_id="object_002",
            geometry_type="cylinder",
            dimensions_m=(0.06, 0.06, 0.07),
            position_m=(0.20, -0.04, 0.785),
            yaw_rad=1.1,
            color_rgb=(0.16, 0.57, 0.88),
        )
        obs = render_synthetic_rgbd((cyl,), simulation_time_s=0.5)
        detections = self.detector.detect(obs)
        self.assertEqual(len(detections), 1)
        det = detections[0]
        self.assertEqual(det.geometry_type, "cylinder")
        self.assertAlmostEqual(det.position_m[0], cyl.position_m[0], delta=0.003)
        self.assertAlmostEqual(det.position_m[1], cyl.position_m[1], delta=0.003)
        self.assertAlmostEqual(det.position_m[2], cyl.position_m[2], delta=0.001)
        self.assertAlmostEqual(det.dimensions_m[0], 0.06, delta=0.006)
        self.assertAlmostEqual(det.dimensions_m[1], 0.06, delta=0.006)
        self.assertAlmostEqual(det.dimensions_m[2], 0.07, delta=0.001)
        self.assertEqual(det.yaw_rad, 0.0)

    def test_multi_object_detection_across_ordered_sparse_and_clutter_scenarios(self) -> None:
        for scenario in ("ordered", "sparse", "clutter"):
            with self.subTest(scenario=scenario):
                specs = generate_scenario(scenario, seed=7, object_count=8)
                obs = render_synthetic_rgbd(specs, simulation_time_s=0.0)
                dets = self.detector.detect(obs)
                self.assertEqual(len(dets), len(specs))

                matched_ids: set[str] = set()
                for spec in specs:
                    best = min(
                        dets,
                        key=lambda d: hypot(
                            d.position_m[0] - spec.position_m[0],
                            d.position_m[1] - spec.position_m[1],
                        ),
                    )
                    matched_ids.add(best.detection_id)
                    xy_err = hypot(
                        best.position_m[0] - spec.position_m[0],
                        best.position_m[1] - spec.position_m[1],
                    )
                    z_err = abs(best.position_m[2] - spec.position_m[2])
                    self.assertLess(xy_err, 0.005)
                    self.assertLess(z_err, 0.002)
                self.assertEqual(len(matched_ids), len(specs))

    def test_touching_adjacent_objects_with_distinct_colors_are_split_in_clutter(self) -> None:
        # Place two boxes with zero pixel gap so their binary foreground masks touch
        box_a = ObjectSpec(
            object_id="object_001",
            geometry_type="box",
            dimensions_m=(0.06, 0.05, 0.08),
            position_m=(-0.20, -0.025, 0.79),
            yaw_rad=0.0,
            color_rgb=(0.90, 0.32, 0.12),
        )
        box_b = ObjectSpec(
            object_id="object_002",
            geometry_type="box",
            dimensions_m=(0.06, 0.05, 0.06),
            position_m=(-0.20, 0.026, 0.78),
            yaw_rad=0.0,
            color_rgb=(0.16, 0.57, 0.88),
        )
        obs = render_synthetic_rgbd((box_a, box_b), simulation_time_s=0.0)
        dets = self.detector.detect(obs)
        self.assertEqual(len(dets), 2)

    def test_out_of_roi_or_fallen_objects_are_ignored(self) -> None:
        on_belt = ObjectSpec(
            object_id="object_001",
            geometry_type="box",
            dimensions_m=(0.06, 0.05, 0.08),
            position_m=(0.0, 0.0, 0.79),
            yaw_rad=0.0,
            color_rgb=(0.90, 0.32, 0.12),
        )
        off_belt_side = replace(on_belt, object_id="object_002", position_m=(0.0, -0.35, 0.79))
        fallen_on_floor = replace(on_belt, object_id="object_003", position_m=(1.1, 0.0, 0.04))
        obs = render_synthetic_rgbd((on_belt, off_belt_side, fallen_on_floor), simulation_time_s=0.0)
        dets = self.detector.detect(obs)
        self.assertEqual(len(dets), 1)
        self.assertAlmostEqual(dets[0].position_m[0], 0.0, delta=0.004)


if __name__ == "__main__":
    unittest.main()
