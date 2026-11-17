"""Emre's simulation environment: seeded objects, two Pandas, and RGB-D capture.

Native dynamics cover missed objects; robot execution and belt contact dynamics
remain integration work for the control/planning modules.
The optional RAI backend is loaded only when constructing a scene.
"""

from .conveyor import ConveyorSimulation, ObjectGroundTruth
from .camera import CameraCalibration, CameraObservation
from .scene import build_dual_arm_scene, build_minimal_scene
from .robots import RobotState
from .targets import TargetBin

__all__ = [
    "CameraCalibration", "CameraObservation", "ConveyorSimulation", "ObjectGroundTruth",
    "build_dual_arm_scene", "build_minimal_scene",
    "RobotState", "TargetBin",
]
