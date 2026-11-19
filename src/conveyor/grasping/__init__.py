"""Geometry-based grasp candidates. Owner: Hüseyin.

Top-down parallel-jaw grasps for upright boxes and cylinders, with gripper-width,
support-height and neighbour-clearance screening. Geometric feasibility only:
IK/KOMO reachability and physical grasp success are verified elsewhere.
"""

from .geometry import (
    ObjectGeometry, geometry_from_ground_truth, geometry_from_spec, yaw_from_quaternion_wxyz,
)
from .top_down import (
    RAI_GRIPPER_FRAME_CONVENTION, GraspCandidate, GraspSelectionResult, GripperConfig, approach_frame_quaternion,
    approach_to_rai_gripper, rai_gripper_quaternion, select_top_down_grasps,
)

__all__ = [
    "GraspCandidate", "GraspSelectionResult", "GripperConfig", "ObjectGeometry", "RAI_GRIPPER_FRAME_CONVENTION",
    "approach_frame_quaternion", "approach_to_rai_gripper", "geometry_from_ground_truth", "geometry_from_spec",
    "rai_gripper_quaternion", "select_top_down_grasps", "yaw_from_quaternion_wxyz",
]
