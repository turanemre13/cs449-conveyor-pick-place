"""Dual-arm object/robot/interception-time assignment. Owner: Hüseyin.

The proposed :class:`DeadlineAwareGreedyScheduler` and the
:class:`FifoNearestRobotScheduler` baseline share one approximate
reachability/timing model (:mod:`.feasibility`). Results are planning-level
estimates; IK/KOMO feasibility and execution belong to planning/control.
"""

from .adapters import (
    default_robot_configs, robot_configs_from_snapshots, schedulable_objects_from_ground_truth,
    schedulable_objects_from_specs,
)
from .baseline import FifoNearestRobotScheduler
from .greedy import DeadlineAwareGreedyScheduler
from .metrics import ScheduleMetrics, input_observation_window_s, summarize
from .models import (
    Assignment, ReachModel, RobotConfig, SchedulableObject, ScheduleResult, SchedulerConfig,
    TimingModel, WorkspaceConfig,
)

__all__ = [
    "Assignment", "DeadlineAwareGreedyScheduler", "FifoNearestRobotScheduler", "ReachModel",
    "RobotConfig", "SchedulableObject", "ScheduleMetrics", "ScheduleResult", "SchedulerConfig",
    "TimingModel", "WorkspaceConfig", "default_robot_configs", "input_observation_window_s", "robot_configs_from_snapshots",
    "schedulable_objects_from_ground_truth", "schedulable_objects_from_specs", "summarize",
]
