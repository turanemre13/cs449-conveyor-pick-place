"""FIFO + nearest-robot baseline scheduler.

Objects are served strictly in arrival order (first in = furthest along the
direction of motion). Each object goes to the robot whose base is nearest to the
object's *current* position; if that robot cannot intercept it, the next
nearest robot is tried. There is no urgency (slack) ordering, no reasoning about
which robot is the object's last chance, and no gap filling. It uses the same
interception/feasibility model as the proposed method, so a comparison isolates
the decision logic.
"""

from __future__ import annotations

from math import hypot

from .base import SchedulerBase


class FifoNearestRobotScheduler(SchedulerBase):
    name = "fifo_nearest_robot"

    def _arrival_key(self, object_id: str) -> tuple:
        state = self.model.objects[object_id].state
        velocity, stationary = self.model.predictor.effective_velocity(state)
        if stationary:
            progress = state.position_m[0]
        else:
            speed = hypot(*velocity)
            progress = sum(p * v for p, v in zip(state.position_m, velocity)) / speed
        return (-progress, object_id)

    def _decide(self) -> None:
        for object_id in sorted(self.pending, key=self._arrival_key):
            position = self.model.objects[object_id].state.position_m
            robots = sorted(
                self.runtime.values(),
                key=lambda r: (hypot(position[0] - r.config.base_position_m[0],
                                     position[1] - r.config.base_position_m[1]), r.robot_id),
            )
            for robot in robots:
                option, _ = self._options(robot.robot_id, object_id)
                if option is not None:
                    self._commit(option)
                    break
