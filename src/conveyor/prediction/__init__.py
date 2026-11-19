"""Future object states at candidate interception times. Owner: Hüseyin.

Constant-velocity prediction ``p(t + tau) = p(t) + v(t) * tau`` on the shared
``TrackedState`` interface, plus adapters that let simulation ground truth
stand in for the tracker during early testing. No RAI or simulation imports.
"""

from .constant_velocity import ConstantVelocityPredictor, PredictedState
from .ground_truth import tracked_state_from_ground_truth, tracked_states_from_ground_truth

__all__ = [
    "ConstantVelocityPredictor", "PredictedState",
    "tracked_state_from_ground_truth", "tracked_states_from_ground_truth",
]
