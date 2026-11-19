"""Panda jaw motion and bilateral geometric contact verification."""

from dataclasses import dataclass, replace
from math import fsum, isfinite

from .panda import PandaPlanningError


CONTACT_TOLERANCE_M = 0.001
_MAX_CONTACT_CORRECTION_M = 0.004


@dataclass(frozen=True)
class FingerClosure:
    joint_frame: str
    pad_frames: tuple[str, str]
    initial_m: float
    closed_m: float
    start_s: float
    end_s: float

    def coordinate_at(self, simulation_time_s: float) -> float:
        fraction = min(1.0, max(0.0, (simulation_time_s - self.start_s) / (self.end_s - self.start_s)))
        return self.initial_m + (self.closed_m - self.initial_m) * fraction

    def contact_clearances(self, object_id: str) -> dict[frozenset[str], float]:
        return {frozenset((pad, object_id)): -CONTACT_TOLERANCE_M for pad in self.pad_frames}


def plan_finger_closure(robot, grasp, start_s: float, end_s: float, *, gap_margin_m: float = 0.0003) -> FingerClosure:
    initial, mimic = robot.finger_joint_positions_m
    # The two symmetric Panda sliders each supply half of the jaw gap.
    closed = grasp.object_width_m / 2 + gap_margin_m
    if (not 0.0 < closed < initial <= 0.04 + 1e-9 or abs(initial - mimic) > 1e-6
            or grasp.required_opening_m > 2 * initial + 1e-6):
        raise PandaPlanningError("The current Panda jaw opening cannot admit the scheduled grasp")
    prefix = robot.gripper_frame.removesuffix("gripper")
    return FingerClosure(robot.finger_joint_names[0], (prefix + "finger1", prefix + "finger2"), initial, closed, start_s, end_s)


def _pad_gaps(config, object_id: str, closure: FingerClosure) -> tuple[float, ...]:
    import robotic as ry

    return tuple(-float(config.eval(ry.FS.negDistance, [pad, object_id])[0][0]) for pad in closure.pad_frames)


def fit_simulated_closure(config, gripper_frame: str, object_id: str, closure: FingerClosure) -> FingerClosure:
    """Refine a camera width using virtual contact at a fixed planned TCP pose."""
    joint = config.getFrame(closure.joint_frame)
    saved = joint.getJointState().copy()

    def gaps_at(coordinate):
        joint.setJointState([coordinate])
        gaps = _pad_gaps(config, object_id, closure)
        if not all(isfinite(gap) for gap in gaps):
            raise PandaPlanningError("Invalid simulated pad distance during closure")
        return gaps

    try:
        if all(0.0 <= gap <= CONTACT_TOLERANCE_M for gap in gaps_at(closure.closed_m)):
            require_bilateral_contact(config, gripper_frame, object_id, closure)
            return closure

        # Seat the nearer pad without modeling penetration that would persist
        # while KOMO plans the attached load. The farther pad is checked below.
        goal_gap = 0.000005
        lower = max(0.0, closure.closed_m - _MAX_CONTACT_CORRECTION_M)
        upper = min(closure.initial_m, closure.closed_m + _MAX_CONTACT_CORRECTION_M)
        if not min(gaps_at(lower)) <= goal_gap <= min(gaps_at(upper)):
            raise PandaPlanningError("Camera width cannot be corrected within the simulated jaw range")
        for _ in range(14):
            middle = (lower + upper) / 2
            if min(gaps_at(middle)) < goal_gap:
                lower = middle
            else:
                upper = middle
        adjusted = replace(closure, closed_m=upper)
        joint.setJointState([adjusted.closed_m])
        require_bilateral_contact(config, gripper_frame, object_id, adjusted)
        return adjusted
    finally:
        joint.setJointState(saved)


def require_bilateral_contact(config, gripper_frame: str, object_id: str, closure: FingerClosure) -> None:
    """Require both pads to bracket the object and reach it within model tolerance."""
    obj = config.getFrame(object_id).getPosition()
    axis = config.getFrame(gripper_frame).getRotationMatrix()[:, 0]
    offsets = []
    for pad, distance in zip(closure.pad_frames, _pad_gaps(config, object_id, closure)):
        delta = config.getFrame(pad).getPosition() - obj
        offsets.append(fsum(v * a for v, a in zip(delta, axis)))
        if not isfinite(distance) or abs(distance) > CONTACT_TOLERANCE_M:
            raise PandaPlanningError(f"Grasp contact not verified at {pad}")
    if offsets[0] * offsets[1] >= 0:
        raise PandaPlanningError("The closed pads do not bracket the object")
