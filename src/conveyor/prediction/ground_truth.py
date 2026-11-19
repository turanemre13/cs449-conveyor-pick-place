"""Use simulation ground truth as a stand-in tracker for early testing.

These adapters duck-type Emre's ``ObjectGroundTruth`` records (attributes
``object_id``, ``simulation_time_s``, ``position_m``, ``velocity_m_s`` and
``motion_state``) so that this package never imports the simulation or RAI.
Ground truth is evaluation data; in the integrated pipeline the same
predictor receives Heysem's Kalman-filter ``TrackedState`` records instead.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from conveyor.types import TrackedState

#: Motion states for which the belt velocity is meaningful for prediction.
PREDICTABLE_MOTION_STATES = ("on_belt",)


def tracked_state_from_ground_truth(record: Any) -> TrackedState:
    """Convert one ground-truth record to a ``TrackedState``.

    Raises ``ValueError`` if the record has no known velocity (for example a
    ``released`` object controlled by a robot).
    """
    velocity = getattr(record, "velocity_m_s", None)
    if velocity is None:
        raise ValueError(
            f"{getattr(record, 'object_id', '?')!r} has no known velocity "
            f"(motion_state={getattr(record, 'motion_state', '?')!r})"
        )
    return TrackedState(
        object_id=record.object_id,
        simulation_time_s=float(record.simulation_time_s),
        position_m=tuple(float(x) for x in record.position_m),
        velocity_m_s=tuple(float(x) for x in velocity),
    )


def tracked_states_from_ground_truth(
    records: Mapping[str, Any] | Iterable[Any],
    *,
    motion_states: tuple[str, ...] = PREDICTABLE_MOTION_STATES,
) -> dict[str, TrackedState]:
    """Convert the objects whose ``motion_state`` is in ``motion_states``.

    By default only objects still carried by the belt are returned: falling,
    settled, exited and released objects are not interception targets.
    """
    items = records.values() if isinstance(records, Mapping) else records
    result: dict[str, TrackedState] = {}
    for record in items:
        if getattr(record, "motion_state", "on_belt") not in motion_states:
            continue
        state = tracked_state_from_ground_truth(record)
        result[state.object_id] = state
    return result
