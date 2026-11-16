# Shared interfaces

`conveyor.types` defines two frozen dataclasses. These are minimal centroid
interfaces, not complete camera, grasp, or trajectory contracts.

| Field | `ObjectObservation` | `TrackedState` |
| --- | --- | --- |
| `object_id: str` | Nonblank detection identifier | Nonblank stable track identifier |
| `simulation_time_s: float` | RGB-D acquisition time | Filter state reference time |
| `position_m: tuple[float, float, float]` | Measured world centroid `(x, y, z)` | Estimated world centroid `(x, y, z)` |
| `velocity_m_s: tuple[float, float, float]` | Not present | Estimated world linear velocity `(vx, vy, vz)` |
| `frame` | Class constant `"world"` | Class constant `"world"` |

## Conventions and validation

- All positions are in the scene's fixed world frame, in metres. Linear
  velocities use metres/second. Future angles use radians and angular velocities
  radians/second. The scene owner must document the actual world-axis convention.
- Timestamps are finite, non-negative seconds since simulation start. They are
  not Unix or wall-clock times. Restart tracking/prediction when the clock resets.
- Vector fields must be immutable tuples of exactly three finite real numbers.
  Integers are accepted; booleans, NaN, and infinity are rejected. Convert arrays
  explicitly at module boundaries. Invalid fields raise `ValueError`.
- Position and velocity components may be negative. The dataclasses cannot verify
  physical units or calibration: producers must convert depth to metres and
  apply camera-to-world transforms before constructing observations.
- `frame` is a class constant, not an initializer argument or serialized dataclass
  field. Any future serialization layer must preserve the world-frame contract.
- Detection IDs need not persist across frames. Tracking performs association
  and assigns track IDs; equal detection/track strings imply no association.
- Constructors validate individual records. The tracking consumer must enforce
  timestamp ordering per stream and define duplicate/out-of-order handling.

```python
from conveyor.types import ObjectObservation, TrackedState

observation = ObjectObservation("detection-1", 1.0, (0.4, 0.0, 0.1))
state = TrackedState("track-1", 1.0, observation.position_m, (0.1, 0.0, 0.0))
```

The state above is synthetic; it does not run a filter. Covariance, orientation,
object dimensions, confidence, camera frames, and robot trajectories are not yet
represented. Agree on these contracts together before extending the records.
When covariance is introduced, document its coordinate frame, state ordering,
and mixed units explicitly.

## Integration entry point

`conveyor.pipeline.run()` is reserved for future orchestration and currently
raises `NotImplementedError`. Subpackages expose no working backend APIs yet.
The synthetic `scripts/interface_demo.py` is the only runnable demo.
