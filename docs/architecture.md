# Architecture

This document describes the intended integration. Only package scaffolding,
validated data records, and the synthetic interface demo exist today.

| Module | Owner | Intended responsibility / handoff |
| --- | --- | --- |
| `simulation` | Doğa | RAI + Panda, conveyor scene, RGB-D frames, calibration, simulation clock |
| `perception` | Emre | OpenCV detection and depth projection; world-frame `ObjectObservation` records |
| `tracking` | Emre | Cross-frame association and constant-velocity Kalman filtering; `TrackedState` records |
| `prediction` | Emre | Estimate object pose at candidate future interception times |
| `grasping` | Hüseyin | Geometry-based grasp candidates from object geometry and predicted state |
| `planning` | Hüseyin | IK/KOMO interception and placement trajectories |
| `control` | Heysem | Execute trajectories and coordinate feedback / optional replanning |
| `pipeline.py` | Heysem | Future orchestration of the above modules |
| `experiments/` | Heysem | Evaluation scenarios, metrics, and reproducible runs |

## Integration flow

1. Simulation advances the shared clock and captures RGB-D data at an acquisition
   timestamp. Camera calibration and the camera-to-world transform accompany it.
2. Perception projects detections to world coordinates and emits observations.
3. Tracking associates detections and estimates position/velocity with a
   constant-velocity Kalman filter at each simulation timestamp.
4. Prediction estimates the object's future state for candidate interception
   times. Grasping selects geometry-based candidates.
5. Planning checks IK/KOMO feasibility for interception and subsequent placement.
6. Control executes the accepted trajectory. New observations feed the loop;
   optional replanning must account for execution progress and planning latency.

This is a design outline, not an implemented loop. `pipeline.run()` raises
`NotImplementedError` until integration is ready. No backend is loaded on import.

## Boundaries and dependencies

`types.py` is shared and standard-library-only. Perception/tracking/prediction
must not import simulation or RAI. NumPy and OpenCV are available through the
`perception` extra; algorithms remain pending. RAI-specific imports belong in
simulation and planning backend code, with actionable errors if unavailable.

Keep RAI scene sources in `scenes/`, demos in `scripts/`, evaluation code in
`experiments/`, report sources in `reports/`, and generated artifacts in ignored
output directories. Pass resource paths explicitly or derive them from source
locations. Do not rely on a developer's home directory.

## Decisions still required

- Verify the course RAI distribution/version, native toolchain, Python version,
  Panda assets, and camera API; record reproducible setup in `requirements-rai.txt`.
- Agree on scene/world axes, camera calibration, image depth units, capture rate,
  simulation step, and clock reset behavior.
- Define object geometry/orientation, covariance layout, grasp, trajectory,
  execution status, and failure/replanning interfaces before implementing them.
- Agree on association, track lifecycle, missing detections, interception timing,
  collision constraints, and replanning thresholds.
- Define repeatable evaluation scenes and metrics such as pick success,
  prediction error, planning time, and processing latency.
