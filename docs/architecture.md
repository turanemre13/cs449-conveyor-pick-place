# Architecture

This document describes the intended dual-arm integration. The implemented
environment has two stationary Panda arms, seeded multi-object layouts,
kinematic belt transport, native falling-object dynamics, calibrated RGB-D
capture, reset and ground truth. The original
static single-Panda scene and synthetic interface demo remain available. The
full processing loop remains pending.

| Module | Owner | Intended responsibility / handoff |
| --- | --- | --- |
| `simulation` | Emre | Two Pandas, conveyor and objects, RGB-D frames, calibration, ground truth, simulation clock and reproducible scenarios |
| `perception` | Heysem | OpenCV detection and depth projection; world-frame `ObjectObservation` records |
| `tracking` | Heysem | Cross-frame association and constant-velocity Kalman filtering; `TrackedState` records |
| `prediction` | Hüseyin | Estimate object pose at candidate future interception times |
| `grasping` | Hüseyin | Geometry-based grasp candidates from object geometry and predicted state |
| `scheduling` (planned) | Hüseyin | Assign objects, robots and interception times using availability, reachability and deadlines |
| `planning` | Doğa | Dual-robot IK/KOMO interception and placement trajectories with collision avoidance |
| `control` | Doğa | Execute trajectories, operate grippers and coordinate feedback / optional replanning |
| `pipeline.py` | Shared | Future orchestration of the above modules |
| `experiments/` | Shared | Evaluation scenarios, metrics, and reproducible runs |

## Integration flow

1. Simulation advances the shared clock and captures RGB-D data at an acquisition
   timestamp. Camera calibration and the camera-to-world transform accompany it.
2. Perception projects detections to world coordinates and emits observations.
3. Tracking associates detections and estimates position/velocity with a
   constant-velocity Kalman filter at each simulation timestamp.
4. Prediction estimates the object's future state for candidate interception
   times. Grasping selects geometry-based candidates.
5. Scheduling assigns each selected object to one robot and interception time,
   accounting for availability, reachability, deadlines and conflicting actions.
6. Planning checks IK/KOMO feasibility for interception and subsequent placement.
7. Control executes the accepted trajectory. New observations feed the loop;
   optional replanning must account for execution progress and planning latency.

This is a design outline, not an implemented loop. `pipeline.run()` raises
`NotImplementedError` until integration is ready. No backend is loaded on import.

## Current simulation milestone

`build_dual_arm_scene()` loads `scenes/dual_arm.g` and adds installed RAI Panda
models with separate `a_` and `b_` frame prefixes. `ConveyorSimulation` wraps the
scene with `reset(seed=None)`, `step(dt)`, simulation time and object ground truth.
`scenarios.py` generates mixed boxes/cylinders for `single`, `ordered`, `sparse`
and `clutter` cases using a local seeded random generator. Rotated planar footprints
have at least 8 mm clearance and 5 mm margin from belt edges at initialization.
The CLI defaults to eight ordered objects: a central lane with small lateral
variation and at least 10 cm longitudinal edge gaps. The seed-7 default has
about 24 cm gaps; oversized ordered batches are rejected rather than compressed.
The entry point `scripts/run_simulation.py` animates the conveyor and objects; it also
supports bounded runs with `--no-view --duration 10`.

World `+z` is up and conveyor travel is along `+x`. The belt is `2.4 m` long and
`0.4 m` wide, centered at `y=0`, with its top at `z=0.75 m`. Robot A mounts at
`(-0.55, -0.60, 0.75) m` and Robot B at `(0.55, -0.60, 0.75) m`, each with yaw
`+pi/2`. This is a preliminary sequential placement on one side of the belt,
not a verified reachability map or coordinated motion plan.

The conveyor chassis is fixed. Visual belt seams loop over the top, around the
rollers and underneath, while objects translate along `+x`. Transport on the
belt is kinematic rather than friction-driven. Both arms stay
in a parked pose with joint 2 at `-1.0 rad`, keeping the waiting grippers clear
of the overhead belt view. `release_object("object_001")` stops the belt code from overwriting
that object's pose, leaving a future controller free to manipulate it; release
alone does not create a physical grasp.

The CLI defaults to `--exit-mode fall`. When an object's centre moves just past
the end, its pose and belt linear velocity are handed to native RAI PhysX CPU
dynamics. Gravity and contact produce tipping, impact, rebound and rolling;
there is no prescribed spin or random impulse. Fallen bodies remain visible and
have live ground-truth poses and velocities through `falling` and `settled`
states. `--exit-mode remove` retains the earlier fully-supported boundary
removal and historical exit records for comparisons.

`falling.py` owns a small separate physics configuration with static floor,
conveyor/support geometry and fallen objects. The native engine is rebuilt only
when a body is added or removed because this wheel has no incremental body
registration API; existing body poses and linear/angular velocities are carried
over. This avoids simulating robot control here. It does not yet model contacts
with still-kinematic belt objects or a gripper-controlled object.

After every object is settled or removed, the default demo waits three
simulation seconds before resetting the batch and clock. `--no-repeat` leaves
fallen objects on the floor. API consumers must reset stateful tracking and
prediction when the clock resets.

The authored floor has a 6×4 m surface at world `z=0`. Scene loading disables
RAI's implicit ground quad using a four-component `Render/floorColor`, so two
coincident floor surfaces no longer compete during orbital viewing.

`camera.py` supplies an overhead `RGBDCamera` using the actual RAI renderer.
`ConveyorSimulation.get_observation()` returns RGB, metric optical-axis depth,
intrinsics, camera-to-world transformation and the simulation acquisition time.
Capture is explicit rather than part of every simulation step, limiting render
cost. Camera observations contain no object ground truth or segmentation IDs.
This provides inputs for Heysem's later perception module, not an implemented
detector or tracker. Rendering requires OpenGL/WSLg, including offscreen capture.

The static baseline remains `build_minimal_scene()` with `scenes/minimal.g` and
`scripts/view_scene.py`. It remains static and does not execute robot trajectories.

`robots.py` exposes immutable actual arm/finger coordinates and base/gripper
poses via `get_robot_states()`. `targets.py` exposes the two open destination
bins, their dimensions and nominal place/approach frames via `get_targets()`.
These interfaces use simulation time, world poses and explicit units; final
snapshots remain readable after close. The target geometries are included in
falling-object physics. Native bottom/wall contact tests do not establish arm
reachability or successful robot grasping: those remain integration checks with
Doğa's planner/controller.

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

- Validate full belt/gripper contact dynamics and control against the selected
  `robotic==0.4.1` backend; the isolated falling scene does not validate them.
- Verify both arms' reachable workspaces and refine placement before adding
  dual-arm scheduling or claiming collision-free coordinated execution.
- Choose an integration capture rate and confirm perception behavior under
  occlusion; camera calibration/depth conventions and clock reset are documented.
- Validate the release handoff with the later grasp controller and decide how
  to add contact-based belt physics if required by the experiments.
- Define covariance layout, grasp, trajectory,
  execution status, and failure/replanning interfaces before implementing them.
- Agree on association, track lifecycle, missing detections, interception timing,
  collision constraints, and replanning thresholds.
- Define repeatable evaluation scenes and metrics such as pick success,
  prediction error, planning time, and processing latency.
