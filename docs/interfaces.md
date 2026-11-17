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

The state above is synthetic; it does not run a filter. These perception/tracking
records do not yet represent covariance, orientation, object dimensions,
confidence, camera frames or robot trajectories. The simulation ground-truth
records described below are a separate source of evaluation data; they do not
replace camera measurements in the future perception pipeline.
When covariance is introduced, document its coordinate frame, state ordering,
and mixed units explicitly.

## Simulation interface

`ConveyorSimulation(belt_speed_m_s=0.12, *, scenario="single", seed=7,
object_count=8, exit_mode="remove")` provides a dual-arm environment. The API
defaults preserve the original single-box boundary-removal case. The CLI
defaults to eight ordered objects with `exit_mode="fall"`. Presets are `single`,
`ordered`, `sparse` and `clutter`. Counts from 1 to 24 apply to non-single
presets, subject to available space. A seed reproduces dimensions, colors,
orientations and initial positions.
The RAI backend is loaded only when constructing a scene or simulation.

| Member | Contract |
| --- | --- |
| `config` | The shared `robotic.Config`; robot frames are prefixed `a_` and `b_` |
| `simulation_time_s` | Seconds since the latest reset |
| `object_specs`, `object_ids` | Initial immutable specs and stable IDs for this scenario |
| `active_object_ids` | IDs still visible, including falling, settled and released objects |
| `all_objects_finished` | Whether every object is settled or removed; released objects keep the batch active |
| `object_at_exit` | Legacy check for whether the first object has left belt transport |
| `reset(seed=None)` | Clear falling bodies, restore the seeded batch and time zero; optionally select a new seed |
| `step(dt)` | Advance belt transport and any falling-body dynamics by `dt` simulation seconds |
| `release_object("object_001")` | Relinquish kinematic/physics ownership; does not create a grasp |
| `get_ground_truth(active_only=False)` | Return stable IDs mapped to current active states and, by default, historical exit snapshots |
| `get_robot_states()` | Immutable snapshots of the two arms, inactive finger joints and world poses |
| `get_targets()` | Authored bin geometry and nominal approach/placement references |
| `get_observation()` | Explicitly render a `CameraObservation` at current simulation time |
| `view(message="")` | Open/update the optional nonblocking viewer, preserving it across resets |
| `close()` | Release resources and retain final object/robot/target snapshots; `reset()` reopens the simulation |

`ObjectSpec` records `object_id`, `geometry_type`, full `dimensions_m`, initial
world `position_m`, `yaw_rad` and `color_rgb`. The pure-standard-library
`generate_scenario(...)` helper performs bounded seeded placement; it does not
import RAI or change global random state. Initial rotated footprints have at
least 8 mm object clearance and 5 mm margin from belt edges. `ordered` uses one
central lane, ascending ID/x order and at least 10 cm longitudinal edge gaps.
An ordered batch that cannot fit those gaps is rejected rather than compressed.

`ObjectGroundTruth` is a frozen snapshot with these fields:

| Field | Meaning |
| --- | --- |
| `object_id: str` | Stable identifier such as `object_001` |
| `simulation_time_s: float` | Current simulation time for active states; exit-event time for archived states |
| `position_m: tuple[float, float, float]` | World position `(x, y, z)` in metres |
| `quaternion_wxyz: tuple[float, float, float, float]` | World orientation in `(w, x, y, z)` order; unitless quaternion components |
| `dimensions_m: tuple[float, float, float]` | Full local extents in metres; cylinder dimensions are `(diameter, diameter, height)` |
| `velocity_m_s: tuple[float, float, float] \| None` | World linear velocity from belt transport or native falling dynamics; unknown after release |
| `motion_state: str` | `on_belt`, `falling`, `settled`, `released`, or `exited` in remove mode |
| `geometry_type: str` | `box` or `cylinder` |
| `exit_time_s: float \| None` | Belt-to-physics handoff time or boundary-removal time; otherwise absent |
| `frame` | Class constant `"world"`; not an initializer or serialized dataclass field |

Pose, dimensions and known velocity use immutable tuples. `on_belt` velocity is
`(belt_speed_m_s, 0, 0)`. In fall mode, the object's centre crosses the belt end
before native PhysX takes ownership with that linear velocity and no initial
spin. `falling` and `settled` states retain current poses, current simulation
timestamps and native velocities; they remain visible in camera observations.
Gravity/contact determine impact and rolling without artificial random kicks.

In remove mode, the object is removed at its last fully supported exit position
and its terminal `exited` snapshot is archived with zero terminal velocity.
This historical event is not a visible object at the current camera time. Use
`active_only=True` to exclude archived removals and obtain live ground truth
synchronized with a camera observation; fallen/settled objects remain included.
`released` velocity is `None` because an external controller owns its motion.
Releasing a falling body also removes native physics ownership, leaving its
current visual pose intact. Release alone does not create a physical grasp.

World `+z` is up and belt travel is `+x`. `build_dual_arm_scene(scene_path=None)`
returns the dual-arm geometry without a stepping loop.
`build_minimal_scene(scene_path=None)` still returns an independent
`robotic.Config` for the original single-arm static scene. The geometry builders
alone do not capture camera data.

The animated demo defaults to replay after all objects are settled or removed,
plus `--replay-delay 3` simulation seconds. It restores the same seeded scene and
resets the simulation clock. Consumers must restart tracking after each reset.
`--no-repeat` disables replay and retains fallen bodies on the floor. This is a
batch replay, not a continuous object stream or physical respawn model.

## Robot and destination interfaces

`get_robot_states()` returns `RobotState` records keyed by `robot_a` and
`robot_b`. Each record supplies `simulation_time_s`, `base_frame`,
`gripper_frame`, their world `*_position_m` and `*_quaternion_wxyz` poses,
`arm_joint_names`, `arm_joint_positions_rad` and `arm_joint_limits_rad`.
The seven arm values and limit pairs follow joint-1 through joint-7 order.
Values are resolved by name, so unrelated active joints do not shift this map.

`finger_joint_names` and `finger_joint_positions_m` report both model prismatic
coordinates, including the inactive mimic joint. These are not physical jaw-gap
or grasp-success measurements. Controller availability and velocity are not
inferred from a stationary scene. Records and their tuples are immutable;
`frame` is the class constant `world` and quaternions use `(w,x,y,z)` order.

`get_targets()` returns `TargetBin` records keyed by `target_bin_a` and
`target_bin_b`. Fields are `target_id`, `preferred_robot_id`,
`simulation_time_s`, `bin_frame`, `position_m`, `quaternion_wxyz`,
`inner_dimensions_m`, `place_frame`, `place_position_m`, `approach_frame` and
`approach_position_m`; `frame` is again `world`. The root pose denotes the
interior-floor centre. Default roots are `(-1.0,-0.85,0.75)` and
`(1.0,-0.85,0.75)` metres, with inner dimensions `(0.30,0.24,0.14)` metres.
Nominal placement references are 0.10 m above that floor and approaches 0.30 m
above it. They are planning reference points, not universal object-release or
gripper poses. Preferred A/B associations do not imply verified IK reachability.

Both getters use current simulation time and retain final snapshots after
`close()`. Reset restores initial arm and inactive-finger coordinates and
clears those snapshots. `scripts/inspect_environment.py` prints/exports these
interfaces without a viewer. `--capture-dir` also writes `robot_states.json`
and `targets.json` separately from RGB-D data.

RAI 0.4.1 needs explicit collision-cache invalidation after frame motion and
structural rebuilding after removals. The simulation handles this while keeping
the same public `Config`. Consumers should retain frame names, not borrowed
`rai.Frame` handles across reset/removal; read new snapshots or resolve names
again. Use `simulation.view(...)` for a GUI that persists across these resets.

## RGB-D observations

`RGBDCamera(config).capture(simulation_time_s)` renders the actual scene through
RAI's offscreen `CameraView`. `ConveyorSimulation.get_observation()` owns this
sensor lazily and supplies the current simulation timestamp. Rendering needs a
working OpenGL/WSLg display, even with the demo's `--no-view` option. Ordinary
stepping does not render.

`CameraObservation` contains `simulation_time_s`, `rgb`, `depth_m` and
`calibration`. RGB is a read-only owned `uint8` array of shape `(360, 640, 3)`;
depth is a read-only owned `float32` array of shape `(360, 640)` in metres.
Invalid/out-of-range depth is `NaN`. Depth is optical-axis z, not distance along
a pixel's viewing ray. Neither object states nor segmentation IDs are embedded
in the observation. No sensor noise or lens distortion is currently simulated.

`CameraCalibration` supplies `width`, `height`, a 3×3 `intrinsics` tuple, a 4×4
`camera_to_world` tuple, `camera_frame="camera_rgbd"`, `world_frame="world"`
and `depth_convention="optical_axis_z_m"`. Pixels have `u` right and `v` down;
optical axes are +x right, +y down and +z forward. The camera is at world
`(0, 0, 2.5) m` with quaternion `(0, 1, 0, 0)`, so optical +x maps to world +x,
optical +y to world -y, and optical +z to world -z.

The camera targets the belt. The deck can naturally occlude fallen objects on
the floor beyond its end; a live ground-truth record does not imply visibility
in this sensor. Move the camera or use the interactive viewer to inspect falls.

`calibration.deproject_pixel(u, v, depth_m)` converts a valid pixel and positive
optical depth into a world point. Perception should use that calibration and
camera measurements; ground truth remains a separate evaluation source.
`--capture-dir` saves the initial RGB-D pair and matching metadata for inspection.

## Integration entry point

`conveyor.pipeline.run()` is reserved for future orchestration and currently
raises `NotImplementedError`. `scripts/interface_demo.py` still constructs
synthetic perception/tracking records only. The separate
`scripts/run_simulation.py` runs the kinematic dual-arm environment, while
`scripts/view_scene.py` displays or checks the static single-arm baseline.
