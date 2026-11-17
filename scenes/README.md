# RAI scenes

`dual_arm.g` contains the current environment: floor, conveyor chassis and
rollers, two robot mounts and a baseline box. `build_dual_arm_scene()` adds two Panda
models supplied by the installed `robotic==0.4.1` package with separate `a_` and
`b_` prefixes. Robot models and meshes are not copied into this repository.

The industrial appearance uses a graphite belt, aluminum-colored chassis,
crossmembers, bearing caps, a small drive housing and pedestal plates. These
`visual_*` details have `contact: 0`; planning and falling dynamics continue to
use the simpler collidable floor, deck, supports and robot geometry.

Initial world conventions (metres and radians):

- `+z` is up and conveyor travel is along `+x`.
- The belt is `2.4 m` long and `0.4 m` wide, centered at `y=0`; its top is at
  `z=0.75 m`.
- Robot A mounts at `(-0.55, -0.60, 0.75) m` and Robot B at
  `(0.55, -0.60, 0.75) m`, each with yaw `+pi/2`, facing the belt.
- Panda frames include `a_panda_base`, `a_gripper`, `b_panda_base` and `b_gripper`.
- The authored floor is 6×4 m with its top at `z=0`. Scene loading disables
  RAI's implicit ground quad, avoiding coincident floor surfaces and orbital
  depth-fighting artifacts.

This sequential placement on the same side is preliminary; it does not guarantee
reachability or safe coordinated motion. The simulation keeps both arms parked
with joint 2 at `-1.0 rad` to clear the camera view; scene builders alone retain
the bundled model's default joint values.
Run `.venv/bin/python scripts/run_simulation.py` from the repository root to
animate the belt with eight ordered objects, or add `--no-view --duration 10` for a bounded run
without a window. The chassis remains still while visual belt seams loop over
the top, around the rollers and along the bottom. Belt transport is kinematic;
unpicked objects use native PhysX gravity/contact after crossing the end in the
default `--exit-mode fall`. Scheduling and robot control remain pending.

`ConveyorSimulation` replaces the baseline box with the seeded `single`,
`ordered`, `sparse` or `clutter` layout from `simulation/scenarios.py`. Boxes and upright cylinders
have varied dimensions and yaw; their rotated footprints remain inside the belt
with initial edge margin and object clearance. `single` preserves the old box.
The ordered default uses a central single file with at least 10 cm longitudinal
edge gaps (about 24 cm for eight seed-7 objects); infeasible counts are rejected.
Clutter remains available as a separate challenge scenario.

Fallen objects remain visible on the floor and may rebound, tumble or roll.
The separate CPU physics configuration includes static floor, belt/supports
and fallen bodies, with inherited belt velocity and no artificial random kicks.
It does not model belt friction, gripper contacts or interactions with objects
still following prescribed belt motion. Replay waits until all objects are
settled or removed, then adds three simulation seconds before resetting time.
Use `--no-repeat` to retain floor objects, or `--exit-mode remove` for the earlier
boundary-removal comparison.

The optional overhead `camera_rgbd` is created lazily for RGB-D capture at
`(0, 0, 2.5) m`, looking downward. It covers the belt using a 640×360 sensor;
calibration and metric optical-axis depth accompany observations. Capture uses
OpenGL/WSLg even without an interactive window. Use `--capture-dir outputs/demo`
to save the initial camera sample separately from the viewer image.

Two open bins stand outboard of the robots: blue `target_bin_a` at
`(-1.0,-0.85,0.75)` and amber `target_bin_b` at `(1.0,-0.85,0.75)` metres.
Their roots denote the interior-floor centres, with `inner_size_m` metadata
`(0.30,0.24,0.14)`. Five collidable plates form each bottom and four walls,
with separate supports; the opening is clear. The falling physics scene copies
this geometry too. `*_place` and `*_approach` frames mark nominal heights
0.10 m and 0.30 m above the interior floor. Placement/approach points and the
preferred robot metadata are preliminary; Doğa's planner must check reach and
choose object/gripper poses.

`minimal.g` and `build_minimal_scene()` preserve the first static baseline with
one robot at `(0, -0.60, 0.75) m`, a conveyor surface and one box. Run
`.venv/bin/python scripts/view_scene.py` to view it, or add `--no-view` to check
scene loading and initial penetrations without a window.

Resolve resources relative to their source files or explicit arguments. Never
use machine-specific absolute paths in committed scene code. These repository
assets are intended for a source checkout and are not bundled into the Python
package.
