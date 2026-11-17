# cs449-conveyor-pick-place
Dynamic robotic pick-and-place of moving objects on a conveyor belt using RAI and Franka Panda.

This four-person CS449 Robotics project targets two Franka Panda arms picking
scattered, moving objects from a conveyor. The decision-making component will
assign objects and interception times to the available robots.

The current simulation contains two Pandas, seeded layouts of moving boxes and
cylinders, and an overhead RGB-D sensor. The default ordered layout spaces
objects along one lane; sparse and cluttered layouts remain available for
experiments. Transport on the belt is kinematic. In the default demo, unpicked
objects fall off the end using native RAI PhysX CPU dynamics, collide with the
floor and may tumble, rebound or roll. Both arms remain parked; perception,
tracking, scheduling and pick-and-place control remain pending.

Two open destination bins now provide nominal placement/approach frames for
Robot A and Robot B. `get_robot_states()` exposes actual joint/finger coordinates
and world poses; `get_targets()` exposes bin geometry and reference points.
Target reachability and full grasp/place trajectories still require planning
integration with Doğa.

## Setup

Use a project-specific virtual environment. The initial RAI backend is
`robotic==0.4.1`, matching the version recorded in the earlier course homework;
the development environment uses WSL Ubuntu 24.04 and Python 3.12. Do not reuse
or modify the homework virtual environment. Run from the repository root:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-rai.txt
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/python scripts/run_simulation.py
```

If `.venv` already exists, skip its creation. The default command opens the RAI
viewer using a working display, such as WSLg. Stop it with Q or Esc in the
viewer, the window's close button, or Ctrl+C in the terminal.
The default demo shows eight ordered objects using seed 7. Longitudinal edge
gaps are about 24 cm for this layout. Unpicked objects fall from the exit and
remain visible on the floor. Once all have settled, the demo waits three
simulation seconds, then replays the same layout and resets simulation time.
The conveyor chassis stays fixed while the visual belt seams loop over the top,
around the end rollers and along the bottom.

For a bounded run without a window:

```bash
.venv/bin/python scripts/run_simulation.py --no-view --duration 10
```

Compare another repeatable arrangement or save an initial camera sample:

```bash
.venv/bin/python scripts/run_simulation.py --scenario sparse --objects 6 --seed 12
.venv/bin/python scripts/run_simulation.py --speed 0.25 --no-repeat
.venv/bin/python scripts/run_simulation.py --scenario clutter
.venv/bin/python scripts/run_simulation.py --capture-dir outputs/demo
```

The capture directory contains `rgb.png`, metric depth and calibration arrays in
`rgbd.npz`, and separate camera, ground-truth and scenario JSON metadata. Open
`rgb.png` to inspect the sensor view; it is separate from the interactive viewer.
RGB-D rendering needs OpenGL/WSLg even when `--no-view` is used. Ordinary
`--no-view` stepping does not render or need a display.

`--scenario` selects `single`, `ordered` (default), `sparse` or `clutter`;
`--objects` sets the count (1–24, ignored by `single`), and `--seed` makes the
layout reproducible. Ordered layouts retain at least 10 cm longitudinal gaps;
counts that cannot fit are rejected with a request to reduce the count.
`--speed` sets non-negative belt speed in metres/second (default `0.12`), `--dt`
sets the simulation step in seconds (default `0.02`), and `--duration` limits a
run in seconds (`0` means unlimited and requires a viewer). `--exit-mode fall`
uses native falling dynamics; `--exit-mode remove` retains the earlier boundary
removal behavior. `--replay-delay 3` controls the wait after a completed batch.
`--no-repeat` keeps fallen objects on the floor instead of resetting. See
[scripts](scripts/README.md) for the static baseline and
more examples.

The core package still uses only the standard library. RAI is optional and loaded
only when building a scene. For interface tests and the synthetic demo:

```bash
.venv/bin/python -m pytest
.venv/bin/python scripts/interface_demo.py
```

Inspect actual robot/target interfaces without a graphics window:

```bash
.venv/bin/python scripts/inspect_environment.py --output outputs/environment.json
```

The export uses metres, radians, simulation seconds and `(w,x,y,z)` quaternions.
It explicitly records that target reachability has not yet been verified.

For future perception/tracking work, install the optional numerical and OpenCV
dependencies with `.venv/bin/python -m pip install -e '.[perception,dev]'`.
The core package and its interface demo do not require RAI.

Panda assets are resolved from the installed RAI package rather than copied from
the homework. See [scenes](scenes/README.md) for scene coordinates and scope.
The fall model inherits belt velocity and lets gravity/contact generate motion;
it adds no artificial random kicks. It uses an isolated physics scene with the
floor, conveyor supports and fallen bodies. It does not yet simulate belt
friction, gripper physics or contacts between fallen bodies and objects still
under kinematic belt control. The RAI pin is not a complete environment lock.

## Layout

```text
src/conveyor/
  __init__.py
  types.py           # Validated observations and tracked states
  pipeline.py        # Explicitly unimplemented integration entry point
  simulation/        # RAI scenes, seeded objects, conveyor motion and RGB-D
  perception/
  tracking/
  prediction/
  grasping/
  planning/
  control/
scenes/              # Authored RAI environment geometry
scripts/             # Runnable demos
tests/               # Interface checks and optional RAI scene/collision checks
experiments/         # Future evaluation code
docs/                # Architecture and interface contracts
reports/             # Report sources
```

The original root-level `simulation/`, `perception/`, `tracking/`, `planning/`,
and `control/` placeholders are preserved. Add Python code under `src/conveyor/`.
Resolve scenes and other resources from explicit arguments or paths relative to
their source files; avoid machine-specific absolute paths. Repository assets
outside `src/` are not bundled into the installed package.

## Team and integration

| Owner | Responsibility |
| --- | --- |
| Emre | Dual-arm simulation, conveyor, objects, RGB-D, reproducible scenarios and environment interfaces |
| Heysem | RGB-D perception and multi-object tracking |
| Hüseyin | Motion prediction, grasp selection, scheduling and robot–object assignment |
| Doğa | Dual-robot IK/KOMO planning, trajectory execution and gripper control |
| All four | Integration, experiments, reports and demonstration |

Planned flow: simulation → RGB-D perception → constant-velocity Kalman tracking
→ interception prediction → geometry-based grasp selection → robot/object/time
assignment → IK/KOMO planning → trajectory execution → simulated feedback and
optional replanning. Scheduling is planned; no scheduling module exists yet.

Shared observations and states use **world-frame** coordinates, **metres**,
**metres/second**, and **simulation seconds**. Angular quantities added later must
use **radians**. Camera-to-world conversion belongs to perception; wall-clock
timing must remain separate from simulation timestamps.

See [architecture](docs/architecture.md) and [interfaces](docs/interfaces.md) for
module boundaries, handoffs, and open integration decisions. Generated outputs
belong in `outputs/` or `experiments/results/`; keep selected report assets in
`reports/`.
