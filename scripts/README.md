# Runnable demos

Run from the repository root after installing the package in editable mode and
the RAI backend from `requirements-rai.txt`:

```bash
.venv/bin/python scripts/run_simulation.py
```

This opens the dual-arm conveyor demo using a working display (WSLg on
Windows/WSL). The arms stay parked while eight mixed boxes/cylinders move along
the belt in a seeded ordered arrangement with visible gaps. The
chassis is fixed; visual belt seams loop across the top, around the rollers and
along the bottom. Belt transport is kinematic; unpicked objects fall using
native RAI PhysX CPU gravity/contact, retaining belt velocity. Impact can cause
rebound, tipping and rolling without artificial random kicks. Pick-and-place
control and belt/gripper contact physics remain pending. Stop with Q or Esc in the viewer, the window's close
button, or Ctrl+C in the terminal.

| Option | Meaning |
| --- | --- |
| `--scenario ordered` | Layout preset: `single`, `ordered` (default), `sparse` or `clutter` |
| `--objects 8` | Object count, 1–24; ordered layouts must fit at least 10 cm edge gaps; `single` uses one |
| `--seed 7` | Seed for repeatable dimensions, orientations and placement |
| `--speed 0.12` | Non-negative conveyor speed in metres/second |
| `--dt 0.02` | Simulation step in seconds |
| `--duration 0` | Run indefinitely by default; a positive value limits the run in seconds |
| `--no-view` | Run without a window; requires a positive `--duration` |
| `--exit-mode fall` | Native falling dynamics (default); `remove` retains boundary removal |
| `--replay-delay 3` | Simulation seconds to wait after all objects settle or are removed |
| `--no-repeat` | Disable replay and keep fallen objects on the floor |
| `--capture-dir outputs/demo` | Save an initial RGB-D pair plus calibration, scenario and separate ground truth |

Example bounded runs:

```bash
.venv/bin/python scripts/run_simulation.py --no-view --duration 10
.venv/bin/python scripts/run_simulation.py --speed 0.25 --no-repeat
.venv/bin/python scripts/run_simulation.py --scenario sparse --objects 6 --seed 12
.venv/bin/python scripts/run_simulation.py --scenario clutter
.venv/bin/python scripts/run_simulation.py --exit-mode remove
.venv/bin/python scripts/run_simulation.py --scenario single
```

In fall mode, objects stay visible after leaving the belt and continue under
native physics until they settle. In remove mode they leave the scene at the
last fully supported boundary. Replay starts only after all have settled or
been removed, plus the requested delay; it restores the same seeded scenario
and simulation time zero. `--no-repeat` lets the settled floor arrangement
remain indefinitely. This is not continuous spawning or physical respawn;
stateful consumers must reset accordingly.
See `--help` for the supported arguments. These commands document usage and do
not imply a validation result.

Save the initial overhead camera view:

```bash
.venv/bin/python scripts/run_simulation.py --capture-dir outputs/demo
```

Open `outputs/demo/rgb.png` for RGB. `rgbd.npz` contains RGB, depth in metres,
intrinsics and camera-to-world arrays; inspect its keys with `numpy.load` when
integrating perception. `camera.json` describes calibration; `ground_truth.json`
and `scenario.json` are separate debugging/evaluation metadata, not perception
inputs. Depth is optical-axis z, and invalid samples are NaN. Capture is at
simulation time zero; this option does not record a full image sequence.

`robot_states.json` adds the actual arm/finger coordinates and base/gripper
poses; `targets.json` adds destination-bin geometry and nominal references.
Both are separate integration metadata, not camera perception inputs.

Inspect/export robot and target interfaces without opening a viewer:

```bash
.venv/bin/python scripts/inspect_environment.py --output outputs/environment.json
```

The output marks reachability as unverified. The blue/amber bins in the main
demo are physical open containers, but no automatic picking or placing is run.

Capture requires OpenGL/WSLg even with `--no-view`; stepping without capture
does not. Rendering adds no Pillow or OpenCV dependency. The camera view and
interactive viewer are separate; no object detection or tracking runs yet.

The earlier single-arm static baseline remains available:

```bash
.venv/bin/python scripts/view_scene.py --no-view
.venv/bin/python scripts/view_scene.py
```

`--no-view` loads and checks that static scene without opening a window.

The independent interface demo prints synthetic records and does not require
RAI:

```bash
.venv/bin/python scripts/interface_demo.py
```
