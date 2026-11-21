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

## Free TCP interception

Run `.venv/bin/python scripts/free_tcp_demo.py` to see a teal gripper approach
one moving box, match its speed, and measure alignment at the time assigned by
the scheduler. The viewer repeats the interception and closes with Q or Esc.
Add `--no-view` for one headless pass or `--speed` to vary the belt speed. Both
Pandas stay parked. The demo measures geometric interception with an open
gripper. Closing the jaws, attaching the box, and Panda motion are later stages.

## Static Panda motion with KOMO

Run `.venv/bin/python scripts/static_panda_demo.py` to see Panda A move to a
fixed pose above the stopped belt while Panda B stays parked. The demo
checks joint limits, joint speed, and separation from collidable geometry at
the playback samples, then reports the achieved TCP pose error. Add `--no-view`
for one headless pass. Close the viewer with Q or Esc.

## Scheduled moving-box interception with Panda

Run `.venv/bin/python scripts/moving_panda_demo.py` to see the selected Panda
approach the orange box on the belt and match its velocity at the scheduled
time. Add `--no-view` for one pass or `--speed 0.20` to try another belt speed.
The jaws remain open during this alignment demo. Position, relative speed,
timing, and orientation errors are measured against the live simulation.

Add `--lift` to close the jaws while tracking the box, confirm bilateral jaw
contact, and lift it approximately 12 cm. The simulator uses a rigid kinematic
attachment once contact is verified. The lift checks clearance for the arm and
the carried box. The viewer replays the cycle after a short pause.

Add `--place` to run the complete pick-and-place sequence for one moving box.
The demo transports it to the selected Panda's bin, checks that it reaches the
floor within tolerance, opens the jaws, and verifies that the box stays inside
while the arm retreats. The box is detached from the gripper at release and
held in its world pose for this kinematic demo.

Add `--perception --place` to schedule the single-box run from live RGB-D and
Kalman tracking. The belt advances for 0.50 s while the camera confirms the
track. Its estimated position, velocity, dimensions and yaw feed scheduling,
and the simulator's frame identity and collision geometry are used for
collision checks and attachment. This mode requires the `perception` extra
and an OpenGL/X11 display, including with `--no-view`.

## Dual-Panda coordination

Run `.venv/bin/python scripts/dual_panda_demo.py` to see both Pandas place
objects from the same running conveyor. The default sparse layout uses seed 21
and a belt speed of 0.022 m/s. The two plans are combined and checked with both
arms, the held objects, and the moving belt objects in one scene before the
viewer opens. Playback applies joint commands and attachment changes at their
recorded simulation times. Use `--serial` for the same layout with one arm
moving at a time and rescheduling after each placement. `--no-view` reports
one batch, and Q or Esc closes the viewer.

Add `--perception` to prepare the concurrent batch from live RAI RGB-D and
Kalman tracks. The camera confirms the two separated objects over 0.50 s and
continues sampling at 5 Hz while the arms approach. Sampling stops when the
first arm starts to descend, where the grippers can occlude the objects.
The controller allows up to two missed frames while the original track ID
can still be recovered. Replanning waits for fresh measurements of both
original tracks. A third missed frame or a track still missing at the descent
checkpoint stops the batch before either arm descends.
Track estimates drive assignment and TCP targets. The simulator binds track
IDs to scene frames, validates both pad distances, and checks both trajectories
together. If two consecutive measurements move a predicted interception point
by more than 5 mm, motion holds at the current approach pose while KOMO builds
and validates a replacement timeline from both current joint states. Playback
resumes only after validation succeeds. If the first replacement is rejected,
the controller tries up to six alternatives using ranked grasp angles and
nearby later times inside the scheduled pick windows. The simulation clock
stays fixed during these attempts. The camera mode requires the `perception`
extra and an OpenGL/X11 display, including with `--no-view`.

The default layout can demonstrate feedback with a speed change at 0.75 s. The
controller compensates the RGB-D tracks for the commanded belt motion.

```bash
.venv/bin/python scripts/dual_panda_demo.py --perception --change-speed-to 0.025
```

## Incoming product stream

Run `.venv/bin/python scripts/dual_panda_demo.py --stream` to see four products
advance from a compact, covered feeder onto the main belt in one run.
Queued products enter beneath the cover and slide out onto the short infeed.
The viewer keeps the final scene when the queue finishes.
Add `--no-view` for a single terminal result or `--objects` to change the
initial queue size. This mode uses the `perception` extra and OpenGL/X11.

RGB-D tracks identify each arriving product. A prioritizes boxes and B
prioritizes cylinders. When the preferred arm cannot make the pick window,
the scheduler can assign the other arm, which uses its own bin. The arms
reserve separate jobs and place products in free bin positions. A preview
with successive camera frames validates the shared motions before playback.

Run `.venv/bin/python scripts/dual_panda_demo.py --showcase` for the nine-product
demo. It starts with alternating boxes and cylinders, then shows three closely
spaced boxes with a sphere before and after the burst. The spheres and any
missed products fall into the end bin using RAI's CPU PhysX dynamics. The display
counts arm placements and products inside the end bin separately. Each cycle
holds its final view for five seconds before it repeats. `--no-view` runs one cycle.
Add `--front-view` to frame both arms, the feeder, and the end bin from the front.
