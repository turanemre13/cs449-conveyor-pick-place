# Doğa — Progress Notes (Dual-Robot Motion Planning & Control)

Status as of 19 November 2026. The RAI demo runs the full pick-and-place cycle
for two moving objects using both Pandas. Work is on `feature/doga-motion-planning`.

## 1. What was implemented

- Built a free-moving tool center point (TCP) interception baseline and Panda
  IK/KOMO trajectories for fixed targets and moving objects. [Demo guide](../scripts/README.md)
- Added jaw closure, lift, transport, bin release, and retreat. The simulator
  attaches an object kinematically after both pads reach it and checks bin
  containment at release. [Pickup control](../src/conveyor/control/panda_pickup.py) · [Placement control](../src/conveyor/control/panda_placement.py)
- Integrated Emre’s RAI scene, Heysem’s RGB-D tracks, and Hüseyin’s prediction,
  grasp selection, and scheduling. Camera estimates drive the grasp targets.
  Simulator state binds track IDs to objects and checks collisions. The two arms
  can run concurrently or in sequence. [RGB-D handoff](../src/conveyor/control/rgbd_panda.py) · [Dual-arm control](../src/conveyor/control/dual_pandas.py)
- Added 5 Hz camera feedback until the grippers start descending. If two
  consecutive observations shift the predicted interception point by more
  than 5 mm, the arms hold while a replacement plan is validated from their
  current joint states. [Feedback playback](../src/conveyor/control/dual_pandas.py)
- Added recovery with the original track IDs after up to two missed approach
  frames. Replanning requires fresh measurements of both tracks. A sustained
  loss or a missing track at the descent checkpoint stops the batch.
- Added up to six alternatives after a rejected replacement. Trials use fresh
  grasp angles or later interception times within the scheduler’s pick windows.
  Both arms and their placements must pass shared validation before playback.

## 2. Verified results

The sparse layout used seed 21 with one box and one cylinder. Completion time
includes the retreat of both arms and is measured in simulation seconds.

| Mode | Placed | Completion time |
| --- | ---: | ---: |
| Concurrent, simulator object states at 0.022 m/s | 2/2 | 11.21 s |
| Concurrent, RGB-D at a constant 0.022 m/s | 2/2 | 12.14 s |
| Concurrent, RGB-D with speed changing from 0.022 to 0.025 m/s | 2/2 | 12.53 s |
| Concurrent, RGB-D with speed changing to 0.025 m/s and two missed cylinder detections | 2/2 | 12.57 s |

The speed-change run replanned at 1.11 simulation seconds. Conservative shape
bounds reduced the wall-clock hold from about 10 s to about 2 s on this layout.
Pairs near their clearance limits still use exact RAI signed distances at each
trajectory sample. The arms stay at the checkpoint if replacement validation
fails. [Clearance checks](../src/conveyor/planning/clearance.py)

The recovery test omitted two cylinder detections from real RAI RGB-D frames.
The same track ID returned and the batch replanned at 1.31 simulation seconds.
The original cylinder grasp failed contact checks. A second alternative
passed in about 2 wall-clock seconds and both objects reached their bins.

The full suite passed **242 tests** with RGB-D enabled. [RGB-D integration tests](../tests/test_rgbd_panda.py)

Run from the repository root with OpenGL/X11 available.

```bash
.venv/bin/python scripts/dual_panda_demo.py --perception --change-speed-to 0.025
CS449_TEST_RGBD=1 .venv/bin/python -m pytest -q
```

## 3. Speed, clutter, and arm assignment checks

| Mode and layout | Belt speed (m/s) | Result |
| --- | ---: | --- |
| RGB-D, sparse, seed 21 | 0.022 and 0.025 | 2/2 at each speed |
| RGB-D, sparse, seed 21 | 0.030, 0.050, and 0.080 | Rejected before motion. The downstream pick ran out of time. |
| RGB-D, sparse, seed 33 | 0.040 | 2/2 in 11.95 s. A picked the cylinder and B picked the box. |
| RGB-D, clutter, seeds 0 and 21 | 0.120 | The camera produced one current track for two objects. |
| Serial, sparse, seed 21 | 0.022 | 2/3 and 2/4 with simulator object states. |
| Serial, clutter, seed 0 | 0.120 | 2/2 with two objects. 0/4 and 0/6 in the denser batches. |

In the serial two-object clutter run, A reached the cylinder first. After A
retreated at 10.78 s, the remaining box was outside A’s reach. B picked it at
13.38 s. Restricting the same scheduler to A alone produced 1/2. Each arm uses
its own bin, and both arms have now placed both shapes in the RGB-D demo across
different layouts. The concurrent camera controller handles one two-object
batch at a time. [Dual-arm control](../src/conveyor/control/dual_pandas.py)

## 4. Incoming product stream

The `--stream` demo feeds four products from a compact covered feeder through one
continuous run. RGB-D tracks drive dispatch as the arms become available.
A prioritizes boxes and B prioritizes cylinders. A backup arm uses its own
bin, with checked free placement positions for successive products.

At 0.12 m/s with seed 7, the arms placed **4/4** in 52.20 simulation seconds.
Their motion overlapped for 10.26 s. A separate test with A unavailable
placed the arriving box using B. The viewer replays the checked camera-driven
preview and holds the final scene when the queue finishes.

## 5. Next steps

1. Test longer feeds and improve track separation for dense layouts.
2. Compare one and two arms, and compare the deadline-aware scheduler with FIFO
   on the same simulated scenarios. Report success rate, throughput, and
   completion time for the team report and demo.
