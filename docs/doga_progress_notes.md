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
  current joint states. [Feedback playback](../src/conveyor/control/dual_pandas.py#L312)

## 2. Verified results

The sparse layout used seed 21 with one box and one cylinder. Completion time
includes the retreat of both arms and is measured in simulation seconds.

| Mode | Placed | Completion time |
| --- | ---: | ---: |
| Concurrent, simulator object states at 0.022 m/s | 2/2 | 11.21 s |
| Concurrent, RGB-D at a constant 0.022 m/s | 2/2 | 12.14 s |
| Concurrent, RGB-D with speed changing from 0.022 to 0.025 m/s | 2/2 | 12.53 s |

The speed-change run replanned at 1.11 simulation seconds. KOMO and validation
took about 10 seconds of wall-clock time while the simulation clock held. Some
larger speed changes stopped the run when scheduling or contact checks failed.
The full suite passed **235 tests** with RGB-D enabled. [RGB-D integration tests](../tests/test_rgbd_panda.py)

Run from the repository root with OpenGL/X11 available.

```bash
.venv/bin/python scripts/dual_panda_demo.py --perception --change-speed-to 0.025
CS449_TEST_RGBD=1 .venv/bin/python -m pytest -q
```

## 3. Next steps

1. Reduce the replanning hold and test recovery from occluded tracks or failed
   replacement trajectories.
2. Test faster belts, more objects, and denser layouts across fixed seeds.
   Record missed picks and clearance failures.
3. Compare one and two arms, and compare the deadline-aware scheduler with FIFO
   on the same simulated scenarios. Report success rate, throughput, and
   completion time for the team report and demo.
