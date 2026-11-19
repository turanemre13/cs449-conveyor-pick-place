# Hüseyin — progress notes (prediction, grasp selection, dual-arm scheduling)

Status as of 18 November 2026 (test results updated 19 November 2026: full
suite 189 passed, 0 failed, 0 skipped in WSL + RAI with RGB-D enabled; see §6),
for the 23 November progress report. Everything
below is implemented in pure Python (no RAI import in the modules) and tested;
RAI is used only in the optional integration tests. All scheduling numbers are
**planning-level estimates**: no robot motion was executed, no IK/KOMO
trajectory was planned for a pick, and no grasp was physically simulated.

Four levels must not be confused:

| Level | Status |
| --- | --- |
| Predicted scheduling feasibility (approximate reach/timing/geometry model) | implemented, reported below |
| Simulated interception (robot moves in RAI and meets the object) | **not done** (needs Doğa's planner/controller) |
| Actual robot execution | **not done** |
| Physical grasp success | **not verified**; every grasp has `physically_verified=False` |

## 1. What was implemented

| Module | Files | Purpose |
| --- | --- | --- |
| Motion prediction | `src/conveyor/prediction/constant_velocity.py`, `ground_truth.py` | Constant-velocity prediction on the shared `TrackedState`; ground-truth adapter standing in for the tracker |
| Grasp selection | `src/conveyor/grasping/geometry.py`, `top_down.py` | `ObjectGeometry`, top-down parallel-jaw candidates for boxes/cylinders, width/height/neighbour-clearance checks, motion-aware ranking, RAI gripper orientation |
| Scheduling | `src/conveyor/scheduling/` (`models`, `feasibility`, `base`, `greedy`, `baseline`, `metrics`, `adapters`, `online`, `streams`) | Deadline- and reachability-aware greedy dual-arm scheduler, FIFO/nearest-robot baseline, metrics, rolling re-planning, synthetic arrival streams |
| Evaluation | `experiments/scheduling_comparison.py` | Reproducible comparison (fixed seeds, identical inputs, paired statistics) |
| Demo | `scripts/huseyin_demo.py` | Terminal demo of the three modules, no RAI/viewer needed |
| Tests | `tests/test_prediction.py`, `test_grasping.py`, `test_scheduling.py`, `test_huseyin_rai_integration.py` | 126 pure-Python tests + 6 RAI/KOMO integration tests |

No teammate module was modified. The `prediction` and `grasping` package
docstrings were updated from "placeholder" (owner Hüseyin, as in
`docs/architecture.md`).

## 2. Review fixes (18 November 2026)

An internal technical review found the issues below; each was reproduced
before the fix and is covered by a regression test.

| ID | Problem (before) | Fix | Regression tests |
| --- | --- | --- | --- |
| G1 | `quaternion_wxyz` used `Rz(ψ)·Rx(π)` (+z **down**). The RAI Panda `gripper` frame has +z towards the wrist (**up** in a top-down grasp): KOMO could not reach the old orientation anywhere. | `quaternion_wxyz` is now the RAI gripper frame `Rz(ψ)` (x = closing axis, z = up). The old frame is kept as `approach_frame_quaternion_wxyz` and converted explicitly: `q_rai = q_approach · Rx(π)`. | `test_grasp_quaternion_matches_rai_panda_gripper_frame`, `test_rai_gripper_quaternion_is_pure_yaw_in_wxyz_order`; RAI: `test_rai_panda_gripper_frame_convention`, `test_komo_accepts_the_grasp_quaternion_but_not_the_old_z_down_frame` |
| G2 | Ranking ignored object motion; on ties the jaws closed **along** the belt (cylinders, square boxes), putting the upstream finger in the object's path during the 0.4 s descent (4.8 cm at 0.12 m/s). | Optional `object_velocity_m_s`; ranking adds `motion_alignment_weight · |sin(closing axis, motion)|` (default 0.25). The scheduler passes the tracked velocity. | `test_moving_*`, `test_motion_*`, `test_scheduled_cylinder_grasp_closes_across_the_belt_with_rai_orientation` |
| S1 | A committed object whose track time was later than `now` crashed planning (backward prediction). | Future-dated tracks are never extrapolated backwards. Uncommitted → rejected `future_dated_track`; committed → commitment kept (robot conflicts, "picked" status) but object reported in `ScheduleResult.excluded_committed`. Stale uncommitted → `stale_track`. | `test_future_dated_committed_track_does_not_crash_and_is_reported` (original input) and 3 related tests |
| S2 | An unassigned object could get the reason `""` (FIFO left a still-feasible object). | Reason `feasible_but_not_selected` when a robot can still intercept it; infeasibility reasons otherwise. | `test_unassigned_but_feasible_objects_are_reported_as_not_selected` (clutter-8, seed 6, 0.05 m/s), `test_rejection_reasons_are_never_empty` |
| S3 | `greedy.py` docstring described a critical-first rule, but the default is `urgency="system"`. | Docstring now describes the implemented rule and all three policies; default unchanged. | — (documentation) |
| S4 | Slack mixed travel time at the interception time with the deadline at the window end (e.g. 5.393 s reported vs 5.299 s consistent). | `latest_start = max_t∈window (t − travel(t))` with both terms at the same sample; `slack = latest_start − t_available`. New field `latest_start_s`. | `test_slack_uses_travel_time_at_the_deadline_not_at_the_interception`, `test_slack_matches_brute_force_*` (4 availabilities × 3 speeds), `test_latest_start_is_independent_of_availability_within_one_window` |
| R | The spherical reach model was optimistic near its boundary. | `ReachModel.safety_margin_m` default 0.0 → **0.03 m**, calibrated with KOMO (see §5). | `test_default_reach_model_keeps_a_calibrated_safety_margin`, `test_overlapping_workspace_survives_the_safety_margin`; RAI: `test_reach_model_boundary_points_are_komo_feasible` |
| M | `throughput_per_min` divided by each method's own makespan: fewer picks finishing earlier could look better. | `throughput_per_min` now uses a fixed, input-derived observation window (same for both methods); the old value is `processing_rate_per_min` (descriptive only). | `test_processing_rate_can_favour_fewer_picks_but_fixed_window_throughput_does_not`, `test_observation_window_is_derived_from_inputs_only`, `test_online_throughput_uses_the_same_window_for_both_methods` |

## 3. Motion prediction

Constant-velocity model in the world frame (metres, seconds since simulation start):

```
p̂(t + τ) = p̂(t) + v̂(t) · τ,      τ = t_target − t ≥ 0
```

`ConstantVelocityPredictor` (input: `conveyor.types.TrackedState`):

- `predict(state, t_target)` → `PredictedState` (position, velocity used, horizon, stationary flag).
- `predict_many(states, t)` for multiple objects with asynchronous track timestamps (duplicate IDs rejected).
- `predict_trajectory(state, times)` for candidate interception times.
- `time_to_reach_coordinate(state, axis, value)` gives deadlines such as the belt exit.
- Speeds ≤ `stationary_speed_m_s` (1 mm/s) are treated as zero; `planar_motion=True` ignores v_z.
- Invalid input (non-finite/negative/backward target times, horizons above `max_horizon_s`, wrong types) → `ValueError`/`TypeError`.

Verified against the RAI kinematic belt: 2 s prediction error 2.8e-16 m.
Until Heysem's tracker is available, `tracked_states_from_ground_truth`
converts Emre's `ObjectGroundTruth` records (only `on_belt` objects).

## 4. Grasp selection

`select_top_down_grasps(geometry, position, neighbors=..., config=GripperConfig(), object_velocity_m_s=None)`

**Frames** (world frame, quaternions `(w, x, y, z)`):

- **TCP** `position_m`: midway between the finger pads = origin of RAI's
  `a_gripper`/`b_gripper` frame (0.1035 m from `panda_hand_joint`, verified in RAI).
- **RAI gripper frame** `quaternion_wxyz`: RAI `panda.g` defines
  `gripper(panda_joint7): d(180 0 1 0) d(135 0 0 1) t(0 0 -.2105)`, so +x is the
  finger opening/closing direction and +z points from the TCP to the wrist.
  A top-down grasp with closing heading ψ is `Rz(ψ)` = `(cos ψ/2, 0, 0, sin ψ/2)`.
  This is the orientation to request for the gripper frame in KOMO
  (equivalently `vectorZ = [0,0,1]`, `vectorX = closing_axis_world`).
- **Approach frame** `approach_frame_quaternion_wxyz`: x = closing axis, z =
  approach direction (down). Readability only; do not send it to RAI.

**Candidates**: boxes — closing along local x or y (ψ or ψ + π/2), wrapped to
(−π/2, π/2]; cylinders — `cylinder_yaw_samples` (8) directions in [0, π).

**Hard checks** (unchanged): required opening = width + 2 × 5 mm ≤ 80 mm
(RAI finger joints 2 × 0.04 m), width ≥ 10 mm; TCP depth min(25 mm, h/2) below
the top, raised to keep fingertips ≥ 3 mm above the belt, ≥ 10 mm contact;
open finger pads (12 × 20 mm) ≥ 5 mm from every neighbour footprint.

**Ranking** = 0.5·opening margin + 0.3·pad contact + 0.2·neighbour clearance
(saturating at 5 cm) **+ 0.25·|sin(closing axis, motion)|** when the object
moves (> 1 mm/s). The motion term only reorders feasible candidates.
Study on 960 generated objects at 0.12 m/s: best grasps with |sin| ≥ 0.9 rise
from 119 (no motion term) to 672 (weight 0.25); a weight of 0.5 gains only 2
more but raises near-limit openings (> 75 mm) from 13 to 19. Trade-off: a
perpendicular closing axis may need a wider opening (less margin for lateral
error) and puts the pads across the belt, where lateral neighbours can be
closer; the clearance check still applies to every candidate.

The preference does **not** replace velocity-matched interception (the gripper
should track the object during descent and closing) or physical grasp
validation. Neighbour heights and the hand body are still ignored (2-D pads only).

## 5. Dual-arm scheduling

### Feasibility model (shared by both methods)

All times are absolute simulation seconds. For robot *i* and object *j*:

- Predicted path p_j(t) is sampled every `time_step_s` until the horizon or until
  all objects leave the pickable belt region (x ∈ [−1.2, 1.15] m, |y| ≤ 0.2 m).
- **Reachability (approximate):** the wrist point (grasp TCP + 0.1034 m) must lie
  within 0.855 − **0.03** m of the shoulder (base + 0.333 m) and ≥ 0.25 m
  horizontally from the base. *Not an IK check.*
  - Calibration: KOMO (RAI `panda.g`, robot A, 251 belt points at TCP heights
    0.775/0.81 m, position + top-down `vectorZ` (+ optional `vectorX`), joint
    limits, **no collision objectives**). Without margin 38–47 points the sphere
    accepted had no IK solution, all within 0.031 m of the sphere boundary.
    With 0.03 m: 0 optimistic points at 5 mm IK tolerance (5 borderline ones at
    1 mm tolerance) and 3–5 IK-solvable points excluded.
  - The robots' shared region remains (|x| ≲ 0.15 m near the robot-side edge);
    the far edge around x ≈ 0 is reachable by neither arm.
- **Travel time:** 0.25 s overhead + |pre-grasp − tool| / 0.4 m/s + 0.4 s descent.
- **Busy time after interception:** close 0.3 s + lift 0.3 s + |bin approach − grasp| / 0.4 m/s + release 0.3 s.
- Earliest feasible interception t: inside the window, `t_available + t_travel(t) ≤ t`,
  no overlapping pick by the other robot within 0.15 m, and a geometrically
  feasible grasp against the neighbours still on the belt at t.
- **Slack:** `S_ij = latest_start_ij − t_available,i`, with
  `latest_start_ij = max over samples t of the reach window (t − t_travel(t))`
  (travel time and target time at the same sample). It is the time the robot's
  departure can still be postponed. The window is geometric, so S is optimistic
  with respect to later neighbour/conflict changes.

### Proposed method: `DeadlineAwareGreedyScheduler`

1. Compute every feasible (robot, object) option and its slack.
2. Dispatch the robot that becomes free first (ties by robot ID).
3. Rank options by urgency (then earliest interception, object ID):
   `urgency="system"` (default) uses `max_k S_kj` over all robots that can still
   catch j (objects only this robot can catch keep their own small slack);
   `"critical"` puts objects no other robot can catch first, then least slack;
   `"robot"` uses this robot's slack only.
4. **Gap filling:** take an earlier object first if the chosen one is still catchable afterwards.
5. Commit, move the robot's availability/position to its bin, repeat until no
   robot has an option. Rejection reasons: `no_geometrically_feasible_grasp`,
   `never_in_reach_of_any_robot_before_exit`, `insufficient_time_before_deadline`,
   `neighbor_clearance_or_robot_conflict`, `stale_track`, `future_dated_track`,
   `feasible_but_not_selected` (only possible for FIFO).

The rule is a heuristic, not optimal: in 250 random 3–5-object instances it
missed 5 objects in total relative to an exhaustive search over commit orders
(FIFO: 9). Minimal counter-example: three objects in robot B's region where
taking the least-slack object first blocks two others.

`schedule(..., committed=[...])` accepts assignments already being executed;
`run_online(...)` re-plans every 0.5 s and commits a robot's next pick only when
it must start moving.

### Baseline: `FifoNearestRobotScheduler`

Objects in arrival order (furthest downstream first); each goes to the robot
whose base is nearest to its current position, falling back to the other robot
if infeasible. Single pass. Same feasibility model → the comparison isolates the
decision logic.

## 6. Test results (actual runs, 18 Nov 2026, after the fixes)

| Environment | Command | Result |
| --- | --- | --- |
| Windows, Python 3.12.4, `venv` | `venv\Scripts\python -m pytest -q` | **142 passed, 10 skipped** (skips = RAI-only tests) |
| WSL Ubuntu 24.04, existing RAI venv `~/venvs/cs449` (no pytest installed there; pytest made importable from a temporary copy, nothing installed) | `python -m pytest tests/test_huseyin_rai_integration.py` | **6 passed** |
| same | full repository suite | **185 passed, 4 skipped** (OpenGL/camera tests) |

### Latest verified run (19 Nov 2026, WSL + RAI, RGB-D enabled)

| Environment | Command | Result |
| --- | --- | --- |
| WSL Ubuntu 24.04.5, WSLg (`DISPLAY=:0`), OpenGL 4.5 via Mesa `llvmpipe` (software rendering), existing RAI venv `~/venvs/cs449`; pytest made importable from a temporary copy, nothing installed | `CS449_TEST_RGBD=1 python -m pytest tests/test_camera.py tests/test_conveyor_falls.py tests/test_environment_interfaces.py tests/test_multiobject_simulation.py -k <the 4 RGB-D tests>` | **4 passed** (all 4 previously skipped RGB-D camera tests) |
| same | `CS449_TEST_RGBD=1 python -m pytest` (full repository suite) | **189 passed, 0 failed, 0 skipped** |

The 4 RGB-D tests are `test_actual_rgbd_depth_calibration_and_pose_updates`,
`test_rgbd_observes_native_fall_when_the_exit_is_in_line_of_sight`,
`test_camera_and_config_survive_removal_and_reset_collision_rebuilds` and
`test_initial_clutter_objects_are_visible_to_the_overhead_camera`.

**What these results do and do not show.** The 189 passing tests are unit and
integration tests: pure-Python checks of prediction, grasp selection and
scheduling, RAI/KOMO kinematic checks (gripper-frame convention, reachability
of grasp poses and reach-model boundary points) and RGB-D rendering of the
simulated scene. They do **not** show physical robotic grasp execution: no
robot has yet moved in simulation to intercept an object, closed its gripper
and lifted it. Interception, robot execution and physical grasp success remain
unverified (see the table at the top) until Doğa's planner/controller is
integrated.

An independent validator (outside the repository) re-checked 720 static and 24
online schedules (positions, reach, robot sequencing, grasp vs neighbours,
robot separation, slack definition): 0 violations.

## 7. Baseline comparison (estimates)

`python experiments/scheduling_comparison.py` (127 s; seeds 0–9; full table and
paired statistics in `outputs/huseyin_scheduling/summary.md`). Scheduled =
objects with a feasible planned interception, summed over 10 seeds.

| Layout (t = 0 batch) | Speed (m/s) | Proposed (after fixes) | FIFO (after fixes) | Before fixes P / F | Paired W/T/L, sign-test p |
| --- | --- | --- | --- | --- | --- |
| clutter, 8 objects | 0.05 | **59 / 80** | 53 / 80 | 63 / 55 | 4/6/0, p = 0.125 |
| clutter, 8 objects | 0.08 | **44 / 80** | 39 / 80 | 46 / 41 | 5/4/1, p = 0.22 |
| clutter, 8 objects | 0.12 | 30 / 80 | 27 / 80 | 32 / 30 | 3/7/0, p = 0.25 |
| clutter, 8 objects | 0.20 | 20 / 80 | 21 / 80 | 21 / 22 | 0/9/1 |
| clutter, 12 objects | 0.05 | **71 / 120** | 59 / 120 | 71 / 60 | 6/4/0, p = 0.031 |
| clutter, 12 objects | 0.08 | 49 / 120 | 45 / 120 | 51 / 46 | 3/7/0, p = 0.25 |
| clutter, 12 objects | 0.12 | 30 / 120 | 32 / 120 | 33 / 32 | 0/8/2 |
| ordered, 8 objects | 0.05 – 0.25 | identical | identical | identical | all ties |
| sparse, 8 objects | 0.05 – 0.25 | within +3 | | within ±2 | |
| online stream, 20 objects (5 seeds) | 0.08 – 0.25 | within ±1 | | within ±1 | no wins at 0.08/0.20/0.25 |

What changed the numbers: the slack fix alone (run with a 0 m margin) changed
totals by at most +3 (sparse 0.16 m/s) and left the clutter rows at 0.05 m/s
unchanged; the 0.03 m reach margin removes boundary interceptions and lowers
both methods (e.g. clutter-8 at 0.05 m/s: 63→59 proposed, 55→53 FIFO). The
G2 grasp ranking does not change feasibility, only which grasp is reported.

Interpretation (honest): the proposed method is never worse in total on clutter
at ≤ 0.08 m/s and is identical on single-file layouts, but the advantage is
small. Only clutter-12 at 0.05 m/s reaches p < 0.05 (a single test among 36
configurations, uncorrected). At ≥ 0.12 m/s results are mixed (between +3 and
−2 objects per configuration) and in the online stream the methods are
equivalent; both are limited by the assumed pick cycle (≈ 3–4 s including the
trip to the bin). Ablation after the fixes (clutter-8, 0.05 m/s): `system` 59,
`robot` 59, `critical` 60, `system` without gap filling 59, FIFO 53 — the gain
comes from slack ordering itself, not from the cross-robot term or gap filling. "Throughput" now uses a fixed window;
the previous makespan-based figure (now `processing_rate_per_min`) is not used
for comparisons. Results of the earlier version must not be quoted any more.

**Prediction sensitivity** (Monte Carlo, 2000 trials, belt 0.12 m/s): with a
velocity error of σ = 5 mm/s the mean position error is 6 mm at 1 s and 25 mm at
4 s; with σ = 10 mm/s 13 mm and 51 mm. Since the opening clearance is 5 mm per
side, interceptions should be re-planned with short horizons (≤ ~1 s).

## 8. Current limitations

- Reachability is a calibrated sphere/annulus, not IK; collisions (belt, other
  arm, self), joint-limit margins and trajectory feasibility are not checked.
- Travel times use a constant Cartesian speed, not KOMO trajectories.
- Grasps are top-down only, for upright boxes/cylinders; tilted or fallen objects are excluded.
- Neighbour clearance is 2-D (ignores neighbour heights and the hand body).
- The interception model assumes a static descent over the predicted point; a
  real pick needs velocity matching (planning/control).
- Inter-robot deconfliction is a distance/time rule (0.15 m), not a link-level collision check.
- Greedy decisions are not optimal (see §5); planning time grows quickly
  (≈ 1 s for 20 objects at a 0.02 s step), which matters for online re-planning.
- Online experiments use ideal constant-velocity motion (no tracker noise);
  committed picks are not re-validated when new objects arrive.
- Object yaw is assumed constant on the belt (true for the kinematic belt).

## 9. Integration needs with teammates

- **Heysem (perception/tracking):** `TrackedState` is used unchanged. Also needed
  per track: geometry type, dimensions and yaw → `ObjectGeometry` (same
  `object_id`, stable over time). Track timestamps must not be later than the
  planning time (otherwise `future_dated_track`). Velocity accuracy matters
  (§7); covariance would allow uncertainty-aware slack.
- **Doğa (planning/control):** use `Assignment.grasp.quaternion_wxyz` as the
  `a_gripper`/`b_gripper` target (RAI convention, z up, x = closing axis) with
  `position_m`/`pregrasp_position_m`, `required_opening_m` and
  `intercept_time_s`; implement velocity-matched descent/closing; replace
  `ReachModel`/`TimingModel` by IK/KOMO feasibility and durations (e.g. a
  callback in `InterceptionModel`); confirm or re-calibrate the 0.03 m margin
  with collision objectives; report robot availability so
  `RobotConfig.available_at_s`/`tool_position_m` and `committed` stay correct.
- **Emre (simulation):** the belt is kinematic — `release_object(...)` must be
  called when a grasp closes (who calls it must be agreed). Robot
  availability/controller status is not exposed yet. In `remove` mode objects
  disappear at `1.2 m − half footprint` (can be < 1.15 m for rotated large boxes).
- **Pipeline:** call `run_online`-style re-planning every cycle with
  `now_s` ≥ the newest track time.
- Note: `src/conveyor/tracking`, `perception`, `planning` and `control`
  docstrings list owners that differ from `docs/architecture.md`; left unchanged.

## 10. How to run

```bash
# Windows (from the repository root)
python -m venv venv
venv\Scripts\python -m pip install -e ".[dev]"
venv\Scripts\python -m pytest -q
venv\Scripts\python scripts\huseyin_demo.py
venv\Scripts\python scripts\huseyin_demo.py --scenario clutter --objects 10 --speed 0.05 --seed 0
venv\Scripts\python experiments\scheduling_comparison.py          # or --quick --output outputs/huseyin_scheduling_quick
```
