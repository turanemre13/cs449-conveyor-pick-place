# Heysem — Progress Notes (Multi-Object RGB-D Perception & Kalman Tracking)

Status as of 19 November 2026, prepared for the 23 November progress report and
team integration. All modules are implemented in pure Python + NumPy + OpenCV
without importing RAI or `conveyor.simulation` at runtime, satisfying the
module boundary contract in `docs/architecture.md`.

## 1. What Was Implemented

| Module | Files | Purpose |
| --- | --- | --- |
| Perception models | `src/conveyor/perception/models.py` | `PerceptionConfig` and `DetectedObject` (world-frame 3-D centroid, full dimensions, planar yaw, box/cylinder classification, colour and `to_observation()`) |
| RGB-D detector | `src/conveyor/perception/detector.py` | `RGBDObjectDetector`: vectorised calibrated depth deprojection, conveyor ROI & height segmentation, colour/height clutter splitting, oriented footprint and shape estimation |
| Synthetic RGB-D | `src/conveyor/perception/synthetic.py` | `render_synthetic_rgbd` and `advance_specs_along_belt`: offline overhead RGB-D renderer matching `RGBDCamera` intrinsics (`640x360`, `fx=fy=288`, `z=2.5 m`) with configurable depth/RGB noise |
| Tracking models | `src/conveyor/tracking/models.py` | `TrackerConfig` and `TrackedObject` (wraps shared `TrackedState` + smoothed 3-D geometry, `to_object_geometry()`, `to_schedulable_object()`) |
| Kalman filter | `src/conveyor/tracking/kalman.py` | `ConstantVelocityKalmanFilter3D`: 6-D state `[x, y, z, vx, vy, vz]^T`, Joseph-form covariance update, planar belt constraint and EMA geometry/orientation smoothing |
| Multi-object tracker | `src/conveyor/tracking/tracker.py` | `MultiObjectTracker`: spatial gating + colour-aware minimum-cost bipartite association, stable track ID assignment (`track_001`, ...), missed-frame lifecycle, and automatic reset on simulation clock reset |
| Demo | `scripts/heysem_demo.py` | Terminal demo on synthetic moving conveyor frames or exported `--capture-dir` (`rgbd.npz`) |
| Evaluation | `experiments/perception_tracking_eval.py` | Reproducible benchmark across `ordered`, `sparse` and `clutter` layouts, belt speeds (`0.05`–`0.20 m/s`), object counts (`8`, `12`) and sensor noise levels |
| Tests | `tests/test_perception.py`, `tests/test_tracking.py` | 14 unit and end-to-end integration tests |

---

## 2. Algorithmic Pipeline

### 2.1 Multi-Object RGB-D Perception (`RGBDObjectDetector`)
1. **Calibrated 3-D Deprojection:** Valid optical-axis depth pixels $d(u, v) > 0$ are deprojected in a single vectorised pass using the camera intrinsics $(f_x, f_y, c_x, c_y)$ and the $4 \times 4$ `camera_to_world` matrix:
   $$\mathbf{p}_{\text{cam}} = \begin{bmatrix} (u - c_x) d / f_x \\ (v - c_y) d / f_y \\ d \end{bmatrix}, \quad \mathbf{p}_{\text{world}} = \mathbf{R}_{\text{c2w}}\,\mathbf{p}_{\text{cam}} + \mathbf{t}_{\text{c2w}}$$
2. **Conveyor ROI & Elevation Filtering:** Pixels outside the belt footprint ($x \in [-1.20, 1.20]\text{ m}$, $y \in [-0.20, 0.20]\text{ m}$) or below/above valid object heights above the belt surface ($z_{\text{belt}} = 0.75\text{ m}$, $h \in [0.015, 0.25]\text{ m}$) are masked out. This removes the floor, conveyor chassis, robot bases, destination bins and objects that have already fallen off the belt.
3. **Clutter Segmentation:** `cv2.connectedComponentsWithStats` extracts foreground blobs. When two adjacent objects in `clutter` have masks that touch by 1–2 pixels, `_split_cluttered_component` checks colour and top-height dispersion within the component and separates distinct modes cleanly.
4. **3-D Pose, Dimensions & Shape (`box` vs `cylinder`):**
   - Top-surface height $z_{\text{top}}$ is estimated from the robust median of the top depth layer (80th percentile reference $\pm 10\text{ mm}$), yielding object height $d_z = z_{\text{top}} - z_{\text{belt}}$ and 3-D centroid height $z_{\text{center}} = z_{\text{belt}} + d_z / 2$.
   - Top-surface world $(x, y)$ points are fitted with `cv2.minAreaRect` corrected for single-pixel cell pitch ($d_{\text{top}} / f_x$).
   - **Shape classification** sweeps bounding box fill ratios (extent) over $[0, \pi/2]$ and checks equal-area disc inlier ratio: upright cylinders exhibit rotationally invariant extent ($\approx \pi/4 \approx 0.785$) and high circular inlier fraction ($\ge 0.965$), whereas upright boxes peak near $1.0$ along their principal axes.

### 2.2 Kalman Filter & Multi-Object Tracking (`MultiObjectTracker`)
1. **State Space:** Each active track runs a 3-D constant-velocity Kalman filter:
   $$\mathbf{x}_k = [x, y, z, v_x, v_y, v_z]^T, \quad \mathbf{F}(\Delta t) = \begin{bmatrix} \mathbf{I}_3 & \Delta t\,\mathbf{I}_3 \\ \mathbf{0}_3 & \mathbf{I}_3 \end{bmatrix}, \quad \mathbf{H} = [\mathbf{I}_3 \;\; \mathbf{0}_3]$$
   with continuous-time white-noise acceleration process covariance $\mathbf{Q}(\Delta t)$ and Joseph-form covariance update $\mathbf{P} \leftarrow (\mathbf{I} - \mathbf{K}\mathbf{H})\mathbf{P}(\mathbf{I} - \mathbf{K}\mathbf{H})^T + \mathbf{K}\mathbf{R}\mathbf{K}^T$.
2. **Data Association & Track Lifecycle:**
   - Gated Euclidean distance ($\le 0.09\text{ m}$) plus RGB distance penalty forms a bipartite cost graph between predicted track positions and current frame detections, solved exactly per connected component via branch-and-bound.
   - Tracks maintain smoothed geometry (`dimensions_m`, `yaw_rad` modulo $\pi$ parallel-jaw symmetry, and majority-voted `geometry_type`).
   - Unmatched tracks survive up to `max_missed_frames = 4` (`0.60 s`) before pruning; tracks crossing `belt_exit_x_m = 1.25 m` are retired; simulation clock resets automatically call `tracker.reset()`.

---

## 3. Experimental Evaluation Results

Command: `PYTHONPATH=src python3 experiments/perception_tracking_eval.py`
(5 random seeds per row, 20 sequential RGB-D frames per run at $\Delta t = 0.05\text{ s}$):

| Scenario | Objects | Belt Speed (m/s) | Depth Noise (mm) | Detection Recall | Mean Pos Error (mm) | Mean Vel Error (mm/s) | Mean Dim Error (mm) | Mean Box Yaw Error (deg) | Shape Accuracy | ID Switches |
| --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `ordered` | 8 | 0.08 | 0.5 | **100.0%** | 0.27 | 0.81 | 1.69 | 0.69° | **100.0%** | **0** |
| `ordered` | 8 | 0.12 | 0.5 | **100.0%** | 0.55 | 1.36 | 1.61 | 1.10° | **100.0%** | **0** |
| `sparse` | 8 | 0.12 | 0.5 | **100.0%** | 0.45 | 1.31 | 1.73 | 0.60° | **100.0%** | **0** |
| `clutter` | 8 | 0.05 | 0.5 | **100.0%** | 0.35 | 0.63 | 1.54 | 0.65° | **100.0%** | **0** |
| `clutter` | 8 | 0.12 | 0.5 | **100.0%** | 0.62 | 1.39 | 1.50 | 0.93° | **100.0%** | **0** |
| `clutter` | 8 | 0.20 | 0.5 | **100.0%** | 0.35 | 0.63 | 1.48 | 0.72° | **100.0%** | **0** |
| `clutter` | 12 | 0.12 | 0.5 | **100.0%** | 0.55 | 1.33 | 1.56 | 1.14° | **100.0%** | **0** |
| `clutter` | 12 | 0.12 | 1.5 | **100.0%** | 0.56 | 1.33 | 1.67 | 0.94° | **100.0%** | **0** |

- **Velocity accuracy vs prediction requirement:** Hüseyin's sensitivity analysis (`docs/huseyin_progress_notes.md`, §7) showed that velocity error $\sigma_v \le 5\text{ mm/s}$ keeps 1-second prediction error within the $5\text{ mm}$ gripper finger clearance. The Kalman filter achieves a mean velocity error of **0.63–1.39 mm/s** across all tested belt speeds and clutter levels, with **0 ID switches**.

---

## 4. Handoff & Integration with Teammates (Hüseyin & Doğa)

Connecting perception + tracking to Hüseyin's prediction, grasp selection and dual-arm scheduler requires only three lines:

```python
from conveyor.perception import RGBDObjectDetector
from conveyor.tracking import MultiObjectTracker

detector = RGBDObjectDetector()
tracker = MultiObjectTracker()

# Inside the simulation loop at each camera frame:
observation = simulation.get_observation()
detections = detector.detect(observation)
tracker.update(detections, observation.simulation_time_s)

# 1) Shared TrackedState list for ConstantVelocityPredictor:
tracked_states = tracker.get_tracked_states()

# 2) Direct SchedulableObject(state=TrackedState, geometry=ObjectGeometry) list
#    for DeadlineAwareGreedyScheduler / FifoNearestRobotScheduler:
schedulable_objects = tracker.schedulable_objects()
schedule_result = scheduler.schedule(schedulable_objects, robots, observation.simulation_time_s)
```

Verified end-to-end with `feature/huseyin-prediction-grasp-scheduling`:
`render_synthetic_rgbd` $\rightarrow$ `RGBDObjectDetector.detect` $\rightarrow$ `MultiObjectTracker.update` $\rightarrow$ `tracker.schedulable_objects()` $\rightarrow$ `ConstantVelocityPredictor` / `select_top_down_grasps` / `DeadlineAwareGreedyScheduler.schedule` executes with zero errors and produces valid dual-arm assignments (`robot_a`, `robot_b`) and RAI gripper quaternions.
