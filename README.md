# cs449-conveyor-pick-place
Dynamic robotic pick-and-place of moving objects on a conveyor belt using RAI and Franka Panda.

This four-person CS449 Robotics project is currently a Python scaffold. Shared
interfaces and a synthetic data demo are implemented; simulation, camera
integration, filtering, prediction, grasp planning, motion planning, and control
are pending.

## Setup

Use Python 3.10 or newer. Run these commands from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pytest
python scripts/interface_demo.py
```

The core package uses only the standard library. For future perception and
tracking work, install the optional numerical and OpenCV dependencies:

```bash
python -m pip install -e '.[perception,dev]'
```

Neither command requires RAI. The dependency ranges are starting points, not a
tested robotics environment lock. Select a Python version supported by the
course RAI build before full integration.

`requirements-rai.txt` is intentionally comments only: the course-provided RAI
distribution, version/commit, native build dependencies, Panda model assets, and
Python compatibility still need verification by the simulation/planning owners.
Installing that file currently installs nothing. Once selected, record the exact
backend setup there. Future backend imports should stay inside the modules or
functions that need them so perception/tracking can run independently.

## Layout

```text
src/conveyor/
  __init__.py
  types.py           # Validated observations and tracked states
  pipeline.py        # Explicitly unimplemented integration entry point
  simulation/
  perception/
  tracking/
  prediction/
  grasping/
  planning/
  control/
scenes/              # Future RAI scene files
scripts/             # Runnable demos
tests/               # pytest interface/import smoke test
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
| Doğa | Simulation: RAI, Franka Panda, conveyor, simulated RGB-D |
| Emre | Perception, tracking, prediction |
| Hüseyin | Geometry-based grasping and IK/KOMO motion planning |
| Heysem | Control, integration, evaluation |

Planned flow: simulation → RGB-D perception → constant-velocity Kalman tracking
→ interception prediction → geometry-based grasp selection → IK/KOMO planning
→ trajectory execution → simulated feedback and optional replanning.

Shared observations and states use **world-frame** coordinates, **metres**,
**metres/second**, and **simulation seconds**. Angular quantities added later must
use **radians**. Camera-to-world conversion belongs to perception; wall-clock
timing must remain separate from simulation timestamps.

See [architecture](docs/architecture.md) and [interfaces](docs/interfaces.md) for
module boundaries, handoffs, and open integration decisions. Generated outputs
belong in `outputs/` or `experiments/results/`; keep selected report assets in
`reports/`.
