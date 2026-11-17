"""Print/export actual robot and target geometry snapshots without a GUI."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from conveyor.simulation import ConveyorSimulation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="also write the JSON snapshot to this path")
    args = parser.parse_args()
    simulation = ConveyorSimulation()
    try:
        data = {
            "simulation_time_s": simulation.simulation_time_s,
            "frame": "world", "distance_unit": "m", "angle_unit": "rad",
            "quaternion_order": "wxyz", "reachability_verified": False,
            "robots": {key: asdict(value) for key, value in simulation.get_robot_states().items()},
            "targets": {key: asdict(value) for key, value in simulation.get_targets().items()},
        }
        output = json.dumps(data, indent=2) + "\n"
        print(output, end="")
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(output, encoding="utf-8")
    finally:
        simulation.close()


if __name__ == "__main__":
    main()
