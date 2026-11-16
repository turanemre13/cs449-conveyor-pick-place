"""Run after editable installation: python scripts/interface_demo.py.

Constructs synthetic interface data only; this is not a simulation.
"""

from conveyor.types import ObjectObservation, TrackedState


def main() -> None:
    observation = ObjectObservation("detection-1", 0.0, (0.4, 0.0, 0.1))
    state = TrackedState("track-1", 0.0, observation.position_m, (0.1, 0.0, 0.0))
    print(observation)
    print(state)


if __name__ == "__main__":
    main()
