"""Build single- and dual-Panda RAI geometry without loading a controller."""

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import robotic as ry


DEFAULT_SCENE_PATH = Path(__file__).resolve().parents[3] / "scenes" / "minimal.g"
DUAL_ARM_SCENE_PATH = DEFAULT_SCENE_PATH.with_name("dual_arm.g")
ROBOT_A_PREFIX = "a_"


def build_minimal_scene(scene_path: str | Path | None = None) -> "ry.Config":
    """Load authored geometry and the backend's bundled Panda with unique names.

    Use an editable installation for the repository's default scene, or pass an
    explicit scene path defining ``world``, ``robot_a_mount``, ``conveyor``, and
    ``object_001`` frames. Each call returns an independent configuration.
    Importing this module does not load RAI.
    """
    path = DEFAULT_SCENE_PATH if scene_path is None else Path(scene_path).resolve()
    return _build_scene(path, ((ROBOT_A_PREFIX, "robot_a_mount"),))


def build_dual_arm_scene(scene_path: str | Path | None = None) -> "ry.Config":
    """Load two independently named Pandas beside the conveyor.

    The robots are sequential along world +x on the same side of the belt.
    This initial placement is not a verified reachability map or motion plan.
    """
    path = DUAL_ARM_SCENE_PATH if scene_path is None else Path(scene_path).resolve()
    return _build_scene(path, (("a_", "robot_a_mount"), ("b_", "robot_b_mount")))


def _build_scene(path: Path, robots: tuple[tuple[str, str], ...]) -> "ry.Config":
    if not path.is_file():
        raise FileNotFoundError(f"RAI scene file not found: {path}")

    try:
        import robotic as ry
    except ImportError as exc:
        raise RuntimeError(
            "RAI could not be imported. In the project environment run: "
            "python -m pip install -r requirements-rai.txt. "
            "If already installed, check the native libraries: "
            "liblapack3, libglut, libglu1-mesa, libxrandr2, libfreetype6. "
            f"Original import error: {exc}"
        ) from exc

    model_path = Path(ry.raiPath("panda/panda.g"))
    if not model_path.is_file():
        raise FileNotFoundError(f"Bundled Panda model not found: {model_path}")

    # RAI 0.4.1 otherwise adds its own z=0 floor, exactly overlapping ours.
    # Its renderer adds that implicit floor only for a three-component color;
    # an RGBA value disables it while retaining our authored/collidable floor.
    # Set this when constructing a scene, never while importing the package.
    ry.params_add({"Render/floorColor": [0.0, 0.0, 0.0, 0.0]})

    config = ry.Config()
    config.addFile(str(path))
    required_frames = {"world", "conveyor", "object_001"}
    required_frames.update(mount for _, mount in robots)
    missing = required_frames - set(config.getFrameNames())
    if missing:
        raise ValueError(f"The scene is missing required frames: {', '.join(sorted(missing))}")
    for prefix, mount in robots:
        config.addFile(str(model_path), prefix)
        base = config.getFrame(f"{prefix}panda_base")
        base.setParent(config.getFrame(mount))
        base.setRelativePosition([0.0, 0.0, 0.0])
        base.setRelativeQuaternion([1.0, 0.0, 0.0, 0.0])
    return config
