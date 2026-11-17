"""RGB-D rendering and calibrated camera geometry, separate from ground truth."""

from dataclasses import dataclass
from math import isfinite
from numbers import Real
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    import robotic as ry


CAMERA_FRAME = "camera_rgbd"


def _finite(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
        raise ValueError(f"{name} must be a finite real number")


@dataclass(frozen=True)
class CameraCalibration:
    """Pinhole model: pixels right/down, optical axes +x right/+y down/+z forward.

    Depth is positive optical-axis z in metres, rather than distance along a ray.
    ``camera_to_world`` maps homogeneous optical-camera points into world.
    """

    width: int
    height: int
    intrinsics: tuple[tuple[float, float, float], ...]
    camera_to_world: tuple[tuple[float, float, float, float], ...]
    camera_frame: str = CAMERA_FRAME
    world_frame: str = "world"
    depth_convention: str = "optical_axis_z_m"

    def deproject_pixel(self, u: float, v: float, depth_m: float) -> tuple[float, float, float]:
        """Convert a valid image pixel and optical z-depth to a world point."""
        for name, value in (("u", u), ("v", v), ("depth_m", depth_m)):
            _finite(name, value)
        if not (0 <= u < self.width and 0 <= v < self.height):
            raise ValueError("Pixel coordinates must lie within the image")
        if depth_m <= 0:
            raise ValueError("depth_m must be positive; invalid rendered depth is NaN")
        fx, fy = self.intrinsics[0][0], self.intrinsics[1][1]
        cx, cy = self.intrinsics[0][2], self.intrinsics[1][2]
        point = ((u - cx) * depth_m / fx, (v - cy) * depth_m / fy, depth_m, 1.0)
        return tuple(sum(row[i] * point[i] for i in range(4)) for row in self.camera_to_world[:3])


@dataclass(frozen=True)
class CameraObservation:
    """One actual rendered RGB-D pair, with no object states or segmentation IDs.

    RGB is uint8 ``(height, width, 3)``. Depth is float32 ``(height, width)``
    optical-axis metres; invalid/out-of-range samples are NaN. Arrays are owned
    read-only snapshots, so another capture cannot mutate earlier observations.
    """

    simulation_time_s: float
    rgb: "np.ndarray"
    depth_m: "np.ndarray"
    calibration: CameraCalibration


class RGBDCamera:
    """Overhead sensor for the 2.4 m belt; rendering needs OpenGL/WSLg.

    The camera is at world (0, 0, 2.5) m, looking down with quaternion
    (w,x,y,z)=(0,1,0,0). Camera +x maps to world +x; camera +y to world -y.
    Initial dimensions are 640x360 and focal length is 0.8*image height.
    The RAI renderer is created only on capture, independently of the GUI view.
    """

    def __init__(self, config: "ry.Config"):
        self.config = config
        self._renderer = None
        self._near_m, self._far_m = .1, 5.0
        if CAMERA_FRAME in config.getFrameNames():
            frame = config.getFrame(CAMERA_FRAME)
        else:
            frame = config.addFrame(CAMERA_FRAME, "world")
        frame.setPosition([0.0, 0.0, 2.5]).setQuaternion([0.0, 1.0, 0.0, 0.0])
        frame.setAttributes({
            "focalLength": .8, "width": 640, "height": 360,
            "zRange": [self._near_m, self._far_m],
        })

    def capture(self, simulation_time_s: float) -> CameraObservation:
        """Render the current scene at its supplied simulation timestamp.

        No depth noise or lens distortion is simulated. This method uses the
        same configuration instance throughout its lifetime; pose updates are
        reflected in each render. Call ``close`` before discarding the sensor.
        """
        _finite("simulation_time_s", simulation_time_s)
        if simulation_time_s < 0:
            raise ValueError("simulation_time_s must be non-negative")
        import numpy as np
        import robotic as ry

        if self._renderer is None:
            self._renderer = ry.CameraView(self.config, True)
        # A simulator reset may rebuild native frames while keeping Config.
        frame = self.config.getFrame(CAMERA_FRAME)
        self._renderer.setCamera(frame)
        rgb, depth = self._renderer.computeImageAndDepth(self.config, False, True)
        rgb = np.array(rgb, dtype=np.uint8, copy=True, order="C")
        depth = np.array(depth, dtype=np.float32, copy=True, order="C")
        invalid = (~np.isfinite(depth)) | (depth <= self._near_m) | (depth >= self._far_m)
        depth[invalid] = np.nan
        fx, fy, cx, cy = (float(x) for x in self._renderer.getFxycxy())
        calibration = CameraCalibration(
            width=int(rgb.shape[1]), height=int(rgb.shape[0]),
            intrinsics=((fx, 0.0, cx), (0.0, fy, cy), (0.0, 0.0, 1.0)),
            camera_to_world=tuple(tuple(float(x) for x in row) for row in frame.getTransform()),
        )
        rgb.setflags(write=False)
        depth.setflags(write=False)
        return CameraObservation(float(simulation_time_s), rgb, depth, calibration)

    def close(self) -> None:
        """Drop the native CameraView handle, releasing its rendering resources.

        RAI 0.4.1 binds no explicit CameraView close method. Its C++ destructor
        runs when this owning Python reference is dropped. A later capture may
        recreate the renderer using the same camera frame.
        """
        self._renderer = None
