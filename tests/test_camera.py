"""Pure camera math plus explicit opt-in tests requiring OpenGL/WSLg."""

import os

import pytest

from conveyor.simulation.camera import CameraCalibration


def test_deprojection_uses_optical_axes_and_world_transform():
    calibration = CameraCalibration(
        width=640, height=360,
        intrinsics=((288., 0., 320.), (0., 288., 180.), (0., 0., 1.)),
        camera_to_world=((1., 0., 0., 0.), (0., -1., 0., 0.),
                         (0., 0., -1., 2.5), (0., 0., 0., 1.)),
    )
    assert calibration.deproject_pixel(320, 180, 1.75) == pytest.approx((0, 0, .75))
    assert calibration.deproject_pixel(464, 108, 2.) == pytest.approx((1., .5, .5))
    for pixel in ((-1, 180, 1), (640, 180, 1), (320, 360, 1), (320, 180, 0),
                  (320, 180, float("nan")), (True, 180, 1)):
        with pytest.raises(ValueError):
            calibration.deproject_pixel(*pixel)


@pytest.mark.skipif(os.environ.get("CS449_TEST_RGBD") != "1", reason="Set CS449_TEST_RGBD=1 with OpenGL/WSLg to render RGB-D")
def test_actual_rgbd_depth_calibration_and_pose_updates():
    np = pytest.importorskip("numpy")
    ry = pytest.importorskip("robotic")
    from conveyor.simulation.camera import RGBDCamera
    from conveyor.simulation.scene import build_dual_arm_scene

    config = build_dual_arm_scene()
    camera = RGBDCamera(config)
    try:
        observation = camera.capture(0.0)
        assert observation.rgb.shape == (360, 640, 3)
        assert observation.rgb.dtype == np.uint8
        assert observation.depth_m.shape == (360, 640)
        assert observation.depth_m.dtype == np.float32
        assert not observation.rgb.flags.writeable
        assert not observation.depth_m.flags.writeable
        assert observation.calibration.depth_convention == "optical_axis_z_m"

        # Independent known geometry: belt top z=.75, camera z=2.5.
        calibration = observation.calibration
        assert observation.depth_m[180, 320] == pytest.approx(1.75, abs=1e-4)
        assert calibration.deproject_pixel(320, 180, observation.depth_m[180, 320]) == pytest.approx((0, 0, .75), abs=1e-4)
        # Box top z=.83 at x=-1.05: depth is 1.67 even far from image centre.
        assert observation.depth_m[180, 139] == pytest.approx(1.67, abs=1e-4)
        assert calibration.deproject_pixel(139, 180, observation.depth_m[180, 139]) == pytest.approx((-1.05, 0, .83), abs=.003)
        for u in (131, 509):
            assert observation.depth_m[180, u] == pytest.approx(1.75, abs=1e-4)

        # Same native CameraView must update poses, not retain a stale scene.
        config.getFrame("object_001").setPosition([-.45, .10, .79])
        moved = camera.capture(2.0)
        assert moved.simulation_time_s == 2.0
        assert moved.depth_m[163, 242] == pytest.approx(1.67, abs=1e-4)
        assert moved.calibration.deproject_pixel(242, 163, moved.depth_m[163, 242]) == pytest.approx((-.45, .10, .83), abs=.004)
        assert moved.depth_m[180, 139] == pytest.approx(1.75, abs=1e-4)
        assert observation.depth_m[180, 139] == pytest.approx(1.67, abs=1e-4)

        config.delFrame("object_001")
        hidden = camera.capture(3.0)
        assert hidden.depth_m[163, 242] == pytest.approx(1.75, abs=1e-4)
        config.addFrame("object_001").setShape(ry.ST.ssBox, [.06, .045, .08, .005]).setPosition([-1.05, 0, .79]).setColor([.9, .32, .12]).setContact(1)
        restored = camera.capture(4.0)
        assert restored.depth_m[180, 139] == pytest.approx(1.67, abs=1e-4)
    finally:
        camera.close()
