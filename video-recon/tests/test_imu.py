import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from videorecon import imu


def test_quaternion_matches_android_formula():
    # Android's getRotationMatrixFromVector, written out.
    x, y, z, w = Rotation.from_euler("xyz", [20, -35, 50], degrees=True).as_quat()
    expected = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    np.testing.assert_allclose(imu.quaternion_to_matrix([x, y, z, w]), expected, atol=1e-12)


def test_device_to_camera_is_right_handed_for_every_orientation():
    for deg in (0, 90, 180, 270):
        r = imu.device_to_camera(deg)
        np.testing.assert_allclose(r @ r.T, np.eye(3), atol=1e-12)
        assert np.isclose(np.linalg.det(r), 1.0)
        # The optical axis is always -Z of the device (a back camera).
        np.testing.assert_allclose(r[:, 2], [0, 0, -1])


def test_device_to_camera_90_puts_image_right_along_minus_device_y():
    r = imu.device_to_camera(90)
    np.testing.assert_allclose(r[:, 0], [0, -1, 0], atol=1e-12)  # image +u -> device -Y
    np.testing.assert_allclose(r[:, 1], [-1, 0, 0], atol=1e-12)  # image +v -> device -X


def test_upright_phone_facing_north_looks_north_in_repo_frame():
    # Device X east, Y up, Z south: columns of R_wd are the device axes in ENU.
    r_wd = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=float)
    r = imu.camera_to_world_gl(r_wd, 90)
    forward = -r[:, 2]  # camera looks down -Z in the GL convention
    np.testing.assert_allclose(forward, [0, 0, -1], atol=1e-12)  # north is -Z (Y up, Z south)
    # The sensor-native image is landscape: world-up appears along image left.
    np.testing.assert_allclose(r[:, 0], [0, -1, 0], atol=1e-12)


def test_gravity_in_device_for_flat_phone():
    np.testing.assert_allclose(imu.gravity_in_device(np.eye(3)), [0, 0, -1])


def test_orientation_track_interpolates_halfway():
    q0 = Rotation.identity().as_quat()
    q1 = Rotation.from_euler("z", 90, degrees=True).as_quat()
    track = imu.OrientationTrack.from_samples(np.array([0, 1_000_000_000]), np.stack([q0, q1]))
    r = track.rotation_at(np.array([500_000_000]))[0]
    np.testing.assert_allclose(
        Rotation.from_matrix(r).as_euler("xyz", degrees=True), [0, 0, 45], atol=1e-9
    )
    rate = track.angular_rate_deg_s()
    np.testing.assert_allclose(rate, [90.0])


def test_orientation_track_rejects_single_sample_and_fixes_sign_flips():
    with pytest.raises(ValueError):
        imu.OrientationTrack(np.array([0]), np.zeros((1, 4)))
    q = Rotation.from_euler("z", [[0], [10], [20]], degrees=True).as_quat()
    q[1] = -q[1]
    track = imu.OrientationTrack.from_samples(np.array([0, 1, 2]), q)
    assert np.all(
        np.einsum("ij,ij->i", track.quaternions_xyzw[:-1], track.quaternions_xyzw[1:]) > 0
    )


def test_world_linear_acceleration_removes_gravity_at_rest():
    r_wd = np.repeat(np.eye(3)[None], 4, axis=0)
    accel = np.tile([0.0, 0.0, imu.GRAVITY_M_S2], (4, 1))
    np.testing.assert_allclose(imu.world_linear_acceleration(r_wd, accel), 0.0, atol=1e-12)


def test_double_integrate_constant_acceleration():
    t = np.arange(0, 2_000_000_001, 5_000_000)  # 200 Hz for 2 s
    a = np.tile([2.0, 0.0, 0.0], (len(t), 1))
    p = imu.double_integrate(t, a, 0, 2_000_000_000)
    np.testing.assert_allclose(p, [0.5 * 2.0 * 4.0, 0, 0], rtol=1e-6)
    assert np.all(np.isnan(imu.double_integrate(t, a, 5_000_000_000, 6_000_000_000)))
