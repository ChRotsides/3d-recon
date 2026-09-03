import numpy as np
from scipy.spatial.transform import Rotation
from videorecon import align, imu


def test_solve_rotation_recovers_known_rotation_under_noise():
    rng = np.random.default_rng(0)
    truth = Rotation.from_euler("xyz", [15, -40, 120], degrees=True).as_matrix()
    source = Rotation.random(40, random_state=1).as_matrix()
    noise = Rotation.from_rotvec(rng.normal(0, np.deg2rad(0.5), (40, 3))).as_matrix()
    target = np.einsum("ij,njk,nkl->nil", truth, source, noise)
    out = align.solve_rotation(target, source)
    err = np.rad2deg(Rotation.from_matrix(out.rotation.T @ truth).magnitude())
    assert err < 0.5
    assert out.median_deg < 1.5 and out.p95_deg < 3.0


def _synthetic_walk(scale_truth: float, seed: int = 0):
    """A camera strolling around; IMU at 200 Hz, keyframes every 0.2 s."""
    rng = np.random.default_rng(seed)
    duration = 20.0
    t_imu = np.arange(0, duration, 1 / 200)
    # Smooth metric trajectory: two sinusoids, walking pace.
    p = np.stack(
        [
            0.8 * np.sin(0.7 * t_imu) + 0.3 * np.sin(1.9 * t_imu),
            0.6 * np.cos(0.5 * t_imu) + 0.2 * np.sin(2.3 * t_imu),
            0.05 * np.sin(1.3 * t_imu),
        ],
        axis=1,
    )
    dt = t_imu[1] - t_imu[0]
    v = np.gradient(p, dt, axis=0)
    a = np.gradient(v, dt, axis=0)
    a_noisy = a + rng.normal(0, 0.05, a.shape) + np.array([0.02, -0.03, 0.01])  # noise and bias
    t_kf = np.arange(0.5, duration - 0.5, 0.2)
    p_kf = np.stack([np.interp(t_kf, t_imu, p[:, i]) for i in range(3)], axis=1)
    return (
        (t_imu * 1e9).astype(np.int64),
        a_noisy,
        (t_kf * 1e9).astype(np.int64),
        p_kf / scale_truth,
    )


def test_scale_from_imu_recovers_metres_per_unit():
    truth = 0.37
    t_imu, a_world, t_kf, pos_units = _synthetic_walk(truth)
    est = align.scale_from_imu(pos_units, t_kf, t_imu, a_world)
    assert est.usable
    assert abs(est.metres_per_unit - truth) / truth < 0.05
    assert est.inliers > 20


def test_scale_from_imu_reports_failure_on_static_camera():
    t_imu = np.arange(0, 5_000_000_000, 5_000_000).astype(np.int64)
    a_world = np.zeros((len(t_imu), 3))
    t_kf = np.arange(0, 5_000_000_000, 200_000_000).astype(np.int64)
    pos = np.zeros((len(t_kf), 3))
    est = align.scale_from_imu(pos, t_kf, t_imu, a_world)
    assert not est.usable


def test_to_metric_frame_scales_centres_and_keeps_rotations_valid():
    r_enu = Rotation.from_euler("z", 30, degrees=True).as_matrix()
    r_wc = Rotation.from_euler("xyz", [10, 20, 30], degrees=True).as_matrix()
    centre = np.array([1.0, 2.0, 3.0])
    w2c = np.eye(4)
    w2c[:3, :3] = r_wc.T
    w2c[:3, 3] = -r_wc.T @ centre
    out = align.to_metric_frame({"a": w2c}, np.array([[1.0, 0.0, 0.0]]), r_enu, 2.0)
    r_gl, t_gl = out.poses["a"]
    np.testing.assert_allclose(t_gl, imu.ENU_TO_Y_UP @ (2.0 * r_enu @ centre))
    np.testing.assert_allclose(r_gl @ r_gl.T, np.eye(3), atol=1e-12)
    assert np.isclose(np.linalg.det(r_gl), 1.0)
    np.testing.assert_allclose(out.points[0], imu.ENU_TO_Y_UP @ (2.0 * r_enu @ [1.0, 0.0, 0.0]))
    # A camera looking along ENU +y (north) ends up looking down -Z in the repo frame.
    look_cv = np.array([0.0, 1.0, 0.0])
    r_cv = np.eye(3)
    r_cv[:, 2] = look_cv
    r_cv[:, 1] = [0, 0, -1]
    r_cv[:, 0] = np.cross(r_cv[:, 1], r_cv[:, 2])
    w2c = np.eye(4)
    w2c[:3, :3] = r_cv.T
    out = align.to_metric_frame({"n": w2c}, np.zeros((0, 3)), np.eye(3), 1.0)
    r_gl, _ = out.poses["n"]
    np.testing.assert_allclose(-r_gl[:, 2], [0, 0, -1], atol=1e-12)


def test_heading_offset_between_game_and_magnetic_tracks():
    base = Rotation.random(30, random_state=3).as_matrix()
    yaw = Rotation.from_euler("z", 42.0, degrees=True).as_matrix()
    mag = np.einsum("ij,njk->nik", yaw, base)
    offset, spread = align.heading_offset_deg(base, mag)
    assert abs(offset - 42.0) < 1e-6
    assert spread < 1e-6


def test_scale_from_floor_recovers_height():
    rng = np.random.default_rng(5)
    # A room 8 x 5 units, floor at y=0, walls with points, cameras 2 units up.
    floor = np.stack(
        [rng.uniform(0, 8, 3000), rng.normal(0, 0.01, 3000), rng.uniform(0, 5, 3000)], 1
    )
    walls = np.stack(
        [rng.uniform(0, 8, 2000), rng.uniform(0, 3.5, 2000), rng.choice([0.0, 5.0], 2000)], 1
    )
    # A table top denser than the floor, and a few outliers underneath the floor.
    table = np.stack(
        [rng.uniform(2, 4, 5000), 1.0 + rng.normal(0, 0.01, 5000), rng.uniform(1, 2, 5000)], 1
    )
    junk = np.stack([rng.uniform(0, 8, 40), rng.uniform(-3, -0.2, 40), rng.uniform(0, 5, 40)], 1)
    pts = np.concatenate([floor, walls, table, junk])
    cams = np.stack(
        [rng.uniform(1, 7, 50), 2.0 + rng.normal(0, 0.05, 50), rng.uniform(1, 4, 50)], 1
    )
    est = align.scale_from_floor(pts, cams, camera_height_m=1.5)
    assert est.usable
    assert abs(est.metres_per_unit - 0.75) < 0.02


def test_scale_estimate_rejects_wide_dispersion():
    est = align.ScaleEstimate(0.01, "imu", 400, 250, 28.9)
    assert not est.usable
