"""Camera mount and IMU synthesis: the two places a frame convention error would hide."""

import numpy as np
from scipy.spatial.transform import Rotation
from sim.sensors import (
    CAM_HEIGHT,
    GRAVITY,
    Camera,
    ImuSim,
    imu_cam_extrinsics,
    integrate_imu,
    intrinsics,
)
from sim.vehicle import Vehicle
from sim.world import World


def test_intrinsics_match_the_field_of_view():
    K = intrinsics(640, 360, 70.0)
    assert abs(2 * np.degrees(np.arctan(320 / K[0])) - 70.0) < 1e-6
    assert K[0] == K[1]


def test_extrinsics_are_a_right_handed_rotation():
    R, p = imu_cam_extrinsics()
    assert abs(np.linalg.det(R) - 1.0) < 1e-9
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)
    assert np.allclose(p, [0.0, 0.0, CAM_HEIGHT])


def test_point_ahead_on_the_optical_axis_projects_to_the_image_centre():
    cam = Camera()
    R_wc, c = cam.pose(np.eye(3), np.zeros(3))
    P = c + R_wc @ np.array([0.0, 0.0, 3.0])  # 3 m along the optical axis
    pc = R_wc.T @ (P - c)
    u = cam.K[0] * pc[0] / pc[2] + cam.K[2]
    v = cam.K[1] * pc[1] / pc[2] + cam.K[3]
    assert abs(u - cam.K[2]) < 1e-6 and abs(v - cam.K[3]) < 1e-6
    assert abs(u - (cam.width - 1) / 2) < 1e-6 and abs(v - (cam.height - 1) / 2) < 1e-6


def test_a_point_above_the_robot_has_negative_image_y():
    cam = Camera()
    R_wc, c = cam.pose(np.eye(3), np.zeros(3))
    P = np.array([1.0, 0.0, c[2] + 1.0])  # a metre ahead and a metre above the camera
    pc = R_wc.T @ (P - c)
    v = cam.K[1] * pc[1] / pc[2] + cam.K[3]
    assert v < cam.K[3], "up in the world must be up (smaller v) in the image"


def test_camera_looks_forward_and_slightly_down():
    cam = Camera()
    R_wc, c = cam.pose(np.eye(3), np.zeros(3))
    axis = R_wc[:, 2]
    assert axis[0] > 0.98 and abs(axis[1]) < 1e-9
    assert abs(axis[2] + np.sin(np.deg2rad(8.0))) < 1e-9
    assert abs(c[2] - CAM_HEIGHT) < 1e-12


def test_base_from_camera_inverts_the_mount():
    cam = Camera()
    for x, y, yaw in ((1.0, 2.0, 0.4), (-3.0, 0.5, -2.2)):
        R_wb = Rotation.from_euler("z", yaw).as_matrix()
        R_wc, c = cam.pose(R_wb, np.array([x, y, 0.0]))
        bx, by, byaw = cam.base_from_camera(R_wc, c)
        assert abs(bx - x) < 1e-9 and abs(by - y) < 1e-9
        assert abs(np.arctan2(np.sin(byaw - yaw), np.cos(byaw - yaw))) < 1e-9


def test_noise_free_imu_reproduces_the_trajectory_over_ten_seconds():
    """A drive with starts, stops, turns and the suspension running, integrated back.

    The synthesis is a central difference of the true pose sequence, so the matching strapdown
    is the leapfrog in sensors.integrate_imu; what the test really checks is the frames, the
    signs and the gravity term, any of which would be wrong by metres, not centimetres.
    """
    world = World.generate(5, "open")
    veh = Vehicle(world, world.start[0], world.start[1], 0.3, rate=200.0, seed=5)
    imu = ImuSim(200.0, noise=0.0, seed=5)
    dt = 1 / 200.0
    samples, Rl, Pl, t = [], [], [], 0.0
    while t < 10.0:
        v = 0.45 if (t % 4.0) < 2.6 else 0.0
        veh.step(v, 0.5 * np.sin(0.8 * t))
        R_wb, p_wb = veh.body_pose()
        Rl.append(R_wb)
        Pl.append(p_wb)
        s = imu.sample(t, R_wb, p_wb)
        if s:
            samples.append(s)
        t += dt
    assert len(samples) > 1800
    # samples[j] is the central difference around pose j+1
    Pl = np.asarray(Pl)
    v0 = (Pl[2] - Pl[0]) / (2 * dt)
    Rs, ps = integrate_imu(samples, Rl[1], Pl[1], v0, dt)
    n = min(len(ps), len(Pl) - 1)
    pos_err = np.linalg.norm(ps[:n] - Pl[1 : 1 + n], axis=1)
    ang_err = [np.degrees(Rotation.from_matrix(Rs[i].T @ Rl[1 + i]).magnitude()) for i in range(n)]
    assert pos_err.max() < 0.02, f"position drift {pos_err.max() * 100:.2f} cm"
    assert max(ang_err) < 0.2, f"attitude drift {max(ang_err):.3f} deg"


def test_imu_reads_gravity_at_rest():
    world = World.generate(5, "open")
    veh = Vehicle(world, world.start[0], world.start[1], 0.0, rate=200.0, seed=1, vibration=False)
    imu = ImuSim(200.0, noise=0.0, seed=1)
    out = None
    for k in range(20):
        veh.step(0.0, 0.0)
        out = imu.sample(k / 200.0, *veh.body_pose()) or out
    assert np.allclose(out[2], [0.0, 0.0, GRAVITY], atol=1e-6)
    assert np.allclose(out[1], 0.0, atol=1e-9)


def test_imu_noise_switch():
    world = World.generate(5, "open")
    outs = {}
    for noise in (0.0, 1.0):
        veh = Vehicle(world, world.start[0], world.start[1], 0.0, rate=200.0, seed=1, vibration=False)
        imu = ImuSim(200.0, noise=noise, seed=1)
        acc = []
        for k in range(400):
            veh.step(0.0, 0.0)
            s = imu.sample(k / 200.0, *veh.body_pose())
            if s:
                acc.append(s[2])
        outs[noise] = np.asarray(acc)
    assert outs[0.0].std(axis=0).max() < 1e-9
    assert 0.1 < outs[1.0][:, 0].std() < 1.0


def test_render_shapes_and_depth_positive():
    world = World.generate(3)
    cam = Camera()
    veh = Vehicle(world, *world.start)
    img, depth = cam.render(world, *cam.pose(*veh.body_pose()), np.random.default_rng(0))
    assert img.shape == (360, 640, 3) and img.dtype == np.uint8
    assert depth.shape == (360, 640) and depth.dtype == np.float32
    assert depth.min() > 0.05 and depth.max() < 20.0
