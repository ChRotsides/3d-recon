"""The differential-drive robot: limits are respected and collisions are refused, not crashed."""

import numpy as np
from sim.vehicle import A_MAX, ALPHA_MAX, OMEGA_MAX, V_MAX, Vehicle
from sim.world import World


class Wall:
    """A world with one obstacle: everything with x > 2 is solid."""

    def collides(self, x, y, yaw=0.0):
        return x > 2.0


def test_speed_and_acceleration_limits():
    veh = Vehicle(Wall(), 0.0, 0.0, 0.0, rate=200.0)
    veh.world = type("W", (), {"collides": staticmethod(lambda *a: False)})()
    prev_v, prev_w = 0.0, 0.0
    for _ in range(600):
        veh.step(10.0, 10.0)  # commands far beyond the limits
        assert abs(veh.v) <= V_MAX + 1e-9
        assert abs(veh.omega) <= OMEGA_MAX + 1e-9
        assert abs(veh.v - prev_v) <= A_MAX / 200.0 + 1e-9
        assert abs(veh.omega - prev_w) <= ALPHA_MAX / 200.0 + 1e-9
        prev_v, prev_w = veh.v, veh.omega
    assert abs(veh.v - V_MAX) < 1e-6 and abs(veh.omega - OMEGA_MAX) < 1e-6


def test_collision_is_refused_and_counted_not_crashed():
    veh = Vehicle(Wall(), 1.0, 0.0, 0.0, rate=200.0)
    for _ in range(2000):
        veh.step(V_MAX, 0.0)
    assert veh.x <= 2.0, "the robot must never end up inside the obstacle"
    assert veh.collisions > 0
    assert np.isfinite(veh.x) and np.isfinite(veh.y)


def test_the_robot_can_still_turn_away_from_a_collision():
    veh = Vehicle(Wall(), 1.99, 0.0, 0.0, rate=200.0)
    for _ in range(400):
        veh.step(V_MAX, OMEGA_MAX)
    assert veh.collisions > 0
    assert abs(veh.yaw) > 0.5, "rotation must not be blocked by a refused translation"


def test_odometry_matches_the_commanded_motion_without_slip():
    veh = Vehicle(type("W", (), {"collides": staticmethod(lambda *a: False)})(), 0.0, 0.0, 0.0, rate=200.0)
    for _ in range(1000):
        veh.step(0.5, 0.0)
    assert abs(veh.y) < 1e-9
    assert 2.2 < veh.x < 2.6  # 5 s at 0.5 m/s minus the ramp-up


def test_suspension_moves_only_when_driving():
    w = World.generate(4, "open")
    veh = Vehicle(w, w.start[0], w.start[1], 0.0, rate=200.0, seed=2)
    for _ in range(400):
        veh.step(0.0, 0.0)
    _, p_rest = veh.body_pose()
    assert abs(p_rest[2]) < 1e-9 and abs(veh.susp.pitch) < 1e-9
    zs = []
    for _ in range(600):
        veh.step(V_MAX, 0.3)
        zs.append(veh.body_pose()[1][2])
    zs = np.asarray(zs)
    assert veh.collisions == 0, "this test needs clear floor to drive on"
    assert 0.001 < zs.max() < 0.01, "the bob is millimetres, not centimetres"
    assert abs(veh.susp.pitch) < np.deg2rad(1.0)


def test_vibration_can_be_switched_off():
    w = World.generate(4, "open")
    veh = Vehicle(w, w.start[0], w.start[1], 0.0, rate=200.0, seed=2, vibration=False)
    for _ in range(600):
        veh.step(V_MAX, 0.3)
    R, p = veh.body_pose()
    assert p[2] == 0.0
    assert abs(R[2, 0]) < 1e-12 and abs(R[2, 1]) < 1e-12  # a pure yaw
