"""Differential-drive ground robot: the true state the simulator advances, and the suspension
wobble that makes VIO scale observable.

Body (= IMU) frame: x forward, y left, z up, origin on the floor under the robot centre.
A planar constant-velocity robot gives an unobservable metric scale to a monocular VIO; the
0.5 deg suspension oscillation, the 5 mm bob and the accelerations at the starts, stops and
turns are what make it observable, so they are part of the *true* motion, not of the noise.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

V_MAX = 0.6
OMEGA_MAX = 1.2
A_MAX = 0.8
ALPHA_MAX = 2.5


def rz(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


class Suspension:
    """Speed-driven pitch/roll oscillation and vertical bob. Phases are integrated, so the
    frequency may follow the speed without the signal jumping."""

    PITCH_AMP = np.deg2rad(0.5)
    BOB_AMP = 0.005

    def __init__(self, seed: int = 0, enabled: bool = True):
        rng = np.random.default_rng(seed + 7717)
        self.ph = rng.uniform(0, 2 * np.pi, 3)
        self.enabled = enabled
        self.pitch = self.roll = self.dz = 0.0

    def update(self, dt: float, speed: float, omega: float) -> None:
        if not self.enabled:
            return
        s = min(1.0, abs(speed) / (0.35 * V_MAX))
        f = 2.0 + 2.0 * min(1.0, abs(speed) / V_MAX)
        self.ph += 2 * np.pi * dt * np.array([f, f * 0.77, f * 1.31])
        self.pitch = self.PITCH_AMP * s * np.sin(self.ph[0])
        self.roll = 0.8 * self.PITCH_AMP * s * np.sin(self.ph[1]) + 0.35 * self.PITCH_AMP * np.clip(omega / OMEGA_MAX, -1, 1)
        self.dz = self.BOB_AMP * s * np.sin(self.ph[2])


@dataclass
class Vehicle:
    world: object
    x: float
    y: float
    yaw: float
    rate: float = 200.0
    slip: float = 0.0
    seed: int = 0
    vibration: bool = True

    def __post_init__(self):
        self.v = 0.0
        self.omega = 0.0
        self.t = 0.0
        self.collisions = 0     # distinct contact events, not blocked ticks
        self.blocked_ticks = 0
        self.distance = 0.0
        self.in_contact = False
        self._rng = np.random.default_rng(self.seed + 991)
        self.susp = Suspension(self.seed, self.vibration)
        self.last_collision_t = -1e9

    @property
    def dt(self) -> float:
        return 1.0 / self.rate

    def step(self, v_cmd: float, omega_cmd: float) -> bool:
        """Advance one tick towards the commands; returns True if a collision was refused."""
        dt = self.dt
        v_cmd = float(np.clip(v_cmd, -V_MAX, V_MAX))
        omega_cmd = float(np.clip(omega_cmd, -OMEGA_MAX, OMEGA_MAX))
        self.v += np.clip(v_cmd - self.v, -A_MAX * dt, A_MAX * dt)
        self.omega += np.clip(omega_cmd - self.omega, -ALPHA_MAX * dt, ALPHA_MAX * dt)
        v, w = self.v, self.omega
        if self.slip > 0:
            v *= 1.0 + self.slip * self._rng.normal()
            w += self.slip * OMEGA_MAX * self._rng.normal() * 0.2
        yaw_mid = self.yaw + 0.5 * w * dt
        nx = self.x + v * np.cos(yaw_mid) * dt
        ny = self.y + v * np.sin(yaw_mid) * dt
        nyaw = self.yaw + w * dt
        hit = self.world.collides(nx, ny, nyaw)
        if hit:
            self.blocked_ticks += 1
            if not self.in_contact:      # one event per contact, not one per 5 ms tick
                self.collisions += 1
            self.in_contact = True
            self.last_collision_t = self.t
            self.v = 0.0  # refuse the translation, but let the robot turn away
            self.yaw = nyaw
        else:
            self.in_contact = False
            self.distance += abs(v) * dt
            self.x, self.y, self.yaw = nx, ny, nyaw
        self.yaw = (self.yaw + np.pi) % (2 * np.pi) - np.pi
        self.susp.update(dt, self.v, self.omega)
        self.t += dt
        return hit

    def body_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """True R_wb (body->world) and body origin in world, including the suspension wobble."""
        p, r = self.susp.pitch, self.susp.roll
        cp, sp, cr, sr = np.cos(p), np.sin(p), np.cos(r), np.sin(r)
        ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
        rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
        return rz(self.yaw) @ ry @ rx, np.array([self.x, self.y, self.susp.dz])

    @property
    def base(self) -> tuple[float, float, float]:
        return self.x, self.y, self.yaw
