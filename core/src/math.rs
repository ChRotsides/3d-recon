//! Minimal vector/pose types. No external linear-algebra crate: the whole point
//! of this crate is that it drops into an iOS and an Android binary without
//! dragging a dependency tree behind it.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Vec3 {
    pub const ZERO: Vec3 = Vec3 {
        x: 0.0,
        y: 0.0,
        z: 0.0,
    };

    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Vec3 { x, y, z }
    }

    pub fn dot(self, o: Vec3) -> f32 {
        self.x * o.x + self.y * o.y + self.z * o.z
    }

    pub fn cross(self, o: Vec3) -> Vec3 {
        Vec3::new(
            self.y * o.z - self.z * o.y,
            self.z * o.x - self.x * o.z,
            self.x * o.y - self.y * o.x,
        )
    }

    pub fn length(self) -> f32 {
        self.dot(self).sqrt()
    }

    pub fn normalized(self) -> Vec3 {
        let l = self.length();
        if l < 1e-9 {
            Vec3::ZERO
        } else {
            self * (1.0 / l)
        }
    }

    pub fn distance(self, o: Vec3) -> f32 {
        (self - o).length()
    }
}

impl std::ops::Add for Vec3 {
    type Output = Vec3;
    fn add(self, o: Vec3) -> Vec3 {
        Vec3::new(self.x + o.x, self.y + o.y, self.z + o.z)
    }
}

impl std::ops::Sub for Vec3 {
    type Output = Vec3;
    fn sub(self, o: Vec3) -> Vec3 {
        Vec3::new(self.x - o.x, self.y - o.y, self.z - o.z)
    }
}

impl std::ops::Mul<f32> for Vec3 {
    type Output = Vec3;
    fn mul(self, s: f32) -> Vec3 {
        Vec3::new(self.x * s, self.y * s, self.z * s)
    }
}

impl std::ops::Neg for Vec3 {
    type Output = Vec3;
    fn neg(self) -> Vec3 {
        Vec3::new(-self.x, -self.y, -self.z)
    }
}

/// A world<-camera pose. `r` is row-major 3x3.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Pose {
    pub r: [f32; 9],
    pub t: Vec3,
}

impl Pose {
    pub fn identity() -> Self {
        Pose {
            r: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            t: Vec3::ZERO,
        }
    }

    /// Rotate a camera-frame vector into the world.
    pub fn rotate(&self, v: Vec3) -> Vec3 {
        Vec3::new(
            self.r[0] * v.x + self.r[1] * v.y + self.r[2] * v.z,
            self.r[3] * v.x + self.r[4] * v.y + self.r[5] * v.z,
            self.r[6] * v.x + self.r[7] * v.y + self.r[8] * v.z,
        )
    }

    /// The camera looks down -Z (plan §5).
    pub fn forward(&self) -> Vec3 {
        Vec3::new(-self.r[2], -self.r[5], -self.r[8])
    }

    pub fn position(&self) -> Vec3 {
        self.t
    }

    /// Relative rotation magnitude in radians, via the trace of R_a^T R_b.
    pub fn angle_to(&self, other: &Pose) -> f32 {
        let mut trace = 0.0f32;
        for i in 0..3 {
            for k in 0..3 {
                trace += self.r[i * 3 + k] * other.r[i * 3 + k];
            }
        }
        (((trace - 1.0) * 0.5).clamp(-1.0, 1.0)).acos()
    }
}

pub fn yaw_rotation(yaw: f32) -> [f32; 9] {
    let (s, c) = yaw.sin_cos();
    [c, 0.0, s, 0.0, 1.0, 0.0, -s, 0.0, c]
}

/// Build a world<-camera rotation for a camera at `yaw` (about +Y) and `pitch`.
pub fn look_rotation(yaw: f32, pitch: f32) -> [f32; 9] {
    let (sy, cy) = yaw.sin_cos();
    let (sp, cp) = pitch.sin_cos();
    // Ry(yaw) * Rx(pitch), row-major.
    [cy, sy * sp, sy * cp, 0.0, cp, -sp, -sy, cy * sp, cy * cp]
}
