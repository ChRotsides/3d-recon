//! UniFFI surface for the mobile clients (ADR-0001, PLAN.md §3.8, §6.1).
//!
//! A deliberately thin layer over the core types rather than annotations on
//! them. Two reasons: the core stays free of FFI concerns and testable as plain
//! Rust, and the clients get an API shaped for what an app actually does —
//! push a frame, get back what to show the user — rather than the internal
//! structure of a voxel grid.
//!
//! This is what ADR-0001 promised. Until it existed both clients hand-maintained
//! their own copy of the keyframe rule, flagged as a stopgap in both files;
//! that is precisely the drift a shared core exists to prevent.

use std::sync::Mutex;

use crate::coverage::CoverageParams;
use crate::math::{Pose, Vec3};
use crate::quality::{
    CompletionThresholds, FrameStats, KeyframeDecision, Nag, QualityParams, TrackingState,
};
use crate::session::{CaptureSession, SessionConfig};
use crate::station::StationParams;

#[derive(uniffi::Record, Clone, Copy)]
pub struct FfiVec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl From<FfiVec3> for Vec3 {
    fn from(v: FfiVec3) -> Vec3 {
        Vec3::new(v.x, v.y, v.z)
    }
}

impl From<Vec3> for FfiVec3 {
    fn from(v: Vec3) -> FfiVec3 {
        FfiVec3 { x: v.x, y: v.y, z: v.z }
    }
}

/// A world<-camera pose. `r` is row-major 3x3, matching PLAN.md §5.
#[derive(uniffi::Record, Clone)]
pub struct FfiPose {
    pub r: Vec<f32>,
    pub t: FfiVec3,
}

#[derive(uniffi::Enum, Clone, Copy, PartialEq, Eq)]
pub enum FfiTracking {
    Normal,
    Limited,
    Relocalizing,
}

impl From<FfiTracking> for TrackingState {
    fn from(t: FfiTracking) -> TrackingState {
        match t {
            FfiTracking::Normal => TrackingState::Normal,
            FfiTracking::Limited => TrackingState::Limited,
            FfiTracking::Relocalizing => TrackingState::Relocalizing,
        }
    }
}

#[derive(uniffi::Enum, Clone, Copy, PartialEq, Eq)]
pub enum FfiDecision {
    Accept,
    RejectBlurry,
    RejectExposure,
    RejectTracking,
    RejectTooSimilar,
    RejectCapReached,
}

impl From<KeyframeDecision> for FfiDecision {
    fn from(d: KeyframeDecision) -> FfiDecision {
        match d {
            KeyframeDecision::Accept => FfiDecision::Accept,
            KeyframeDecision::RejectBlurry => FfiDecision::RejectBlurry,
            KeyframeDecision::RejectExposure => FfiDecision::RejectExposure,
            KeyframeDecision::RejectTracking => FfiDecision::RejectTracking,
            KeyframeDecision::RejectTooSimilar => FfiDecision::RejectTooSimilar,
            KeyframeDecision::RejectCapReached => FfiDecision::RejectCapReached,
        }
    }
}

#[derive(uniffi::Enum, Clone, Copy, PartialEq, Eq)]
pub enum FfiNag {
    TooClose,
    TooFast,
    Textureless,
    CeilingMissing,
    TrackingLost,
    DarkRoom,
}

impl From<Nag> for FfiNag {
    fn from(n: Nag) -> FfiNag {
        match n {
            Nag::TooClose => FfiNag::TooClose,
            Nag::TooFast => FfiNag::TooFast,
            Nag::Textureless => FfiNag::Textureless,
            Nag::CeilingMissing => FfiNag::CeilingMissing,
            Nag::TrackingLost => FfiNag::TrackingLost,
            Nag::DarkRoom => FfiNag::DarkRoom,
        }
    }
}

/// The user-facing copy for a nag, so the two clients cannot word it
/// differently — capture guidance is a product surface, not a debug string.
#[uniffi::export]
pub fn nag_message(nag: FfiNag) -> String {
    let n = match nag {
        FfiNag::TooClose => Nag::TooClose,
        FfiNag::TooFast => Nag::TooFast,
        FfiNag::Textureless => Nag::Textureless,
        FfiNag::CeilingMissing => Nag::CeilingMissing,
        FfiNag::TrackingLost => Nag::TrackingLost,
        FfiNag::DarkRoom => Nag::DarkRoom,
    };
    n.message().to_string()
}

#[derive(uniffi::Record, Clone)]
pub struct FfiFrame {
    pub timestamp_s: f64,
    pub pose: FfiPose,
    pub tracking: FfiTracking,
    pub blur_score: f32,
    pub exposure_clip_ratio: f32,
    pub median_depth_m: f32,
    pub angular_velocity_rad_s: f32,
    pub sparse_point_count: u32,
    pub lux: f32,
}

#[derive(uniffi::Record, Clone, Copy)]
pub struct FfiStation {
    pub position: FfiVec3,
    pub expected_gain: f32,
    pub walk_distance_m: f32,
}

#[derive(uniffi::Record, Clone)]
pub struct FfiGuidance {
    pub decision: FfiDecision,
    pub nag: Option<FfiNag>,
    /// Where to draw the AR floor ring, if anywhere.
    pub station: Option<FfiStation>,
    pub floor_coverage: f32,
    pub wall_coverage: f32,
    pub ceiling_coverage: f32,
    pub complete: bool,
    pub keyframes: u32,
}

/// Errors the clients have to handle rather than crash on.
#[derive(uniffi::Error, Debug, thiserror::Error)]
pub enum FfiError {
    #[error("pose rotation must have 9 elements, got {count}")]
    BadPose { count: u32 },
    #[error("points and normals must be the same length ({points} vs {normals})")]
    LengthMismatch { points: u32, normals: u32 },
}

/// One capture session. Mirrors `CaptureSession`; the mutex is because UniFFI
/// objects are shared immutably across the FFI boundary.
#[derive(uniffi::Object)]
pub struct FfiCaptureSession {
    inner: Mutex<CaptureSession>,
}

#[uniffi::export]
impl FfiCaptureSession {
    /// `gravity` is the world-frame down vector, as in PLAN.md §5.
    #[uniffi::constructor]
    pub fn new(gravity: FfiVec3) -> Self {
        FfiCaptureSession {
            inner: Mutex::new(CaptureSession::new(SessionConfig::default(), gravity.into())),
        }
    }

    /// The floor height from ARKit/ARCore plane detection. Better than
    /// inferring it, and available before any depth has run.
    pub fn set_floor_height(&self, y: f32) {
        self.inner.lock().unwrap().set_floor_height(y);
    }

    /// Push one frame plus the on-device depth pass's surface samples.
    ///
    /// `points` and `normals` are flat xyz triples. Flat rather than a vector of
    /// records because this is called at 1-2 Hz with thousands of points, and
    /// UniFFI copies every record across the boundary.
    pub fn observe(
        &self,
        frame: FfiFrame,
        points: Vec<f32>,
        normals: Vec<f32>,
    ) -> Result<FfiGuidance, FfiError> {
        if frame.pose.r.len() != 9 {
            return Err(FfiError::BadPose { count: frame.pose.r.len() as u32 });
        }
        if points.len() != normals.len() {
            return Err(FfiError::LengthMismatch {
                points: points.len() as u32,
                normals: normals.len() as u32,
            });
        }
        let mut r = [0f32; 9];
        r.copy_from_slice(&frame.pose.r);

        let stats = FrameStats {
            timestamp_s: frame.timestamp_s,
            pose: Pose { r, t: frame.pose.t.into() },
            tracking: frame.tracking.into(),
            blur_score: frame.blur_score,
            exposure_clip_ratio: frame.exposure_clip_ratio,
            median_depth_m: frame.median_depth_m,
            angular_velocity_rad_s: frame.angular_velocity_rad_s,
            sparse_point_count: frame.sparse_point_count,
            lux: frame.lux,
        };

        let pts: Vec<Vec3> = points.chunks_exact(3).map(|c| Vec3::new(c[0], c[1], c[2])).collect();
        let nrm: Vec<Vec3> = normals.chunks_exact(3).map(|c| Vec3::new(c[0], c[1], c[2])).collect();

        let update = self.inner.lock().unwrap().observe_frame(&stats, &pts, &nrm);
        Ok(FfiGuidance {
            decision: update.decision.into(),
            nag: update.nag.map(Into::into),
            station: update.station.map(|s| FfiStation {
                position: s.position.into(),
                expected_gain: s.expected_gain,
                walk_distance_m: s.walk_distance_m,
            }),
            floor_coverage: update.status.floor_coverage,
            wall_coverage: update.status.wall_coverage,
            ceiling_coverage: update.status.ceiling_coverage,
            complete: update.status.complete,
            keyframes: update.keyframes,
        })
    }

    /// Call after ARKit/ARCore reports a successful relocalisation, so the next
    /// frame is not rejected for being too similar to a pre-loss keyframe.
    pub fn on_relocalised(&self) {
        self.inner.lock().unwrap().on_relocalised();
    }

    /// Where to send a user who has lost tracking (PLAN.md §6.1).
    pub fn last_known_good_position(&self) -> Option<FfiVec3> {
        self.inner.lock().unwrap().last_known_good_position().map(Into::into)
    }
}

/// The tuned constants, exposed so a client can display them and so the two
/// platforms cannot disagree about what "too fast" means.
#[derive(uniffi::Record, Clone, Copy)]
pub struct FfiParams {
    pub voxel_m: f32,
    pub q_min: f32,
    pub d_opt_m: f32,
    pub min_rotation_deg: f32,
    pub min_translation_m: f32,
    pub max_keyframes: u32,
    pub too_fast_deg_per_s: f32,
    pub too_close_depth_m: f32,
    pub floor_target: f32,
    pub wall_target: f32,
    pub ceiling_target: f32,
}

#[uniffi::export]
pub fn default_params() -> FfiParams {
    let c = CoverageParams::default();
    let q = QualityParams::default();
    let _s = StationParams::default();
    let t = CompletionThresholds::default();
    FfiParams {
        voxel_m: c.voxel_m,
        q_min: c.q_min,
        d_opt_m: c.d_opt_m,
        min_rotation_deg: q.min_rotation_rad.to_degrees(),
        min_translation_m: q.min_translation_m,
        max_keyframes: q.max_keyframes,
        too_fast_deg_per_s: q.too_fast_rad_per_s.to_degrees(),
        too_close_depth_m: q.too_close_depth_m,
        floor_target: t.floor,
        wall_target: t.walls,
        ceiling_target: t.ceiling,
    }
}
