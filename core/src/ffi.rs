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
    /// Nothing new in it — plan §7's "adaptive selection".
    RejectRedundant,
    /// Waiting for the user to aim at, or settle on, the requested shot.
    RejectOffTarget,
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
            KeyframeDecision::RejectRedundant => FfiDecision::RejectRedundant,
            KeyframeDecision::RejectOffTarget => FfiDecision::RejectOffTarget,
        }
    }
}

impl From<FfiDecision> for KeyframeDecision {
    fn from(d: FfiDecision) -> KeyframeDecision {
        match d {
            FfiDecision::Accept => KeyframeDecision::Accept,
            FfiDecision::RejectBlurry => KeyframeDecision::RejectBlurry,
            FfiDecision::RejectExposure => KeyframeDecision::RejectExposure,
            FfiDecision::RejectTracking => KeyframeDecision::RejectTracking,
            FfiDecision::RejectTooSimilar => KeyframeDecision::RejectTooSimilar,
            FfiDecision::RejectCapReached => KeyframeDecision::RejectCapReached,
            FfiDecision::RejectRedundant => KeyframeDecision::RejectRedundant,
            FfiDecision::RejectOffTarget => KeyframeDecision::RejectOffTarget,
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

/// The user-facing copy for a per-frame verdict, for the same reason as
/// `nag_message`: what a rejection means, and what to do about it, is the
/// core's to say.
#[uniffi::export]
pub fn decision_message(decision: FfiDecision) -> String {
    KeyframeDecision::from(decision).message().to_string()
}

/// True when the frame became a keyframe. Exported so a client cannot drift
/// from the core about what counts as captured — the clients draw a running
/// accepted/rejected tally from this.
#[uniffi::export]
pub fn decision_accepted(decision: FfiDecision) -> bool {
    KeyframeDecision::from(decision).accepted()
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
    /// Retain this frame's image: it is the best candidate for the current
    /// viewpoint so far. The core never sees pixels, so the client is told
    /// which frame to keep rather than handing frames in.
    pub hold: bool,
    /// Write the retained image as a keyframe now, using [`Self::commit_pose`].
    ///
    /// This refers to an *earlier* frame than the one just offered — the one
    /// the client was holding — so the pose travels with the instruction
    /// rather than being taken from the current frame.
    pub commit_held: bool,
    /// Pose of the frame being committed, world<-camera.
    pub commit_pose: Option<FfiPose>,
    pub nag: Option<FfiNag>,
    /// Where to draw the AR floor ring, if anywhere.
    pub station: Option<FfiStation>,
    pub floor_coverage: f32,
    pub wall_coverage: f32,
    pub ceiling_coverage: f32,
    pub complete: bool,
    pub keyframes: u32,
    /// Set while a scripted capture is running.
    pub shot: Option<FfiShot>,
}

#[derive(uniffi::Enum, Clone, Copy, PartialEq, Eq)]
pub enum FfiShotHeight {
    Bottom,
    Middle,
    Top,
}

#[derive(uniffi::Enum, Clone, Copy, PartialEq, Eq)]
pub enum FfiShotState {
    /// Aimed and steady; the frame is being taken.
    Take,
    /// Pointing somewhere else — turn towards the marker.
    NotAimed,
    /// On target but still moving — hold still.
    NotSteady,
    /// On target and steady, but still standing where the centre pass was shot.
    /// The second phase exists for the baseline; without moving there is none.
    NotMoved,
    /// Nothing left to shoot.
    Finished,
}

/// The shot the script is asking for.
///
/// `direction` is a world-space unit vector rather than yaw and pitch: the
/// client has to place a marker in the scene, and handing it angles would mean
/// re-deriving the basis on both platforms — which is how two clients start
/// disagreeing about where "the second corner" is.
#[derive(uniffi::Record, Clone, Copy)]
pub struct FfiShot {
    pub direction: FfiVec3,
    pub height: FfiShotHeight,
    /// True once the user has left the centre and is shooting for parallax.
    pub at_station: bool,
    pub state: FfiShotState,
    pub taken: u32,
    pub total: u32,
    /// Shots the user gave up on. Kept beside `taken` so a client cannot show
    /// progress without showing what it cost.
    pub skipped: u32,
}

/// Where to go, and which way to turn, to get tracking back.
#[derive(uniffi::Record, Clone, Copy)]
pub struct FfiRecovery {
    /// Last good camera position, ARCore world frame. Only drawable while
    /// ARCore's world is still valid — after a relocalisation it is stale.
    pub position: FfiVec3,
    /// Last good view direction, same caveat.
    pub forward: FfiVec3,
    /// Degrees to turn, positive clockwise. Absent without a compass.
    pub turn_deg: Option<f32>,
}

/// What to tell the user when tracking is lost.
///
/// Wording lives here for the same reason the nags do, but the reason is
/// sharper in this case: this is the only instruction given at a moment when
/// the screen is showing nothing trustworthy, and two clients disagreeing
/// about which way "right" is would send users in opposite directions.
#[uniffi::export]
pub fn recovery_message(recovery: FfiRecovery) -> String {
    match recovery.turn_deg {
        // Below the noise floor of an indoor compass: claiming a precise turn
        // here would be false precision, and the user is already facing near
        // enough the right way.
        Some(d) if d.abs() < 20.0 => "Facing the right way — walk back".to_string(),
        Some(d) if d > 0.0 => format!("Turn right about {}°, then walk back", d.round() as i32),
        Some(d) => format!("Turn left about {}°, then walk back", -d.round() as i32),
        None => "Go back to where you were standing".to_string(),
    }
}

/// A shot the user reported as unobtainable, for the capture bundle.
///
/// Yaw is relative to where the capture started, matching `Shot`; absolute
/// world yaw is not knowable at the point the script is built, and rewriting
/// these to world angles later would invent a precision they do not have.
#[derive(uniffi::Record, Clone, Copy)]
pub struct FfiSkippedShot {
    pub yaw_deg: f32,
    pub height: FfiShotHeight,
    pub at_station: bool,
}

/// A drawable snapshot of the coverage grid (plan §6.1).
///
/// Flat arrays rather than a vector of records for the same reason `observe`
/// takes them: this crosses the FFI boundary several times a second with
/// thousands of entries, and UniFFI copies every record.
#[derive(uniffi::Record, Clone)]
pub struct FfiCoverageCloud {
    /// Voxel centres as flat xyz triples, world frame.
    pub points: Vec<f32>,
    /// Per voxel, `quality / q_min` clamped to 0..1 — 1.0 has cleared the bar.
    /// This is what the client colours by, so a user can see which surfaces are
    /// merely *seen* and which are actually *done*.
    pub fill: Vec<f32>,
    /// Voxels in the grid before subsampling, so the client can say so.
    pub total: u32,
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
            hold: update.hold,
            commit_held: update.committed.is_some(),
            commit_pose: update.committed.map(|p| FfiPose {
                r: p.r.to_vec(),
                t: p.t.into(),
            }),
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
            shot: if update.shot.is_none() && update.shot_progress.1 > 0 {
                Some(FfiShot {
                    direction: FfiVec3 { x: 0.0, y: 0.0, z: 0.0 },
                    height: FfiShotHeight::Middle,
                    at_station: true,
                    state: FfiShotState::Finished,
                    taken: update.shot_progress.0,
                    total: update.shot_progress.1,
                    skipped: update.shots_skipped,
                })
            } else { update.shot.map(|s| FfiShot {
                direction: update.shot_direction.unwrap_or(crate::math::Vec3::ZERO).into(),
                height: match s.height {
                    crate::script::ShotHeight::Bottom => FfiShotHeight::Bottom,
                    crate::script::ShotHeight::Middle => FfiShotHeight::Middle,
                    crate::script::ShotHeight::Top => FfiShotHeight::Top,
                },
                at_station: s.phase == crate::script::ShotPhase::Station,
                state: match update.shot_outcome {
                    Some(crate::script::ShotOutcome::Take) => FfiShotState::Take,
                    Some(crate::script::ShotOutcome::NotSteady) => FfiShotState::NotSteady,
                    Some(crate::script::ShotOutcome::NotMoved) => FfiShotState::NotMoved,
                    Some(crate::script::ShotOutcome::Finished) => FfiShotState::Finished,
                    _ => FfiShotState::NotAimed,
                },
                taken: update.shot_progress.0,
                total: update.shot_progress.1,
                skipped: update.shots_skipped,
            })},
        })
    }

    /// Take the picture. Returns the pose of the frame written, so the client
    /// knows which of the images it is holding to save, or null when the
    /// camera has nothing usable to offer right now.
    ///
    /// Nothing vetoes this: the user can see the scene and the marker, and a
    /// shutter that refuses a deliberate press is the worst thing a capture
    /// app can do.
    pub fn capture_now(&self) -> Option<FfiPose> {
        self.inner.lock().unwrap().capture_now().map(|p| FfiPose {
            r: p.r.to_vec(),
            t: p.t.into(),
        })
    }

    /// Whether a frame is ready to be taken, for the shutter button's state.
    pub fn has_candidate(&self) -> bool {
        self.inner.lock().unwrap().has_candidate()
    }

    /// Commit the candidate still held, at the end of a capture. Returns true
    /// when the client should write the image it is holding.
    pub fn flush(&self) -> bool {
        self.inner.lock().unwrap().flush().is_some()
    }

    /// Compass heading for this frame, degrees clockwise from magnetic north.
    ///
    /// Fed separately from `observe` because it arrives on the sensor's
    /// schedule rather than the camera's, and because a device without a
    /// usable magnetometer simply never calls it.
    pub fn set_heading(&self, degrees: f32) {
        self.inner.lock().unwrap().set_heading(degrees);
    }

    /// Where to go to get tracking back, or `None` before the first keyframe.
    pub fn recovery(&self) -> Option<FfiRecovery> {
        self.inner.lock().unwrap().recovery().map(|r| FfiRecovery {
            position: r.position.into(),
            forward: r.forward.into(),
            turn_deg: r.turn_deg,
        })
    }

    /// The user cannot get the shot being asked for; move on to the next.
    ///
    /// Returns the abandoned shot so the client can write it into the bundle,
    /// or `None` if the script had already finished — the button is live for
    /// as long as the capture is, and the last shot can land in between.
    pub fn skip_shot(&self) -> Option<FfiSkippedShot> {
        self.inner.lock().unwrap().skip_shot().map(|s| FfiSkippedShot {
            yaw_deg: s.yaw_rad.to_degrees(),
            height: match s.height {
                crate::script::ShotHeight::Bottom => FfiShotHeight::Bottom,
                crate::script::ShotHeight::Middle => FfiShotHeight::Middle,
                crate::script::ShotHeight::Top => FfiShotHeight::Top,
            },
            at_station: s.phase == crate::script::ShotPhase::Station,
        })
    }

    /// Every shot given up on so far, for the bundle written at the end.
    pub fn skipped_shots(&self) -> Vec<FfiSkippedShot> {
        let guard = self.inner.lock().unwrap();
        let Some(script) = guard.script() else {
            return Vec::new();
        };
        script
            .skipped_shots()
            .map(|s| FfiSkippedShot {
                yaw_deg: s.yaw_rad.to_degrees(),
                height: match s.height {
                    crate::script::ShotHeight::Bottom => FfiShotHeight::Bottom,
                    crate::script::ShotHeight::Middle => FfiShotHeight::Middle,
                    crate::script::ShotHeight::Top => FfiShotHeight::Top,
                },
                at_station: s.phase == crate::script::ShotPhase::Station,
            })
            .collect()
    }

    /// Switch this session to a scripted capture (aim, hold, shoot).
    ///
    /// Takes effect on the next frame, whose heading becomes the script's yaw
    /// origin — so call it when the user is standing where they mean to start.
    pub fn start_script(&self) {
        self.inner.lock().unwrap().start_script(crate::script::ScriptParams::default());
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

    /// The coverage grid as points the client can draw over the real surfaces.
    ///
    /// A capture where the user cannot see what has been mapped is one where
    /// they find out what they missed after they have left the room, so this
    /// exists to be drawn rather than logged.
    ///
    /// Subsampling above `target_points` is by a hash of the voxel key, not by
    /// iteration order or reservoir sampling: a `HashMap` yields its entries in
    /// an arbitrary order that changes as it grows, so taking the first N would
    /// redraw a *different* subset every frame and the overlay would boil. A key
    /// hash keeps the chosen subset stable as long as the stride is, so points
    /// stay put and only thin out as the room grows.
    ///
    /// `target_points` is therefore a budget, not a hard cap — a hash-modulo
    /// keeps *about* that many, varying by a few percent either way. Trading the
    /// exact count for a stable subset is the whole point; a caller sizing a
    /// buffer should read the returned length rather than assume the budget.
    pub fn coverage_cloud(&self, target_points: u32) -> FfiCoverageCloud {
        let session = self.inner.lock().unwrap();
        let grid = session.grid();
        let q_min = grid.params().q_min;
        let voxels = grid.all_voxels();
        let total = voxels.len();

        let stride = if target_points == 0 || total <= target_points as usize {
            1u64
        } else {
            (total as u64).div_ceil(target_points as u64)
        };

        let expected = total / stride as usize + 1;
        let mut points = Vec::with_capacity(expected * 3);
        let mut fill = Vec::with_capacity(expected);
        for (key, vox) in voxels {
            if stride > 1 && hash_key(key) % stride != 0 {
                continue;
            }
            let c = grid.centre_of(key);
            points.push(c.x);
            points.push(c.y);
            points.push(c.z);
            fill.push((vox.quality / q_min).clamp(0.0, 1.0));
        }
        FfiCoverageCloud { points, fill, total: total as u32 }
    }
}

/// A cheap integer mix, so the subsample is a deterministic function of the
/// voxel's position rather than of hash-map internals.
fn hash_key(k: crate::coverage::VoxelKey) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for v in [k.0, k.1, k.2] {
        h ^= v as i64 as u64;
        h = h.wrapping_mul(0x100_0000_01b3);
    }
    h
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

#[cfg(test)]
mod tests {
    use super::*;

    fn session() -> FfiCaptureSession {
        FfiCaptureSession::new(FfiVec3 { x: 0.0, y: -1.0, z: 0.0 })
    }

    /// A frame that clears every quality gate, so the geometry is integrated.
    fn frame(t: f64, pos: FfiVec3) -> FfiFrame {
        FfiFrame {
            timestamp_s: t,
            pose: FfiPose {
                r: vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                t: pos,
            },
            tracking: FfiTracking::Normal,
            blur_score: 300.0,
            exposure_clip_ratio: 0.0,
            median_depth_m: 2.0,
            angular_velocity_rad_s: 0.2,
            sparse_point_count: 200,
            lux: 200.0,
        }
    }

    /// A wall of surface points at z = 0, facing +z towards the camera.
    fn wall(n: i32) -> (Vec<f32>, Vec<f32>) {
        let mut pts = Vec::new();
        let mut nrm = Vec::new();
        for i in 0..n {
            for j in 0..n {
                pts.extend_from_slice(&[i as f32 * 0.05, j as f32 * 0.05, 0.0]);
                nrm.extend_from_slice(&[0.0, 0.0, 1.0]);
            }
        }
        (pts, nrm)
    }

    #[test]
    fn coverage_cloud_is_empty_before_any_geometry() {
        let s = session();
        let cloud = s.coverage_cloud(4096);
        assert_eq!(cloud.total, 0);
        assert!(cloud.points.is_empty() && cloud.fill.is_empty());
    }

    #[test]
    fn coverage_cloud_pairs_one_fill_with_each_point() {
        let s = session();
        let (p, n) = wall(20);
        // Two viewpoints: geometry is credited when the held frame is written,
        // which happens once the camera has moved on from it.
        s.observe(frame(0.0, FfiVec3 { x: 0.5, y: 0.5, z: 2.0 }), p, n).unwrap();
        s.capture_now().expect("the shutter should take the held frame");
        let cloud = s.coverage_cloud(4096);
        assert!(cloud.total > 0, "a wall in view must produce voxels");
        assert_eq!(cloud.points.len(), cloud.fill.len() * 3);
        assert!(cloud.fill.iter().all(|f| (0.0..=1.0).contains(f)));
    }

    #[test]
    fn fill_rises_towards_one_as_a_voxel_is_re_observed() {
        let s = session();
        let (p, n) = wall(20);
        s.observe(frame(0.0, FfiVec3 { x: 0.5, y: 0.5, z: 2.0 }), p.clone(), n.clone()).unwrap();
        s.capture_now().expect("first shot");
        let first = s.coverage_cloud(4096).fill.iter().cloned().fold(0.0f32, f32::max);
        // A second view from elsewhere, taken deliberately as before.
        s.observe(frame(1.0, FfiVec3 { x: 0.5, y: 0.5, z: 1.2 }), p, n).unwrap();
        s.capture_now().expect("second shot");
        let second = s.coverage_cloud(4096).fill.iter().cloned().fold(0.0f32, f32::max);
        assert!(second > first, "a second view should raise fill: {first} -> {second}");
        assert!(second <= 1.0, "fill is clamped at the quality bar");
    }

    /// The point of hashing the key rather than taking the first N: the overlay
    /// must not redraw a different subset of voxels on every frame.
    #[test]
    fn subsampling_picks_the_same_voxels_every_call() {
        let s = session();
        let (p, n) = wall(40);
        // Two viewpoints: geometry is credited when the held frame is written.
        s.observe(frame(0.0, FfiVec3 { x: 0.5, y: 0.5, z: 2.0 }), p, n).unwrap();
        s.capture_now().expect("the shutter should take the held frame");
        let budget = 50;
        let a = s.coverage_cloud(budget);
        let b = s.coverage_cloud(budget);
        assert!(a.total > budget, "test needs more voxels than the budget to be meaningful");
        // A budget, not a cap — see coverage_cloud. Close enough that a client
        // sizing a buffer from it is not surprised, and never everything.
        assert!(
            a.fill.len() <= (budget as f32 * 1.5) as usize && a.fill.len() < a.total as usize,
            "kept {} of {} against a budget of {budget}", a.fill.len(), a.total,
        );
        assert!(!a.points.is_empty(), "subsampling must not drop everything");
        let mut pa = a.points.clone();
        let mut pb = b.points.clone();
        pa.sort_by(f32::total_cmp);
        pb.sort_by(f32::total_cmp);
        assert_eq!(pa, pb, "the same grid must yield the same subset");
    }
}
