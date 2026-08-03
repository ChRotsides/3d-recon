//! Keyframe selection, quality gates and the live-nag state machine (plan §6.1).
//!
//! Keyframe on `>=15 deg rotation OR >=30 cm translation` since the last one,
//! AND blur below threshold, AND tracking normal. Hard cap 150 per room, which
//! is also the cost control from plan §7.

use crate::math::{Pose, Vec3};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrackingState {
    Normal,
    Limited,
    Relocalizing,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Nag {
    TooClose,
    TooFast,
    Textureless,
    CeilingMissing,
    TrackingLost,
    DarkRoom,
}

impl Nag {
    /// Copy shown to the user. Concrete instructions, not diagnoses.
    pub fn message(self) -> &'static str {
        match self {
            Nag::TooClose => "Step back a little",
            Nag::TooFast => "Slow down",
            Nag::Textureless => "Include a corner or doorway in the frame",
            Nag::CeilingMissing => "Tilt up",
            Nag::TrackingLost => "Go back to where you were standing",
            Nag::DarkRoom => "Turn on the lights",
        }
    }

    /// Lower value wins when several fire at once. Tracking loss trumps
    /// everything because nothing else can be fixed while it persists.
    fn priority(self) -> u8 {
        match self {
            Nag::TrackingLost => 0,
            Nag::DarkRoom => 1,
            Nag::TooFast => 2,
            Nag::TooClose => 3,
            Nag::Textureless => 4,
            Nag::CeilingMissing => 5,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct QualityParams {
    pub min_rotation_rad: f32,
    pub min_translation_m: f32,
    /// Variance of Laplacian below this is too blurry to keep.
    pub min_blur_score: f32,
    pub max_exposure_clip_ratio: f32,
    pub max_keyframes: u32,
    // Nag thresholds, plan §6.1.
    pub too_close_depth_m: f32,
    pub too_fast_rad_per_s: f32,
    pub min_sparse_points: u32,
    pub min_ceiling_coverage: f32,
    pub min_lux: f32,
    pub nag_cooldown_s: f32,
}

impl Default for QualityParams {
    fn default() -> Self {
        QualityParams {
            min_rotation_rad: 15.0_f32.to_radians(),
            min_translation_m: 0.30,
            min_blur_score: 100.0,
            max_exposure_clip_ratio: 0.25,
            max_keyframes: 150,
            too_close_depth_m: 0.80,
            too_fast_rad_per_s: 60.0_f32.to_radians(),
            min_sparse_points: 25,
            min_ceiling_coverage: 0.40,
            min_lux: 25.0,
            nag_cooldown_s: 3.0,
        }
    }
}

/// Everything the guidance layer needs to know about one camera frame.
#[derive(Debug, Clone, Copy)]
pub struct FrameStats {
    pub timestamp_s: f64,
    pub pose: Pose,
    pub tracking: TrackingState,
    pub blur_score: f32,
    pub exposure_clip_ratio: f32,
    pub median_depth_m: f32,
    pub angular_velocity_rad_s: f32,
    pub sparse_point_count: u32,
    pub lux: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeyframeDecision {
    Accept,
    /// Rejected, with the reason — surfaced in capture telemetry so a high
    /// rejection rate on real devices is visible rather than guessed at.
    RejectBlurry,
    RejectExposure,
    RejectTracking,
    RejectTooSimilar,
    RejectCapReached,
}

impl KeyframeDecision {
    pub fn accepted(self) -> bool {
        matches!(self, KeyframeDecision::Accept)
    }
}

#[derive(Debug, Clone)]
pub struct QualityGate {
    params: QualityParams,
    last_keyframe: Option<Pose>,
    accepted: u32,
    last_nag: Option<(Nag, f64)>,
}

impl QualityGate {
    pub fn new(params: QualityParams) -> Self {
        QualityGate { params, last_keyframe: None, accepted: 0, last_nag: None }
    }

    pub fn accepted_count(&self) -> u32 {
        self.accepted
    }

    pub fn params(&self) -> QualityParams {
        self.params
    }

    /// Decide whether to keep this frame as a keyframe.
    ///
    /// Order matters: the cheap hard gates run before the baseline check, so a
    /// blurry frame during a fast sweep never resets the baseline and lets a
    /// second blurry frame through behind it.
    pub fn consider(&mut self, f: &FrameStats) -> KeyframeDecision {
        if self.accepted >= self.params.max_keyframes {
            return KeyframeDecision::RejectCapReached;
        }
        if f.tracking != TrackingState::Normal {
            return KeyframeDecision::RejectTracking;
        }
        if f.blur_score < self.params.min_blur_score {
            return KeyframeDecision::RejectBlurry;
        }
        if f.exposure_clip_ratio > self.params.max_exposure_clip_ratio {
            return KeyframeDecision::RejectExposure;
        }
        let far_enough = match self.last_keyframe {
            None => true,
            Some(prev) => {
                let moved = prev.position().distance(f.pose.position());
                let rotated = prev.angle_to(&f.pose);
                moved >= self.params.min_translation_m
                    || rotated >= self.params.min_rotation_rad
            }
        };
        if !far_enough {
            return KeyframeDecision::RejectTooSimilar;
        }
        self.last_keyframe = Some(f.pose);
        self.accepted += 1;
        KeyframeDecision::Accept
    }

    /// At most one nag at a time, with a cooldown (plan §6.1).
    pub fn nag(&mut self, f: &FrameStats, ceiling_coverage: f32) -> Option<Nag> {
        let mut candidates: Vec<Nag> = Vec::new();
        if f.tracking != TrackingState::Normal {
            candidates.push(Nag::TrackingLost);
        }
        if f.lux < self.params.min_lux {
            candidates.push(Nag::DarkRoom);
        }
        if f.angular_velocity_rad_s > self.params.too_fast_rad_per_s {
            candidates.push(Nag::TooFast);
        }
        if f.median_depth_m > 0.0 && f.median_depth_m < self.params.too_close_depth_m {
            candidates.push(Nag::TooClose);
        }
        if f.sparse_point_count < self.params.min_sparse_points {
            candidates.push(Nag::Textureless);
        }
        if ceiling_coverage < self.params.min_ceiling_coverage {
            candidates.push(Nag::CeilingMissing);
        }
        candidates.sort_by_key(|n| n.priority());
        let next = *candidates.first()?;

        if let Some((prev, at)) = self.last_nag {
            let elapsed = (f.timestamp_s - at) as f32;
            // The cooldown suppresses repeats of the *same* nag; a
            // higher-priority one interrupts immediately, because telling
            // someone to slow down is useless once tracking has already gone.
            if elapsed < self.params.nag_cooldown_s && next.priority() >= prev.priority() {
                return None;
            }
        }
        self.last_nag = Some((next, f.timestamp_s));
        Some(next)
    }

    pub fn reset_after_relocalisation(&mut self) {
        self.last_keyframe = None;
    }
}

/// Room-complete rule from plan §6.1.
#[derive(Debug, Clone, Copy)]
pub struct CompletionThresholds {
    pub floor: f32,
    pub walls: f32,
    pub ceiling: f32,
}

impl Default for CompletionThresholds {
    fn default() -> Self {
        CompletionThresholds { floor: 0.95, walls: 0.90, ceiling: 0.70 }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct CompletionStatus {
    pub floor_coverage: f32,
    pub wall_coverage: f32,
    pub ceiling_coverage: f32,
    pub has_productive_station: bool,
    pub complete: bool,
}

pub fn assess_completion(
    grid: &crate::coverage::CoverageGrid,
    up: Vec3,
    has_productive_station: bool,
    thresholds: CompletionThresholds,
) -> CompletionStatus {
    let floor = grid.directional_coverage(up, 0.85);
    let ceiling = grid.directional_coverage(-up, 0.85);
    let walls = grid.worst_wall_direction_coverage(up);
    CompletionStatus {
        floor_coverage: floor,
        wall_coverage: walls,
        ceiling_coverage: ceiling,
        has_productive_station,
        complete: floor >= thresholds.floor
            && walls >= thresholds.walls
            && ceiling >= thresholds.ceiling
            && !has_productive_station,
    }
}
