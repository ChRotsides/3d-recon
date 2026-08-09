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
    /// Waiting for the user to aim at, or hold still on, the requested shot.
    RejectOffTarget,
    /// Sharp, well-placed, and shows nothing the capture does not already have
    /// to the required quality.
    ///
    /// Plan §7 pairs the 150-keyframe cap with "adaptive selection", and this is
    /// it. Without it the budget is spent by a purely *local* rule — moved far
    /// enough, sharp enough — which happily takes a fifth view of a wall that is
    /// already done while the ceiling stays empty. The cap then ends the capture
    /// rather than the room-complete rule, which is what happened on the first
    /// real capture: 150 frames, and coverage that never moved.
    RejectRedundant,
}

impl KeyframeDecision {
    pub fn accepted(self) -> bool {
        matches!(self, KeyframeDecision::Accept)
    }

    /// Copy shown to the user for this frame, in the same voice as `Nag`:
    /// what to do, not what went wrong.
    ///
    /// This lives here rather than in the clients because ADR-0001 puts the
    /// wording with the decision. A client that phrased `RejectTooSimilar` as
    /// "duplicate frame" would be telling the user their frame was bad, when
    /// what the core means is that the *viewpoint* was — the fix is to step
    /// sideways, and only the core knows that.
    pub fn message(self) -> &'static str {
        match self {
            KeyframeDecision::Accept => "Captured",
            KeyframeDecision::RejectBlurry => "Too blurry — hold still",
            KeyframeDecision::RejectExposure => "Too bright — angle away from the window",
            KeyframeDecision::RejectTracking => "Tracking lost — go back and wait",
            KeyframeDecision::RejectTooSimilar => "Same spot — take a step, don't just turn",
            KeyframeDecision::RejectCapReached => "Frame budget spent — finish the capture",
            KeyframeDecision::RejectOffTarget => "Line up the marker",
            KeyframeDecision::RejectRedundant => "Already covered — aim somewhere new",
        }
    }
}

/// Tuning for the best-of-window selector.
#[derive(Debug, Clone, Copy)]
pub struct SelectionParams {
    /// Longest a candidate may be held before it is committed anyway.
    ///
    /// Bounds the wait when the user stops moving: without it a stationary
    /// phone would hold its candidate forever waiting for a viewpoint change
    /// that never comes, and the capture would silently stop collecting.
    pub max_hold_s: f64,
    /// Blur below which a frame is not worth holding at any price. Far below
    /// the old accept threshold: this is "unusable", not "imperfect".
    pub unusable_blur: f32,
    /// Clipping above which the frame is mostly blown highlights.
    pub unusable_exposure: f32,
    /// Weight on exposure clipping when scoring, relative to sharpness.
    pub exposure_penalty: f32,
    /// Weight on angular velocity, per radian per second.
    pub motion_penalty: f32,
}

impl Default for SelectionParams {
    fn default() -> Self {
        SelectionParams {
            // Long enough to span the pause between two deliberate movements,
            // short enough that a slow pan still yields frames steadily.
            max_hold_s: 1.0,
            unusable_blur: 20.0,
            unusable_exposure: 0.60,
            exposure_penalty: 400.0,
            motion_penalty: 120.0,
        }
    }
}

/// What the client should do with the frame it just offered.
///
/// The client holds the pixels, so it has to be told which frame to keep; the
/// core never sees an image. Both flags can be set at once — the common case
/// at a viewpoint change is "write the one you were holding, now hold this".
#[derive(Debug, Clone, Copy, Default)]
pub struct Selection {
    /// Retain this frame's image: it is the best candidate so far.
    pub hold: bool,
    /// The candidate just committed, if any. The client writes the image it
    /// retained when this frame was held — which is an *earlier* frame than
    /// the one being offered, and why the stats travel back with it.
    pub committed: Option<FrameStats>,
}

impl Selection {
    pub fn commit_held(&self) -> bool {
        self.committed.is_some()
    }
}

/// A candidate frame, held while the window is open.
#[derive(Debug, Clone, Copy)]
struct Candidate {
    stats: FrameStats,
    score: f32,
}

#[derive(Debug, Clone)]
pub struct QualityGate {
    params: QualityParams,
    selection: SelectionParams,
    last_keyframe: Option<Pose>,
    accepted: u32,
    last_nag: Option<(Nag, f64)>,
    held: Option<Candidate>,
    window_opened_s: f64,
}

impl QualityGate {
    pub fn new(params: QualityParams) -> Self {
        QualityGate {
            params,
            selection: SelectionParams::default(),
            last_keyframe: None,
            accepted: 0,
            last_nag: None,
            held: None,
            window_opened_s: 0.0,
        }
    }

    /// How good this frame is as a picture of whatever it is pointed at.
    ///
    /// Sharpness, docked for blown highlights and for motion. A single number
    /// because the choice being made is a comparison between two frames of the
    /// *same* view seconds apart, where "which is less smeared" is the whole
    /// question. It is deliberately not a pass mark: nothing is rejected for
    /// scoring low, it just loses to something better.
    pub fn score(&self, f: &FrameStats) -> f32 {
        f.blur_score
            - f.exposure_clip_ratio * self.selection.exposure_penalty
            - f.angular_velocity_rad_s * self.selection.motion_penalty
    }

    /// Whether a frame is a *good* candidate — never whether it is allowed.
    ///
    /// Nothing is excluded. A frame failing this still becomes the candidate
    /// when there is nothing better, because the alternative is telling
    /// someone who deliberately pressed the shutter that they may not have a
    /// picture. Recovering a dark or smeared frame is the reconstruction's
    /// problem, and it has techniques for it; refusing to take one leaves
    /// nothing to recover from.
    fn viable(&self, f: &FrameStats) -> bool {
        f.tracking == TrackingState::Normal
            && f.sparse_point_count >= self.params.min_sparse_points
            && f.lux >= self.params.min_lux
            && f.blur_score >= self.selection.unusable_blur
            && f.exposure_clip_ratio <= self.selection.unusable_exposure
    }

    /// True once the held candidate is looking at something the current frame
    /// no longer is — the moment to commit it and start a new window.
    fn window_closed(&self, held: &Candidate, f: &FrameStats) -> bool {
        if f.timestamp_s - self.window_opened_s >= self.selection.max_hold_s {
            return true;
        }
        let moved = held.stats.pose.position().distance(f.pose.position());
        let rotated = held.stats.pose.angle_to(&f.pose);
        moved >= self.params.min_translation_m || rotated >= self.params.min_rotation_rad
    }

    /// Offer a frame to the selector.
    ///
    /// Keeps the best frame of the last moment ready, and commits nothing on
    /// its own — [`Self::capture_now`] does that, when the user presses the
    /// shutter. The selector's job is no longer deciding *whether* to take a
    /// picture, only *which* of the last second's frames is the one worth
    /// keeping when the user says take it. That removes the shutter lag
    /// without taking the decision away from them.
    ///
    /// A stale candidate is dropped rather than committed: once the view has
    /// moved on, the held frame is a picture of somewhere the user is no
    /// longer pointing, and pressing the shutter should never save that.
    pub fn offer(&mut self, f: &FrameStats) -> Selection {
        if let Some(held) = self.held {
            if self.window_closed(&held, f) {
                self.held = None;
            }
        }

        let mut out = Selection::default();
        // A poor frame still scores, just far below any usable one, so it can
        // never displace a good candidate and can always stand in for a
        // missing one. There is always something to take a picture of.
        let score = if self.viable(f) { self.score(f) } else { self.score(f) - 1.0e6 };
        let better = match self.held {
            None => true,
            Some(held) => score > held.score,
        };
        if better {
            if self.held.is_none() {
                self.window_opened_s = f.timestamp_s;
            }
            self.held = Some(Candidate { stats: *f, score });
            out.hold = true;
        }
        out
    }

    /// Take the picture. Commits the best frame of the last moment.
    ///
    /// Unconditional by design: no aim test, no steadiness test, no coverage
    /// veto. The user is looking at the scene and the target marker and has
    /// judged them aligned, which is a better instrument than a 17-degree
    /// tolerance — and being refused after deliberately pressing a shutter is
    /// the single most frustrating thing a capture app can do.
    /// Never refuses. The only way this returns `None` is if no frame has
    /// reached the core at all, which means the camera is not running.
    ///
    /// The keyframe cap is deliberately not consulted. It is a cost ceiling
    /// for a capture that took its own pictures; against a finger on a shutter
    /// it would be a button that stops working at 150 with no explanation.
    pub fn capture_now(&mut self) -> Option<FrameStats> {
        let held = self.held.take()?;
        self.commit(&held.stats);
        Some(held.stats)
    }

    /// Whether there is a frame ready to be taken, for the shutter's state.
    pub fn has_candidate(&self) -> bool {
        self.held.is_some()
    }

    /// Commit whatever is held, if anything — for the end of a capture, so the
    /// last viewpoint is not silently dropped along with the window it was in.
    pub fn flush(&mut self) -> Option<FrameStats> {
        match self.held.take() {
            Some(held) if self.accepted < self.params.max_keyframes => {
                self.commit(&held.stats);
                Some(held.stats)
            }
            _ => None,
        }
    }

    /// Pose of the candidate currently held, for the client to write with it.
    pub fn held_pose(&self) -> Option<Pose> {
        self.held.map(|c| c.stats.pose)
    }

    pub fn accepted_count(&self) -> u32 {
        self.accepted
    }

    pub fn params(&self) -> QualityParams {
        self.params
    }

    /// Decide whether to keep this frame, recording it if so.
    pub fn consider(&mut self, f: &FrameStats) -> KeyframeDecision {
        let decision = self.evaluate(f);
        if decision.accepted() {
            self.commit(f);
        }
        decision
    }

    /// Record an accepted keyframe as the new baseline.
    ///
    /// Split from [`Self::evaluate`] so that a later gate — the coverage-gain
    /// test, which cannot run until the frame's geometry is known — can veto a
    /// frame without leaving the baseline advanced to one that was never kept.
    /// Advancing it anyway would let the *next* frame through on a comparison
    /// against a keyframe that does not exist in the bundle.
    pub fn commit(&mut self, f: &FrameStats) {
        self.last_keyframe = Some(f.pose);
        self.accepted += 1;
    }

    /// The hard gates only: cap, tracking, blur, exposure.
    ///
    /// Split out for scripted capture, where the *script* decides when to shoot
    /// and the "moved 30 cm or turned 15°" rule would fight it — a shot asked
    /// for from a fixed spot is by definition not far from the last one.
    pub fn evaluate_shot(&self, f: &FrameStats) -> KeyframeDecision {
        if self.accepted >= self.params.max_keyframes {
            return KeyframeDecision::RejectCapReached;
        }
        if f.tracking != TrackingState::Normal {
            return KeyframeDecision::RejectTracking;
        }
        if f.sparse_point_count < self.params.min_sparse_points
            || f.angular_velocity_rad_s > self.params.too_fast_rad_per_s
            || f.lux < self.params.min_lux
        {
            return KeyframeDecision::RejectTracking;
        }
        if f.blur_score < self.params.min_blur_score {
            return KeyframeDecision::RejectBlurry;
        }
        if f.exposure_clip_ratio > self.params.max_exposure_clip_ratio {
            return KeyframeDecision::RejectExposure;
        }
        KeyframeDecision::Accept
    }

    /// Decide whether to keep this frame as a keyframe, without recording it.
    ///
    /// Order matters: the cheap hard gates run before the baseline check, so a
    /// blurry frame during a fast sweep never resets the baseline and lets a
    /// second blurry frame through behind it.
    pub fn evaluate(&self, f: &FrameStats) -> KeyframeDecision {
        if self.accepted >= self.params.max_keyframes {
            return KeyframeDecision::RejectCapReached;
        }
        if f.tracking != TrackingState::Normal {
            return KeyframeDecision::RejectTracking;
        }
        if f.sparse_point_count < self.params.min_sparse_points
            || f.angular_velocity_rad_s > self.params.too_fast_rad_per_s
            || f.lux < self.params.min_lux
        {
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
        // The held candidate must go with it. ARCore relocalises by moving the
        // world origin, so a frame captured before the rebase carries a pose
        // expressed in a frame that no longer exists — committing it afterwards
        // would place its geometry somewhere the camera never was. Dropping one
        // good picture is the cheap side of that trade.
        self.held = None;
    }
}

/// Room-complete rule from plan §6.1.
#[derive(Debug, Clone, Copy)]
pub struct CompletionThresholds {
    pub floor: f32,
    pub walls: f32,
    pub ceiling: f32,
    /// Fewest keyframes a room can be reconstructed from.
    pub min_keyframes: usize,
    /// Furthest apart two keyframes must be. This is the capture's parallax
    /// budget: below it every view shares a camera centre and there is nothing
    /// to triangulate, however many pictures were taken.
    pub min_baseline_m: f32,
    /// Typical translation between consecutive keyframes. Guards the case the
    /// max baseline cannot see — a user who walks to the far corner, stands
    /// there, and turns has a fine max baseline and no parallax anywhere.
    pub min_median_step_m: f32,
    /// Yaw sectors that must hold a keyframe.
    pub min_captured_sectors: usize,
    /// Fraction of judged frames that must have had healthy tracking.
    pub min_tracking_health: f32,
    /// Median sparse features over accepted keyframes. A capture of blank
    /// plaster registers nothing downstream however sharp the pictures are.
    pub min_median_features: u32,
}

impl Default for CompletionThresholds {
    fn default() -> Self {
        CompletionThresholds {
            floor: 0.95,
            walls: 0.90,
            ceiling: 0.70,
            min_keyframes: 20,
            // A room-sized scene needs metre-scale parallax; the failed real
            // capture managed 1.97 m of *extent* but only 1.3 m of usable box.
            min_baseline_m: 2.0,
            // About a pace. Chosen so that turning on the spot cannot satisfy it.
            min_median_step_m: 0.25,
            min_captured_sectors: 8,
            min_tracking_health: 0.80,
            min_median_features: 40,
        }
    }
}

/// Why a capture is not finished yet, in the order worth telling the user.
///
/// A reason rather than a bare boolean because "not complete" is useless as an
/// instruction. The whole point of moving completion off shot-count is that the
/// app can now say which of these is outstanding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Incomplete {
    /// Tracking has been unhealthy for much of the capture.
    TrackingUnhealthy,
    /// Too little texture in view for anything to register later.
    Featureless,
    /// Not enough frames yet.
    TooFewKeyframes,
    /// The camera has not travelled far enough for triangulation.
    NotEnoughParallax,
    /// Frames are being taken from the same spot — rotation, not translation.
    TurningNotWalking,
    /// Whole directions of the room hold no keyframe.
    DirectionsMissing,
    /// Directions were skipped, or attempted and yielded nothing usable.
    UnresolvedDirections,
    /// Surfaces have been seen but not yet well enough.
    CoverageThin,
    /// There is a better place to stand.
    StationOutstanding,
}

impl Incomplete {
    /// What to do about it. Translation is phrased as a distance to walk,
    /// deliberately: "turn to the next wall" is the instruction that produced
    /// captures with no parallax in them.
    pub fn message(self) -> &'static str {
        match self {
            Incomplete::TrackingUnhealthy => "Tracking is struggling — move to a textured area",
            Incomplete::Featureless => "Too little detail here — include a corner or furniture",
            Incomplete::TooFewKeyframes => "Keep going — not enough frames yet",
            Incomplete::NotEnoughParallax => "Walk to the other side of the room",
            Incomplete::TurningNotWalking => "Move 40–80 cm sideways, don't turn on the spot",
            Incomplete::DirectionsMissing => "Some directions have no frames yet",
            Incomplete::UnresolvedDirections => "Some directions were skipped or unusable",
            Incomplete::CoverageThin => "Keep working the surfaces that are still amber",
            Incomplete::StationOutstanding => "Walk to the ring",
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct CompletionStatus {
    pub floor_coverage: f32,
    pub wall_coverage: f32,
    pub ceiling_coverage: f32,
    pub has_productive_station: bool,
    pub complete: bool,
    /// The first outstanding requirement, or `None` when complete.
    pub blocker: Option<Incomplete>,
    pub keyframes: usize,
    pub max_baseline_m: f32,
    pub median_step_m: f32,
    pub captured_sectors: usize,
    pub unresolved_sectors: usize,
    pub tracking_health: f32,
}

/// Decide whether the room has actually been captured.
///
/// Ordered from "nothing else can work until this is fixed" downwards, so the
/// blocker reported is the one worth acting on rather than the first that
/// happens to be checked. Geometry gates come before coverage gates because a
/// capture with no parallax cannot be rescued by more pictures of the walls.
pub fn assess_completion(
    grid: &crate::coverage::CoverageGrid,
    up: Vec3,
    has_productive_station: bool,
    thresholds: CompletionThresholds,
    capture: &crate::capture::CaptureStats,
) -> CompletionStatus {
    let floor = grid.directional_coverage(up, 0.85);
    let ceiling = grid.directional_coverage(-up, 0.85);
    let walls = grid.worst_wall_direction_coverage(up);

    let blocker = if capture.tracking_health() < thresholds.min_tracking_health {
        Some(Incomplete::TrackingUnhealthy)
    } else if capture.keyframes() > 0 && capture.median_features() < thresholds.min_median_features {
        Some(Incomplete::Featureless)
    } else if capture.keyframes() < thresholds.min_keyframes {
        Some(Incomplete::TooFewKeyframes)
    } else if capture.max_baseline_m() < thresholds.min_baseline_m {
        Some(Incomplete::NotEnoughParallax)
    } else if capture.median_step_m() < thresholds.min_median_step_m {
        Some(Incomplete::TurningNotWalking)
    } else if capture.captured_sectors() < thresholds.min_captured_sectors {
        Some(Incomplete::DirectionsMissing)
    } else if capture.unresolved_sectors() > 0 {
        // A skipped or unusable direction is never "covered". It can end the
        // capture — the user may genuinely be unable to get it — but it cannot
        // be reported as done, because the reconstruction will be guessing there.
        Some(Incomplete::UnresolvedDirections)
    } else if floor < thresholds.floor || walls < thresholds.walls || ceiling < thresholds.ceiling {
        Some(Incomplete::CoverageThin)
    } else if has_productive_station {
        Some(Incomplete::StationOutstanding)
    } else {
        None
    };

    CompletionStatus {
        floor_coverage: floor,
        wall_coverage: walls,
        ceiling_coverage: ceiling,
        has_productive_station,
        complete: blocker.is_none(),
        blocker,
        keyframes: capture.keyframes(),
        max_baseline_m: capture.max_baseline_m(),
        median_step_m: capture.median_step_m(),
        captured_sectors: capture.captured_sectors(),
        unresolved_sectors: capture.unresolved_sectors(),
        tracking_health: capture.tracking_health(),
    }
}
