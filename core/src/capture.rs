//! What the capture has actually collected, as opposed to what it was asked for.
//!
//! Completion used to mean "the coverage bars are full and there is no better
//! station to send the user to". That is a statement about the *voxels that
//! happened to be observed*, and a capture can satisfy it having barely moved:
//! the grid does not know that thirty views of one wall from one spot cannot be
//! triangulated. A real capture passed every coverage gate, was declared ready,
//! and then reconstructed a tabletop as the room's ceiling.
//!
//! So the session records the things a reconstruction actually needs — how far
//! the camera travelled, which directions it faced from where, whether tracking
//! was healthy, how much texture there was to track — and completion is decided
//! against those. A capture of thirty frames with a metre of travel should fail.
//! Twenty-two well-distributed frames should pass.

use crate::math::{Pose, Vec3};
use crate::quality::TrackingState;

/// Number of yaw sectors the room is divided into for coverage bookkeeping.
/// Twelve gives 30-degree wedges — narrower than a phone's ~75-degree lens, so
/// a sector cannot be credited by a view that merely clipped its edge.
pub const YAW_SECTORS: usize = 12;

/// What happened to one direction of the room.
///
/// Four states, not two, because "we tried and could not" and "we never looked"
/// call for different things: the first is a fact about the room to record, the
/// second is work outstanding. Collapsing them is what let a skipped wall count
/// as covered.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SectorState {
    /// Never looked at.
    #[default]
    Unseen,
    /// Frames were offered here but none were usable — too dark, too smeared,
    /// tracking down, nothing to track.
    Attempted,
    /// The user reported this direction as unobtainable.
    Skipped,
    /// At least one keyframe was written looking this way.
    Captured,
}

impl SectorState {
    /// Higher wins when a sector is updated: a captured sector is not demoted
    /// by a later bad frame, and a skip does not erase a shot already taken.
    fn rank(self) -> u8 {
        match self {
            SectorState::Unseen => 0,
            SectorState::Attempted => 1,
            SectorState::Skipped => 2,
            SectorState::Captured => 3,
        }
    }

    fn raise_to(&mut self, other: SectorState) {
        if other.rank() > self.rank() {
            *self = other;
        }
    }
}

/// Running record of what the capture has collected.
#[derive(Debug, Clone)]
pub struct CaptureStats {
    up: Vec3,
    /// Positions of accepted keyframes, in order.
    positions: Vec<Vec3>,
    /// Yaw sector of each accepted keyframe, for overlap bookkeeping.
    sectors: [SectorState; YAW_SECTORS],
    /// Sparse feature counts at accepted keyframes.
    feature_counts: Vec<u32>,
    frames_seen: u64,
    frames_tracking_ok: u64,
    /// Largest distance between any two accepted keyframes, metres.
    max_baseline_m: f32,
}

impl CaptureStats {
    pub fn new(up: Vec3) -> Self {
        CaptureStats {
            up: up.normalized(),
            positions: Vec::new(),
            sectors: [SectorState::Unseen; YAW_SECTORS],
            feature_counts: Vec::new(),
            frames_seen: 0,
            frames_tracking_ok: 0,
            max_baseline_m: 0.0,
        }
    }

    /// Yaw sector a direction falls in, measured about the up axis.
    pub fn sector_of(&self, forward: Vec3) -> usize {
        // Any two axes perpendicular to up will do; the absolute origin is
        // irrelevant because every check here is about spread, not bearing.
        let e0 = crate::coverage::pick_tangent(self.up);
        let e1 = self.up.cross(e0);
        let flat = forward - self.up * forward.dot(self.up);
        if flat.length() < 1e-6 {
            // Straight up or down belongs to no sector; charge it to the first
            // rather than dividing by zero. Ceiling and floor coverage are
            // measured by the grid, not here.
            return 0;
        }
        let angle = flat.dot(e1).atan2(flat.dot(e0));
        let turns = (angle / std::f32::consts::TAU).rem_euclid(1.0);
        ((turns * YAW_SECTORS as f32) as usize).min(YAW_SECTORS - 1)
    }

    /// Every frame the session judged, whether or not it was kept.
    pub fn observe_frame(&mut self, stats: &crate::quality::FrameStats, usable: bool) {
        self.frames_seen += 1;
        if stats.tracking == TrackingState::Normal {
            self.frames_tracking_ok += 1;
        }
        if !usable {
            let s = self.sector_of(stats.pose.forward());
            self.sectors[s].raise_to(SectorState::Attempted);
        }
    }

    /// A frame that became a keyframe.
    pub fn accept_keyframe(&mut self, pose: &Pose, feature_count: u32) {
        let p = pose.position();
        for prev in &self.positions {
            self.max_baseline_m = self.max_baseline_m.max(prev.distance(p));
        }
        self.positions.push(p);
        self.feature_counts.push(feature_count);
        let s = self.sector_of(pose.forward());
        self.sectors[s].raise_to(SectorState::Captured);
    }

    /// A direction the user reported as unobtainable.
    pub fn skip_direction(&mut self, forward: Vec3) {
        let s = self.sector_of(forward);
        self.sectors[s].raise_to(SectorState::Skipped);
    }

    pub fn keyframes(&self) -> usize {
        self.positions.len()
    }

    pub fn max_baseline_m(&self) -> f32 {
        self.max_baseline_m
    }

    pub fn sectors(&self) -> &[SectorState; YAW_SECTORS] {
        &self.sectors
    }

    pub fn captured_sectors(&self) -> usize {
        self.sectors.iter().filter(|s| **s == SectorState::Captured).count()
    }

    /// Sectors that were tried and yielded nothing usable, plus ones the user
    /// gave up on. These are the ones that must not read as covered.
    pub fn unresolved_sectors(&self) -> usize {
        self.sectors
            .iter()
            .filter(|s| matches!(s, SectorState::Attempted | SectorState::Skipped))
            .count()
    }

    /// Fraction of judged frames where ARCore was tracking normally.
    pub fn tracking_health(&self) -> f32 {
        if self.frames_seen == 0 {
            return 1.0;
        }
        self.frames_tracking_ok as f32 / self.frames_seen as f32
    }

    /// Median sparse feature count over accepted keyframes. Median rather than
    /// mean because one richly textured bookcase should not paper over a
    /// capture that was otherwise pointed at blank plaster.
    pub fn median_features(&self) -> u32 {
        if self.feature_counts.is_empty() {
            return 0;
        }
        let mut v = self.feature_counts.clone();
        v.sort_unstable();
        v[v.len() / 2]
    }

    /// Translation between consecutive keyframes, metres — the parallax the
    /// reconstruction actually gets. Rotation contributes nothing here, which
    /// is the point: turning on the spot produces views without triangulation.
    pub fn median_step_m(&self) -> f32 {
        if self.positions.len() < 2 {
            return 0.0;
        }
        let mut steps: Vec<f32> = self
            .positions
            .windows(2)
            .map(|w| w[0].distance(w[1]))
            .collect();
        steps.sort_by(|a, b| a.partial_cmp(b).unwrap());
        steps[steps.len() / 2]
    }
}
