//! The object the mobile clients actually drive.
//!
//! One `CaptureSession` per room. Swift and Kotlin push frames in and read
//! guidance out; everything stateful lives here so the two clients cannot drift
//! apart in behaviour.

use crate::coverage::{CoverageGrid, CoverageParams};
use crate::math::{Pose, Vec3};
use crate::quality::{
    assess_completion, CompletionStatus, CompletionThresholds, FrameStats, KeyframeDecision,
    Nag, QualityGate, QualityParams,
};
use crate::station::{next_best_station, Station, StationParams};

#[derive(Debug, Clone, Copy)]
pub struct SessionConfig {
    pub coverage: CoverageParams,
    pub quality: QualityParams,
    pub station: StationParams,
    pub completion: CompletionThresholds,
    /// Recompute the suggested station at most this often (plan §6.1: every 2 s).
    pub station_interval_s: f64,
}

impl Default for SessionConfig {
    fn default() -> Self {
        SessionConfig {
            coverage: CoverageParams::default(),
            quality: QualityParams::default(),
            station: StationParams::default(),
            completion: CompletionThresholds::default(),
            station_interval_s: 2.0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct GuidanceUpdate {
    pub decision: KeyframeDecision,
    pub nag: Option<Nag>,
    pub station: Option<Station>,
    pub status: CompletionStatus,
    pub keyframes: u32,
}

pub struct CaptureSession {
    config: SessionConfig,
    grid: CoverageGrid,
    gate: QualityGate,
    up: Vec3,
    floor_y: f32,
    floor_seen: bool,
    last_station: Option<Station>,
    last_station_at: f64,
    last_good_pose: Option<Pose>,
}

impl CaptureSession {
    pub fn new(config: SessionConfig, gravity: Vec3) -> Self {
        let up = (-gravity).normalized();
        CaptureSession {
            grid: CoverageGrid::new(config.coverage),
            gate: QualityGate::new(config.quality),
            config,
            up,
            floor_y: 0.0,
            floor_seen: false,
            last_station: None,
            last_station_at: f64::NEG_INFINITY,
            last_good_pose: None,
        }
    }

    pub fn grid(&self) -> &CoverageGrid {
        &self.grid
    }

    pub fn up(&self) -> Vec3 {
        self.up
    }

    pub fn floor_height(&self) -> f32 {
        self.floor_y
    }

    /// The client's estimate of the floor plane, from ARKit/ARCore plane
    /// detection. Better than inferring it from the point cloud, and it is
    /// available before any depth has been run.
    pub fn set_floor_height(&mut self, y: f32) {
        self.floor_y = y;
        self.floor_seen = true;
    }

    /// Push one frame's worth of observed geometry plus its statistics.
    ///
    /// `points`/`normals` are the on-device depth pass (DA3-Small at 1-2 Hz,
    /// plan §3.6) back-projected to the world — deliberately *not* every frame.
    pub fn observe_frame(
        &mut self,
        stats: &FrameStats,
        points: &[Vec3],
        normals: &[Vec3],
    ) -> GuidanceUpdate {
        let decision = self.gate.consider(stats);

        if decision.accepted() {
            self.last_good_pose = Some(stats.pose);
            let cam = stats.pose.position();
            for (p, n) in points.iter().zip(normals.iter()) {
                self.grid.observe(*p, *n, cam, stats.blur_score);
            }
            if !self.floor_seen {
                // Fall back to the lowest observed up-facing surface.
                for (p, n) in points.iter().zip(normals.iter()) {
                    if n.normalized().dot(self.up) > 0.85 {
                        let h = p.dot(self.up);
                        if !self.floor_seen || h < self.floor_y {
                            self.floor_y = h;
                            self.floor_seen = true;
                        }
                    }
                }
            }
        }

        let ceiling_coverage = self.grid.directional_coverage(-self.up, 0.85);
        let nag = self.gate.nag(stats, ceiling_coverage);

        if stats.timestamp_s - self.last_station_at >= self.config.station_interval_s {
            let user = self.last_good_pose.unwrap_or(stats.pose).position();
            self.last_station = next_best_station(
                &self.grid,
                user,
                self.floor_y,
                self.up,
                self.config.station,
            );
            self.last_station_at = stats.timestamp_s;
        }

        let status = assess_completion(
            &self.grid,
            self.up,
            self.last_station.is_some(),
            self.config.completion,
        );

        GuidanceUpdate {
            decision,
            nag,
            station: self.last_station,
            status,
            keyframes: self.gate.accepted_count(),
        }
    }

    /// Call after ARKit/ARCore reports a successful relocalisation, so the next
    /// frame is not rejected for being too similar to a pre-loss keyframe.
    pub fn on_relocalised(&mut self) {
        self.gate.reset_after_relocalisation();
    }

    /// Where to send a user who has lost tracking (plan §6.1 TRACKING_LOST).
    pub fn last_known_good_position(&self) -> Option<Vec3> {
        self.last_good_pose.map(|p| p.position())
    }
}
