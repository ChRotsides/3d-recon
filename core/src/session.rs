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
use crate::script::{CaptureScript, ScriptParams, ShotOutcome, ShotPhase};
use crate::station::{next_best_station, Station, StationParams};

#[derive(Debug, Clone, Copy)]
pub struct SessionConfig {
    pub coverage: CoverageParams,
    pub quality: QualityParams,
    pub station: StationParams,
    pub completion: CompletionThresholds,
    /// Recompute the suggested station at most this often (plan §6.1: every 2 s).
    pub station_interval_s: f64,
    /// A frame showing less than this fraction of not-yet-covered surface is
    /// redundant. Plan §7's "adaptive selection": the keyframe budget is a cost
    /// ceiling, and spending it on views of finished walls is what makes a
    /// capture end at the cap instead of at room-complete.
    pub min_novel_fraction: f32,
    /// Fewest contributing samples before the redundancy test is believed. A
    /// frame that happens to catch a sliver of geometry must not be discarded on
    /// the strength of it, and a device with no depth at all must still capture.
    pub min_novel_samples: usize,
}

impl Default for SessionConfig {
    fn default() -> Self {
        SessionConfig {
            coverage: CoverageParams::default(),
            quality: QualityParams::default(),
            station: StationParams::default(),
            completion: CompletionThresholds::default(),
            station_interval_s: 2.0,
            min_novel_fraction: 0.10,
            min_novel_samples: 200,
        }
    }
}

#[derive(Debug, Clone)]
pub struct GuidanceUpdate {
    pub decision: KeyframeDecision,
    /// The shot the script is asking for, if a script is running.
    pub shot: Option<crate::script::Shot>,
    /// World direction to aim in for that shot.
    pub shot_direction: Option<Vec3>,
    /// Why the current frame did not satisfy it.
    pub shot_outcome: Option<ShotOutcome>,
    /// `(taken, total)` over the whole script.
    pub shot_progress: (u32, u32),
    /// Shots the user has given up on. Shown next to the progress count so
    /// "20 of 24" cannot be read as a complete room when four were abandoned.
    pub shots_skipped: u32,
    /// Retain this frame's image: best candidate for the current viewpoint.
    pub hold: bool,
    /// The pose of the frame just committed, if one was. The client writes the
    /// image it retained, not the frame in front of it.
    pub committed: Option<Pose>,
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
    /// What the capture has actually collected — the evidence completion is
    /// decided from, rather than how many scripted shots were ticked off.
    capture: crate::capture::CaptureStats,
    /// Geometry of the frame currently held for the shutter.
    ///
    /// Kept here rather than in the gate so the gate stays a pure decision
    /// over statistics. It has to be kept *somewhere*: the coverage grid must
    /// be credited with the points of the frame actually written, and by the
    /// time the user presses the shutter that frame is several frames in the
    /// past — its depth samples are long gone from the client.
    held_points: Vec<Vec3>,
    held_normals: Vec<Vec3>,
    /// Compass heading reported by the client for the current frame, degrees
    /// clockwise from magnetic north.
    heading_deg: Option<f32>,
    /// Heading at the last accepted keyframe.
    ///
    /// Kept *alongside* `last_good_pose` rather than derived from it, because
    /// the two survive different things. ARCore relocalises by moving the world
    /// origin, so after a relocalisation a remembered world-frame direction
    /// points somewhere that no longer means anything — real captures have
    /// jumped 5.6 m across one. Magnetic heading is measured against the earth,
    /// not against ARCore's world, so it is still valid on the other side of a
    /// rebase, and it is available while the pose is not trackable at all.
    last_good_heading_deg: Option<f32>,
    script: Option<CaptureScript>,
    /// Held until the first frame, which is what defines "where the user is
    /// facing" — the script's yaw origin.
    pending_script: Option<ScriptParams>,
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
            capture: crate::capture::CaptureStats::new((-gravity).normalized()),
            held_points: Vec::new(),
            held_normals: Vec::new(),
            heading_deg: None,
            last_good_heading_deg: None,
            script: None,
            pending_script: None,
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
        // The first frame is what defines "ahead" for the whole script, so the
        // script cannot be built until one arrives.
        if self.script.is_none() {
            if let Some(params) = self.pending_script.take() {
                self.script =
                    Some(CaptureScript::centre_pass(params, self.up, stats.pose.forward()));
            }
        }

        let mut shot_outcome = None;
        let mut redundant = false;
        let usable_here = stats.tracking == crate::quality::TrackingState::Normal;
        self.capture.observe_frame(stats, usable_here);

        // One selector for both modes. Frames are no longer accepted or
        // rejected on the spot: the best of each viewpoint is held and written
        // when the view moves on, which is what a phone camera has always done
        // with the burst it was already capturing. Under the old rule a real
        // capture kept 8 frames out of 1843 — the user was being asked to hit
        // thresholds the device is better placed to hit for them.
        // Every viable frame is offered, whatever it is pointed at. The
        // script still draws a target, but it no longer decides when a picture
        // is taken — the user does, and the selector's only job is to have the
        // sharpest frame of the last moment ready for them.
        // Redundancy is now *reported*, never enforced. It was a budget rule
        // for a capture that took its own pictures; with a shutter it would
        // mean the user presses take-picture in a well-covered room and
        // nothing happens, which is the one thing a camera must never do.
        redundant = self.is_redundant(stats, points, normals);
        let selection = self.gate.offer(stats);
        if selection.hold {
            self.held_points.clear();
            self.held_points.extend_from_slice(points);
            self.held_normals.clear();
            self.held_normals.extend_from_slice(normals);
        }
        if self.script.is_some() {
            shot_outcome = self.script.as_ref().map(|s| s.evaluate(stats));
        }

        if let Some(committed) = selection.committed {
            self.capture
                .accept_keyframe(&committed.pose, committed.sparse_point_count);
            self.last_good_pose = Some(committed.pose);
            self.last_good_heading_deg = self.heading_deg;
            if let Some(s) = self.script.as_mut() {
                s.commit(committed.pose.position());
            }
        }

        // Telemetry only, now that selection is continuous: this reports what
        // became of the frame, and nothing in the UI depends on it.
        let mut decision = if selection.committed.is_some() {
            KeyframeDecision::Accept
        } else if !usable_here {
            KeyframeDecision::RejectTracking
        } else if redundant {
            KeyframeDecision::RejectRedundant
        } else if shot_outcome.is_some() && shot_outcome != Some(ShotOutcome::Take) {
            KeyframeDecision::RejectOffTarget
        } else if selection.hold {
            KeyframeDecision::Accept
        } else {
            KeyframeDecision::RejectTooSimilar
        };

        let _ = &mut decision;

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
            &self.capture,
        );

        // The centre pass is structure without parallax, so a second position is
        // always required once it is shot.
        //
        // Deliberately *not* conditional on the station solver having something
        // to suggest. Twelve shots from one spot make a room look thoroughly
        // covered — every surface seen, from a good distance, in focus — and the
        // solver duly reports nothing worth walking to. It is still an
        // unreconstructable capture, because every one of those views shares a
        // camera centre. Coverage completeness and reconstructability are
        // different properties, and only one of them is visible to the grid.
        if let Some(script) = self.script.as_mut() {
            if script.phase_finished(ShotPhase::Centre) && !script.has_phase(ShotPhase::Station) {
                script.add_station();
            } else if script.evaluate(stats) == ShotOutcome::Finished && !status.complete {
                // Target count is guidance progress, never a completion rule.
                // If the supported surface coverage is still short, add another
                // translated pass instead of leaving a finished script that can
                // neither accept frames nor tell the user where to look.
                script.add_station();
            }
        }

        let (shot, shot_direction, shot_progress, shots_skipped) = match self.script.as_ref() {
            None => (None, None, (0, 0), 0),
            Some(s) => {
                let (done, total) = s.progress();
                (
                    s.current().copied(),
                    s.current().map(|c| s.direction_of(c)),
                    (done as u32, total as u32),
                    s.skipped() as u32,
                )
            }
        };

        GuidanceUpdate {
            decision,
            shot,
            shot_direction,
            shot_outcome,
            shot_progress,
            shots_skipped,
            hold: selection.hold,
            committed: selection.committed.map(|c| c.pose),
            nag,
            station: self.last_station,
            status,
            keyframes: self.gate.accepted_count(),
        }
    }

    /// Run a scripted capture: aim, hold, shoot, rather than sweep and hope.
    ///
    /// Takes effect on the next frame, whose heading becomes the script's yaw
    /// origin. Leaving this unset keeps the free-form rule, which is what plan
    /// §6.3's continuous walk between rooms needs.
    pub fn start_script(&mut self, params: ScriptParams) {
        self.pending_script = Some(params);
        self.script = None;
    }

    pub fn script(&self) -> Option<&CaptureScript> {
        self.script.as_ref()
    }

    /// The user cannot get the shot the script is asking for. Give up on that
    /// one and offer the next.
    ///
    /// Deliberately not conditional on any measurement: the whole point is
    /// that the obstruction is something the device cannot see — a wardrobe,
    /// a doorway too narrow to back through, a neighbour's fence beyond the
    /// window. The user is the sensor here, and refusing their assessment
    /// leaves them stuck on a shot that will never be satisfied.
    ///
    /// Returns the abandoned shot so the client can record it in the bundle.
    pub fn skip_shot(&mut self) -> Option<crate::script::Shot> {
        self.script.as_mut()?.skip_current()
    }

    /// Whether this frame shows the capture anything it does not already have.
    ///
    /// Two deliberate abstentions. With no geometry at all the answer is "keep
    /// it": a device without depth support would otherwise have every frame
    /// rejected and capture nothing. With too few contributing samples the
    /// frame is also kept, because discarding an image on the evidence of a
    /// handful of pixels risks losing a view the cloud reconstruction wanted —
    /// the coverage grid is a proxy for what the photogrammetry needs, and a
    /// proxy should not get the benefit of the doubt.
    fn is_redundant(&self, stats: &FrameStats, points: &[Vec3], normals: &[Vec3]) -> bool {
        if points.is_empty() {
            return false;
        }
        let (novel, contributing) = self.grid.novel_fraction(
            points,
            normals,
            stats.pose.position(),
            stats.blur_score,
        );
        contributing >= self.config.min_novel_samples && novel < self.config.min_novel_fraction
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

    /// Where the camera was looking when tracking was last good.
    ///
    /// World frame, so it is only meaningful while ARCore's world still is —
    /// see `recovery`, which is the frame-independent half of the same answer.
    pub fn last_known_good_forward(&self) -> Option<Vec3> {
        self.last_good_pose.map(|p| p.forward())
    }

    /// Take the picture the user just asked for.
    ///
    /// Commits the best frame of the last moment and advances the script past
    /// the shot they were lining up. Nothing vetoes it: they can see the scene
    /// and the marker, and refusing a deliberate shutter press is worse than
    /// keeping an imperfect frame.
    ///
    /// Returns the pose of the frame written, so the client knows which of the
    /// images it is holding to save, or `None` if no frame is ready — the
    /// camera has nothing usable right now.
    pub fn capture_now(&mut self) -> Option<Pose> {
        let committed = self.gate.capture_now()?;
        self.capture
            .accept_keyframe(&committed.pose, committed.sparse_point_count);
        self.last_good_pose = Some(committed.pose);
        self.last_good_heading_deg = self.heading_deg;
        if let Some(s) = self.script.as_mut() {
            s.commit(committed.pose.position());
        }
        self.observe_held_geometry(&committed);
        Some(committed.pose)
    }

    /// Credit the coverage grid with the written frame's surfaces.
    fn observe_held_geometry(&mut self, committed: &FrameStats) {
        let cam = committed.pose.position();
        for (p, n) in self.held_points.iter().zip(self.held_normals.iter()) {
            self.grid.observe(*p, *n, cam, committed.blur_score);
        }
        if !self.floor_seen {
            // Fall back to the lowest observed up-facing surface.
            for (p, n) in self.held_points.iter().zip(self.held_normals.iter()) {
                if n.normalized().dot(self.up) > 0.85 {
                    let h = p.dot(self.up);
                    if !self.floor_seen || h < self.floor_y {
                        self.floor_y = h;
                        self.floor_seen = true;
                    }
                }
            }
        }
        self.held_points.clear();
        self.held_normals.clear();
    }

    /// Whether a frame is ready for the shutter, for the button's state.
    pub fn has_candidate(&self) -> bool {
        self.gate.has_candidate()
    }

    /// Commit whatever candidate is still held, at the end of a capture.
    ///
    /// Without this the last viewpoint of every capture is silently dropped —
    /// its frame is held waiting for a move that never comes, because the user
    /// has stopped recording.
    pub fn flush(&mut self) -> Option<Pose> {
        let committed = self.gate.flush()?;
        self.capture
            .accept_keyframe(&committed.pose, committed.sparse_point_count);
        self.last_good_pose = Some(committed.pose);
        self.observe_held_geometry(&committed);
        Some(committed.pose)
    }

    /// The client's compass reading for this frame, degrees clockwise from
    /// magnetic north. Optional: a device without a usable magnetometer simply
    /// never calls this, and `recovery` then returns no bearing.
    pub fn set_heading(&mut self, deg: f32) {
        self.heading_deg = Some(deg.rem_euclid(360.0));
    }

    /// How to get back to where tracking was working.
    ///
    /// The turn is computed from compass headings rather than from the poses,
    /// deliberately. When tracking is lost there is no trustworthy current pose
    /// to measure a world-frame bearing against — which is exactly the moment
    /// this is needed. Two magnetometer readings can still be differenced, and
    /// the difference survives a relocalisation that rebases ARCore's world.
    ///
    /// Indoors an absolute compass bearing is worth little — steel and wiring
    /// bend it by tens of degrees — but the *difference* between two readings
    /// taken in the same room minutes apart shares most of that error and
    /// cancels it. So this reports a relative turn and never a heading.
    pub fn recovery(&self) -> Option<Recovery> {
        let target = self.last_good_pose?;
        let turn_deg = match (self.heading_deg, self.last_good_heading_deg) {
            (Some(now), Some(then)) => {
                // Signed shortest turn, positive clockwise (to the user's right).
                let mut d = then - now;
                while d > 180.0 {
                    d -= 360.0;
                }
                while d < -180.0 {
                    d += 360.0;
                }
                Some(d)
            }
            _ => None,
        };
        Some(Recovery {
            position: target.position(),
            forward: target.forward(),
            turn_deg,
        })
    }
}

/// Where the user has to get back to, for a capture to continue.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Recovery {
    /// Last good camera position, ARCore world frame.
    pub position: Vec3,
    /// Last good view direction, ARCore world frame.
    pub forward: Vec3,
    /// Degrees to turn, positive clockwise. `None` when the device has no
    /// compass, or when no heading was recorded at the last good frame.
    pub turn_deg: Option<f32>,
}
