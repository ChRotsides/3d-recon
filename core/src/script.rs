//! Scripted capture: aim, hold, shoot — instead of sweeping and hoping.
//!
//! The free-form rule (plan §6.1: 30 cm or 15° since the last keyframe) asks the
//! user to keep moving and keeps whatever is sharp enough. On real captures that
//! spends most of the budget badly: the largest single rejection category was
//! motion blur — 166 to 406 frames thrown away per capture — because a walking
//! sweep in indoor light is blurred more often than not, and the frames that do
//! survive land wherever the user happened to be pointing.
//!
//! A script asks for a specific shot at a time and waits for the device to be
//! still. Deliberate stills remove the blur category outright, corners bound the
//! room's geometry in a dozen frames, and "3 of 12" is something a person can
//! act on in a way that a coverage percentage is not.
//!
//! **A script from one position cannot reconstruct.** Every shot would share a
//! camera centre, and with no baseline there is no triangulation — the harness
//! refuses such bundles outright (`test_validator_rejects_a_pure_rotation_capture`:
//! "No parallax, no reconstruction"). So the script is in two phases: the centre
//! pass for structure and speed, then shots from stations the solver picks,
//! which is where the parallax comes from. The centre pass alone is a fast room
//! outline, not a capture.

use crate::coverage::pick_tangent;
use crate::math::Vec3;
use crate::quality::FrameStats;

/// Which part of the script a shot belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShotPhase {
    /// Standing where the capture began, turning on the spot.
    Centre,
    /// Standing where the next-best-station solver sent the user.
    Station,
}

/// How high to aim. Named rather than numeric because it is a user-facing
/// instruction before it is an angle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShotHeight {
    /// Down at the wall-floor junction, which bounds the footprint.
    Bottom,
    /// Level, which is most of the wall.
    Middle,
    /// Up at the wall-ceiling junction, which bounds the height and is the
    /// part users skip unprompted — hence the plan's `CEILING_MISSING` nag.
    Top,
}

impl ShotHeight {
    /// Pitch in radians. Not symmetric: from 1.5 m in a 2.5 m room the ceiling
    /// junction is a steeper look up than the floor junction is down.
    pub fn pitch_rad(self) -> f32 {
        match self {
            ShotHeight::Bottom => -0.61, // ~35 deg down
            ShotHeight::Middle => 0.0,
            ShotHeight::Top => 0.52,     // ~30 deg up
        }
    }
}

/// Whether a shot is still owed, and if not, why not.
///
/// Tri-state rather than a `done` flag because "taken" and "given up on" are
/// different facts about a room and the harness has to be able to tell them
/// apart. Collapsing them would let a capture that skipped its way to the end
/// report the same progress as one that shot everything.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShotState {
    Pending,
    Taken,
    /// The user said this shot is not obtainable — furniture in the way, a
    /// wall too close to back away from, a corner behind a wardrobe. Rooms
    /// like that exist, and a script with no way out of one is a script the
    /// user abandons entirely, which costs the whole capture rather than
    /// one direction of it.
    Skipped,
}

#[derive(Debug, Clone, Copy)]
pub struct Shot {
    pub phase: ShotPhase,
    /// Yaw about the up axis, measured from the direction the user faced when
    /// the capture started. Relative because the room's orientation is unknown
    /// at the first frame — which is exactly when this has to be shown.
    pub yaw_rad: f32,
    pub height: ShotHeight,
    pub state: ShotState,
}

impl Shot {
    /// Resolved one way or the other; the script will not ask for it again.
    pub fn done(&self) -> bool {
        self.state != ShotState::Pending
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ScriptParams {
    /// How close to the target direction counts as aimed.
    pub aim_tolerance_rad: f32,
    /// Above this the device is still moving and the shot would smear.
    pub steady_rad_per_s: f32,
    /// Yaw directions in the centre pass.
    ///
    /// Set by the camera, not by the room's shape. A phone's horizontal field
    /// of view is about 75 degrees (measured: 74.6 on the S23 that shot the
    /// first of these), so four directions at 90 degrees apart leave a 15
    /// degree wedge between every pair that nothing ever sees. The first
    /// scripted capture reconstructed as disconnected patches for exactly that
    /// reason — one patch per shot, gaps in between.
    ///
    /// Twelve gives 30 degree spacing and about 60% horizontal overlap. The
    /// generous overlap is intentional: depth alignment and feature matching
    /// both need shared surfaces, not merely touching fields of view.
    pub centre_yaws: usize,
    /// Yaw directions asked for at each station. Fewer than the centre pass —
    /// the station is there for baseline — but still overlapping.
    pub station_yaws: usize,
    /// How far the user must have moved from the centre before a station shot
    /// counts. Roughly a pace: enough baseline to triangulate a room-sized
    /// scene, small enough to ask of someone in a cluttered lounge.
    pub min_baseline_m: f32,
}

impl Default for ScriptParams {
    fn default() -> Self {
        ScriptParams {
            // Generous: this is "point roughly there", not a shooting gallery.
            // Too tight and the user fights the reticle instead of capturing.
            aim_tolerance_rad: 0.30, // ~17 deg
            steady_rad_per_s: 0.35,
            centre_yaws: 12,
            station_yaws: 10,
            min_baseline_m: 0.8,
        }
    }
}

/// What happened to a frame offered to the script.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShotOutcome {
    /// Aimed, steady: take it.
    Take,
    /// Pointing somewhere else.
    NotAimed,
    /// Aimed but still moving.
    NotSteady,
    /// Aimed and steady, but still standing where the centre pass was shot.
    ///
    /// The reason the second phase exists is the baseline. Without this check a
    /// user can satisfy every "station" shot without moving, producing exactly
    /// the single-viewpoint bundle the harness refuses.
    NotMoved,
    /// Nothing left to shoot in this phase.
    Finished,
}

pub struct CaptureScript {
    params: ScriptParams,
    shots: Vec<Shot>,
    /// The world direction yaw is measured from — where the user first looked.
    reference: Vec3,
    up: Vec3,
    /// Where the centre pass was shot from, set by its first committed shot.
    origin: Option<Vec3>,
}

impl CaptureScript {
    /// Start a centre pass, with yaw measured from where the user is facing.
    pub fn centre_pass(params: ScriptParams, up: Vec3, facing: Vec3) -> Self {
        let up = up.normalized();
        // Project the initial heading onto the floor plane; looking slightly up
        // or down at the start must not tilt every subsequent target.
        let flat = facing - up * facing.dot(up);
        let reference = if flat.length() < 1e-3 { pick_tangent(up) } else { flat.normalized() };

        let mut shots = Vec::new();
        for i in 0..params.centre_yaws {
            let yaw = (i as f32) * std::f32::consts::TAU / (params.centre_yaws as f32);
            // Snake vertically. Ending one heading at the same height where
            // the next begins avoids the old ceiling-to-floor diagonal jump,
            // which was about 78 degrees and destroyed frame overlap.
            let heights = if i % 2 == 0 {
                [ShotHeight::Bottom, ShotHeight::Middle, ShotHeight::Top]
            } else {
                [ShotHeight::Top, ShotHeight::Middle, ShotHeight::Bottom]
            };
            for height in heights {
                shots.push(Shot {
                    phase: ShotPhase::Centre,
                    yaw_rad: yaw,
                    height,
                    state: ShotState::Pending,
                });
            }
        }
        CaptureScript { params, shots, reference, up, origin: None }
    }

    /// Append the shots asked for at a station the solver has chosen.
    ///
    /// Fewer than the centre pass, and aimed *between* the directions it
    /// already shot: the point of standing somewhere else is the baseline, not
    /// another inventory of the same views.
    ///
    /// The offset is half of the **centre** pass's step, not half of the
    /// station's own. With four centre directions and three here, offsetting by
    /// half a station step would land one shot at 180° — straight back down a
    /// direction already covered.
    pub fn add_station(&mut self) {
        let n = self.params.station_yaws.max(1);
        let step = std::f32::consts::TAU / (n as f32);
        let offset = std::f32::consts::TAU / (2.0 * self.params.centre_yaws.max(1) as f32);
        for i in 0..n {
            let yaw = step * (i as f32) + offset;
            let heights = if i % 2 == 0 {
                [ShotHeight::Bottom, ShotHeight::Middle]
            } else {
                [ShotHeight::Middle, ShotHeight::Bottom]
            };
            for height in heights {
                shots_push(&mut self.shots, ShotPhase::Station, yaw, height);
            }
        }
    }

    pub fn current(&self) -> Option<&Shot> {
        self.shots.iter().find(|s| !s.done())
    }

    /// `(taken, total)` for the whole script. Skipped shots are *not* taken:
    /// the counter has to keep meaning "frames you actually got".
    pub fn progress(&self) -> (usize, usize) {
        (
            self.shots.iter().filter(|s| s.state == ShotState::Taken).count(),
            self.shots.len(),
        )
    }

    /// How many shots the user has given up on.
    pub fn skipped(&self) -> usize {
        self.shots.iter().filter(|s| s.state == ShotState::Skipped).count()
    }

    /// The abandoned shots, for the client to record in the bundle. The harness
    /// cannot otherwise distinguish a direction nobody could reach from one the
    /// script never asked for.
    pub fn skipped_shots(&self) -> impl Iterator<Item = &Shot> {
        self.shots.iter().filter(|s| s.state == ShotState::Skipped)
    }

    pub fn has_phase(&self, phase: ShotPhase) -> bool {
        self.shots.iter().any(|s| s.phase == phase)
    }

    pub fn phase_finished(&self, phase: ShotPhase) -> bool {
        self.shots.iter().filter(|s| s.phase == phase).all(|s| s.done())
    }

    /// The world-space direction the user should point the camera in.
    pub fn direction_of(&self, shot: &Shot) -> Vec3 {
        let e0 = self.reference;
        let e1 = self.up.cross(e0);
        let (sy, cy) = shot.yaw_rad.sin_cos();
        let (sp, cp) = shot.height.pitch_rad().sin_cos();
        ((e0 * cy + e1 * sy) * cp + self.up * sp).normalized()
    }

    /// Offer a frame to the script without consuming it.
    pub fn evaluate(&self, stats: &FrameStats) -> ShotOutcome {
        let Some(shot) = self.current() else {
            return ShotOutcome::Finished;
        };
        let want = self.direction_of(shot);
        let have = stats.pose.forward();
        if have.dot(want).clamp(-1.0, 1.0).acos() > self.params.aim_tolerance_rad {
            return ShotOutcome::NotAimed;
        }
        if stats.angular_velocity_rad_s > self.params.steady_rad_per_s {
            return ShotOutcome::NotSteady;
        }
        if shot.phase == ShotPhase::Station {
            if let Some(origin) = self.origin {
                if origin.distance(stats.pose.position()) < self.params.min_baseline_m {
                    return ShotOutcome::NotMoved;
                }
            }
        }
        ShotOutcome::Take
    }

    /// Mark the current shot taken, from `position`.
    pub fn commit(&mut self, position: Vec3) {
        if self.origin.is_none() {
            self.origin = Some(position);
        }
        if let Some(s) = self.shots.iter_mut().find(|s| !s.done()) {
            s.state = ShotState::Taken;
        }
    }

    /// Give up on the current shot and move to the next one.
    ///
    /// Only this shot: the heights at a direction fail independently, and the
    /// common case is a ceiling corner hidden by a wardrobe whose floor
    /// junction is perfectly visible. Skipping the whole bearing there would
    /// throw away two obtainable shots to avoid one that is not.
    ///
    /// Returns the shot that was abandoned, or `None` if nothing was pending.
    pub fn skip_current(&mut self) -> Option<Shot> {
        let s = self.shots.iter_mut().find(|s| !s.done())?;
        s.state = ShotState::Skipped;
        Some(*s)
    }

    /// Nothing left pending. Skipped shots count as resolved — a room with an
    /// unreachable corner must still be able to finish.
    pub fn finished(&self) -> bool {
        self.shots.iter().all(|s| s.done())
    }
}

fn shots_push(shots: &mut Vec<Shot>, phase: ShotPhase, yaw_rad: f32, height: ShotHeight) {
    shots.push(Shot { phase, yaw_rad, height, state: ShotState::Pending });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::{look_rotation, Pose};
    use crate::quality::TrackingState;

    const UP: Vec3 = Vec3 { x: 0.0, y: 1.0, z: 0.0 };

    fn facing_north() -> Vec3 {
        Vec3::new(0.0, 0.0, -1.0)
    }

    fn script() -> CaptureScript {
        CaptureScript::centre_pass(ScriptParams::default(), UP, facing_north())
    }

    /// A frame looking in `dir`, steady and sharp unless told otherwise.
    fn aimed_at(s: &CaptureScript, dir: Vec3, angular: f32) -> FrameStats {
        // Recover yaw/pitch of `dir` in the script's own basis so the test
        // drives the same convention the script reads.
        let e0 = s.reference;
        let e1 = s.up.cross(e0);
        let pitch = dir.normalized().dot(s.up).clamp(-1.0, 1.0).asin();
        let yaw = dir.dot(e1).atan2(dir.dot(e0));
        FrameStats {
            timestamp_s: 0.0,
            pose: Pose { r: look_rotation(yaw, pitch), t: Vec3::ZERO },
            tracking: TrackingState::Normal,
            blur_score: 300.0,
            exposure_clip_ratio: 0.0,
            median_depth_m: 2.0,
            angular_velocity_rad_s: angular,
            sparse_point_count: 200,
            lux: 200.0,
        }
    }

    #[test]
    fn the_centre_pass_is_every_direction_at_three_heights() {
        let s = script();
        let (done, total) = s.progress();
        assert_eq!(done, 0);
        assert_eq!(total, 36, "12 overlapping directions x bottom/middle/top");
        assert!(s.shots.iter().all(|x| x.phase == ShotPhase::Centre));
    }

    /// Measured horizontal field of view of the phone that shot the first
    /// scripted capture. Shots spaced wider than this see disjoint cones.
    const CAMERA_FOV_DEG: f32 = 74.6;

    #[test]
    fn neighbouring_shots_overlap_rather_than_merely_abut() {
        // The first scripted capture used 90 degree spacing against a 74.6
        // degree lens and reconstructed as disconnected patches — one per shot.
        let spacing = 360.0 / (ScriptParams::default().centre_yaws as f32);
        assert!(
            spacing < CAMERA_FOV_DEG * 0.75,
            "shots {spacing} deg apart leave a gap in a {CAMERA_FOV_DEG} deg lens"
        );
    }

    #[test]
    fn consecutive_targets_never_make_a_large_diagonal_jump() {
        let mut s = script();
        s.add_station();
        for phase in [ShotPhase::Centre, ShotPhase::Station] {
            let phase_shots: Vec<_> = s.shots.iter().filter(|shot| shot.phase == phase).collect();
            for pair in phase_shots.windows(2) {
                let a = s.direction_of(pair[0]);
                let b = s.direction_of(pair[1]);
                let separation_deg = a.dot(b).clamp(-1.0, 1.0).acos().to_degrees();
                assert!(
                    separation_deg <= 37.0,
                    "{phase:?} targets are {separation_deg:.1} degrees apart"
                );
            }
        }
    }

    #[test]
    fn heights_are_grouped_so_the_user_tilts_rather_than_circles() {
        // Three full turns to collect bottom, then middle, then top would be a
        // miserable way to shoot a room. Consecutive shots share a direction.
        let s = script();
        for (i, chunk) in s.shots.chunks(3).enumerate() {
            assert!(chunk.iter().all(|x| x.yaw_rad == chunk[0].yaw_rad));
            if i % 2 == 0 {
                assert_eq!(chunk[0].height, ShotHeight::Bottom);
                assert_eq!(chunk[2].height, ShotHeight::Top);
            } else {
                assert_eq!(chunk[0].height, ShotHeight::Top);
                assert_eq!(chunk[2].height, ShotHeight::Bottom);
            }
        }
    }

    #[test]
    fn yaw_is_measured_from_wherever_the_user_started() {
        // The room's orientation is unknown at the first frame, so the first
        // shot must be straight ahead rather than at some world axis.
        let s = script();
        let first = s.direction_of(s.current().unwrap());
        // First shot is Bottom, so it points ahead and down.
        let flat = Vec3::new(first.x, 0.0, first.z).normalized();
        assert!(flat.dot(facing_north()) > 0.99, "{flat:?}");
        assert!(first.y < 0.0, "the first shot should look down at the floor junction");
    }

    #[test]
    fn a_shot_is_taken_only_when_aimed_and_steady() {
        let mut s = script();
        let want = s.direction_of(s.current().unwrap());

        let elsewhere = Vec3::new(1.0, 0.0, 0.0);
        assert_eq!(s.evaluate(&aimed_at(&s, elsewhere, 0.1)), ShotOutcome::NotAimed);

        // Aimed but still swinging: this is precisely the frame that comes back
        // blurred, and blur was the largest rejection category on real captures.
        assert_eq!(s.evaluate(&aimed_at(&s, want, 1.2)), ShotOutcome::NotSteady);

        assert_eq!(s.evaluate(&aimed_at(&s, want, 0.05)), ShotOutcome::Take);
        s.commit(Vec3::ZERO);
        assert_eq!(s.progress().0, 1);
    }

    #[test]
    fn the_whole_centre_pass_can_be_shot() {
        let mut s = script();
        for _ in 0..36 {
            let want = s.direction_of(s.current().expect("a shot should remain"));
            assert_eq!(s.evaluate(&aimed_at(&s, want, 0.05)), ShotOutcome::Take);
            s.commit(Vec3::ZERO);
        }
        assert!(s.phase_finished(ShotPhase::Centre));
        assert_eq!(s.evaluate(&aimed_at(&s, facing_north(), 0.0)), ShotOutcome::Finished);
    }

    #[test]
    fn a_station_adds_fewer_shots_offset_from_the_centre_directions() {
        // The point of moving is the baseline, not a second inventory of the
        // same directions, so a station asks for less and looks elsewhere.
        let mut s = script();
        s.add_station();
        let (_, total) = s.progress();
        assert_eq!(total, 36 + 20);
        let station: Vec<_> = s.shots.iter().filter(|x| x.phase == ShotPhase::Station).collect();
        assert!(station.iter().all(|x| x.height != ShotHeight::Top), "top is the centre pass's job");
        for c in s.shots.iter().filter(|x| x.phase == ShotPhase::Centre) {
            for t in &station {
                assert!((c.yaw_rad - t.yaw_rad).abs() > 1e-3, "station repeats a centre direction");
            }
        }
    }

    #[test]
    fn station_shots_refuse_to_be_taken_without_moving() {
        // The entire reason for a second phase is the baseline. A user who
        // turns on the spot and satisfies every "station" shot has produced the
        // single-viewpoint bundle the harness refuses, and would be told the
        // capture was complete.
        let mut s = script();
        for _ in 0..36 {
            let want = s.direction_of(s.current().unwrap());
            assert_eq!(s.evaluate(&aimed_at(&s, want, 0.05)), ShotOutcome::Take);
            s.commit(Vec3::ZERO);
        }
        s.add_station();

        let want = s.direction_of(s.current().unwrap());
        let mut still_there = aimed_at(&s, want, 0.05);
        still_there.pose.t = Vec3::new(0.1, 0.0, 0.1); // shuffled, not moved
        assert_eq!(s.evaluate(&still_there), ShotOutcome::NotMoved);

        let mut moved = aimed_at(&s, want, 0.05);
        moved.pose.t = Vec3::new(1.2, 0.0, 0.0);
        assert_eq!(s.evaluate(&moved), ShotOutcome::Take);
    }

    #[test]
    fn skipping_gives_up_on_one_shot_and_offers_the_next() {
        // "I cannot get this from this angle" is about the shot in front of the
        // user, not the bearing: the floor junction below a wardrobe is usually
        // shootable even when the ceiling corner above it is not.
        let mut s = script();
        let first = *s.current().unwrap();
        let skipped = s.skip_current().expect("a shot was pending");

        assert_eq!(skipped.height, first.height);
        assert_eq!(skipped.state, ShotState::Skipped);

        let next = s.current().expect("the next height should be offered");
        assert_eq!(next.yaw_rad, first.yaw_rad, "same direction, next height");
        assert_ne!(next.height, first.height);
    }

    #[test]
    fn a_skipped_shot_is_not_counted_as_taken() {
        // The counter is the user's evidence that the room was shot. If a skip
        // advanced it, skipping every shot would read as a complete capture.
        let mut s = script();
        s.skip_current();
        assert_eq!(s.progress().0, 0, "skipping is not capturing");
        assert_eq!(s.skipped(), 1);
        assert_eq!(s.progress().1, 36, "the shot stays in the denominator");
    }

    #[test]
    fn a_capture_whose_last_shots_are_unreachable_can_still_finish() {
        // A room with a genuinely unreachable corner must be completable, or
        // the user abandons the whole capture rather than the one direction.
        let mut s = script();
        for _ in 0..35 {
            let want = s.direction_of(s.current().unwrap());
            s.evaluate(&aimed_at(&s, want, 0.05));
            s.commit(Vec3::ZERO);
        }
        assert!(!s.finished());
        s.skip_current();
        assert!(s.finished(), "a skipped shot is resolved, not still owed");
        assert_eq!(s.progress(), (35, 36));
        assert_eq!(s.skipped(), 1);
    }

    #[test]
    fn skipped_shots_are_reportable_for_the_bundle() {
        // The harness has to tell a direction nobody could reach from one the
        // script never asked for; that means the client has to write them down.
        let mut s = script();
        s.skip_current();
        s.skip_current();
        let abandoned: Vec<_> = s.skipped_shots().collect();
        assert_eq!(abandoned.len(), 2);
        assert!(abandoned.iter().all(|x| x.state == ShotState::Skipped));
    }

    #[test]
    fn skipping_with_nothing_pending_is_not_an_error() {
        // The button exists while the script is running; the script can finish
        // between the user deciding to press it and the press arriving.
        let mut s = script();
        for _ in 0..36 {
            s.commit(Vec3::ZERO);
        }
        assert!(s.finished());
        assert_eq!(s.skip_current().map(|x| x.state), None);
        assert_eq!(s.skipped(), 0);
    }

    #[test]
    fn tilting_the_phone_at_the_start_does_not_tilt_every_target() {
        // Someone starting with the phone angled down must not have the whole
        // script rotated under them.
        let level = CaptureScript::centre_pass(ScriptParams::default(), UP, facing_north());
        let tilted = CaptureScript::centre_pass(
            ScriptParams::default(), UP, Vec3::new(0.0, -0.6, -1.0).normalized(),
        );
        let a = level.direction_of(level.current().unwrap());
        let b = tilted.direction_of(tilted.current().unwrap());
        assert!(a.dot(b) > 0.999, "{a:?} vs {b:?}");
    }
}
