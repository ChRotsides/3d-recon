//! Shared capture-guidance core (plan §3.8, §6.1).
//!
//! This is where the interesting math lives: the coverage voxel grid, the
//! next-best-station solver and the quality gates. Keeping it in one Rust crate
//! behind UniFFI means iOS and Android cannot drift apart in capture behaviour,
//! it stays testable on a desktop, and it runs unmodified on a Jetson for
//! ground-truth work.
//!
//! Everything here is in the canonical frame from plan §5: right-handed, Y-up,
//! world<-camera poses, camera looking down -Z. **ARCore's frame is converted
//! at the client boundary, not here.**

// UniFFI scaffolding must live at the crate root.
uniffi::setup_scaffolding!();

pub mod coverage;
pub mod ffi;
pub mod math;
pub mod quality;
pub mod session;
pub mod station;

pub use coverage::{CoverageGrid, CoverageParams, Voxel, VoxelKey};
pub use math::{look_rotation, yaw_rotation, Pose, Vec3};
pub use quality::{
    assess_completion, CompletionStatus, CompletionThresholds, FrameStats, KeyframeDecision,
    Nag, QualityGate, QualityParams, TrackingState,
};
pub use session::{CaptureSession, GuidanceUpdate, SessionConfig};
pub use station::{next_best_station, sample_floor_candidates, Station, StationParams};

#[cfg(test)]
mod tests {
    use super::*;

    /// A closed box room, sampled as oriented surface points. Stands in for the
    /// on-device depth pass so the guidance logic can be tested without one.
    fn room_samples(w: f32, d: f32, h: f32, step: f32) -> (Vec<Vec3>, Vec<Vec3>) {
        let mut pts = Vec::new();
        let mut nrm = Vec::new();
        let mut x = 0.0;
        while x <= w {
            let mut z = 0.0;
            while z <= d {
                pts.push(Vec3::new(x, 0.0, z));
                nrm.push(Vec3::new(0.0, 1.0, 0.0));
                pts.push(Vec3::new(x, h, z));
                nrm.push(Vec3::new(0.0, -1.0, 0.0));
                z += step;
            }
            x += step;
        }
        let mut y = 0.0;
        while y <= h {
            let mut x = 0.0;
            while x <= w {
                pts.push(Vec3::new(x, y, 0.0));
                nrm.push(Vec3::new(0.0, 0.0, 1.0));
                pts.push(Vec3::new(x, y, d));
                nrm.push(Vec3::new(0.0, 0.0, -1.0));
                x += step;
            }
            let mut z = 0.0;
            while z <= d {
                pts.push(Vec3::new(0.0, y, z));
                nrm.push(Vec3::new(1.0, 0.0, 0.0));
                pts.push(Vec3::new(w, y, z));
                nrm.push(Vec3::new(-1.0, 0.0, 0.0));
                z += step;
            }
            y += step;
        }
        (pts, nrm)
    }

    fn frame(t: f64, pos: Vec3) -> FrameStats {
        FrameStats {
            timestamp_s: t,
            pose: Pose { r: math::yaw_rotation(0.0), t: pos },
            tracking: TrackingState::Normal,
            blur_score: 300.0,
            exposure_clip_ratio: 0.0,
            median_depth_m: 2.0,
            angular_velocity_rad_s: 0.2,
            sparse_point_count: 200,
            lux: 200.0,
        }
    }

    // -- coverage ---------------------------------------------------------

    #[test]
    fn back_facing_surfaces_are_not_credited() {
        let mut g = CoverageGrid::new(CoverageParams::default());
        // Camera behind the surface: normal points away from it.
        g.observe(Vec3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0),
                  Vec3::new(0.0, 0.0, -2.0), 300.0);
        assert!(g.is_empty(), "a back-facing observation must not count");
    }

    #[test]
    fn oblique_views_are_worth_less_than_head_on() {
        let p = Vec3::ZERO;
        let n = Vec3::new(0.0, 0.0, 1.0);
        let mut head_on = CoverageGrid::new(CoverageParams::default());
        head_on.observe(p, n, Vec3::new(0.0, 0.0, 2.0), 300.0);
        let mut oblique = CoverageGrid::new(CoverageParams::default());
        oblique.observe(p, n, Vec3::new(1.9, 0.0, 0.6), 300.0);
        let a = head_on.all_voxels()[0].1.quality;
        let b = oblique.all_voxels()[0].1.quality;
        assert!(a > b, "head-on {a} should beat oblique {b}");
    }

    #[test]
    fn distance_weight_is_flat_up_to_d_opt_then_falls_away() {
        // The near side is deliberately *not* penalised: a 2.45 m ceiling seen
        // from 1.5 m is 0.95 m away, and a symmetric weight would make ceilings
        // permanently uncoverable. See CoverageGrid::distance_weight.
        let g = CoverageGrid::new(CoverageParams::default());
        assert_eq!(g.distance_weight(0.2), 0.0, "arm's length is not a useful view");
        assert!(g.distance_weight(0.5) > 0.0 && g.distance_weight(0.5) < 1.0);
        assert_eq!(g.distance_weight(0.95), 1.0, "a normal ceiling must count fully");
        assert_eq!(g.distance_weight(2.0), 1.0);
        assert!(g.distance_weight(3.5) < g.distance_weight(2.0));
        assert_eq!(g.distance_weight(6.0), 0.0, "beyond d_opt + d_range is worthless");
    }

    #[test]
    fn blurry_frames_contribute_less() {
        let p = Vec3::ZERO;
        let n = Vec3::new(0.0, 0.0, 1.0);
        let cam = Vec3::new(0.0, 0.0, 2.0);
        let mut sharp = CoverageGrid::new(CoverageParams::default());
        sharp.observe(p, n, cam, 400.0);
        let mut blurry = CoverageGrid::new(CoverageParams::default());
        blurry.observe(p, n, cam, 20.0);
        assert!(sharp.all_voxels()[0].1.quality > blurry.all_voxels()[0].1.quality);
    }

    #[test]
    fn two_good_views_reach_the_coverage_bar() {
        let p = Vec3::ZERO;
        let n = Vec3::new(0.0, 0.0, 1.0);
        let mut g = CoverageGrid::new(CoverageParams::default());
        for dx in [-0.2f32, 0.2] {
            g.observe(p, n, Vec3::new(dx, 0.0, 2.0), 400.0);
        }
        let (key, _) = g.all_voxels()[0];
        assert!(g.is_covered(key), "q_min is meant to mean ~two decent views");
    }

    // -- station solver ---------------------------------------------------

    #[test]
    fn station_is_inside_the_room_and_clear_of_walls() {
        let mut g = CoverageGrid::new(CoverageParams::default());
        let (pts, nrm) = room_samples(4.0, 3.0, 2.5, 0.1);
        // Observe from one corner only, leaving the far side uncovered.
        let cam = Vec3::new(0.6, 1.5, 0.6);
        for (p, n) in pts.iter().zip(nrm.iter()) {
            g.observe(*p, *n, cam, 300.0);
        }
        let s = next_best_station(&g, cam, 0.0, Vec3::new(0.0, 1.0, 0.0),
                                 StationParams::default())
            .expect("a partially covered room must offer a station");
        let p = StationParams::default();
        assert!(s.position.x > p.wall_clearance_m - 0.05, "{:?}", s.position);
        assert!(s.position.x < 4.0 - p.wall_clearance_m + 0.05, "{:?}", s.position);
        assert!(s.position.z > p.wall_clearance_m - 0.05, "{:?}", s.position);
        assert!(s.position.z < 3.0 - p.wall_clearance_m + 0.05, "{:?}", s.position);
    }

    #[test]
    fn walking_penalty_prefers_the_nearer_of_two_equal_stations() {
        let mut g = CoverageGrid::new(CoverageParams::default());
        let (pts, nrm) = room_samples(6.0, 3.0, 2.5, 0.15);
        let cam = Vec3::new(3.0, 1.5, 1.5);
        for (p, n) in pts.iter().zip(nrm.iter()) {
            g.observe(*p, *n, cam, 60.0); // low quality everywhere -> lots of gain
        }
        let near = Vec3::new(1.5, 0.0, 1.5);
        let s = next_best_station(&g, near, 0.0, Vec3::new(0.0, 1.0, 0.0),
                                  StationParams::default())
            .unwrap();
        let mut greedy = StationParams::default();
        greedy.lambda = 0.0;
        let s0 = next_best_station(&g, near, 0.0, Vec3::new(0.0, 1.0, 0.0), greedy).unwrap();
        assert!(
            s.walk_distance_m <= s0.walk_distance_m + 1e-3,
            "lambda>0 walked {} vs greedy {}",
            s.walk_distance_m,
            s0.walk_distance_m
        );
    }

    #[test]
    fn a_fully_covered_room_offers_no_station() {
        let mut g = CoverageGrid::new(CoverageParams::default());
        let (pts, nrm) = room_samples(4.0, 3.0, 2.5, 0.1);
        for cam in [
            Vec3::new(1.2, 1.5, 1.0),
            Vec3::new(2.8, 1.5, 1.0),
            Vec3::new(1.2, 1.5, 2.0),
            Vec3::new(2.8, 1.5, 2.0),
            Vec3::new(2.0, 1.5, 1.5),
        ] {
            for (p, n) in pts.iter().zip(nrm.iter()) {
                g.observe(*p, *n, cam, 400.0);
            }
        }
        let s = next_best_station(&g, Vec3::new(2.0, 0.0, 1.5), 0.0,
                                  Vec3::new(0.0, 1.0, 0.0), StationParams::default());
        assert!(s.is_none(), "well-covered room still suggested {s:?}");
    }

    // -- quality gates ----------------------------------------------------

    #[test]
    fn keyframes_need_motion() {
        let mut q = QualityGate::new(QualityParams::default());
        assert!(q.consider(&frame(0.0, Vec3::ZERO)).accepted());
        assert_eq!(
            q.consider(&frame(0.1, Vec3::new(0.05, 0.0, 0.0))),
            KeyframeDecision::RejectTooSimilar
        );
        assert!(q.consider(&frame(0.2, Vec3::new(0.35, 0.0, 0.0))).accepted());
    }

    #[test]
    fn rotation_alone_triggers_a_keyframe() {
        let mut q = QualityGate::new(QualityParams::default());
        let mut a = frame(0.0, Vec3::ZERO);
        assert!(q.consider(&a).accepted());
        a.timestamp_s = 0.5;
        a.pose.r = math::yaw_rotation(20.0_f32.to_radians());
        assert!(q.consider(&a).accepted(), "20 deg of yaw should pass the 15 deg gate");
    }

    #[test]
    fn blurry_frames_never_become_keyframes() {
        let mut q = QualityGate::new(QualityParams::default());
        let mut f = frame(0.0, Vec3::ZERO);
        f.blur_score = 10.0;
        assert_eq!(q.consider(&f), KeyframeDecision::RejectBlurry);
        assert_eq!(q.accepted_count(), 0);
    }

    #[test]
    fn a_blurry_frame_does_not_reset_the_baseline() {
        // Regression guard: if a rejected frame updated `last_keyframe`, the
        // next sharp frame would be measured from the wrong place.
        let mut q = QualityGate::new(QualityParams::default());
        assert!(q.consider(&frame(0.0, Vec3::ZERO)).accepted());
        let mut blurry = frame(0.1, Vec3::new(1.0, 0.0, 0.0));
        blurry.blur_score = 5.0;
        assert!(!q.consider(&blurry).accepted());
        let mut small = frame(0.2, Vec3::new(1.05, 0.0, 0.0));
        small.blur_score = 300.0;
        assert!(q.consider(&small).accepted(), "should measure from the origin keyframe");
    }

    #[test]
    fn keyframe_cap_is_enforced() {
        let mut p = QualityParams::default();
        p.max_keyframes = 3;
        let mut q = QualityGate::new(p);
        for i in 0..10 {
            q.consider(&frame(i as f64, Vec3::new(i as f32, 0.0, 0.0)));
        }
        assert_eq!(q.accepted_count(), 3);
    }

    #[test]
    fn tracking_loss_beats_every_other_nag() {
        let mut q = QualityGate::new(QualityParams::default());
        let mut f = frame(0.0, Vec3::ZERO);
        f.tracking = TrackingState::Relocalizing;
        f.median_depth_m = 0.3;
        f.angular_velocity_rad_s = 5.0;
        assert_eq!(q.nag(&f, 1.0), Some(Nag::TrackingLost));
    }

    #[test]
    fn the_same_nag_respects_its_cooldown() {
        let mut q = QualityGate::new(QualityParams::default());
        let mut f = frame(0.0, Vec3::ZERO);
        f.median_depth_m = 0.3;
        assert_eq!(q.nag(&f, 1.0), Some(Nag::TooClose));
        f.timestamp_s = 1.0;
        assert_eq!(q.nag(&f, 1.0), None, "3 s cooldown not honoured");
        f.timestamp_s = 4.0;
        assert_eq!(q.nag(&f, 1.0), Some(Nag::TooClose));
    }

    #[test]
    fn a_more_urgent_nag_interrupts_the_cooldown() {
        let mut q = QualityGate::new(QualityParams::default());
        let mut f = frame(0.0, Vec3::ZERO);
        f.median_depth_m = 0.3;
        assert_eq!(q.nag(&f, 1.0), Some(Nag::TooClose));
        f.timestamp_s = 0.5;
        f.tracking = TrackingState::Limited;
        assert_eq!(q.nag(&f, 1.0), Some(Nag::TrackingLost));
    }

    // -- session ----------------------------------------------------------

    #[test]
    fn a_capture_converges_to_complete() {
        // Nine standing positions on a ~1.2 m grid — what the station solver
        // walks a user through in a 4x3 m room. Four corner positions is *not*
        // a complete capture and the metric correctly says so: from 1.5 m, a
        // 2.5 m ceiling is only well seen from more or less underneath it.
        let mut s = CaptureSession::new(SessionConfig::default(), Vec3::new(0.0, -1.0, 0.0));
        s.set_floor_height(0.0);
        let (pts, nrm) = room_samples(4.0, 3.0, 2.5, 0.1);

        let mut t = 0.0f64;
        let mut last = None;
        let mut n = 0usize;
        for i in 0..3 {
            for j in 0..3 {
                let x = 0.8 + (i as f32) * 1.2;
                let z = 0.8 + (j as f32) * 0.7;
                let mut f = frame(t, Vec3::new(x, 1.5, z));
                f.pose.r = math::yaw_rotation((n as f32) * 0.7);
                last = Some(s.observe_frame(&f, &pts, &nrm));
                t += 3.0;
                n += 1;
            }
        }
        let u = last.unwrap();
        assert_eq!(u.keyframes, 9);
        assert!(u.status.floor_coverage >= 0.95, "{:?}", u.status);
        assert!(u.status.wall_coverage >= 0.90, "{:?}", u.status);
        assert!(u.status.ceiling_coverage >= 0.70, "{:?}", u.status);
        assert!(u.station.is_none(), "a covered room should stop suggesting stations");
        assert!(u.status.complete, "{:?}", u.status);
    }

    #[test]
    fn a_one_sided_capture_is_not_complete_and_gets_a_station() {
        let mut s = CaptureSession::new(SessionConfig::default(), Vec3::new(0.0, -1.0, 0.0));
        s.set_floor_height(0.0);
        let (pts, nrm) = room_samples(6.0, 4.0, 2.5, 0.12);
        let f = frame(0.0, Vec3::new(0.7, 1.5, 0.7));
        let u = s.observe_frame(&f, &pts, &nrm);
        assert!(!u.status.complete);
        assert!(u.station.is_some(), "an incomplete room must suggest somewhere to stand");
    }

    #[test]
    fn relocalisation_clears_the_keyframe_baseline() {
        let mut s = CaptureSession::new(SessionConfig::default(), Vec3::new(0.0, -1.0, 0.0));
        let (pts, nrm) = room_samples(4.0, 3.0, 2.5, 0.3);
        let f = frame(0.0, Vec3::new(2.0, 1.5, 1.5));
        assert!(s.observe_frame(&f, &pts, &nrm).decision.accepted());
        let f2 = frame(0.1, Vec3::new(2.01, 1.5, 1.5));
        assert!(!s.observe_frame(&f2, &pts, &nrm).decision.accepted());
        s.on_relocalised();
        assert!(s.observe_frame(&f2, &pts, &nrm).decision.accepted());
    }

    #[test]
    fn gravity_defines_up_regardless_of_sign_convention() {
        let s = CaptureSession::new(SessionConfig::default(), Vec3::new(0.0, -9.81, 0.0));
        assert!((s.up().y - 1.0).abs() < 1e-6);
    }
}
