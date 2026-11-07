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

pub mod capture;
pub mod coverage;
pub mod ffi;
pub mod floorplan;
pub mod math;
pub mod quality;
pub mod script;
pub mod session;
pub mod station;

pub use coverage::{CoverageGrid, CoverageParams, Voxel, VoxelKey};
pub use floorplan::{
    floor_hit, FloorHit, FloorPlan, FloorPlanParams, Footprint, Mark, MarkKind, MarkOutcome,
    MarkQuality, MarkRejection, Point2,
};
pub use math::{look_rotation, yaw_rotation, Pose, Vec3};
pub use quality::{
    assess_completion, CompletionStatus, CompletionThresholds, FrameStats, KeyframeDecision, Nag,
    QualityGate, QualityParams, TrackingState,
};
pub use script::{CaptureScript, ScriptParams, Shot, ShotHeight, ShotOutcome, ShotPhase};
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
            pose: Pose {
                r: math::yaw_rotation(0.0),
                t: pos,
            },
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
        g.observe(
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(0.0, 0.0, -2.0),
            300.0,
        );
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
        assert_eq!(
            g.distance_weight(0.2),
            0.0,
            "arm's length is not a useful view"
        );
        assert!(g.distance_weight(0.5) > 0.0 && g.distance_weight(0.5) < 1.0);
        assert_eq!(
            g.distance_weight(0.95),
            1.0,
            "a normal ceiling must count fully"
        );
        assert_eq!(g.distance_weight(2.0), 1.0);
        assert!(g.distance_weight(3.5) < g.distance_weight(2.0));
        assert_eq!(
            g.distance_weight(6.0),
            0.0,
            "beyond d_opt + d_range is worthless"
        );
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
        assert!(
            g.is_covered(key),
            "q_min is meant to mean ~two decent views"
        );
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
        let s = next_best_station(
            &g,
            cam,
            0.0,
            Vec3::new(0.0, 1.0, 0.0),
            StationParams::default(),
        )
        .expect("a partially covered room must offer a station");
        let p = StationParams::default();
        assert!(s.position.x > p.wall_clearance_m - 0.05, "{:?}", s.position);
        assert!(
            s.position.x < 4.0 - p.wall_clearance_m + 0.05,
            "{:?}",
            s.position
        );
        assert!(s.position.z > p.wall_clearance_m - 0.05, "{:?}", s.position);
        assert!(
            s.position.z < 3.0 - p.wall_clearance_m + 0.05,
            "{:?}",
            s.position
        );
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
        let s = next_best_station(
            &g,
            near,
            0.0,
            Vec3::new(0.0, 1.0, 0.0),
            StationParams::default(),
        )
        .unwrap();
        let greedy = StationParams {
            lambda: 0.0,
            ..Default::default()
        };
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
        let s = next_best_station(
            &g,
            Vec3::new(2.0, 0.0, 1.5),
            0.0,
            Vec3::new(0.0, 1.0, 0.0),
            StationParams::default(),
        );
        assert!(s.is_none(), "well-covered room still suggested {s:?}");
    }

    // -- selection ---------------------------------------------------------

    /// Frames of the same view, arriving at 30 Hz, with varying sharpness.
    fn burst(blurs: &[f32]) -> Vec<FrameStats> {
        blurs
            .iter()
            .enumerate()
            .map(|(i, &b)| {
                let mut f = frame(i as f64 * 0.033, Vec3::new(0.0, 1.5, 0.0));
                f.blur_score = b;
                f
            })
            .collect()
    }

    #[test]
    fn the_sharpest_frame_of_a_view_is_the_one_kept() {
        // The whole point. Under the old rule each frame was judged against a
        // pass mark alone, so a run of slightly-soft frames yielded nothing
        // even though one of them was clearly the best of the bunch.
        let mut g = QualityGate::new(QualityParams::default());
        let frames = burst(&[120.0, 90.0, 460.0, 150.0]);
        let mut held_at = None;
        for (i, f) in frames.iter().enumerate() {
            if g.offer(f).hold {
                held_at = Some(i);
            }
        }
        assert_eq!(held_at, Some(2), "the 460 should be the one retained");
        assert_eq!(
            g.held_pose().map(|p| p.position()),
            Some(Vec3::new(0.0, 1.5, 0.0))
        );
    }

    #[test]
    fn a_soft_frame_is_kept_when_it_is_all_there_is() {
        // A wall photographed slightly soft is worth incomparably more than
        // the same wall not photographed at all, and the old gate threw it
        // away at blur 99.
        let mut g = QualityGate::new(QualityParams::default());
        let mut f = frame(0.0, Vec3::ZERO);
        f.blur_score = 60.0; // under the old min_blur_score of 100
        assert!(
            g.offer(&f).hold,
            "a usable-but-soft frame must still be held"
        );
        assert!(g.capture_now().is_some(), "and must be takeable");
        assert_eq!(g.accepted_count(), 1);
    }

    #[test]
    fn a_bright_window_does_not_cost_the_wall_it_is_in() {
        let mut g = QualityGate::new(QualityParams::default());
        let mut f = frame(0.0, Vec3::ZERO);
        f.exposure_clip_ratio = 0.35; // over the old max of 0.25
        assert!(g.offer(&f).hold);
    }

    #[test]
    fn even_a_hopeless_frame_can_be_taken() {
        // Nothing is excluded. Smeared, blown, dark, tracking down — if the
        // user presses the shutter they get the picture, and recovering it is
        // the reconstruction's problem. Refusing leaves nothing to recover.
        for mut bad in [
            frame(0.0, Vec3::ZERO),
            frame(0.0, Vec3::ZERO),
            frame(0.0, Vec3::ZERO),
        ] {
            bad.blur_score = 3.0;
            bad.exposure_clip_ratio = 0.95;
            bad.lux = 1.0;
            bad.tracking = TrackingState::Limited;
            let mut g = QualityGate::new(QualityParams::default());
            assert!(
                g.offer(&bad).hold,
                "a bad frame must still become the candidate"
            );
            assert!(g.capture_now().is_some(), "the shutter was refused");
        }
    }

    #[test]
    fn a_poor_frame_never_displaces_a_good_one() {
        // Permissive about what may be taken, still opinionated about which of
        // several frames is the one worth keeping.
        let mut g = QualityGate::new(QualityParams::default());
        let mut good = frame(0.0, Vec3::ZERO);
        good.blur_score = 400.0;
        g.offer(&good);

        let mut smeared = frame(0.03, Vec3::ZERO);
        smeared.blur_score = 4.0;
        g.offer(&smeared);

        let taken = g.capture_now().expect("a frame was ready");
        assert_eq!(
            taken.blur_score, 400.0,
            "the shutter took the smeared frame"
        );
    }

    #[test]
    fn nothing_is_written_without_the_shutter() {
        // The app takes no pictures on its own. A frame is only ever held.
        let mut g = QualityGate::new(QualityParams::default());
        for i in 0..90 {
            g.offer(&frame(
                i as f64 * 0.033,
                Vec3::new(i as f32 * 0.02, 0.0, 0.0),
            ));
        }
        assert_eq!(
            g.accepted_count(),
            0,
            "frames were taken without being asked for"
        );
        assert!(g.has_candidate(), "but one should be ready for the shutter");
    }

    #[test]
    fn a_stale_candidate_is_dropped_rather_than_written() {
        // Pressing the shutter must never save a picture of somewhere the user
        // stopped pointing a second ago.
        let mut g = QualityGate::new(QualityParams::default());
        let mut sharp = frame(0.0, Vec3::ZERO);
        sharp.blur_score = 900.0;
        assert!(g.offer(&sharp).hold);

        // Moved on: the old candidate is no longer what the camera sees.
        let mut elsewhere = frame(0.2, Vec3::new(0.5, 0.0, 0.0));
        elsewhere.blur_score = 120.0;
        assert!(
            g.offer(&elsewhere).hold,
            "the new view becomes the candidate"
        );
        let taken = g.capture_now().expect("a frame was ready");
        assert_eq!(
            taken.pose.position(),
            Vec3::new(0.5, 0.0, 0.0),
            "the shutter took the stale frame instead of the live one"
        );
    }

    #[test]
    fn the_shutter_takes_the_sharpest_frame_of_the_moment() {
        // What the selector is for, now that it no longer decides when: the
        // press lands on whichever recent frame was least smeared, so a small
        // wobble at the moment of tapping costs nothing.
        let mut g = QualityGate::new(QualityParams::default());
        for (i, b) in [140.0f32, 620.0, 180.0].iter().enumerate() {
            let mut f = frame(i as f64 * 0.033, Vec3::ZERO);
            f.blur_score = *b;
            g.offer(&f);
        }
        let taken = g.capture_now().expect("a frame was ready");
        assert_eq!(taken.blur_score, 620.0);
        assert_eq!(g.accepted_count(), 1);
    }

    #[test]
    fn the_shutter_is_never_refused_for_aim_or_steadiness() {
        // A deliberate press on a wobbly, badly aimed frame still takes a
        // picture. The user can see the scene; the app cannot overrule them.
        let mut g = QualityGate::new(QualityParams::default());
        let mut swinging = frame(0.0, Vec3::ZERO);
        swinging.angular_velocity_rad_s = 2.5;
        swinging.blur_score = 45.0;
        g.offer(&swinging);
        assert!(g.capture_now().is_some(), "the shutter was refused");
    }

    #[test]
    fn the_keyframe_cap_does_not_stop_the_shutter() {
        // The cap is a cost ceiling, and it used to silently refuse frames
        // past it. Against a finger on a button that is a control that stops
        // working with no explanation, so it counts rather than blocks.
        let params = QualityParams {
            max_keyframes: 2,
            ..Default::default()
        };
        let mut g = QualityGate::new(params);
        for i in 0..10 {
            g.offer(&frame(i as f64 * 0.5, Vec3::new(i as f32, 0.0, 0.0)));
            assert!(g.capture_now().is_some(), "shutter refused at frame {i}");
        }
        assert_eq!(g.accepted_count(), 10);
    }

    #[test]
    fn scoring_prefers_the_steadier_of_two_equally_sharp_frames() {
        let g = QualityGate::new(QualityParams::default());
        let mut swinging = frame(0.0, Vec3::ZERO);
        swinging.angular_velocity_rad_s = 0.9;
        let mut still = frame(0.0, Vec3::ZERO);
        still.angular_velocity_rad_s = 0.05;
        assert!(g.score(&still) > g.score(&swinging));
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
        assert!(q
            .consider(&frame(0.2, Vec3::new(0.35, 0.0, 0.0)))
            .accepted());
    }

    #[test]
    fn rotation_alone_triggers_a_keyframe() {
        let mut q = QualityGate::new(QualityParams::default());
        let mut a = frame(0.0, Vec3::ZERO);
        assert!(q.consider(&a).accepted());
        a.timestamp_s = 0.5;
        a.pose.r = math::yaw_rotation(20.0_f32.to_radians());
        assert!(
            q.consider(&a).accepted(),
            "20 deg of yaw should pass the 15 deg gate"
        );
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
    fn weak_tracking_signals_pause_scripted_and_freeform_capture() {
        let q = QualityGate::new(QualityParams::default());
        for mut weak in [
            frame(0.0, Vec3::ZERO),
            frame(0.1, Vec3::ZERO),
            frame(0.2, Vec3::ZERO),
        ] {
            weak.sparse_point_count = 0;
            assert_eq!(q.evaluate(&weak), KeyframeDecision::RejectTracking);
            assert_eq!(q.evaluate_shot(&weak), KeyframeDecision::RejectTracking);
        }

        let mut moving = frame(1.0, Vec3::ZERO);
        moving.angular_velocity_rad_s = 2.0;
        assert_eq!(q.evaluate_shot(&moving), KeyframeDecision::RejectTracking);

        let mut dark = frame(2.0, Vec3::ZERO);
        dark.lux = 1.0;
        assert_eq!(q.evaluate_shot(&dark), KeyframeDecision::RejectTracking);
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
        assert!(
            q.consider(&small).accepted(),
            "should measure from the origin keyframe"
        );
    }

    #[test]
    fn keyframe_cap_is_enforced() {
        let p = QualityParams {
            max_keyframes: 3,
            ..Default::default()
        };
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
                s.capture_now();
                t += 3.0;
                n += 1;
            }
        }
        let u = last.unwrap();
        assert!(u.status.floor_coverage >= 0.95, "{:?}", u.status);
        assert!(u.status.wall_coverage >= 0.90, "{:?}", u.status);
        assert!(u.status.ceiling_coverage >= 0.70, "{:?}", u.status);
        assert!(
            u.station.is_none(),
            "a covered room should stop suggesting stations"
        );

        // Full coverage bars are no longer enough on their own. Nine stations
        // is a thorough sweep of a small room and it still has too few frames
        // to reconstruct from, which is the whole point of moving completion
        // off coverage: the grid cannot tell how much evidence produced it.
        assert!(!u.status.complete, "{:?}", u.status);
        assert_eq!(
            u.status.blocker,
            Some(crate::quality::Incomplete::TooFewKeyframes)
        );
        assert!(u.status.max_baseline_m >= 2.0, "{:?}", u.status);
        assert!(u.status.median_step_m >= 0.25, "{:?}", u.status);
        assert_eq!(u.status.unresolved_sectors, 0, "{:?}", u.status);
    }

    #[test]
    fn a_thorough_capture_completes() {
        // The same room worked properly: enough viewpoints, walked rather than
        // pivoted, and every direction faced from somewhere.
        let mut s = CaptureSession::new(SessionConfig::default(), Vec3::new(0.0, -1.0, 0.0));
        s.set_floor_height(0.0);
        // A room big enough that twenty viewpoints still each show something
        // new: in a small one the redundancy rule (plan §7) correctly stops the
        // capture short of the keyframe minimum, which is a real tension
        // between the two and not something to paper over in a test.
        let (pts, nrm) = room_samples(7.0, 5.0, 2.5, 0.1);

        let mut t = 0.0f64;
        let mut last = None;
        let mut n = 0usize;
        for i in 0..8 {
            for j in 0..8 {
                let x = 0.7 + (i as f32) * 0.80;
                let z = 0.7 + (j as f32) * 0.50;
                // Two frames per viewpoint: the selector needs a window to pick
                // from, and a real capture is a video, not a slideshow.
                for k in 0..2 {
                    let mut f = frame(t, Vec3::new(x, 1.5, z));
                    f.pose.r = math::yaw_rotation((n as f32) * 0.55);
                    f.blur_score = 200.0 + (k as f32) * 100.0;
                    last = Some(s.observe_frame(&f, &pts, &nrm));
                    t += 0.05;
                }
                s.capture_now();
                t += 1.5;
                n += 1;
            }
        }
        let u = last.unwrap();
        assert!(u.status.keyframes >= 20, "{:?}", u.status);
        assert!(u.status.complete, "{:?}", u.status);
        assert_eq!(u.status.blocker, None);
    }

    // -- adaptive selection (plan §7) --------------------------------------

    #[test]
    fn a_frame_showing_nothing_new_is_rejected_as_redundant() {
        let mut s = CaptureSession::new(SessionConfig::default(), Vec3::new(0.0, -1.0, 0.0));
        s.set_floor_height(0.0);
        let (pts, nrm) = room_samples(4.0, 3.0, 2.5, 0.1);

        // Cover the room properly first — the same nine stations that reach
        // room-complete, since "redundant" only means anything once there is
        // something to be redundant *with*.
        let mut t = 0.0;
        let mut n = 0usize;
        for i in 0..3 {
            for j in 0..3 {
                let mut f = frame(
                    t,
                    Vec3::new(0.8 + (i as f32) * 1.2, 1.5, 0.8 + (j as f32) * 0.7),
                );
                f.pose.r = math::yaw_rotation((n as f32) * 0.7);
                s.observe_frame(&f, &pts, &nrm);
                s.capture_now();
                t += 3.0;
                n += 1;
            }
        }

        // Another look at the same room from the middle of it: sharp, tracking
        // fine, and showing nothing new.
        let f = frame(t, Vec3::new(2.0, 1.5, 1.5));
        let u = s.observe_frame(&f, &pts, &nrm);
        assert_eq!(
            u.decision,
            KeyframeDecision::RejectRedundant,
            "a covered room should still be reported as redundant"
        );

        // ...but reporting is all it does. The user pressing the shutter on a
        // covered room still gets a picture: a camera that ignores a
        // deliberate press is broken, whatever the coverage grid thinks.
        let before = u.keyframes;
        assert!(
            s.capture_now().is_some(),
            "the shutter was refused as redundant"
        );
        let after = s.observe_frame(&frame(t + 0.1, Vec3::new(2.0, 1.5, 1.5)), &pts, &nrm);
        assert_eq!(after.keyframes, before + 1);
    }

    #[test]
    fn the_first_frame_is_never_redundant() {
        let mut s = CaptureSession::new(SessionConfig::default(), Vec3::new(0.0, -1.0, 0.0));
        let (pts, nrm) = room_samples(4.0, 3.0, 2.5, 0.1);
        let u = s.observe_frame(&frame(0.0, Vec3::new(2.0, 1.5, 1.5)), &pts, &nrm);
        assert_eq!(
            u.decision,
            KeyframeDecision::Accept,
            "an empty grid has nothing covered"
        );
    }

    #[test]
    fn a_device_without_depth_still_captures() {
        // The whole test is the abstention: with no geometry the redundancy
        // rule cannot judge, and must not reject. A device with no depth
        // support would otherwise record nothing at all.
        let mut s = CaptureSession::new(SessionConfig::default(), Vec3::new(0.0, -1.0, 0.0));
        let mut t = 0.0;
        for i in 0..5 {
            let f = frame(t, Vec3::new(1.0 + (i as f32) * 0.5, 1.5, 1.5));
            let u = s.observe_frame(&f, &[], &[]);
            assert_eq!(
                u.decision,
                KeyframeDecision::Accept,
                "frame {i} was dropped"
            );
            t += 3.0;
        }
    }

    #[test]
    fn a_frame_revealing_new_surface_survives_a_covered_room() {
        // The rule must not become "stop once mostly done". A frame pointed at
        // the part that is still missing has to get through, or the capture can
        // never finish the thing it is being sent back for.
        let mut s = CaptureSession::new(SessionConfig::default(), Vec3::new(0.0, -1.0, 0.0));
        s.set_floor_height(0.0);
        let (pts, nrm) = room_samples(4.0, 3.0, 2.5, 0.1);
        let mut t = 0.0;
        for i in 0..6 {
            let mut f = frame(t, Vec3::new(1.0 + (i as f32) * 0.5, 1.5, 1.5));
            f.pose.r = math::yaw_rotation((i as f32) * 0.9);
            s.observe_frame(&f, &pts, &nrm);
            t += 3.0;
        }

        // A surface the capture has never seen — an alcove off the far wall.
        let mut new_pts = Vec::new();
        let mut new_nrm = Vec::new();
        let mut x = 5.0f32;
        while x <= 6.0 {
            let mut y = 0.0f32;
            while y <= 2.0 {
                new_pts.push(Vec3::new(x, y, 3.5));
                new_nrm.push(Vec3::new(0.0, 0.0, -1.0));
                y += 0.05;
            }
            x += 0.05;
        }
        let f = frame(t, Vec3::new(5.5, 1.5, 1.8));
        let u = s.observe_frame(&f, &new_pts, &new_nrm);
        assert_eq!(
            u.decision,
            KeyframeDecision::Accept,
            "a frame showing unseen surface must be kept"
        );
    }

    #[test]
    fn a_scripted_capture_shoots_the_centre_pass_then_asks_for_parallax() {
        use crate::script::{ScriptParams, ShotPhase};

        let mut s = CaptureSession::new(SessionConfig::default(), Vec3::new(0.0, -1.0, 0.0));
        s.set_floor_height(0.0);
        s.start_script(ScriptParams::default());
        let (pts, nrm) = room_samples(4.0, 3.0, 2.5, 0.1);

        // A frame pointing somewhere the script did not ask for is held, not
        // taken — the whole point of a script is that it chooses the view.
        let mut t = 0.0;
        let mut f = frame(t, Vec3::new(2.0, 1.5, 1.5));
        f.pose.r = math::look_rotation(2.5, 0.0);
        let u = s.observe_frame(&f, &pts, &nrm);
        assert_eq!(u.decision, KeyframeDecision::RejectOffTarget);
        assert_eq!(
            u.shot_progress,
            (0, 36),
            "twelve overlapping directions x three heights"
        );

        // Shoot the pass by aiming where each shot asks. A shot is now
        // satisfied by holding roughly on target for a moment rather than by
        // one frame arriving perfect, so each is fed a short burst — which is
        // what pointing a phone at something actually produces.
        for _ in 0..36 {
            let want = u_direction(&s);
            let before = s.script().unwrap().progress().0;
            t += 0.05;
            let mut f = frame(t, Vec3::new(2.0, 1.5, 1.5));
            f.pose.r = look_along(want, Vec3::new(0.0, 1.0, 0.0));
            f.angular_velocity_rad_s = 0.05;
            s.observe_frame(&f, &pts, &nrm);
            // The user lines up the marker and presses the shutter; the script
            // advances because a picture was taken, not because a gate passed.
            s.capture_now()
                .expect("the shutter should take the held frame");
            assert!(
                s.script().unwrap().progress().0 > before,
                "pressing the shutter must advance the script"
            );
        }

        // One more frame, as the camera keeps delivering after the last shot:
        // the script appends its station phase while handling a frame.
        t += 0.05;
        let mut f = frame(t, Vec3::new(2.0, 1.5, 1.5));
        f.pose.r = look_along(Vec3::new(0.0, 0.0, -1.0), Vec3::new(0.0, 1.0, 0.0));
        s.observe_frame(&f, &pts, &nrm);

        let script = s.script().expect("script running");
        assert!(script.phase_finished(ShotPhase::Centre));
        // Standing still gives no baseline, so the script must now send the
        // user somewhere else rather than declare the room captured.
        assert!(
            script.has_phase(ShotPhase::Station),
            "after the centre pass the script must ask for parallax"
        );
    }

    fn u_direction(s: &CaptureSession) -> Vec3 {
        let script = s.script().expect("script running");
        script.direction_of(script.current().expect("a shot remains"))
    }

    /// A world<-camera rotation whose forward (-Z, plan §5) points along `dir`.
    ///
    /// Built from the axes rather than from yaw and pitch: deriving angles here
    /// would mean re-implementing the script's basis in the test, and a test
    /// that repeats the code it checks proves only that it is self-consistent.
    fn look_along(dir: Vec3, up: Vec3) -> [f32; 9] {
        let f = dir.normalized();
        let z = f * -1.0; // camera +Z points backwards
        let x = up.cross(z).normalized();
        let y = z.cross(x);
        // Row-major, columns are the camera axes in world coordinates.
        [x.x, y.x, z.x, x.y, y.y, z.y, x.z, y.z, z.z]
    }

    #[test]
    fn a_one_sided_capture_is_not_complete_and_gets_a_station() {
        let mut s = CaptureSession::new(SessionConfig::default(), Vec3::new(0.0, -1.0, 0.0));
        s.set_floor_height(0.0);
        let (pts, nrm) = room_samples(6.0, 4.0, 2.5, 0.12);
        // Geometry reaches the grid when the picture is taken, not when it is
        // seen — so the shutter has to be pressed before there is coverage.
        s.observe_frame(&frame(0.0, Vec3::new(0.7, 1.5, 0.7)), &pts, &nrm);
        s.capture_now();
        let u = s.observe_frame(&frame(2.5, Vec3::new(1.3, 1.5, 0.7)), &pts, &nrm);
        assert!(!u.status.complete);
        assert!(
            u.station.is_some(),
            "an incomplete room must suggest somewhere to stand"
        );
    }

    #[test]
    fn relocalisation_discards_the_frame_held_across_it() {
        // ARCore relocalises by moving the world origin. A candidate captured
        // before the rebase carries a pose in a frame that no longer exists,
        // and committing it afterwards would put its geometry where the camera
        // never was — a worse outcome than losing one picture.
        let mut s = CaptureSession::new(SessionConfig::default(), Vec3::new(0.0, -1.0, 0.0));
        let (pts, nrm) = room_samples(4.0, 3.0, 2.5, 0.3);
        s.observe_frame(&frame(0.0, Vec3::new(2.0, 1.5, 1.5)), &pts, &nrm);

        s.on_relocalised();

        // The pre-rebase candidate is gone, so the shutter has nothing to take.
        assert!(
            s.capture_now().is_none(),
            "a stale candidate must not be written"
        );
        assert!(s.last_known_good_position().is_none());

        // And the capture carries on normally from the next frame.
        s.observe_frame(&frame(3.0, Vec3::new(3.2, 1.5, 1.5)), &pts, &nrm);
        assert!(
            s.capture_now().is_some(),
            "capture resumes after the rebase"
        );
    }

    /// Drive a session far enough to have a remembered good frame.
    fn session_with_a_good_frame(heading_deg: f32) -> CaptureSession {
        let mut s = CaptureSession::new(SessionConfig::default(), Vec3::new(0.0, -1.0, 0.0));
        let (pts, nrm) = room_samples(4.0, 3.0, 2.5, 0.3);
        s.set_heading(heading_deg);
        s.observe_frame(&frame(0.0, Vec3::new(2.0, 1.5, 1.5)), &pts, &nrm);
        s.capture_now()
            .expect("the shutter should take the held frame");
        assert!(
            s.last_known_good_position().is_some(),
            "a keyframe should have been written"
        );
        s
    }

    #[test]
    fn recovery_points_back_to_the_last_good_frame() {
        let mut s = session_with_a_good_frame(90.0);
        s.set_heading(60.0);
        let r = s.recovery().expect("a good frame was recorded");
        assert_eq!(r.position, Vec3::new(2.0, 1.5, 1.5));
        // Was facing 90, now facing 60: turn 30 clockwise to get back.
        assert!(
            (r.turn_deg.unwrap() - 30.0).abs() < 1e-3,
            "{:?}",
            r.turn_deg
        );
    }

    #[test]
    fn recovery_turns_the_short_way_round_north() {
        // The bug this exists to prevent: 350 -> 10 is a 20 degree turn right,
        // not a 340 degree turn left, and a user told the latter walks away
        // from the room.
        let mut s = session_with_a_good_frame(10.0);
        s.set_heading(350.0);
        assert!((s.recovery().unwrap().turn_deg.unwrap() - 20.0).abs() < 1e-3);

        let mut s = session_with_a_good_frame(350.0);
        s.set_heading(10.0);
        assert!((s.recovery().unwrap().turn_deg.unwrap() + 20.0).abs() < 1e-3);
    }

    #[test]
    fn recovery_never_asks_for_more_than_a_half_turn() {
        for then in [0.0, 45.0, 120.0, 200.0, 275.0, 359.0] {
            for now in [0.0, 30.0, 90.0, 181.0, 300.0, 359.9] {
                let mut s = session_with_a_good_frame(then);
                s.set_heading(now);
                let t = s.recovery().unwrap().turn_deg.unwrap();
                assert!(
                    (-180.0..=180.0).contains(&t),
                    "then {then} now {now} -> {t}"
                );
            }
        }
    }

    #[test]
    fn recovery_offers_a_position_even_without_a_compass() {
        // A device with no usable magnetometer must still be told where to
        // stand; only the bearing is unavailable.
        let mut s = CaptureSession::new(SessionConfig::default(), Vec3::new(0.0, -1.0, 0.0));
        let (pts, nrm) = room_samples(4.0, 3.0, 2.5, 0.3);
        s.observe_frame(&frame(0.0, Vec3::new(2.0, 1.5, 1.5)), &pts, &nrm);
        s.capture_now()
            .expect("the shutter should take the held frame");
        let r = s.recovery().expect("position is known without a heading");
        assert_eq!(r.turn_deg, None);
        assert_eq!(r.position, Vec3::new(2.0, 1.5, 1.5));
    }

    #[test]
    fn there_is_nothing_to_recover_to_before_the_first_keyframe() {
        let s = CaptureSession::new(SessionConfig::default(), Vec3::new(0.0, -1.0, 0.0));
        assert!(s.recovery().is_none());
    }

    #[test]
    fn gravity_defines_up_regardless_of_sign_convention() {
        let s = CaptureSession::new(SessionConfig::default(), Vec3::new(0.0, -9.81, 0.0));
        assert!((s.up().y - 1.0).abs() < 1e-6);
    }
}
