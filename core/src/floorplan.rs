//! Floor plan by pointing.
//!
//! The user walks the room, aims the reticle at each place where two walls meet
//! the floor, and presses the shutter. Those marks are the polygon.
//!
//! # Why marking beats inferring
//!
//! The archived automatic path (`harness/recon/layout/floorplan.py`) rasterised
//! observed floor points, traced a contour, fitted wall planes and intersected
//! them. `archive/outputs/*/layout.json` is what it produced: a twenty-vertex
//! footprint wobbling at 5 cm. Two things defeated it, and neither is a tuning
//! problem:
//!
//! - **No sensor can tell a wardrobe from the wall behind it.** Depth sees one
//!   surface and it is the wardrobe's. A person pointing at a corner is
//!   asserting *this is the room boundary*, which is information the capture
//!   does not otherwise contain.
//! - **Plane membership by normal and offset claims the whole room** — see
//!   `HANDOFF.md` §3b, where one plane took 3,877 cells spread over 13.2 x 8.5 m
//!   at 3.1% fill. Here membership is declared by the user, so the problem does
//!   not arise.
//!
//! # Why the ray meets the floor rather than the depth map
//!
//! The corner is *on the floor*, and the floor's height is already tracked from
//! ARCore's largest horizontal plane. Intersecting the aim ray with that plane
//! is exact analytic geometry from the pose alone.
//!
//! Reading depth at the reticle instead would sample the one place depth is
//! worst: a concave corner is a mixed pixel across two surfaces, and on a phone
//! without a time-of-flight sensor (`depthSensor=DO_NOT_USE` on the S23 Ultra
//! that shot every capture in this repo) that depth is motion stereo, which
//! needs texture the corner of a painted room does not have. Depth is still
//! worth having as a *veto* — see [`FloorHit::disagrees_with_depth`] — but not
//! as the source.
//!
//! # The error model, which is the whole design
//!
//! A camera `h` above the floor, aiming down at horizontal distance `d`, moves
//! its floor intersection by
//!
//! ```text
//! de/dtheta = h / sin^2(theta) = h + d^2/h
//! ```
//!
//! per radian of aim error — flat within an eye height, then quadratic. At
//! h = 1.4 m, one degree of aim error and a floor known to 2 cm:
//!
//! | horizontal distance | one frame | 12-frame hold |
//! |---|---|---|
//! | 1 m | 4.0 cm | 1.8 cm |
//! | 2 m | 8.0 cm | 3.6 cm |
//! | 3 m | 14.4 cm | 5.8 cm |
//! | 6 m | 48 cm | — beyond the limit |
//!
//! So "walk the room" is not politeness, it is the accuracy mechanism, and
//! [`FloorPlanParams::max_mark_range_m`] enforces it.
//!
//! **One frame is not enough.** A 4 m wall at the old PLAN's 1.5% target has a
//! 6 cm budget, and a single frame spends 14 cm of it at the range limit. So a
//! mark is the median of the burst of poses held steady before the press —
//! [`floor_hit_burst`] — which is the same trick `capture.rs` already plays with
//! the shutter, applied to aim instead of sharpness. Note what does *not*
//! improve: the floor-height term is one estimate shared by every frame in the
//! burst, so it cannot average away, and past a few frames it is what limits a
//! close mark. [`FloorHit::sigma_m`] reports the total either way, so the client
//! can draw the uncertainty shrinking as the user approaches rather than
//! announcing a failure afterwards.

use crate::math::{Pose, Vec3};

/// A point on the floor plane, in metres, as (x, z). Y is dropped: everything
/// here is a plan view and carrying a constant third coordinate around invites
/// somebody to use it.
pub type Point2 = (f32, f32);

#[derive(Debug, Clone, Copy)]
pub struct FloorPlanParams {
    /// Refuse marks whose floor intersection is further than this.
    ///
    /// Error grows as `d^2` past an eye height, so this is the single number
    /// that sets the plan's accuracy. At 3 m, 1.4 m eye height and one degree
    /// of aim error a corner lands within ~14 cm from one frame and ~6 cm from
    /// a steady hold; at 6 m it is ~48 cm and the polygon is fiction.
    pub max_mark_range_m: f32,
    /// Beyond this a mark is taken but flagged [`MarkQuality::Marginal`].
    pub good_mark_range_m: f32,
    /// Combined aim and pose uncertainty, one sigma.
    ///
    /// One degree. The reticle is at the screen centre — on the optical axis,
    /// where there is no lens distortion and no fingertip covering 1.4 degrees
    /// of field — so this is how steadily a person holds a phone plus ARCore's
    /// own rotation error, not touch precision.
    pub aim_sigma_rad: f32,
    /// One sigma on the tracked floor height.
    ///
    /// Enters the mark laterally scaled by `d/h`, so it matters most exactly
    /// where aim error already does. Largely *systematic* within one session —
    /// a floor estimated 2 cm low biases every mark outward rather than
    /// scattering them — which is why holding steady cannot average it away,
    /// and why it is what limits a close mark once the aim term has been.
    pub floor_sigma_m: f32,
    /// Two marks closer than this are one press registered twice.
    pub min_corner_spacing_m: f32,
    /// How near the first corner a later mark must land to read as closing the
    /// loop rather than as a new corner. Generous: it is compared against a
    /// corner the user is standing in front of and deliberately re-marking.
    pub close_tolerance_m: f32,
    /// A mark is vetoed if the depth at the reticle disagrees with the
    /// ray-plane distance by more than this fraction — the user is aiming at a
    /// box, a step or a rug, not at the floor.
    pub depth_veto_fraction: f32,
}

impl Default for FloorPlanParams {
    fn default() -> Self {
        FloorPlanParams {
            max_mark_range_m: 3.0,
            good_mark_range_m: 2.0,
            aim_sigma_rad: 0.0175, // 1 degree
            floor_sigma_m: 0.02,
            min_corner_spacing_m: 0.25,
            close_tolerance_m: 0.75,
            depth_veto_fraction: 0.20,
        }
    }
}

/// How much to trust one mark, decided here so both clients say the same thing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MarkQuality {
    Good,
    /// Taken, but the user should have stepped closer.
    Marginal,
}

/// Where the aim ray meets the floor, and how well it is known.
#[derive(Debug, Clone, Copy)]
pub struct FloorHit {
    pub position: Point2,
    /// Horizontal distance from the camera to the intersection.
    pub distance_m: f32,
    /// Straight-line distance, which is what a depth reading at the reticle
    /// would measure.
    pub range_m: f32,
    /// One-sigma lateral uncertainty of `position`, in metres.
    pub sigma_m: f32,
    pub quality: MarkQuality,
}

impl FloorHit {
    /// Whether a depth sample at the reticle contradicts this intersection.
    ///
    /// A veto, not a source. Agreement proves nothing much — both would agree
    /// on a rug — but a 20% disagreement means the reticle is on something that
    /// is not the floor, which is the common way to mark a corner wrongly.
    pub fn disagrees_with_depth(&self, depth_m: f32, params: FloorPlanParams) -> bool {
        if depth_m <= 0.0 {
            return false; // no return from the sensor is not evidence of anything
        }
        (depth_m - self.range_m).abs() > params.depth_veto_fraction * self.range_m
    }
}

/// Intersect a camera's aim ray with the horizontal plane `y = floor_y`.
///
/// The ray is the camera's forward axis, because the reticle is drawn at the
/// screen centre. Returns `None` when the ray does not descend to the floor at
/// all — the user is looking level or up.
pub fn floor_hit(pose: &Pose, floor_y: f32, params: FloorPlanParams) -> Option<FloorHit> {
    let origin = pose.position();
    let h = origin.y - floor_y;
    if h <= 1e-3 {
        return None; // camera at or below the floor: nothing to intersect
    }
    let dir = pose.forward().normalized();
    // Strictly descending. Without this the intersection races to infinity as
    // the ray flattens, and a mark 400 m away is not a rejection the user can
    // act on — it is a number that breaks every downstream bound.
    if dir.y > -1e-4 {
        return None;
    }
    let t = (floor_y - origin.y) / dir.y;
    let p = origin + dir * t;
    let d = ((p.x - origin.x).powi(2) + (p.z - origin.z).powi(2)).sqrt();

    // d(distance)/d(theta) = h / sin^2(theta) = h + d^2/h, exact rather than
    // the large-d approximation, so it stays right when aiming straight down.
    let from_aim = (h + d * d / h) * params.aim_sigma_rad;
    // A floor estimated wrong by dy slides the intersection by dy * d/h.
    let from_floor = params.floor_sigma_m * d / h;
    let sigma = (from_aim * from_aim + from_floor * from_floor).sqrt();

    Some(FloorHit {
        position: (p.x, p.z),
        distance_m: d,
        range_m: (d * d + h * h).sqrt(),
        sigma_m: sigma,
        quality: if d <= params.good_mark_range_m {
            MarkQuality::Good
        } else {
            MarkQuality::Marginal
        },
    })
}

/// How many frames of a steady hold are allowed to count towards averaging.
///
/// Hand tremor is not independent between consecutive frames at 30-60 Hz —
/// there is low-frequency sway underneath it — so `1/sqrt(n)` is optimistic and
/// gets more so the longer the burst. The cap keeps the reported sigma a modest
/// claim rather than one that improves without limit as the user dithers.
const MAX_AVERAGED_FRAMES: usize = 16;

/// The floor intersection of a burst of poses held steady before the press.
///
/// Component-wise median rather than mean: one frame taken as the thumb lands
/// is a large outlier, and a median of fifteen ignores it where a mean would
/// carry a fifteenth of it into the corner.
///
/// The aim term of the sigma shrinks with the burst; the floor-height term does
/// not, because every frame in the burst intersected the *same* estimated
/// plane. Dividing that by `sqrt(n)` too would be the familiar mistake of
/// averaging away a systematic error.
pub fn floor_hit_burst(poses: &[Pose], floor_y: f32, params: FloorPlanParams) -> Option<FloorHit> {
    let hits: Vec<(Vec3, FloorHit)> = poses
        .iter()
        .filter_map(|p| floor_hit(p, floor_y, params).map(|h| (p.position(), h)))
        .collect();
    let (origin, _) = *hits.last()?;

    let mut xs: Vec<f32> = hits.iter().map(|(_, h)| h.position.0).collect();
    let mut zs: Vec<f32> = hits.iter().map(|(_, h)| h.position.1).collect();
    let position = (median(&mut xs), median(&mut zs));

    // Geometry is re-derived from where the user stood at the press, not
    // averaged: the distance that matters is the one they actually marked from.
    let h = origin.y - floor_y;
    let d = distance2((origin.x, origin.z), position);
    let n = hits.len().clamp(1, MAX_AVERAGED_FRAMES) as f32;
    let from_aim = (h + d * d / h) * params.aim_sigma_rad / n.sqrt();
    let from_floor = params.floor_sigma_m * d / h;

    Some(FloorHit {
        position,
        distance_m: d,
        range_m: (d * d + h * h).sqrt(),
        sigma_m: (from_aim * from_aim + from_floor * from_floor).sqrt(),
        quality: if d <= params.good_mark_range_m {
            MarkQuality::Good
        } else {
            MarkQuality::Marginal
        },
    })
}

fn median(values: &mut [f32]) -> f32 {
    values.sort_by(f32::total_cmp);
    let n = values.len();
    if n % 2 == 1 {
        values[n / 2]
    } else {
        0.5 * (values[n / 2 - 1] + values[n / 2])
    }
}

/// What the user is asserting with a press.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MarkKind {
    /// Two walls and the floor meet here.
    Corner,
    /// A point along the floor line of the wall currently being walked. Used
    /// where the corner itself cannot be approached; see [`FloorPlan::solve`].
    WallPoint,
}

#[derive(Debug, Clone, Copy)]
pub struct Mark {
    pub kind: MarkKind,
    pub position: Point2,
    pub sigma_m: f32,
    /// Where the user stood. Kept because loop-closure drift is distributed by
    /// walked distance, not by mark count.
    pub from: Vec3,
    /// True for a corner the user declared unreachable: its position is a
    /// placeholder and [`FloorPlan::solve`] replaces it with the intersection
    /// of the two adjacent wall lines.
    pub inferred: bool,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MarkRejection {
    /// The floor height is not known yet — no horizontal plane is tracked.
    NoFloor,
    /// The aim ray never descends to the floor.
    NotLookingDown,
    TooFar {
        distance_m: f32,
        limit_m: f32,
    },
    /// Indistinguishable from the previous mark: one press registered twice.
    TooClose {
        spacing_m: f32,
    },
    /// Depth at the reticle says the reticle is not on the floor.
    NotFloor {
        depth_m: f32,
        expected_m: f32,
    },
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MarkOutcome {
    Accepted {
        index: usize,
        position: Point2,
        sigma_m: f32,
        quality: MarkQuality,
    },
    /// The mark landed on the first corner: the walk is a closed loop, and the
    /// gap between the two is measured drift. Carries the gap *before* it is
    /// distributed, because that number is the plan's error bar and must not be
    /// quietly absorbed by the correction that follows.
    ClosedLoop {
        drift_m: f32,
    },
    Rejected(MarkRejection),
}

/// A room outline under construction.
#[derive(Debug, Clone, Default)]
pub struct FloorPlan {
    params: FloorPlanParams,
    floor_y: Option<f32>,
    marks: Vec<Mark>,
    closed: bool,
    drift_m: Option<f32>,
}

impl FloorPlan {
    pub fn new(params: FloorPlanParams) -> Self {
        FloorPlan {
            params,
            floor_y: None,
            marks: Vec::new(),
            closed: false,
            drift_m: None,
        }
    }

    pub fn params(&self) -> FloorPlanParams {
        self.params
    }

    /// Set from ARCore's lowest tracked horizontal plane, as the client already
    /// does for the station solver.
    pub fn set_floor_height(&mut self, y: f32) {
        self.floor_y = Some(y);
    }

    pub fn floor_height(&self) -> Option<f32> {
        self.floor_y
    }

    pub fn marks(&self) -> &[Mark] {
        &self.marks
    }

    pub fn corner_count(&self) -> usize {
        self.marks
            .iter()
            .filter(|m| m.kind == MarkKind::Corner)
            .count()
    }

    pub fn is_closed(&self) -> bool {
        self.closed
    }

    /// Measured loop-closure drift, once the walk has been closed.
    ///
    /// `None` until then, and deliberately not defaulted to zero: an unclosed
    /// walk has an *unknown* error, which is a different statement from a small
    /// one.
    pub fn drift_m(&self) -> Option<f32> {
        self.drift_m
    }

    /// Live preview for the reticle, every frame. The same intersection the
    /// shutter would commit, so what the user aims at is what they get.
    pub fn preview(&self, pose: &Pose) -> Option<FloorHit> {
        floor_hit(pose, self.floor_y?, self.params)
    }

    /// Commit a mark at the reticle.
    ///
    /// `poses` is the burst the client held while the user steadied on the
    /// corner, oldest first, ending with the pose at the press. A single pose
    /// is accepted and simply gets no averaging — and, correctly, a sigma that
    /// says so.
    ///
    /// `depth_m` is the sensor's reading at the reticle, or `None` where there
    /// is no return; it can only veto, never place.
    pub fn mark(&mut self, kind: MarkKind, poses: &[Pose], depth_m: Option<f32>) -> MarkOutcome {
        let Some(floor_y) = self.floor_y else {
            return MarkOutcome::Rejected(MarkRejection::NoFloor);
        };
        let Some(hit) = floor_hit_burst(poses, floor_y, self.params) else {
            return MarkOutcome::Rejected(MarkRejection::NotLookingDown);
        };
        let Some(pose) = poses.last() else {
            return MarkOutcome::Rejected(MarkRejection::NotLookingDown);
        };
        if hit.distance_m > self.params.max_mark_range_m {
            return MarkOutcome::Rejected(MarkRejection::TooFar {
                distance_m: hit.distance_m,
                limit_m: self.params.max_mark_range_m,
            });
        }
        if let Some(d) = depth_m {
            if hit.disagrees_with_depth(d, self.params) {
                return MarkOutcome::Rejected(MarkRejection::NotFloor {
                    depth_m: d,
                    expected_m: hit.range_m,
                });
            }
        }

        // Closing the loop is checked before the double-press guard, because a
        // deliberate re-mark of the first corner is *supposed* to land on top of
        // an existing one and would otherwise be rejected as a stutter.
        if kind == MarkKind::Corner && self.corner_count() >= 3 {
            if let Some(first) = self.marks.iter().find(|m| m.kind == MarkKind::Corner) {
                let gap = distance2(first.position, hit.position);
                if gap <= self.params.close_tolerance_m {
                    let from = pose.position();
                    self.close_loop(hit.position, from);
                    return MarkOutcome::ClosedLoop { drift_m: gap };
                }
            }
        }

        if let Some(prev) = self.marks.last() {
            let spacing = distance2(prev.position, hit.position);
            if spacing < self.params.min_corner_spacing_m {
                return MarkOutcome::Rejected(MarkRejection::TooClose { spacing_m: spacing });
            }
        }

        self.marks.push(Mark {
            kind,
            position: hit.position,
            sigma_m: hit.sigma_m,
            from: pose.position(),
            inferred: false,
        });
        MarkOutcome::Accepted {
            index: self.marks.len() - 1,
            position: hit.position,
            sigma_m: hit.sigma_m,
            quality: hit.quality,
        }
    }

    /// Declare that the corner the user is walking towards cannot be reached.
    ///
    /// Furniture in front of a corner is ordinary, and a plan that cannot be
    /// finished without standing in a wardrobe is one the user abandons — the
    /// same reasoning as `script.rs`'s per-shot skip. The vertex is kept as a
    /// placeholder and recovered in [`Self::solve`] by intersecting the floor
    /// lines of the two walls that meet there, each of which can be marked from
    /// anywhere along its length.
    pub fn mark_blocked_corner(&mut self) {
        self.marks.push(Mark {
            kind: MarkKind::Corner,
            position: (0.0, 0.0),
            sigma_m: f32::INFINITY,
            from: Vec3::ZERO,
            inferred: true,
        });
    }

    /// Undo the last mark. A misplaced corner is visible immediately on the
    /// minimap, so the fix belongs in the capture rather than in a later edit.
    pub fn undo(&mut self) -> Option<Mark> {
        self.closed = false;
        self.drift_m = None;
        self.marks.pop()
    }

    /// Close the walk onto its first corner and distribute the gap.
    ///
    /// The gap is accumulated VIO drift over the walk, so it is spread in
    /// proportion to *distance walked*, not to corner index — five corners
    /// along one wall accumulate no more drift between them than two do. The
    /// walked path is sampled only at the marks, so it under-measures the true
    /// route, but only the ratios enter the correction.
    fn close_loop(&mut self, remark: Point2, remark_from: Vec3) {
        let Some(first) = self
            .marks
            .iter()
            .find(|m| m.kind == MarkKind::Corner)
            .copied()
        else {
            return;
        };
        let gap = (remark.0 - first.position.0, remark.1 - first.position.1);
        self.drift_m = Some((gap.0 * gap.0 + gap.1 * gap.1).sqrt());

        // Cumulative walked distance at each mark, and on to the re-mark.
        let mut cumulative = Vec::with_capacity(self.marks.len());
        let mut total = 0.0f32;
        let mut previous: Option<Vec3> = None;
        for m in &self.marks {
            if m.inferred {
                // No standing position was recorded for a corner nobody stood
                // at; it inherits the distance of the mark before it.
                cumulative.push(total);
                continue;
            }
            if let Some(p) = previous {
                total += p.distance(m.from);
            }
            previous = Some(m.from);
            cumulative.push(total);
        }
        if let Some(p) = previous {
            total += p.distance(remark_from);
        }
        if total <= 1e-6 {
            self.closed = true;
            return;
        }

        for (m, s) in self.marks.iter_mut().zip(cumulative) {
            if m.inferred {
                continue; // placed by intersection later, from corrected walls
            }
            let f = s / total;
            m.position.0 -= gap.0 * f;
            m.position.1 -= gap.1 * f;
        }
        self.closed = true;
    }

    /// Resolve the marks into a closed polygon.
    ///
    /// Corners marked directly are used as marked. A corner is recovered by
    /// intersecting its two adjacent wall lines when it was declared blocked,
    /// or when both adjacent walls carry extra [`MarkKind::WallPoint`] marks —
    /// two points measured from close range beat one corner measured from
    /// whatever distance the furniture allowed.
    ///
    /// This is fit-then-intersect, which is what the archived Python did and
    /// what `HANDOFF.md` §3b warns about. The difference is membership: there,
    /// a plane claimed every cell sharing its normal and offset, so one wall
    /// swallowed the room. Here the user declared which wall each point belongs
    /// to by marking it while standing at it, and the fit is over a handful of
    /// deliberate points rather than a million incidental ones.
    pub fn solve(&self) -> Option<Footprint> {
        let corner_indices: Vec<usize> = self
            .marks
            .iter()
            .enumerate()
            .filter(|(_, m)| m.kind == MarkKind::Corner)
            .map(|(i, _)| i)
            .collect();
        let n = corner_indices.len();
        if n < 3 {
            return None;
        }

        // Wall i runs from corner i to corner i+1, and owns every wall point
        // marked between them. The wall closing the loop owns the points after
        // the last corner, which is where they land when the user marks the
        // final stretch on the way back to the start.
        let mut walls: Vec<Option<Line2>> = Vec::with_capacity(n);
        for c in 0..n {
            let start = corner_indices[c];
            let end = corner_indices[(c + 1) % n];
            let mut own: Vec<Mark> = Vec::new();
            let mut i = (start + 1) % self.marks.len();
            while i != end {
                if self.marks[i].kind == MarkKind::WallPoint {
                    own.push(self.marks[i]);
                }
                i = (i + 1) % self.marks.len();
            }
            // A wall with two deliberate points of its own is fitted from those
            // alone. Folding the endpoint corners in as well would let a single
            // sloppy corner mark tilt the very line meant to correct it — the
            // wall points exist precisely because the corner could not be
            // marked well.
            let pts: Vec<Mark> = if own.len() >= 2 {
                own
            } else {
                let mut with_ends = own;
                for e in [start, end] {
                    if !self.marks[e].inferred {
                        with_ends.push(self.marks[e]);
                    }
                }
                with_ends
            };
            walls.push(fit_line(&pts));
        }

        let mut points: Vec<Point2> = Vec::with_capacity(n);
        for c in 0..n {
            let marked = &self.marks[corner_indices[c]];
            let incoming = walls[(c + n - 1) % n];
            let outgoing = walls[c];
            // Both adjacent walls independently determined, or no usable mark
            // of its own. One wall point on a side is not enough: that wall
            // still needs the corner marks to be fitted at all, so the
            // intersection would be partly re-deriving the corner from itself.
            let wall_points_here = self.wall_points_around(&corner_indices, c) >= 2;
            let prefer_intersection = marked.inferred || wall_points_here;
            let resolved = if prefer_intersection {
                match (incoming, outgoing) {
                    (Some(a), Some(b)) => intersect(a, b),
                    _ => None,
                }
            } else {
                None
            };
            let fallback = if marked.inferred {
                None
            } else {
                Some(marked.position)
            };
            // A blocked corner whose walls could not be fitted, or came out
            // parallel, resolves to nothing. Dropping it silently would shrink
            // the room by a whole corner and still return a plausible polygon,
            // so the whole solve fails instead.
            points.push(resolved.or(fallback)?);
        }

        Some(Footprint::new(points, self.drift_m))
    }

    /// How many wall points were marked on the two walls meeting at corner `c`.
    fn wall_points_around(&self, corner_indices: &[usize], c: usize) -> usize {
        let n = corner_indices.len();
        let prev = corner_indices[(c + n - 1) % n];
        let here = corner_indices[c];
        let next = corner_indices[(c + 1) % n];
        let count_between = |from: usize, to: usize| {
            let mut i = (from + 1) % self.marks.len();
            let mut k = 0;
            while i != to {
                if self.marks[i].kind == MarkKind::WallPoint {
                    k += 1;
                }
                i = (i + 1) % self.marks.len();
            }
            k
        };
        count_between(prev, here).min(count_between(here, next))
    }
}

/// The finished outline.
#[derive(Debug, Clone)]
pub struct Footprint {
    /// Closed polygon of floor corners as (x, z) in metres, wound
    /// counter-clockwise when plotted as a plan with x to the right and z up
    /// the page — that is, positive shoelace area. Not repeated at the end.
    pub points: Vec<Point2>,
    pub area_m2: f32,
    pub perimeter_m: f32,
    /// Length of each wall, `wall_lengths[i]` running from `points[i]` to
    /// `points[i + 1]`.
    pub wall_lengths_m: Vec<f32>,
    /// Measured loop-closure drift, if the walk was closed. See
    /// [`FloorPlan::drift_m`] for why this is an `Option`.
    pub drift_m: Option<f32>,
    /// Whether the outline crosses itself, which means corners were marked out
    /// of order. Reported rather than repaired: the fix is one undo by someone
    /// who can see the room, and any automatic reordering would be a guess.
    pub self_intersecting: bool,
}

impl Footprint {
    fn new(mut points: Vec<Point2>, drift_m: Option<f32>) -> Self {
        let signed = signed_area(&points);
        if signed < 0.0 {
            points.reverse();
        }
        let n = points.len();
        let wall_lengths_m: Vec<f32> = (0..n)
            .map(|i| distance2(points[i], points[(i + 1) % n]))
            .collect();
        Footprint {
            area_m2: signed.abs(),
            perimeter_m: wall_lengths_m.iter().sum(),
            self_intersecting: is_self_intersecting(&points),
            wall_lengths_m,
            drift_m,
            points,
        }
    }

    /// Whether a plan-view point is inside the room. Ray casting, so it is
    /// correct for the L-shapes and alcoves that a convex test would mangle.
    pub fn contains(&self, p: Point2) -> bool {
        let n = self.points.len();
        let mut inside = false;
        let mut j = n - 1;
        for i in 0..n {
            let (xi, zi) = self.points[i];
            let (xj, zj) = self.points[j];
            if (zi > p.1) != (zj > p.1) {
                let x = xi + (p.1 - zi) / (zj - zi) * (xj - xi);
                if p.0 < x {
                    inside = !inside;
                }
            }
            j = i;
        }
        inside
    }

    /// Distance from `p` to the nearest wall. Unsigned, so callers combine it
    /// with [`Self::contains`] — which is what a station's clearance test wants
    /// anyway.
    pub fn clearance(&self, p: Point2) -> f32 {
        let n = self.points.len();
        (0..n)
            .map(|i| point_segment_distance(p, self.points[i], self.points[(i + 1) % n]))
            .fold(f32::INFINITY, f32::min)
    }
}

// -- 2D helpers -------------------------------------------------------------

/// A line as a point and a unit direction. Infinite: walls are intersected, and
/// a corner routinely lies beyond the last point marked along either wall.
#[derive(Debug, Clone, Copy)]
pub struct Line2 {
    pub point: Point2,
    pub direction: Point2,
}

/// Total-least-squares fit through 2 or more marks, weighted by `1/sigma^2`.
///
/// Ordinary least squares would need a choice of independent axis and blow up
/// on the wall that happens to run along it; the principal axis of the centred
/// covariance has no preferred direction.
///
/// The weighting is what makes mixing marks from different distances safe. A
/// point marked at 3 m carries roughly a ninth the weight of one marked at 1 m,
/// which is the ratio their uncertainties already record — so the user can mark
/// a far wall opportunistically without that mark degrading a line their closer
/// ones had pinned down.
fn fit_line(marks: &[Mark]) -> Option<Line2> {
    if marks.len() < 2 {
        return None;
    }
    let weight = |m: &Mark| {
        // An inferred corner has infinite sigma and is never passed here, but a
        // zero sigma from a degenerate pose would be an infinite weight.
        let s = m.sigma_m.max(1e-4);
        1.0 / (s * s)
    };
    let total: f32 = marks.iter().map(weight).sum();
    if !total.is_finite() || total <= 0.0 {
        return None;
    }
    let mx = marks.iter().map(|m| m.position.0 * weight(m)).sum::<f32>() / total;
    let mz = marks.iter().map(|m| m.position.1 * weight(m)).sum::<f32>() / total;
    let (mut sxx, mut sxz, mut szz) = (0.0f32, 0.0f32, 0.0f32);
    for m in marks {
        let w = weight(m);
        let (dx, dz) = (m.position.0 - mx, m.position.1 - mz);
        sxx += w * dx * dx;
        sxz += w * dx * dz;
        szz += w * dz * dz;
    }
    if (sxx + szz) / total < 1e-12 {
        return None; // every point on top of every other
    }
    let theta = 0.5 * (2.0 * sxz).atan2(sxx - szz);
    Some(Line2 {
        point: (mx, mz),
        direction: (theta.cos(), theta.sin()),
    })
}

/// Intersect two infinite lines, or `None` if they are within ~1.7 degrees of
/// parallel — where the intersection runs away and a corner placed there is
/// worse than no corner at all.
fn intersect(a: Line2, b: Line2) -> Option<Point2> {
    let denom = a.direction.0 * b.direction.1 - a.direction.1 * b.direction.0;
    if denom.abs() < 0.03 {
        return None;
    }
    let dx = b.point.0 - a.point.0;
    let dz = b.point.1 - a.point.1;
    let t = (dx * b.direction.1 - dz * b.direction.0) / denom;
    Some((a.point.0 + a.direction.0 * t, a.point.1 + a.direction.1 * t))
}

fn distance2(a: Point2, b: Point2) -> f32 {
    ((a.0 - b.0).powi(2) + (a.1 - b.1).powi(2)).sqrt()
}

fn signed_area(points: &[Point2]) -> f32 {
    let n = points.len();
    let mut sum = 0.0;
    for i in 0..n {
        let (x0, z0) = points[i];
        let (x1, z1) = points[(i + 1) % n];
        sum += x0 * z1 - x1 * z0;
    }
    sum * 0.5
}

fn point_segment_distance(p: Point2, a: Point2, b: Point2) -> f32 {
    let (vx, vz) = (b.0 - a.0, b.1 - a.1);
    let len_sq = vx * vx + vz * vz;
    if len_sq < 1e-12 {
        return distance2(p, a);
    }
    let t = (((p.0 - a.0) * vx + (p.1 - a.1) * vz) / len_sq).clamp(0.0, 1.0);
    distance2(p, (a.0 + vx * t, a.1 + vz * t))
}

fn is_self_intersecting(points: &[Point2]) -> bool {
    let n = points.len();
    if n < 4 {
        return false;
    }
    for i in 0..n {
        for j in (i + 1)..n {
            // Adjacent edges share a vertex by construction, and the first and
            // last edges share one too.
            if j == i + 1 || (i == 0 && j == n - 1) {
                continue;
            }
            if segments_cross(
                points[i],
                points[(i + 1) % n],
                points[j],
                points[(j + 1) % n],
            ) {
                return true;
            }
        }
    }
    false
}

fn segments_cross(p1: Point2, p2: Point2, p3: Point2, p4: Point2) -> bool {
    let d = |a: Point2, b: Point2, c: Point2| (b.0 - a.0) * (c.1 - a.1) - (b.1 - a.1) * (c.0 - a.0);
    let (d1, d2, d3, d4) = (d(p3, p4, p1), d(p3, p4, p2), d(p1, p2, p3), d(p1, p2, p4));
    ((d1 > 0.0) != (d2 > 0.0)) && ((d3 > 0.0) != (d4 > 0.0))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::look_rotation;

    const EYE: f32 = 1.4;

    /// A camera at `(x, EYE, z)` aiming at yaw/pitch, floor at y = 0.
    fn aim(x: f32, z: f32, yaw: f32, pitch: f32) -> Pose {
        Pose {
            r: look_rotation(yaw, pitch),
            t: Vec3::new(x, EYE, z),
        }
    }

    /// A camera standing at `from` and aiming exactly at the floor point
    /// `target`, which is how a user who has lined the reticle up is posed.
    fn aim_at(from: Point2, target: Point2) -> Pose {
        let (dx, dz) = (target.0 - from.0, target.1 - from.1);
        let d = (dx * dx + dz * dz).sqrt();
        // look_rotation's yaw is measured so that forward is
        // (-sin y * cos p, ...), i.e. yaw 0 looks down -Z.
        let yaw = (-dx).atan2(-dz);
        let pitch = (-EYE).atan2(d);
        Pose {
            r: look_rotation(yaw, pitch),
            t: Vec3::new(from.0, EYE, from.1),
        }
    }

    fn plan() -> FloorPlan {
        let mut p = FloorPlan::new(FloorPlanParams::default());
        p.set_floor_height(0.0);
        p
    }

    #[test]
    fn aiming_straight_down_lands_underfoot() {
        let hit = floor_hit(
            &aim(2.0, -3.0, 0.0, -std::f32::consts::FRAC_PI_2),
            0.0,
            FloorPlanParams::default(),
        )
        .expect("a ray pointing at the floor should meet it");
        assert!(hit.distance_m < 1e-3, "{:?}", hit.position);
        assert!((hit.position.0 - 2.0).abs() < 1e-3);
        assert!((hit.position.1 + 3.0).abs() < 1e-3);
        // Pointing straight down, aim error tips the ray sideways by h * sigma
        // and no more. The large-d approximation d^2/h would say zero, which
        // would advertise a perfect measurement.
        assert!(hit.sigma_m > 0.5 * EYE * FloorPlanParams::default().aim_sigma_rad);
    }

    #[test]
    fn the_helper_actually_hits_what_it_aims_at() {
        // Everything below depends on this, so it is checked directly rather
        // than assumed: a sign error in the test rig would quietly relocate
        // every corner and still produce a plausible-looking room.
        for target in [(1.0, 0.0), (0.0, 2.0), (-1.5, -2.0), (2.0, -1.0)] {
            let hit = floor_hit(&aim_at((0.0, 0.0), target), 0.0, FloorPlanParams::default())
                .expect("aimed at the floor");
            assert!(
                distance2(hit.position, target) < 1e-3,
                "aimed at {target:?}, hit {:?}",
                hit.position
            );
        }
    }

    #[test]
    fn a_level_or_upward_ray_has_no_floor_intersection() {
        // Not a very distant one. Letting the intersection race to infinity
        // puts a corner hundreds of metres away in the polygon, and every
        // bound derived from it is then meaningless.
        let params = FloorPlanParams::default();
        assert!(floor_hit(&aim(0.0, 0.0, 0.0, 0.0), 0.0, params).is_none());
        assert!(floor_hit(&aim(0.0, 0.0, 0.0, 0.3), 0.0, params).is_none());
    }

    #[test]
    fn uncertainty_grows_faster_than_linearly_with_distance() {
        // The reason the user is asked to walk the room. Not purely quadratic:
        // the `h` term dominates within about an eye height, so a mark at 1 m
        // is no better than four times one at 4 m even though the distance
        // ratio says sixteen. Quoting the d^2/h approximation instead would
        // advertise 1.2 cm at 1 m, which is three times better than the truth.
        let params = FloorPlanParams::default();
        let sigma = |target: Point2| {
            floor_hit(&aim_at((0.0, 0.0), target), 0.0, params)
                .unwrap()
                .sigma_m
        };
        let (near, mid, far) = (sigma((1.0, 0.0)), sigma((2.0, 0.0)), sigma((4.0, 0.0)));
        assert!(
            far / near > 4.0,
            "quadrupling the distance did more than quadruple the error"
        );
        assert!(far / mid > 2.5, "{mid} -> {far}");
        // The absolute numbers are the ones the module docs table quotes, and
        // the docs are the argument for the range limit.
        assert!((near - 0.040).abs() < 0.003, "1 m single frame: {near}");
        assert!((mid - 0.080).abs() < 0.003, "2 m single frame: {mid}");
        assert!((far - 0.232).abs() < 0.005, "4 m single frame: {far}");
    }

    #[test]
    fn a_steady_hold_averages_the_aim_error_down_but_not_the_floor_error() {
        // The reason a mark is a burst. Without it a mark at the 3 m limit
        // carries 14 cm, which is more than the whole error budget of a 4 m
        // wall at the old PLAN's 1.5% target.
        let params = FloorPlanParams::default();
        let target = (3.0, 0.0);
        let one = floor_hit(&aim_at((0.0, 0.0), target), 0.0, params).unwrap();
        let burst: Vec<Pose> = (0..12).map(|_| aim_at((0.0, 0.0), target)).collect();
        let held = floor_hit_burst(&burst, 0.0, params).unwrap();

        assert!(one.sigma_m > 0.14, "single frame at 3 m: {}", one.sigma_m);
        assert!(
            held.sigma_m < 0.06,
            "12-frame hold at 3 m: {}",
            held.sigma_m
        );
        // Not sqrt(12) better overall: the floor-height term is one estimate
        // shared by every frame in the burst and cannot average away. If this
        // ever reaches sqrt(12) the systematic term has been divided too.
        let ratio = one.sigma_m / held.sigma_m;
        assert!(
            ratio < 12f32.sqrt(),
            "claimed {ratio}x from averaging, which is too good"
        );
        assert!(ratio > 2.0, "only {ratio}x from a 12-frame hold");
    }

    #[test]
    fn the_burst_takes_a_median_so_one_jolted_frame_does_not_move_the_corner() {
        // The frame taken as the thumb lands is the one that is wrong, and it
        // is always the last one. A mean would carry a twelfth of it.
        let params = FloorPlanParams::default();
        let target = (2.0, 0.0);
        let mut burst: Vec<Pose> = (0..11).map(|_| aim_at((0.0, 0.0), target)).collect();
        burst.push(aim_at((0.0, 0.0), (2.0, 0.6))); // the jolt
        let held = floor_hit_burst(&burst, 0.0, params).unwrap();
        assert!(
            distance2(held.position, target) < 0.01,
            "a jolted last frame moved the mark to {:?}",
            held.position
        );
    }

    #[test]
    fn a_burst_of_one_gets_no_averaging_and_says_so() {
        let params = FloorPlanParams::default();
        let pose = aim_at((0.0, 0.0), (2.0, 0.0));
        let single = floor_hit(&pose, 0.0, params).unwrap();
        let burst = floor_hit_burst(std::slice::from_ref(&pose), 0.0, params).unwrap();
        assert!((single.sigma_m - burst.sigma_m).abs() < 1e-6);
        assert!(distance2(single.position, burst.position) < 1e-6);
    }

    #[test]
    fn a_mark_beyond_the_range_limit_is_refused_with_the_distance() {
        // Refused rather than taken and flagged: at 5 m the corner is a guess,
        // and a plan that quietly contains guesses cannot be told from one that
        // does not.
        let mut p = plan();
        let outcome = p.mark(MarkKind::Corner, &[aim_at((0.0, 0.0), (5.0, 0.0))], None);
        match outcome {
            MarkOutcome::Rejected(MarkRejection::TooFar {
                distance_m,
                limit_m,
            }) => {
                assert!((distance_m - 5.0).abs() < 0.05);
                assert_eq!(limit_m, 3.0);
            }
            other => panic!("expected TooFar, got {other:?}"),
        }
        assert_eq!(p.corner_count(), 0);
    }

    #[test]
    fn a_marginal_mark_is_taken_but_says_so() {
        let mut p = plan();
        match p.mark(MarkKind::Corner, &[aim_at((0.0, 0.0), (2.5, 0.0))], None) {
            MarkOutcome::Accepted { quality, .. } => assert_eq!(quality, MarkQuality::Marginal),
            other => panic!("expected an accepted mark, got {other:?}"),
        }
    }

    #[test]
    fn depth_can_veto_a_mark_but_never_places_one() {
        let mut p = plan();
        let pose = aim_at((0.0, 0.0), (1.5, 0.0));
        let expected = floor_hit(&pose, 0.0, p.params()).unwrap().range_m;

        // Aiming at the top of a box: the sensor is much closer than the floor.
        match p.mark(MarkKind::Corner, &[pose], Some(expected * 0.5)) {
            MarkOutcome::Rejected(MarkRejection::NotFloor { .. }) => {}
            other => panic!("expected NotFloor, got {other:?}"),
        }
        // No return is not evidence of anything, and must not block the mark.
        assert!(matches!(
            p.mark(MarkKind::Corner, &[pose], Some(0.0)),
            MarkOutcome::Accepted { .. }
        ));
        // Agreeing depth changes nothing about where the corner goes.
        let mut q = plan();
        let a = q.mark(MarkKind::Corner, &[pose], Some(expected));
        let mut r = plan();
        let b = r.mark(MarkKind::Corner, &[pose], None);
        match (a, b) {
            (
                MarkOutcome::Accepted { position: pa, .. },
                MarkOutcome::Accepted { position: pb, .. },
            ) => assert!(distance2(pa, pb) < 1e-6),
            other => panic!("expected two accepted marks, got {other:?}"),
        }
    }

    #[test]
    fn one_press_registered_twice_is_rejected() {
        let mut p = plan();
        p.mark(MarkKind::Corner, &[aim_at((0.0, 0.0), (1.0, 0.0))], None);
        let again = p.mark(MarkKind::Corner, &[aim_at((0.0, 0.02), (1.0, 0.01))], None);
        assert!(matches!(
            again,
            MarkOutcome::Rejected(MarkRejection::TooClose { .. })
        ));
        assert_eq!(p.corner_count(), 1);
    }

    /// Walk a 4 x 3 m room, standing 1 m diagonally inside each corner.
    fn walk_rectangle(p: &mut FloorPlan, w: f32, d: f32) {
        let corners = [(0.0, 0.0), (w, 0.0), (w, d), (0.0, d)];
        let inset = [(0.7, 0.7), (-0.7, 0.7), (-0.7, -0.7), (0.7, -0.7)];
        for (c, i) in corners.iter().zip(inset) {
            let from = (c.0 + i.0, c.1 + i.1);
            let outcome = p.mark(MarkKind::Corner, &[aim_at(from, *c)], None);
            assert!(
                matches!(outcome, MarkOutcome::Accepted { .. }),
                "{outcome:?}"
            );
        }
    }

    #[test]
    fn four_corners_make_a_room_with_the_right_area_and_walls() {
        let mut p = plan();
        walk_rectangle(&mut p, 4.0, 3.0);
        let f = p.solve().expect("four corners are a polygon");
        assert_eq!(f.points.len(), 4);
        assert!((f.area_m2 - 12.0).abs() < 0.01, "area {}", f.area_m2);
        assert!(
            (f.perimeter_m - 14.0).abs() < 0.01,
            "perimeter {}",
            f.perimeter_m
        );
        let mut walls = f.wall_lengths_m.clone();
        walls.sort_by(f32::total_cmp);
        assert!(
            (walls[0] - 3.0).abs() < 0.01 && (walls[3] - 4.0).abs() < 0.01,
            "{walls:?}"
        );
        assert!(!f.self_intersecting);
    }

    #[test]
    fn fewer_than_three_corners_is_not_a_room() {
        let mut p = plan();
        p.mark(MarkKind::Corner, &[aim_at((0.0, 0.0), (1.0, 0.0))], None);
        p.mark(MarkKind::Corner, &[aim_at((1.0, 1.0), (2.0, 1.0))], None);
        assert!(p.solve().is_none());
    }

    #[test]
    fn the_footprint_winds_the_same_way_whichever_way_the_user_walked() {
        // Downstream code takes the sign of the area for granted; a user who
        // happens to walk anticlockwise must not invert it.
        let mut clockwise = plan();
        walk_rectangle(&mut clockwise, 4.0, 3.0);
        let a = clockwise.solve().unwrap();

        let mut widdershins = plan();
        let corners = [(0.0, 0.0), (0.0, 3.0), (4.0, 3.0), (4.0, 0.0)];
        let inset = [(0.7, 0.7), (0.7, -0.7), (-0.7, -0.7), (-0.7, 0.7)];
        for (c, i) in corners.iter().zip(inset) {
            widdershins.mark(
                MarkKind::Corner,
                &[aim_at((c.0 + i.0, c.1 + i.1), *c)],
                None,
            );
        }
        let b = widdershins.solve().unwrap();

        assert!(signed_area(&a.points) > 0.0);
        assert!(signed_area(&b.points) > 0.0);
        assert!((a.area_m2 - b.area_m2).abs() < 0.01);
    }

    #[test]
    fn corners_marked_out_of_order_are_reported_not_repaired() {
        // A bowtie has an area and a perimeter and looks like a room to
        // anything that only reads those. The user can see the room and can fix
        // it with one undo; guessing the intended order here cannot.
        let mut p = plan();
        let corners = [(0.0, 0.0), (4.0, 0.0), (0.0, 3.0), (4.0, 3.0)];
        let inset = [(0.7, 0.7), (-0.7, 0.7), (0.7, -0.7), (-0.7, -0.7)];
        for (c, i) in corners.iter().zip(inset) {
            p.mark(
                MarkKind::Corner,
                &[aim_at((c.0 + i.0, c.1 + i.1), *c)],
                None,
            );
        }
        assert!(p.solve().unwrap().self_intersecting);
    }

    #[test]
    fn an_l_shaped_room_survives_the_inside_test() {
        // Six corners with one reflex among them. A convex containment test
        // would report the missing quadrant as part of the room and send the
        // user to stand in the neighbouring flat.
        let mut p = plan();
        let corners = [
            (0.0, 0.0),
            (4.0, 0.0),
            (4.0, 2.0),
            (2.0, 2.0),
            (2.0, 4.0),
            (0.0, 4.0),
        ];
        for c in corners {
            // Stand a metre inside the room from each corner, roughly.
            let from = (
                c.0 + if c.0 > 1.0 { -0.8 } else { 0.8 },
                c.1 + if c.1 > 2.0 { -0.8 } else { 0.8 },
            );
            let outcome = p.mark(MarkKind::Corner, &[aim_at(from, c)], None);
            assert!(
                matches!(outcome, MarkOutcome::Accepted { .. }),
                "{c:?}: {outcome:?}"
            );
        }
        let f = p.solve().unwrap();
        assert!(!f.self_intersecting);
        assert!((f.area_m2 - 12.0).abs() < 0.05, "area {}", f.area_m2);
        assert!(f.contains((1.0, 1.0)));
        assert!(f.contains((3.0, 1.0)));
        assert!(f.contains((1.0, 3.0)));
        assert!(
            !f.contains((3.0, 3.0)),
            "the missing arm of the L is not the room"
        );
        assert!(!f.contains((5.0, 1.0)));
    }

    #[test]
    fn clearance_measures_the_nearest_wall() {
        let mut p = plan();
        walk_rectangle(&mut p, 4.0, 3.0);
        let f = p.solve().unwrap();
        assert!((f.clearance((2.0, 1.5)) - 1.5).abs() < 0.01);
        assert!((f.clearance((0.5, 1.5)) - 0.5).abs() < 0.01);
    }

    #[test]
    fn closing_the_loop_measures_drift_and_distributes_it() {
        // The corners are marked as if VIO had drifted 12 cm over the walk:
        // each successive corner is displaced a little further in +x.
        let mut p = plan();
        let corners = [(0.0, 0.0), (4.0, 0.0), (4.0, 3.0), (0.0, 3.0)];
        let inset = [(0.7, 0.7), (-0.7, 0.7), (-0.7, -0.7), (0.7, -0.7)];
        for (k, (c, i)) in corners.iter().zip(inset).enumerate() {
            let creep = 0.03 * k as f32;
            let target = (c.0 + creep, c.1);
            p.mark(
                MarkKind::Corner,
                &[aim_at((c.0 + i.0 + creep, c.1 + i.1), target)],
                None,
            );
        }
        // Back to the first corner, which now reads 12 cm off.
        let outcome = p.mark(MarkKind::Corner, &[aim_at((0.82, 0.7), (0.12, 0.0))], None);
        let drift = match outcome {
            MarkOutcome::ClosedLoop { drift_m } => drift_m,
            other => panic!("expected ClosedLoop, got {other:?}"),
        };
        assert!((drift - 0.12).abs() < 0.02, "drift {drift}");
        assert!(p.is_closed());
        assert_eq!(
            p.corner_count(),
            4,
            "the re-mark is a measurement, not a fifth corner"
        );

        let f = p.solve().unwrap();
        assert_eq!(f.drift_m.map(|d| (d * 100.0).round()), Some(12.0));
        // The correction pulls the far corners back, so the room is its true
        // 4 m rather than the 4.12 m the drifting walk measured.
        let widest = f.points.iter().map(|p| p.0).fold(f32::MIN, f32::max);
        assert!(
            widest < 4.05,
            "drift was not distributed: widest x {widest}"
        );
    }

    #[test]
    fn an_unclosed_walk_reports_unknown_drift_rather_than_none() {
        // Zero would claim a measurement that was never made. This is the same
        // rule HANDOFF.md sec 7 applies to residuals: do not quote a number the
        // method did not establish.
        let mut p = plan();
        walk_rectangle(&mut p, 4.0, 3.0);
        assert_eq!(p.drift_m(), None);
        assert_eq!(p.solve().unwrap().drift_m, None);
    }

    #[test]
    fn a_blocked_corner_is_recovered_from_the_two_walls_that_meet_there() {
        // The wardrobe case. The corner at (4, 3) cannot be approached, so the
        // user marks the floor line of each adjacent wall instead, from
        // wherever it is reachable.
        let mut p = plan();
        p.mark(MarkKind::Corner, &[aim_at((0.7, 0.7), (0.0, 0.0))], None);
        p.mark(MarkKind::Corner, &[aim_at((3.3, 0.7), (4.0, 0.0))], None);
        // Two points up the wall x = 4, marked while walking beside it.
        p.mark(MarkKind::WallPoint, &[aim_at((3.3, 1.0), (4.0, 1.0))], None);
        p.mark(MarkKind::WallPoint, &[aim_at((3.3, 2.0), (4.0, 2.0))], None);
        p.mark_blocked_corner();
        // Two points along the wall z = 3, coming back the other way.
        p.mark(MarkKind::WallPoint, &[aim_at((3.0, 2.3), (3.0, 3.0))], None);
        p.mark(MarkKind::WallPoint, &[aim_at((1.5, 2.3), (1.5, 3.0))], None);
        p.mark(MarkKind::Corner, &[aim_at((0.7, 2.3), (0.0, 3.0))], None);

        let f = p.solve().expect("the blocked corner is recoverable");
        assert_eq!(f.points.len(), 4);
        let recovered = f
            .points
            .iter()
            .copied()
            .min_by(|a, b| distance2(*a, (4.0, 3.0)).total_cmp(&distance2(*b, (4.0, 3.0))))
            .unwrap();
        assert!(
            distance2(recovered, (4.0, 3.0)) < 0.05,
            "recovered corner {recovered:?} should be near (4, 3)"
        );
        assert!((f.area_m2 - 12.0).abs() < 0.1, "area {}", f.area_m2);
    }

    #[test]
    fn a_blocked_corner_with_nothing_to_intersect_fails_loudly() {
        // Dropping it would silently cut a corner off the room and still return
        // a polygon that looks fine.
        let mut p = plan();
        p.mark(MarkKind::Corner, &[aim_at((0.7, 0.7), (0.0, 0.0))], None);
        p.mark(MarkKind::Corner, &[aim_at((3.3, 0.7), (4.0, 0.0))], None);
        p.mark_blocked_corner();
        p.mark(MarkKind::Corner, &[aim_at((0.7, 2.3), (0.0, 3.0))], None);
        assert!(p.solve().is_none());
    }

    #[test]
    fn wall_points_override_a_corner_marked_from_further_away() {
        // Two points measured at 1 m beat one corner measured at 3 m, which is
        // the whole reason the escape hatch is also allowed on corners that
        // were reachable.
        let mut p = plan();
        p.mark(MarkKind::Corner, &[aim_at((0.7, 0.7), (0.0, 0.0))], None);
        p.mark(MarkKind::WallPoint, &[aim_at((1.5, 0.7), (1.5, 0.0))], None);
        p.mark(MarkKind::WallPoint, &[aim_at((3.0, 0.7), (3.0, 0.0))], None);
        // The corner itself, mis-marked 20 cm into the room from a distance.
        p.mark(MarkKind::Corner, &[aim_at((3.4, 1.0), (4.0, 0.2))], None);
        p.mark(MarkKind::WallPoint, &[aim_at((3.3, 1.5), (4.0, 1.5))], None);
        p.mark(MarkKind::WallPoint, &[aim_at((3.3, 2.5), (4.0, 2.5))], None);
        p.mark(MarkKind::Corner, &[aim_at((3.3, 2.3), (4.0, 3.0))], None);
        p.mark(MarkKind::Corner, &[aim_at((0.7, 2.3), (0.0, 3.0))], None);

        let f = p.solve().unwrap();
        let corrected = f
            .points
            .iter()
            .copied()
            .min_by(|a, b| distance2(*a, (4.0, 0.0)).total_cmp(&distance2(*b, (4.0, 0.0))))
            .unwrap();
        assert!(
            distance2(corrected, (4.0, 0.0)) < 0.05,
            "the fitted walls should override the sloppy corner mark, got {corrected:?}"
        );
    }

    #[test]
    fn undo_removes_the_last_mark_and_reopens_the_walk() {
        let mut p = plan();
        walk_rectangle(&mut p, 4.0, 3.0);
        p.mark(MarkKind::Corner, &[aim_at((0.7, 0.7), (0.0, 0.0))], None);
        assert!(p.is_closed());
        p.undo();
        assert!(
            !p.is_closed(),
            "undoing past the closing mark reopens the walk"
        );
        assert_eq!(p.drift_m(), None);
    }

    #[test]
    fn without_a_floor_height_nothing_can_be_marked() {
        // The station solver has the same dependency and the client already
        // feeds it from ARCore's lowest horizontal plane; before that arrives
        // there is no plane to intersect and a guess would silently scale the
        // whole room.
        let mut p = FloorPlan::new(FloorPlanParams::default());
        assert!(p.preview(&aim_at((0.0, 0.0), (1.0, 0.0))).is_none());
        assert!(matches!(
            p.mark(MarkKind::Corner, &[aim_at((0.0, 0.0), (1.0, 0.0))], None),
            MarkOutcome::Rejected(MarkRejection::NoFloor)
        ));
    }

    #[test]
    fn a_floor_below_the_camera_by_any_amount_still_works() {
        // Floor height is whatever ARCore says, not zero: a capture started on
        // an upper storey has a floor at 3.2 m and the arithmetic must not care.
        let mut p = FloorPlan::new(FloorPlanParams::default());
        p.set_floor_height(3.2);
        let pose = Pose {
            r: look_rotation(0.0, (-EYE).atan2(1.0)),
            t: Vec3::new(0.0, 3.2 + EYE, 0.0),
        };
        let hit = p.preview(&pose).expect("the floor is below the camera");
        assert!((hit.distance_m - 1.0).abs() < 1e-3, "{}", hit.distance_m);
    }

    /// A mark at `position` known to `sigma`, for the fitting tests.
    fn at(position: Point2, sigma_m: f32) -> Mark {
        Mark {
            kind: MarkKind::WallPoint,
            position,
            sigma_m,
            from: Vec3::ZERO,
            inferred: false,
        }
    }

    #[test]
    fn fitting_a_line_needs_two_distinct_points() {
        assert!(fit_line(&[at((1.0, 1.0), 0.02)]).is_none());
        assert!(fit_line(&[at((1.0, 1.0), 0.02), at((1.0, 1.0), 0.02)]).is_none());
        let l = fit_line(&[
            at((0.0, 0.0), 0.02),
            at((1.0, 0.0), 0.02),
            at((2.0, 0.0), 0.02),
        ])
        .unwrap();
        assert!(
            l.direction.1.abs() < 1e-5,
            "a wall along x should point along x"
        );
    }

    #[test]
    fn a_far_mark_does_not_drag_a_line_its_close_neighbours_pinned_down() {
        // Two marks at 2 cm and one 20 cm mark 10 cm off the wall. Unweighted,
        // the centroid moves 3.3 cm and the line tilts; weighted by 1/sigma^2
        // the far mark carries a hundredth of the weight.
        let good = [at((0.0, 0.0), 0.02), at((2.0, 0.0), 0.02)];
        let with_far = [
            at((0.0, 0.0), 0.02),
            at((2.0, 0.0), 0.02),
            at((4.0, 0.10), 0.20),
        ];
        let a = fit_line(&good).unwrap();
        let b = fit_line(&with_far).unwrap();
        let tilt = (a.direction.1 - b.direction.1).abs();
        assert!(tilt < 0.01, "the far mark tilted the wall by {tilt}");
        assert!(b.point.1.abs() < 0.01, "and shifted it to {:?}", b.point);
    }

    #[test]
    fn near_parallel_walls_do_not_produce_a_corner_on_the_horizon() {
        let a = Line2 {
            point: (0.0, 0.0),
            direction: (1.0, 0.0),
        };
        let b = Line2 {
            point: (0.0, 1.0),
            direction: (1.0, 0.001),
        };
        assert!(intersect(a, b).is_none());
        let c = Line2 {
            point: (0.0, 1.0),
            direction: (0.0, 1.0),
        };
        assert!(intersect(a, c).is_some());
    }
}
