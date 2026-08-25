//! Coverage voxel grid (plan §6.1).
//!
//! ```text
//! q(v) = sum_kf  w_angle(v,kf) * w_dist(v,kf) * w_blur(kf)
//! w_angle = max(0, cos theta)^2
//! w_dist  = clamp(1 - |d - d_opt| / d_range, 0, 1)
//! w_blur  = sigmoid(blur_score)
//! ```
//!
//! The grid is sparse (hash map keyed by voxel index) rather than dense. A dense
//! 5 cm grid over a generous room bounding box is ~1 M voxels, which is fine for
//! memory but wasteful to walk every frame on a phone — and the room's extent is
//! not known when the session starts.

use std::collections::HashMap;

use crate::math::Vec3;

pub const DEFAULT_VOXEL_M: f32 = 0.05;

#[derive(Debug, Clone, Copy)]
pub struct CoverageParams {
    pub voxel_m: f32,
    /// Optimal observation distance, plan §6.1.
    pub d_opt_m: f32,
    pub d_range_m: f32,
    /// Below this, coverage falls off — too close means no parallax and a
    /// sliver of the room per frame.
    pub d_near_full_m: f32,
    pub d_near_zero_m: f32,
    /// A voxel counts as covered at this quality, ~= two decent views.
    pub q_min: f32,
    /// Blur score (variance of Laplacian) at which w_blur == 0.5.
    pub blur_midpoint: f32,
    pub blur_steepness: f32,
}

impl Default for CoverageParams {
    fn default() -> Self {
        CoverageParams {
            voxel_m: DEFAULT_VOXEL_M,
            d_opt_m: 2.0,
            d_range_m: 2.5,
            d_near_full_m: 0.8,
            d_near_zero_m: 0.3,
            // Plan §6.1 starts this at 2.0 and says to tune it in Phase 3. It
            // cannot stay at 2.0: w_angle, w_dist and w_blur are each strictly
            // below 1, so *no* pair of views can ever sum to 2.0 and every
            // voxel would read as uncovered forever. 1.8 leaves two good views
            // real headroom. See docs/execution-status.md.
            q_min: 1.8,
            blur_midpoint: 120.0,
            blur_steepness: 0.03,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VoxelKey(pub i32, pub i32, pub i32);

#[derive(Debug, Clone, Copy, Default)]
pub struct Voxel {
    pub quality: f32,
    pub observations: u32,
    /// Running mean of the observed surface normal, used by the layout hints
    /// and by the wall-direction completeness check.
    pub normal: Vec3,
}

#[derive(Debug, Clone)]
pub struct CoverageGrid {
    params: CoverageParams,
    voxels: HashMap<VoxelKey, Voxel>,
    min: Vec3,
    max: Vec3,
    initialised: bool,
}

impl CoverageGrid {
    pub fn new(params: CoverageParams) -> Self {
        CoverageGrid {
            params,
            voxels: HashMap::new(),
            min: Vec3::ZERO,
            max: Vec3::ZERO,
            initialised: false,
        }
    }

    pub fn params(&self) -> CoverageParams {
        self.params
    }

    pub fn len(&self) -> usize {
        self.voxels.len()
    }

    pub fn is_empty(&self) -> bool {
        self.voxels.is_empty()
    }

    pub fn key_of(&self, p: Vec3) -> VoxelKey {
        let v = self.params.voxel_m;
        VoxelKey(
            (p.x / v).floor() as i32,
            (p.y / v).floor() as i32,
            (p.z / v).floor() as i32,
        )
    }

    pub fn centre_of(&self, k: VoxelKey) -> Vec3 {
        let v = self.params.voxel_m;
        Vec3::new(
            (k.0 as f32 + 0.5) * v,
            (k.1 as f32 + 0.5) * v,
            (k.2 as f32 + 0.5) * v,
        )
    }

    pub fn get(&self, k: VoxelKey) -> Option<&Voxel> {
        self.voxels.get(&k)
    }

    pub fn bounds(&self) -> (Vec3, Vec3) {
        (self.min, self.max)
    }

    fn blur_weight(&self, blur_score: f32) -> f32 {
        let z = (blur_score - self.params.blur_midpoint) * self.params.blur_steepness;
        1.0 / (1.0 + (-z).exp())
    }

    /// Distance weight, asymmetric about `d_opt`.
    ///
    /// Plan §6.1 writes this symmetrically as
    /// `clamp(1 - |d - d_opt| / d_range, 0, 1)`. Taken literally that is wrong
    /// for indoor capture: with d_opt = 2 m, a 2.45 m ceiling viewed from a
    /// 1.5 m eye height sits 0.95 m away and scores 0.53, so a ceiling could
    /// never reach the coverage bar — `CEILING_MISSING` would nag forever and
    /// room-complete would never fire. Depth is not *less* reliable at 1 m than
    /// at 2 m; only very close range genuinely hurts, by killing parallax and
    /// shrinking the footprint per frame. So the near side is flat down to
    /// `d_near_full_m` and only then falls away.
    pub fn distance_weight(&self, d: f32) -> f32 {
        let p = self.params;
        if d <= p.d_near_zero_m {
            0.0
        } else if d < p.d_near_full_m {
            (d - p.d_near_zero_m) / (p.d_near_full_m - p.d_near_zero_m)
        } else if d <= p.d_opt_m {
            1.0
        } else {
            (1.0 - (d - p.d_opt_m) / p.d_range_m).clamp(0.0, 1.0)
        }
    }

    /// Integrate one observed surface sample.
    ///
    /// `point` is the world-space surface position, `normal` its outward
    /// normal, `camera` the observing camera position.
    pub fn observe(&mut self, point: Vec3, normal: Vec3, camera: Vec3, blur_score: f32) {
        let to_cam = camera - point;
        let d = to_cam.length();
        if d < 1e-4 {
            return;
        }
        let dir = to_cam * (1.0 / d);

        let cos_theta = normal.normalized().dot(dir);
        if cos_theta <= 0.0 {
            return; // observing the back face; contributes nothing
        }
        let w_angle = cos_theta * cos_theta;

        let w_dist = self.distance_weight(d);
        if w_dist <= 0.0 {
            return;
        }
        let w_blur = self.blur_weight(blur_score);
        let q = w_angle * w_dist * w_blur;
        if q <= 0.0 {
            return;
        }

        if !self.initialised {
            self.min = point;
            self.max = point;
            self.initialised = true;
        } else {
            self.min = Vec3::new(
                self.min.x.min(point.x),
                self.min.y.min(point.y),
                self.min.z.min(point.z),
            );
            self.max = Vec3::new(
                self.max.x.max(point.x),
                self.max.y.max(point.y),
                self.max.z.max(point.z),
            );
        }

        let key = self.key_of(point);
        let v = self.voxels.entry(key).or_default();
        v.quality += q;
        v.observations += 1;
        v.normal = v.normal + normal.normalized();
    }

    /// How much of what this frame sees is still short of the quality bar.
    ///
    /// Returns `(novel_fraction, contributing_samples)`. A sample contributes
    /// only if it would actually change the grid — front-facing, in range, not
    /// hopelessly blurred — which is the same test [`Self::observe`] applies, so
    /// a frame is judged on the geometry it would really add rather than on
    /// every pixel it happens to contain. The count comes back too because a
    /// verdict from nine samples is not a verdict.
    ///
    /// Deliberately *not* the same thing as the station solver's expected gain:
    /// that ranks places a user could walk to, over the whole grid. This asks a
    /// narrower question about one frame that already exists, so it needs no
    /// visibility approximation — the samples themselves are the evidence.
    pub fn novel_fraction(
        &self,
        points: &[Vec3],
        normals: &[Vec3],
        camera: Vec3,
        blur_score: f32,
    ) -> (f32, usize) {
        let w_blur = self.blur_weight(blur_score);
        let mut contributing = 0usize;
        let mut novel = 0usize;
        for (p, n) in points.iter().zip(normals.iter()) {
            let to_cam = camera - *p;
            let d = to_cam.length();
            if d < 1e-4 {
                continue;
            }
            let cos_theta = n.normalized().dot(to_cam * (1.0 / d));
            if cos_theta <= 0.0 {
                continue;
            }
            if self.distance_weight(d) <= 0.0 {
                continue;
            }
            if cos_theta * cos_theta * self.distance_weight(d) * w_blur <= 0.0 {
                continue;
            }
            contributing += 1;
            let short_of_the_bar = self
                .voxels
                .get(&self.key_of(*p))
                .is_none_or(|v| v.quality < self.params.q_min);
            if short_of_the_bar {
                novel += 1;
            }
        }
        if contributing == 0 {
            return (0.0, 0);
        }
        (novel as f32 / contributing as f32, contributing)
    }

    pub fn is_covered(&self, k: VoxelKey) -> bool {
        self.voxels.get(&k).is_some_and(|v| v.quality >= self.params.q_min)
    }

    pub fn covered_count(&self) -> usize {
        self.voxels.values().filter(|v| v.quality >= self.params.q_min).count()
    }

    /// Fraction of *observed* voxels that reach the quality bar.
    ///
    /// Deliberately not "fraction of the room", because the room's true extent
    /// is unknown mid-capture — reporting against a guess would make the
    /// progress bar jump backwards every time a new corner comes into view.
    pub fn coverage_ratio(&self) -> f32 {
        if self.voxels.is_empty() {
            return 0.0;
        }
        self.covered_count() as f32 / self.voxels.len() as f32
    }

    /// Coverage restricted to surfaces whose normal points a given way, used
    /// for the per-surface targets in the room-complete rule.
    pub fn directional_coverage(&self, axis: Vec3, min_cos: f32) -> f32 {
        let axis = axis.normalized();
        let mut total = 0usize;
        let mut covered = 0usize;
        for v in self.voxels.values() {
            let n = v.normal.normalized();
            if n.dot(axis) >= min_cos {
                total += 1;
                if v.quality >= self.params.q_min {
                    covered += 1;
                }
            }
        }
        if total == 0 {
            0.0
        } else {
            covered as f32 / total as f32
        }
    }

    /// Coverage of the four horizontal quadrants of wall direction. Returns the
    /// worst of them, which is what the plan's "all four wall directions >=90%"
    /// rule needs.
    pub fn worst_wall_direction_coverage(&self, up: Vec3) -> f32 {
        let up = up.normalized();
        let mut buckets = [(0usize, 0usize); 4];
        for v in self.voxels.values() {
            let n = v.normal.normalized();
            let horizontal = n - up * n.dot(up);
            if horizontal.length() < 0.5 {
                continue; // floor or ceiling
            }
            let h = horizontal.normalized();
            // Any orthonormal pair spanning the floor plane works; the buckets
            // only need to be consistent, not aligned to the room.
            let e0 = pick_tangent(up);
            let e1 = up.cross(e0);
            let angle = h.dot(e1).atan2(h.dot(e0));
            let idx = (((angle + std::f32::consts::PI)
                / (std::f32::consts::PI * 0.5))
                .floor() as usize)
                .min(3);
            buckets[idx].0 += 1;
            if v.quality >= self.params.q_min {
                buckets[idx].1 += 1;
            }
        }
        let mut worst = 1.0f32;
        let mut any = false;
        for (total, covered) in buckets {
            if total < 10 {
                continue; // an unseen direction is handled by overall coverage
            }
            any = true;
            worst = worst.min(covered as f32 / total as f32);
        }
        if any {
            worst
        } else {
            0.0
        }
    }

    pub fn uncovered_voxels(&self) -> Vec<(VoxelKey, Voxel)> {
        self.voxels
            .iter()
            .filter(|(_, v)| v.quality < self.params.q_min)
            .map(|(k, v)| (*k, *v))
            .collect()
    }

    pub fn all_voxels(&self) -> Vec<(VoxelKey, Voxel)> {
        self.voxels.iter().map(|(k, v)| (*k, *v)).collect()
    }
}

pub fn pick_tangent(up: Vec3) -> Vec3 {
    let tmp = if up.x.abs() > 0.9 {
        Vec3::new(0.0, 0.0, 1.0)
    } else {
        Vec3::new(1.0, 0.0, 0.0)
    };
    (tmp - up * tmp.dot(up)).normalized()
}
