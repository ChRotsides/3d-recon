//! Next-best-station solver (plan §6.1).
//!
//! Explicitly *not* an RL next-best-view method. Those solve a harder problem —
//! an autonomous agent with unconstrained 6-DoF placement. We have a human who
//! must physically stand somewhere on the floor, at roughly a fixed height, and
//! who will not walk into a wall. That reduces the search to a direct
//! information-gain scan over floor positions:
//!
//! 1. Sample candidates on the floor plane on a 40 cm grid, >=60 cm from any
//!    wall, inside already-observed free space.
//! 2. At standing height, integrate expected coverage gain over a full yaw
//!    sweep.
//! 3. Penalise walking distance: `score = gain - lambda * distance`.
//! 4. Emit the best candidate as an AR floor ring.

use crate::coverage::{CoverageGrid, Voxel, VoxelKey};
use crate::math::Vec3;

#[derive(Debug, Clone, Copy)]
pub struct StationParams {
    pub grid_m: f32,
    pub wall_clearance_m: f32,
    pub standing_height_m: f32,
    /// Walking-distance penalty per metre.
    pub lambda: f32,
    /// Ignore voxels beyond this from a candidate — they will not be observed
    /// well from there anyway.
    pub max_useful_range_m: f32,
    pub min_useful_range_m: f32,
    /// Stop suggesting stations once the best one offers less than this
    /// fraction of the total remaining gain.
    pub min_gain_fraction: f32,
    pub max_candidates: usize,
}

impl Default for StationParams {
    fn default() -> Self {
        StationParams {
            grid_m: 0.40,
            wall_clearance_m: 0.60,
            standing_height_m: 1.50,
            lambda: 0.30,
            max_useful_range_m: 4.0,
            min_useful_range_m: 0.5,
            min_gain_fraction: 0.02,
            max_candidates: 400,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Station {
    pub position: Vec3,
    pub expected_gain: f32,
    pub walk_distance_m: f32,
    pub score: f32,
}

/// Suggest where the user should stand next.
///
/// `floor_y` is the floor height in world units and `up` the world up axis.
/// Returns `None` when no candidate offers meaningful gain — which is one of
/// the room-complete conditions.
pub fn next_best_station(
    grid: &CoverageGrid,
    user_position: Vec3,
    floor_y: f32,
    up: Vec3,
    params: StationParams,
) -> Option<Station> {
    let uncovered = grid.uncovered_voxels();
    if uncovered.is_empty() {
        return None;
    }
    // Plan §6.1 stops when no station offers gain above 2% of the *total*
    // volume. Measuring against the remaining volume instead would never
    // terminate: as the room fills in, the bar it has to clear shrinks with it,
    // so the last few unreachable corners would keep the capture open forever.
    let total_volume: f32 = grid.len() as f32;

    let candidates = sample_floor_candidates(grid, floor_y, up, params);
    if candidates.is_empty() {
        return None;
    }

    let up = up.normalized();
    let mut best: Option<Station> = None;
    for c in candidates {
        let eye = c + up * params.standing_height_m;
        let gain = expected_gain(&uncovered, eye, grid, params);
        let walk = c.distance(user_position);
        let score = gain - params.lambda * walk;
        if best.is_none() || score > best.unwrap().score {
            best = Some(Station {
                position: c,
                expected_gain: gain,
                walk_distance_m: walk,
                score,
            });
        }
    }

    let b = best?;
    if b.expected_gain < params.min_gain_fraction * total_volume {
        return None;
    }
    Some(b)
}

/// Expected coverage gain from a full yaw sweep at `eye`.
///
/// A full sweep sees every direction, so instead of ray-casting per yaw we sum
/// the per-voxel gain over everything in range. Visibility is approximated by
/// the observed surface normal facing the candidate — a cheap stand-in for
/// ray-casting the partial mesh, and one that cannot send the user somewhere
/// that only *looks* productive through a wall, because a voxel behind a wall
/// has its normal pointing away.
fn expected_gain(
    uncovered: &[(VoxelKey, Voxel)],
    eye: Vec3,
    grid: &CoverageGrid,
    params: StationParams,
) -> f32 {
    let mut gain = 0.0f32;
    let p = grid.params();
    for (key, vox) in uncovered {
        let centre = grid.centre_of(*key);
        let to_eye = eye - centre;
        let d = to_eye.length();
        if !(params.min_useful_range_m..=params.max_useful_range_m).contains(&d) {
            continue;
        }
        let dir = to_eye * (1.0 / d);
        let n = vox.normal.normalized();
        let cos_theta = n.dot(dir);
        if cos_theta <= 0.0 {
            continue;
        }
        let w_angle = cos_theta * cos_theta;
        let potential = w_angle * grid.distance_weight(d);
        // Only the shortfall to the quality bar counts as gain — re-observing a
        // voxel that is nearly covered is worth very little.
        let deficit = (p.q_min - vox.quality).max(0.0) / p.q_min;
        gain += potential * deficit;
    }
    gain
}

/// Floor positions a human could actually stand on.
pub fn sample_floor_candidates(
    grid: &CoverageGrid,
    floor_y: f32,
    up: Vec3,
    params: StationParams,
) -> Vec<Vec3> {
    let (min, max) = grid.bounds();
    if grid.is_empty() {
        return Vec::new();
    }
    let up = up.normalized();
    let e0 = crate::coverage::pick_tangent(up);
    let e1 = up.cross(e0);

    // Wall voxels, projected to the floor plane, define the obstacles.
    let mut walls: Vec<(f32, f32)> = Vec::new();
    for (key, vox) in grid.all_voxels() {
        let n = vox.normal.normalized();
        if n.dot(up).abs() > 0.5 {
            continue; // floor or ceiling
        }
        let c = grid.centre_of(key);
        walls.push((c.dot(e0), c.dot(e1)));
    }
    if walls.is_empty() {
        return Vec::new();
    }

    let (mut lo0, mut hi0) = (f32::MAX, f32::MIN);
    let (mut lo1, mut hi1) = (f32::MAX, f32::MIN);
    for (a, b) in &walls {
        lo0 = lo0.min(*a);
        hi0 = hi0.max(*a);
        lo1 = lo1.min(*b);
        hi1 = hi1.max(*b);
    }
    let _ = (min, max);

    let mut out = Vec::new();
    let step = params.grid_m;
    let clearance_sq = params.wall_clearance_m * params.wall_clearance_m;
    let mut a = lo0;
    while a <= hi0 && out.len() < params.max_candidates {
        let mut b = lo1;
        while b <= hi1 && out.len() < params.max_candidates {
            let far_enough = walls
                .iter()
                .all(|(wa, wb)| {
                    let da = a - wa;
                    let db = b - wb;
                    da * da + db * db >= clearance_sq
                });
            if far_enough {
                out.push(e0 * a + e1 * b + up * floor_y);
            }
            b += step;
        }
        a += step;
    }
    out
}
