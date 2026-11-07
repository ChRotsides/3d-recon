# Persistent map recovery and depth quality

The 2D exploration view beside the orbit view uses light-grey observed space,
black unknown space, a green map-position dot and white wall outlines. A horizontal
slice within 60 cm of camera height uses cross-view-confirmed depth rays to reveal
space only as far as measured surfaces. Approximately vertical surface normals
need two supported observations before receiving a white wall mark. Furniture can
also create vertical boundaries; these are observed surfaces, not an architectural
floor-plan classification. The initial grid is 10 cm; it coarsens uniformly above
60,000 discovered cells. Pose and explored space share the persistent map frame.

An estimator restart with neither geometry nor saved references now starts an
empty aligned frame, allowing the first pose/reference to form. Existing geometry
or references still require relocalization. This fixes the empty-map deadlock in
which the UI said TRACKING but pose publication and first depth were blocked.
RELOCALIZING now appears when local tracking succeeds but map alignment is pending.
The on-phone JNI regression checks both empty restart and geometry retention;
69 JVM tests pass including exploration boundaries and wall-support checks.

Brief tracking-quality dips pause fusion without invalidating the existing rigid
map transform. Healthy tracking in the same estimator resumes pose publication
and depth directly. Explicit pose discontinuities and tracker restarts require
verified relocalization. Previously every rejected frame triggered relocalization,
which locked a six-frame physical scan despite later healthy tracking. The phone
regression now checks brief-dip continuity and true-discontinuity rejection.
The camera overlay displays at most 180 ranked dots; this only reduces visual
clutter and does not remove measurements from the tracker.

Automatic recovery retains the current native TSDF and up to 32 feature keyframes.
The local OpenVINS estimator can restart after a three-second health failure, at
most once per 15 seconds. The existing cloud stays visible and fusion pauses.
Return to a saved view and translate sideways at least 15 cm to check alignment
and scale. Explicit RESET still saves the old cloud and starts an independent map.

XFeat descriptors are requested at 1 Hz during normal tracking and up to 2.5 Hz
during recovery, within the existing adaptive inference budget. A keyframe needs
20 descriptor-to-landmark associations across six ray-grid cells. Matching uses
mutual nearest neighbours, a ratio test, PnP RANSAC, and reprojection checks.
Three consistent matches spanning at least 0.4 seconds, plus a translated baseline,
are required. Tilt disagreement and inconsistent translation scale are rejected.
Only a gravity-preserving rigid transform is applied to the published camera pose
and landmarks; the local EKF state and covariance remain internally consistent.
Depth processing starts as soon as local tracking is healthy and map alignment is
valid; it does not wait for a recovery keyframe. Descriptors for reference creation
are sampled at projected tracked-landmark locations, rather than requiring a new
detector peak to land within three pixels of a KLT track. If a retained map loses
alignment and no reference can be matched, the cloud remains held; the UI explains
how to recover or reset. Depth attempt/integration/support counters expose the
actual pipeline progress, including provisional depth awaiting a second view.

Relative depth requires at least eight agreeing feature anchors across four of
twelve image regions, with at least 65% agreement. Scale or inverse-depth affine
fits use robust hypotheses and inlier refinement. Inverse fits also require depth
variation. Metric predictions are checked against available feature anchors.
There is no reuse of old scale through missing or contradictory evidence.

Depth is reprojected from a previous translated view with a nearest-surface
z-buffer. Agreement within max(6 cm, 2.5% of depth) gives full weight. Newly seen
or occluding pixels remain provisional; contradictory farther surfaces are
rejected. Provisional observations cannot alter confirmed surfaces, carve free
space, or accumulate enough weight to become confirmed by themselves. They stay
in the TSDF awaiting additional evidence. Invalid borders and depth discontinuities
are excluded during both block allocation and voxel updates.

The displayed and exported surface uses one existing, best-supported sample per
10 cm spatial cell. It does not average across adjacent surfaces. If more than
100,000 cells remain, all cells are coarsened together until they fit. Every occupied
coarse region contributes; no arbitrary prefix of hash blocks is selected. This
reduces output density, not the internal TSDF voxel resolution. Unsupported areas
need another view before appearing in the confirmed cloud. Raw comparison mode
remains an intentionally unfiltered diagnostic view.

## Evidence

`Android/data/com.recon.videoscan/files/depth-evidence/map-*/depth-*.bin.gz` captures
each completed depth prediction (including rejected fits), its exact timestamp,
estimator generation, map transform revision, model, fusion decision/reason, pose,
source and rectified intrinsics, distortion model, feature landmarks, raw prediction,
corrected millimetre depth, confidence mask, rectified RGBA and tracking grayscale.
Frame CSV poses remain in the local estimator frame; depth evidence poses are in
the persistent map frame. Do not mix these coordinate frames without alignment.

The asynchronous queue holds two frames. A full queue drops evidence, not tracking;
the summary records drops. Recording stops at a conservative 256 MiB uncompressed
budget per map; new sessions stop being recorded after the evidence root reaches
1 GiB on disk. Existing captures are never deleted. Automatic tracker restarts
share the map evidence folder. Pausing the activity closes the writer.

Decode a frame with `python tools/read_depth_evidence.py FILE --output /tmp/frame`.
It writes JSON metadata and NPZ arrays. Binary version 1 is gzip-compressed Java
DataOutputStream (big endian): magic DPTH, version, time(double), generation(long),
revision(long), model/reason(writeUTF), fused(boolean), four dimensions(int), then
length-prefixed pose(float), K(float), source calibration(double), landmarks(float),
prediction(float), depth(unsigned short), confidence/RGBA/gray(bytes). Confidence
values are 0 rejected, 64 provisional, 255 cross-view supported. Rejected fits can
have empty depth/confidence arrays.

## Limits

This is relocalization and conservative fusion, not bundle adjustment or a plane
constraint that forces a room to be square. Already fused drift is not undone.
Consistent monocular model bias can survive the cross-view test. A fresh physical
scan, with wall straightness measurements and recorded depth evidence, is needed
to validate the room geometry and phone performance. Existing recordings cannot
establish whether model distortion, pose drift, or changing depth scale dominated.

## Verification (2026-09-06)

- Debug APK built successfully; 66 JVM tests passed, including robust depth fitting,
  translated-view confidence, evidence serialization and quota preservation.
- `tsdf_quality_test` passed on the host: provisional surfaces stay unconfirmed,
  supported plane depth stays accurate, and spatial coarsening preserves separate
  left/centre/right regions under a tight point budget.
- `map_relocalizer_test` passed on the connected SM-S918B: known map transform
  recovered after repeated matches; unrelated descriptors and wrong tilt rejected.
- Python evidence decoder passed a binary round-trip check.
- The Activity recovery instrumentation was updated for map retention and compiled;
  it was not run in this validation. The application APK was not installed or launched.
- These are synthetic checks; physical scan accuracy and sustained performance
  remain unmeasured for this change.
