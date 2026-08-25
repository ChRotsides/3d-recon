# Capture to point cloud, the path that worked

> **Current best RGB-only result:** the later feature-stitch → wrapped panorama
> → six-face virtual cubemap → Depth Anything workflow is recorded exactly in
> [`image-only-panorama-cubemap-point-cloud.md`](image-only-panorama-cubemap-point-cloud.md).
> The document below describes the earlier ARCore-assisted experiment and is
> retained as historical evidence.

Written 2026-08-09, from the first run that produced a usable 360 panorama and a
recognisable 3D point cloud of a real room on this hardware. It records what was
done, the numbers it produced, and — more usefully — the two places where the
obvious approach failed and what replaced it.

The subject capture is `bundle_e4a5da81-4e98-45a5-86e6-73a68f7ef5f2`: one room,
Samsung S23 Ultra, 27 keyframes, ~2 minutes.

---

## 1. The capture

The single change that made this work was **removing every gate from the
shutter**. Before it, the app decided when to take a picture and refused most
frames: one measured capture kept **8 keyframes out of 1,843 frames judged**, a
0.4% accept rate, because the user was being asked to satisfy an aim tolerance
(17°), a steadiness threshold (0.35 rad/s), a blur floor (100), an exposure
ceiling (0.25) and a redundancy veto — all at once, by hand.

Now the script draws a target square and the user presses **Take picture**.
Nothing can refuse the press. A poor frame is scored a million below a good one
so it never displaces a better candidate, but it is always available to be
taken. The core keeps the sharpest frame of the last ~1 second so the press
lands on that rather than on the smear the finger made, which removes shutter
lag without removing the decision from the user.

What that produced, first try:

| | |
|---|---|
| keyframes | 27 |
| capture mode | `arcore_depth`, 27/27 depth frames |
| yaw coverage | 9 directions × 3 heights, 16° → 356° — a full circle |
| pitch range | −40° … +25° |
| max baseline | 1.23 m |
| blur | min **9**, median 130, max 697 |
| skipped shots | 0 |

The blur minimum matters: a frame scoring 9 was kept. The old gate discarded
anything under 100. It was the only view of that direction, and a soft picture
of a wall is worth incomparably more than no picture of it.

**For a panorama, turn on the spot.** Stitching wants one centre of projection,
so rotation is right and translation is the enemy — the exact opposite of what
reconstruction needs. The two goals genuinely pull in opposite directions and
one capture cannot be optimal for both.

## 2. Pull the bundle

```bash
adb pull /sdcard/Android/data/com.recon.poselogger/files/captures/bundle_<uuid> \
    data/captures/
```

Over the WireGuard tunnel this runs at ~0.2 MB/s; a 27-frame bundle is ~1.8 MB.

## 3. The 360 — project by pose, not by feature matching

**OpenCV's stitcher could not do this.** `cv2.Stitcher_PANORAMA` assumes every
frame shares one centre of projection and silently discards whatever it cannot
reconcile. On these 27 frames it kept a handful and produced a 1252×893 crop —
a corner of the room, not a sphere.

Two things were learned fighting it:

- **Strict-first on the confidence threshold.** Loosening
  `setPanoConfidenceThresh` to 0.6 to be "more permissive" made things *worse*:
  it admits marginal pairs into the bundle adjustment, the camera-rotation solve
  diverges, and LAPACK floods stderr with `DLASCLS` errors. Three captures that
  "failed to stitch" succeeded at the default 1.0. One of them went from
  1188×927 at conf 0.6 to **3089×1546** at conf 1.0 — the loose threshold had
  been discarding most of the capture.
- **It still cannot close a 360 with any threshold**, because the operator
  translated 1.23 m and the rotation-only model cannot absorb that.

The replacement ignores feature matching entirely and uses the poses the bundle
already records. ARCore's *rotation* is good, and translation barely affects
direction:

> For every pixel of an equirectangular output, form the world ray, transform it
> into each camera by that frame's pose, project with that frame's intrinsics,
> and blend every frame that can see it — weighted by distance from the frame
> edge (feather) and by sharpness, so a soft frame loses where it overlaps a
> crisp one.

Nothing is discarded for failing to match. Same principle as the shutter: use
what was captured rather than refuse it.

```
27 keyframes -> 4096x2048 equirectangular
27 frames contributed · 72.0% of the sphere covered
wrote equirect_e4a5da81.jpg  4096x1587
```

**All 27 contributed.** The empty poles are cropped, because a hand-held sweep
produces a band around the horizon and leaving black caps makes it look like a
failure rather than like what it is. 72% rather than 100% is exactly the −40°…
+25° pitch range running out.

Convention, from PLAN.md §5 — right-handed, Y-up, world←camera poses, camera
looking down −Z with +X right and +Y up. Image rows run downwards, hence the
sign flip on Y when projecting.

> **Ghosting is expected and is the parallax.** Distant walls and ceiling align
> cleanly; the near sofa and the door frame appear twice and semi-transparent.
> Projection by rotation assumes the camera never moved, and it moved 1.23 m.
> Parallax falls off with distance, so near objects ghost and far ones do not.

## 4. The point cloud — DA3, merged the same way

```bash
source ~/ai/env.sh kepler
python scripts/capture/da3_cloud.py \
    data/captures/bundle_<uuid> out/cloud.bin small 900000
```

DA3-Small runs on the GTX TITANs in **6.7 s for 27 frames** (FP32; see
`docs/` on the Kepler environment). Each frame's depth is back-projected through
its own intrinsics and placed by its own pose — merged exactly like the
panorama, nothing rejected for disagreeing with neighbours.

```
899,991 points from 27 frames
890,991 after trimming the far 1%
cloud.bin 13.4 MB
extent 11.0 × 5.38 × 14.36 m
```

Depth is sampled at the model's working resolution and the **image is sampled on
the same grid** rather than the depth being resized — resizing depth interpolates
across discontinuities and smears object edges out into space.

The far 1% is trimmed because monocular depth invents distance beyond windows
and doorways, and a handful of runaway points would otherwise set the viewer's
entire scale.

## 5. Viewers

Both are dependency-free single files, served by `python -m http.server`.

- **`viewer/panorama.html`** — panorama viewer. Canvas-based pan and zoom, reads a
  `manifest.json` the stitcher writes so new captures appear without an HTML
  edit. Wrapping past the seam is declared per-source (`"wrap": true`) rather
  than guessed from aspect ratio — this 360 is 2.58:1 after pole-cropping and
  would have just missed a 2.6 threshold.
- **`viewer/cloud.html`** — WebGL point cloud. Orbit, zoom, pan, a point-size slider in
  millimetres, and a **texture on/off** toggle. Texture off shades by height,
  which reads structure far better once colour stops competing with shape.

The cloud arrives as one flat buffer — `float32 xyz` for every point, then
`uint8 rgb` — and goes straight into GPU buffers. At ~900k points a PLY text
parse would cost more than the download. Layout check:
`890,991 × 12 + 890,991 × 3 = 13,364,865` = the file size exactly.

> **Point size must be a physical radius.** The first version used
> `gl_PointSize = uSize * 300.0 / w`, an arbitrary constant that made every point
> ~62 px across at a 12 m view — nearly a million overlapping blobs, i.e. a solid
> wall of colour. It is now `radius_metres × focal_px / depth`, so a point is a
> disc of a stated size in the room and keeps it through zoom and window resize.

## 6. What the cloud gets wrong, measured

The extent — 11.0 × 5.4 × 14.4 m — is not this room. Every frame carries a real
metric ARCore depth map, which makes a decisive instrument: is DA3's error one
global scale, a per-frame scale, or shape?

```
per-frame scale (ARCore / DA3), 27 frames
  median 1.025            <- global scale is already right, within 2.5%
  min 0.571   max 1.349   <- frames disagree with each other by 2.36x
  after one global scale: 7.7% median residual, 29.6% p90, 44.3% worst
  within-frame ratio spread (68% band), median: 0.542
```

Three conclusions:

1. **Global scale is not the problem.** The existing sparse-point anchoring gets
   the overall metre right to 2.5%.
2. **Per-frame scale is.** `kf_0007` predicts 4.89 m where ARCore measures 2.84 —
   1.75× too far, with a *low* within-frame spread of 0.254, so that frame is
   uniformly stretched. Each frame is anchored independently against a few sparse
   points, and 1.23 m of baseline makes those anchors weak.
3. **The larger error is inside each frame.** The within-frame spread (0.542)
   exceeds the between-frame residual (7.7%). DA3's depth *shape* disagrees with
   ARCore, not only its scale — `kf_0005` spreads 1.473. **No single multiplier
   per frame can fix that.**

### The ranked fix list this implies

1. **Robust per-frame affine fit to ARCore depth.** Monocular models are
   affine-invariant — `d_metric ≈ a·d_pred + b` — and only `a` is being fitted,
   against sparse points, when both could be fitted against hundreds of thousands
   of measured pixels. Removes the 2.36× frame disagreement *and* much of the
   within-frame bend. Biggest win, cheapest.
2. **Consistency rejection.** Drop DA3 pixels that still disagree with ARCore
   beyond tolerance; keep DA3 where ARCore has no return, which is the whole
   value of having it — dense and sharp where the sensor is blank.
3. **Flying-pixel removal.** Discard points on steep local depth gradients, the
   comet-tails stretched along view rays at object edges.
4. **Voxel downsample.** One point per ~1 cm cell: cleans the pile-up where many
   frames hit the same surface, and cuts the 13.4 MB at the same time.

None of these were applied to the cloud described above. It is the unfiltered
result.

---

## 6a. Fix 1 applied, and what it revealed — 2026-08-10

Fix 1 is implemented in `recon/depth/anchor.py` and on by default
(`DA3Config.anchor_to_reference_depth`). It does what §6 predicted, and in doing
so it disproves §6's headline claim.

### It works, on every metric it was aimed at

Per frame, `d_metric ≈ a·d_pred + b` fitted robustly over every measured pixel
— Theil–Sen start, then Huber-reweighted least squares. Both variants, same 27
frames:

| | small before | small after | large before | large after |
|---|---|---|---|---|
| frame-to-frame scale spread | 2.36× | **1.11×** | 2.58× | **1.08×** |
| within-frame spread | 0.541 | **0.228** | 0.445 | **0.201** |
| median absolute error | 0.358 m | **0.184 m** | 0.579 m | **0.156 m** |
| residual after one global scale | 7.7% | **0.8%** | 15.1% | **0.9%** |

Held-out check, fitting on alternating 8×8 tiles and scoring on all pixels:
identical to four significant figures. Two parameters cannot memorise 300k
constraints, and now that is shown rather than argued.

Disparity space scores slightly better than depth space (large: 0.183 vs 0.201
spread, 0.135 vs 0.156 m). It is **not** the default, because `1/(a/d + b)`
silently caps depth at `1/b` — about 7.4 m on this capture. That flatters the
error on a small room and would quietly truncate a corridor.

### Two frames in six wanted to lie

Six of 27 frames fitted `a` between 0.012 and 0.16 with `b` near 2.7 m — the
line collapsing to *discard the prediction and place the whole frame at one
constant depth*. Every one **lowered** its residual by doing so (kf_0007:
0.79 m → 0.15 m). Scored against the sensor it is an improvement; as geometry it
is a flat wall where a room was, and a residual test cannot tell the difference.

Hence `MIN_CORRELATION`: a frame is corrected only if the prediction and the
measurement agree about its *shape* first, independent of scale. That recovers
23 of 27 and refuses the rest, which keep their sparse anchor.

### The extent was never a depth problem

The cloud still comes out ~10.5 × 6.9 × 11.7 m. The control settles why —
build the cloud from **ARCore's measured depth alone, no model at all**:

```
ARCore-only cloud       8,294,400 pts   extent 10.46 × 6.02 × 11.58 m
DA3-large + anchor      8,294,400 pts   extent 10.51 × 6.91 × 11.72 m
```

**The measurement §6 used as ground truth produces the same extent as the model
it was condemning.** So §6's "the extent is not this room, therefore DA3 is
wrong" does not follow, and the ranked fix list was aimed at the wrong target.

Where the extent actually comes from, over all 8.29M measured pixels:

```
p50 2.72 m   p90 4.09 m   p99 5.92 m
beyond 4 m: 11.0%    beyond 6 m: 0.9%    beyond 8 m: 0.01%
```

The room is a normal room — 2.7 m typical, 99% of it inside 6 m. The extent is
set by the 0.9% beyond 6 m, and those are not scattered noise: bearing them from
the capture centre puts **every one of them in a single 90° arc**, 48k of 73k
inside one 30° sector, and none at all across the other 270°. That is a doorway
or an opening, seen through, and correctly measured.

So the honest reading is that **extent is a bad summary statistic for a room
with an opening in it**, and it is what made the cloud look broken. What remains
genuinely worth doing from the §6 list is 3 (flying pixels) and 4 (voxel
downsample); 2 has nothing left to reject, since ARCore returns depth for 100%
of pixels in this capture, not the sparse subset §6 assumed.

## 6b. The holes were self-inflicted — 2026-08-10

The cloud looked full of holes. It was, and the builder was making them.

`da3_cloud.py` kept a uniform random `max_points / frames` pixels per frame —
33,333 of 307,200, **one in nine** — and a random sample of a surface leaves
gaps at every scale. Measured against the cells the full 8.2M points occupy:

```
890,991 random points  ->    576,156 distinct 2 cm cells   39% of the surface
                       ->    798,200 distinct 1 cm cells   19% of the surface
```

Four in five surface cells were empty because the points were discarded, not
because nothing saw them. The replacement (`recon/cloud.py`) keeps **every**
pixel, then reduces by voxel grid — one point per occupied cell, at the mean
position and colour of its members, so it denoises as well as thins:

```
8,235,753 points from 27 frames (58,647 flying pixels dropped at depth edges)
8,153,395 after trimming the far 1%
1,455,617 after a 2.0 cm voxel grid          21.8 MB
```

Surface coverage on a fixed 2 cm grid goes **576,156 -> 1,211,508 cells**
against a ceiling of 1,482,775. (The residual gap is a half-cell registration
artefact — the two clouds are gridded from different origins — not missing
surface.)

Note the trade the grid makes. At 3 cm it is *724,311 points and 10.9 MB* —
a complete surface in a **smaller** file than the 13.4 MB speckled one it
replaces. Density is now a dial (`choose_voxel_size` picks the finest grid that
fits the point budget) rather than a consequence of how many times a random
draw happened to land on a given patch of wall.

**Half of the holes were in the renderer.** `cloud.html` defaulted to a 10 mm
disc. Points on a 20 mm grid drawn at 10 mm cannot tile it — a complete surface
renders as a sieve, and the fault reads as missing data. The viewer now takes
`voxel_m` from the manifest and defaults the disc to 0.75x the cell, because
discs on a grid have to overlap to cover it and the diagonal neighbour is what
decides.

**What is still a real hole**, and what no filter fixes: surfaces no camera saw.
The pitch range was −40°…+25°, so there is no ceiling directly overhead and no
floor directly underfoot; and anything occluded from all 27 viewpoints — behind
furniture, inside the doorway — was never measured. Those close by capturing
them, or by meshing, not by processing.

## 6c. Cleanup now uses evidence, not distance — 2026-08-10

The next cleanup pass is frame-aware. Every point carries its source keyframe
into voxelisation; pixels from one frame are averaged first, then every frame
gets one vote in the final Huber-weighted centre. This prevents a close or dense
view from pulling a surface away from several sparser views.

A voxel is removed only when both are true:

- fewer than two distinct frames support it;
- fewer than three cells in its 26-neighbourhood are occupied.

The second condition matters. Requiring two frames everywhere would erase a
real wall or doorway seen from only one angle. Local coherence preserves those
surfaces while isolated speckles and thin tails disappear. A final conservative
dispersion check removes voxels whose per-frame samples still span more than
0.75 of the voxel size. The manifest records every kept and removed count under
`cleanup`, so the filter cannot silently improve its appearance by deleting an
unknown fraction of the capture.

The old radial far-1% deletion is gone. Section 6a established that its target
was predominantly a real opening, not model noise. `cloud.bin` and the exact
`min`/`max` bounds now retain it; separate coordinate-wise 1st–99th percentile
`view_*` bounds only control the viewer's initial camera and height shading.

Validated on the cached 27-frame DA3-Large result at a 2 cm grid:

```
8,235,753 input points
1,525,996 occupied voxels
1,505,510 kept
   20,485 isolated low-support voxels removed
        1 high-dispersion voxel removed
1,465,571 coherent low-support voxels explicitly preserved
```

That is intentionally conservative: cleanup removed 1.34% of occupied cells,
not a convenient fixed percentage. Retaining the far geometry increased the
exact depth-axis extent from the previously clipped ~11.7 m to 14.09 m, while
the robust viewer extent is 12.44 m.

The builder also accepts `--depth-cache DEPTH.npz`, so thresholds and voxel
sizes can be compared from one fixed model result instead of paying for another
inference run or accidentally comparing two different predictions.

## 7. Reproducing it

Everything below lives in the repo: builders in `scripts/capture/`, viewer
pages in `viewer/`. Both builders import `recon`, so they need one of the
environments from the Kepler/CPU setup active.

```bash
# 1. capture on the phone: stand still, turn, press Take picture at each target
adb pull .../captures/bundle_<uuid> data/captures/

source ~/ai/env.sh kepler          # GPU env + shared model store

# 2. 360, projected by pose
python scripts/capture/equirect.py \
    data/captures/bundle_<uuid> out/sessions/equirect.jpg 4096

# 3. point cloud
python scripts/capture/da3_cloud.py \
    data/captures/bundle_<uuid> out/cloud.bin small 900000

# 4. serve the viewers next to the artefacts they read
cp viewer/panorama.html out/index.html
cp viewer/cloud.html    out/cloud.html
cd out && python -m http.server 8777 --bind 0.0.0.0
#   http://10.10.0.1:8777/           panoramas   (WireGuard address)
#   http://10.10.0.1:8777/cloud.html point cloud
```

Two more, for diagnosis rather than output:

```bash
# feature-based stitching per capture, strict-first confidence ladder;
# writes the manifest.json that panorama.html reads
python scripts/capture/stitch_sessions.py <dir-of-bundles> out/sessions

# DA3 vs ARCore depth, per frame, before and after the affine anchor — §6 and §6a.
# Second argument is the variant; the model runs once and both corrections are
# fitted from that one prediction.
python scripts/capture/depth_diag.py data/captures/bundle_<uuid> [small|large]
```

`panorama.html` expects `sessions/manifest.json` beside it; `cloud.html` expects
`cloud.bin` and `cloud.json`. Both are single self-contained files with no
external dependencies, so they work from any static server.

## 8. Things that cost time, worth not repeating

- **`cargo test` does not rebuild the cdylib.** Generating UniFFI bindings from
  `target/debug/librecon_core.so` right after `cargo test` silently produces
  bindings without the new API. Run `cargo build` first.
- **Python buffers stdout when piped.** A long stitch job ran 15 minutes with an
  empty log and no way to see progress. Use `python -u`.
- **`git stash` on a file with pre-existing uncommitted work reverts all of it,**
  not just the changes you were A/B testing. Copy to a scratch directory instead.
- **Headless Chrome cannot screenshot ~900k WebGL points** — software GL only,
  and it does not finish. The cloud viewer's rendering is unverified by
  screenshot; its data layout was verified arithmetically instead.
- **Wireless-debugging ports randomise** on every restart of the phone's Wi-Fi.
  Pairing persists; only the port moves. A full TCP scan of the device finds it.
