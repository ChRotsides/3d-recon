# Capture to point cloud, the path that worked

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

# DA3 vs ARCore depth, per frame — the measurement in §6
python scripts/capture/depth_diag.py data/captures/bundle_<uuid>
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
