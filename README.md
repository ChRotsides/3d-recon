# Room capture → 360 panorama + point cloud

Phone capture with no LiDAR, turned into a 360 equirectangular panorama and a
metric 3D point cloud of the room. **This tree contains only the path that has
been made to work end to end on real hardware**, documented with its numbers and
its failure modes in [`docs/capture-to-cloud.md`](docs/capture-to-cloud.md).

Everything else — the TSDF/floorplan/planes/masking/splat reconstruction
pipeline, the FastAPI service, the Babylon walkthrough viewer, the iOS client —
is in `archive/` (git-ignored, nothing deleted) to be built back on top of this
once it earns its place. The original architecture it came from is
[`room-scanner-implementation-plan.md`](room-scanner-implementation-plan.md).

## The path

```
Android app  →  adb pull  →  scripts/capture/*.py  →  viewer/*.html
(core/ Rust)     bundle       equirect + DA3 cloud     panorama + cloud
```

```bash
# 1. Capture: stand still, turn, press "Take picture" at each target.
adb pull /sdcard/Android/data/com.recon.poselogger/files/captures/bundle_<uuid> \
    data/captures/

source ~/ai/env.sh kepler          # GPU env + shared model store

# 2. 360 panorama, projected by pose (not by feature matching — see the doc)
python scripts/capture/equirect.py \
    data/captures/bundle_<uuid> out/sessions/equirect.jpg 4096

# 3. Point cloud, DA3-Small
python scripts/capture/da3_cloud.py \
    data/captures/bundle_<uuid> out/cloud.bin small 900000

# 4. Serve the viewers next to the artefacts they read
cp viewer/panorama.html out/index.html
cp viewer/cloud.html    out/cloud.html
cd out && python -m http.server 8777 --bind 0.0.0.0
```

The cloud builder cleans by default: it removes flying depth-edge pixels, gives
each frame one vote inside a voxel, and rejects a cell only when it lacks both
multi-frame support and occupied neighbours. It does **not** delete a global
distance percentile; exact geometry is retained while robust percentile bounds
frame the initial viewer camera. Optional arguments after the point budget are
`anchor_space`, `voxel_m`, `min_frame_support`, `min_neighbours`, and
`max_spread_voxels`. Add `--depth-cache DEPTH.npz` to rebuild or tune the cloud
without rerunning DA3.

## Layout

```
clients/android/     Kotlin + ARCore capture app — manual shutter, guidance HUD
core/                Shared Rust capture core + UniFFI bindings the app calls
harness/recon/       The Python the capture path needs, and nothing more
  schemas.py           SessionBundle wire types
  io/bundle.py         reads a pulled bundle
  depth/da3.py         Depth Anything 3 adapter
  torch_compat.py      Kepler (sm_35) torch shims
scripts/capture/     equirect.py, da3_cloud.py + two diagnostics
viewer/              panorama.html, cloud.html — dependency-free single files
docs/                capture-to-cloud.md — what worked, measured
archive/             everything outside that path (git-ignored)
```

## Tests

```bash
pytest harness/tests -q      # bundle/schema + Kepler-torch compatibility
cd core && cargo test        # the shared capture core
cd clients/android && ./gradlew :app:testDebugUnitTest
```

## Known-wrong, measured

The cloud comes out 11.0 × 5.4 × 14.4 m, which is not this room. Global scale is
already right to 2.5%; the error is per-frame scale (frames disagree by 2.36×)
and depth *shape* within each frame. `docs/capture-to-cloud.md` §6 has the
measurement and the ranked fix list — a robust per-frame affine fit against the
ARCore depth every frame already carries is the first and cheapest win.
