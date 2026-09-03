# Offline VIO on a videoscan recording (step 1 of the on-phone plan)

Runs OpenVINS over `video.mp4 + imu.csv` of a recording and compares the
trajectory with the COLMAP + sensor reference produced by `videorecon`.
OpenVINS is GPL and is used here only as a measurement; a BSD VIO (Basalt,
Kimera-VIO, OKVIS2) is what would ship on the phone.

## Build (no ROS, no sudo)

```bash
conda create -p ~/ai/vio/env -c conda-forge cmake ninja eigen "opencv>=4.8" ceres-solver boost-cpp libboost-devel compilers
git clone --depth 1 https://github.com/rpng/open_vins.git ~/ai/vio/open_vins
cp video-recon/vio/run_offline.cpp ~/ai/vio/open_vins/ov_msckf/src/
printf '\nadd_executable(run_offline src/run_offline.cpp)\ntarget_link_libraries(run_offline ov_msckf_lib ${thirdparty_libraries})\n' >> ~/ai/vio/open_vins/ov_msckf/cmake/ROS1.cmake
cd ~/ai/vio/open_vins/ov_msckf && mkdir build && cd build
cmake -G Ninja -DENABLE_ROS=OFF -DCMAKE_BUILD_TYPE=Release -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
  -DCMAKE_PREFIX_PATH=$CONDA_PREFIX -DCMAKE_CXX_FLAGS="-include cassert" ..
ninja run_offline
```

## Run

```bash
python video-recon/vio/convert_recording.py data/recordings/rec_X out/vio/rec_X 0.5   # half-res PNGs + merged IMU
run_offline video-recon/vio/config/estimator_config.yaml out/vio/rec_X out/vio/rec_X.txt
python video-recon/vio/compare_vio_colmap.py out/vio/rec_X.txt out/video/rec_X data/recordings/rec_X
python video-recon/vio/make_viewer.py out/vio/index.html "run=out/vio/rec_X.aligned.npz"   # serve: python -m http.server 8765 --directory out/vio
```

`config/` holds the phone configuration: mono, online calibration of
extrinsics, intrinsics and time offset, dynamic initialisation (the walk never
stands still). `estimator_config_static_init.yaml` is the static-init variant
with a lowered jerk threshold; both work on the reference walk.

## Result on rec_20260903-070045 (SM-S918B, 67 s walk)

| | COLMAP reference | VIO (dynamic init) |
|---|---|---|
| rotation residual vs phone gamerotv, median / p95 | 2.73° / 5.82° | 2.08° / 4.59° |
| rotation residual VIO vs COLMAP (after one global rotation), median / p95 | – | 2.0° / 6.8° |
| position error after Sim(3) alignment, rmse / median / max | – | 35 / 30 / 77 cm |
| position error with VIO metric scale kept (SE(3)), rmse / median | – | 80 / 75 cm |
| metric scale vs floor-based reference | 1.0 (assumed 1.3 m camera height) | 1.61× larger |
| online-calibrated focal (960 px wide) | 605 (COLMAP-refined, pinhole) | 642 (factory prior 639) |
| camera-IMU time offset | – | 15.4 ms |
| runtime | 64 min | 28 s for 67 s of video (2.4× real time on 4 threads) |

Poses start about 9 s into the walk (initialisation). The metric scale from the
IMU is not trustworthy on this walk: the motion is slow and smooth, so scale is
weakly observable, the same reason the pipeline's own IMU scale estimator was
rejected. Rotation is at parity with COLMAP; shape is not yet, with a 30 cm
median error over a 6 m room. No loop closure or global refinement was run.

## Walk viewer: video beside the 3D pose

```bash
python video-recon/vio/export_walk.py out/vio/rec_X.txt out/video/rec_X data/recordings/rec_X www/walk.json
# www/ needs: walk.html, three.min.js + OrbitControls.js + PLYLoader.js (three r147), video.mp4 and mesh.ply (symlinks)
python video-recon/vio/serve.py 8765 www      # Range-capable, so the video can seek
```

`walk.html` plays the recording upright on the left and, on the right, the fused
mesh with the VIO path (orange), the COLMAP keyframe path (blue) and a frustum
per source following the video's current time.

## Live depth in the walk viewer

`serve.py` also answers `GET /depth?rec=<rec>&run=<run>&t=<video s>`: it decodes that
frame from the recording, resizes it to a 362 px short side, and runs DA3-Small through
the repo adapter, conditioned on the VIO poses of the last four requested frames (a
sliding window, so the depth is in the VIO's metric frame). Nothing is computed ahead
of playback or cached. `walk2.html` requests depth as the video plays (about one
request per second on the Kepler TITAN, 0.8 to 1.0 s of model time each), shows the map
under the video, and back-projects every 4th pixel through the pose into a growing
point cloud coloured from the video frame. "camera POV" puts the 3D camera at the
followed run's pose so the cloud is seen from where the phone was.

## Voxel-hash TSDF in the page (`tsdf.js`)

The live depth is fused rather than appended: `tsdf.js` is a dependency-free
voxel-hash TSDF (8³ blocks in a hash map, projective integration with a
truncation band of 4 voxels, weighted running average of distance and colour,
flying-pixel rejection at depth edges, zero-crossing extraction per touched
block). It is the algorithm intended for the phone. In the viewer: "TSDF
fusion" toggles it against the raw append, voxel size 2/3/4 cm; surfaces are
shown once a voxel has two observations. Measured in the page at 3 cm on the
new walk: about 25 ms to integrate a 322×181 map, 60 ms to re-extract touched
blocks, 20 MB for 3,600 blocks after 18 frames.

### Serving to a laptop through a port forward

The recordings are 24 Mbit/s H.264 and do not stream through a forward that
measured 9 Mbit/s. `walk2.html` plays `<rec>/video_proxy.mp4` (960×540, 2 Mbit/s,
made with ffmpeg `-vf scale=960:540 -b:v 2000k -movflags +faststart`) and falls back
to the original. The depth endpoint decodes low-resolution requests from the proxy
too, returns deflate-compressed bodies, and the page keeps two requests in flight.
Rendering is capped (main view 25 fps, POV panel 10 fps, video redraw per video
frame, no antialiasing, pixel ratio 1) because compositing two WebGL canvases was
the main-thread cost on an integrated GPU.
