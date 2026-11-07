# Roadmap

Planned work, roughly in priority order.

## Live Scan (on-device)

- [ ] Reliable handheld start: tracking within five seconds of picking up the phone, without a scripted motion
- [ ] Reduce drift on long walks; detect and recover from tracking loss without a second, displaced surface
- [ ] Relocalization against saved views, so a scan can be resumed
- [ ] Camera/IMU calibration flow for phones other than the S23 Ultra
- [ ] Export the fused scan as a mesh (GLB) as well as PLY segments
- [ ] Release build with a one-step model download, instead of a manual export

## Panorama and point cloud

- [ ] Robust per-frame affine depth fit against sensor depth for every frame (the measured top fix)
- [ ] Consistency rejection of depth pixels that disagree across frames
- [ ] Multi-room capture: stitch rooms through doorways into one house model
- [ ] Floor plan extraction (walls, doors, room dimensions) from the fused cloud

## Simulator

- [ ] More worlds (multi-floor, outdoor courtyard) and seeds for the pose x depth sweep
- [ ] Headless CI run of the closed loop with regression thresholds

## Project

- [ ] CI for Python tests, the Rust core and Android unit tests
- [ ] Packaged install (`pip install`) for the capture pipeline
- [ ] Getting-started guide with a sample capture bundle
