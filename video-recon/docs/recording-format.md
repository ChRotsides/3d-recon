# Recording format (`videoscan-recording-v1`)

One directory per recording, `rec_<yyyymmdd-HHMMSS>/`, under
`Android/data/com.recon.videoscan/files/recordings/` on the phone.

## `video.mp4`

H.264, 1920×1080 (largest 16:9 size ≤ 1080p the camera offers for
`MediaRecorder`), 30 fps, 24 Mbit/s. **Sensor-native orientation**: no
orientation hint is written, so a portrait-held phone produces landscape frames
in which world-up points to the image's left. The pipeline never rotates them;
the intrinsics and the device→camera rotation account for it.

Video stabilisation and optical stabilisation are off. Focus is fixed at
0.4 dioptres (2.5 m) when the lens can be driven; zoom ratio is 1.0.

## `frames.csv`

One row per frame the sensor exposed, from `CaptureResult`:

```
frame_number,sensor_timestamp_ns,exposure_ns,focus_diopters,iso
```

`sensor_timestamp_ns` is `SENSOR_TIMESTAMP`, on `elapsedRealtimeNanos` when the
camera reports `SENSOR_INFO_TIMESTAMP_SOURCE = REALTIME` (recorded in
`meta.json`; the loader refuses other clocks). The encoder may drop the odd
frame, so the video's container timestamps are matched to these rows by nearest
neighbour with a robustly fitted offset; the manifest reports the match rate.

## `imu.csv`

```
timestamp_ns,sensor,x,y,z,w
```

`sensor` ∈ `acc` (m/s², includes gravity), `gyr` (rad/s), `mag` (µT),
`rotv` (rotation vector, magnetometer-aided), `gamerotv` (rotation vector
without magnetometer), `grav`, `linacc`. `w` is the quaternion scalar for the
two rotation vectors and empty otherwise. Everything is registered at
`SENSOR_DELAY_FASTEST`; typical rates are 100–500 Hz.

The rotation vectors are `(x, y, z, w)` quaternions whose rotation matrix maps
device axes to East-North-Up, exactly as `SensorManager.getRotationMatrixFromVector`.

## `meta.json`

```json
{
  "format": "videoscan-recording-v1",
  "device": {"manufacturer": "...", "model": "...", "os": "...", "app": "..."},
  "camera": {
    "id": "0",
    "sensor_orientation_deg": 90,
    "timestamp_source": "REALTIME",
    "active_array": {"width": 4000, "height": 3000},
    "pre_correction_active_array": {"width": 4032, "height": 3024},
    "intrinsic_calibration": [fx, fy, cx, cy, s],
    "lens_distortion_android": [k1, k2, k3, k4, k5],
    "focal_length_mm": 6.3,
    "physical_size_mm": {"width": 9.8, "height": 7.3},
    "hardware_level": "LEVEL_3",
    "focus_mode": "fixed", "focus_diopters": 0.4,
    "video_stabilization": "off", "optical_stabilization": "off"
  },
  "video": {"width": 1920, "height": 1080, "fps": 30, "bit_rate": 24000000, "file": "video.mp4"},
  "start_elapsed_realtime_ns": ..., "start_wall_clock_ms": ...,
  "frames_logged": 1800, "duration_ns": 60000000000
}
```

`intrinsic_calibration` is in pixels of the pre-correction active array. The
video is treated as a centre crop of that array to 16:9 followed by a uniform
scale; `videorecon.recording.intrinsics_from_meta` does the arithmetic. When the
device does not publish a calibration, `focal_length_mm / physical width × pixels`
is used instead and COLMAP refines it.

`lens_distortion_android` is kept for reference only. Android defines it as a
*correction* in normalised coordinates, the inverse of OpenCV's model, and the
main camera's frames are already close to rectilinear; the pipeline uses a
pinhole model and lets bundle adjustment absorb the rest.
