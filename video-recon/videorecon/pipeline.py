"""Recording -> mesh, end to end, with a manifest of every number that mattered."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from videorecon import align, depth, frames, fuse, imu, sfm
from videorecon.recording import Recording, match_video_to_capture_results, video_frame_pts_ns


@dataclass
class Options:
    orientation_sensor: str = "gamerotv"
    min_rotation_deg: float = 3.0
    max_interval_s: float = 0.25
    sharpest_window: int = 3
    max_long_edge: int = 1600
    max_keyframes: int = 400
    depth_variant: str = "base"
    #: Match every pair up to this many keyframes; sequential (with loop
    #: candidates) beyond it. Exhaustive is what closes loops without a
    #: vocabulary tree, and 12,720 pairs took under two minutes on CPU.
    exhaustive_below: int = 450
    run_depth: bool = True
    run_fusion: bool = True
    voxel_m: float = 0.02
    depth_trunc_m: float = 6.0
    force: bool = False
    #: Trust the magnetometer track for absolute heading (north = -Z in the output).
    use_magnetic_heading: bool = True
    #: Height the phone was held at, for the floor-based scale (metres).
    camera_height_m: float = 1.3
    #: "auto": IMU when its spread is small, else floor. Or force "imu" / "floor".
    scale_source: str = "auto"
    #: Rotate keyframes upright (by the sensor orientation) before the depth model.
    depth_upright: bool = True
    #: DA3 frames per window; None keeps the adapter's default (clamped to 4 on Kepler).
    depth_window: int | None = 8


@dataclass
class Manifest:
    recording: str
    output: str
    started_at: float = field(default_factory=time.time)
    stages: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def save(self, path: Path) -> None:
        payload = asdict(self)
        payload["finished_at"] = time.time()
        path.write_text(json.dumps(payload, indent=2, default=_jsonable))


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def reconstruct(
    recording_dir: Path, out_dir: Path, opts: Options | None = None, *, log=print
) -> Manifest:
    opts = opts or Options()
    out_dir.mkdir(parents=True, exist_ok=True)
    # A manifest means "finished"; a rerun must not look finished while it works.
    (out_dir / "manifest.json").unlink(missing_ok=True)
    manifest = Manifest(str(recording_dir), str(out_dir))
    t_start = time.time()

    # -- 1. What the phone recorded ------------------------------------------
    rec = Recording.load(recording_dir)
    intr_video = rec.intrinsics()
    rates = rec.sensor_rates_hz()
    log(
        f"recording: {rec.meta.get('device', {}).get('model', '?')}, video "
        f"{intr_video.width}x{intr_video.height}, {len(rec.frames)} capture results, "
        f"sensors {', '.join(f'{k} {v:.0f} Hz' for k, v in sorted(rates.items()))}"
    )
    log(
        f"intrinsics ({intr_video.source}): fx {intr_video.fx:.1f} fy {intr_video.fy:.1f} "
        f"cx {intr_video.cx:.1f} cy {intr_video.cy:.1f}"
    )

    pts_ns = video_frame_pts_ns(rec.video_path)
    timing = match_video_to_capture_results(pts_ns, rec.frames[:, 1])
    log(
        f"timing: {len(pts_ns)} video frames, {timing.matched_fraction * 100:.1f}% matched "
        "to capture results within a quarter frame (median |residual| "
        f"{np.median(np.abs(timing.residual_ns)) / 1e6:.2f} ms)"
    )
    if timing.matched_fraction < 0.9:
        manifest.warnings.append(
            f"only {timing.matched_fraction * 100:.0f}% of video frames matched a capture timestamp"
        )

    sensor = opts.orientation_sensor
    if not rec.has_sensor(sensor):
        fallback = "rotv" if sensor == "gamerotv" else "gamerotv"
        manifest.warnings.append(f"no {sensor} samples; using {fallback}")
        sensor = fallback
    track = rec.orientation_track(sensor)
    r_wd = track.rotation_at(timing.timestamps_ns)  # (V, 3, 3) device -> ENU
    r_dc = imu.device_to_camera(rec.sensor_orientation_deg)
    r_wc_enu = np.einsum("nij,jk->nik", r_wd, r_dc)  # camera(cv) -> ENU
    manifest.stages["recording"] = {
        "video_frames": int(len(pts_ns)),
        "capture_results": int(len(rec.frames)),
        "sensor_rates_hz": rates,
        "orientation_sensor": sensor,
        "matched_fraction": timing.matched_fraction,
        "intrinsics_video": asdict(intr_video),
        "sensor_orientation_deg": rec.sensor_orientation_deg,
    }

    # -- 2. Keyframes ---------------------------------------------------------
    images_dir = out_dir / "images"
    keyframes_json = out_dir / "keyframes.json"
    if keyframes_json.exists() and images_dir.exists() and not opts.force:
        kf = json.loads(keyframes_json.read_text())
        indices = np.asarray(kf["indices"], dtype=np.int64)
        names = {int(k): v for k, v in kf["names"].items()}
        intr = sfm.Intrinsics(
            **{**kf["intrinsics"], "distortion": tuple(kf["intrinsics"]["distortion"])}
        )
        log(f"keyframes: reusing {len(indices)} from {keyframes_json}")
    else:
        blur = frames.score_all_frames(rec.video_path)
        n = min(len(blur), len(timing.timestamps_ns))
        selection = frames.select_keyframes(
            timing.timestamps_ns[:n],
            r_wc_enu[:n],
            blur[:n],
            min_rotation_deg=opts.min_rotation_deg,
            max_interval_s=opts.max_interval_s,
            sharpest_window=opts.sharpest_window,
        )
        indices = selection.indices
        if len(indices) > opts.max_keyframes:
            keep = np.linspace(0, len(indices) - 1, opts.max_keyframes).round().astype(int)
            indices = indices[np.unique(keep)]
            manifest.warnings.append(f"keyframes thinned to {len(indices)} (max_keyframes)")
        reasons = {r: selection.reasons.count(r) for r in set(selection.reasons)}
        log(
            f"keyframes: {len(indices)} of {n} frames ({reasons}); "
            f"median blur {np.median(blur):.0f}"
        )
        extracted = frames.extract_frames(
            rec.video_path, indices, images_dir, intr_video, max_long_edge=opts.max_long_edge
        )
        names = extracted.names
        intr = extracted.intrinsics
        keyframes_json.write_text(
            json.dumps(
                {
                    "indices": indices.tolist(),
                    "names": {str(k): v for k, v in names.items()},
                    "intrinsics": asdict(intr),
                    "reasons": reasons,
                    "blur": blur[indices].tolist(),
                }
            )
        )
    kf_times = timing.timestamps_ns[indices]
    kf_r_wc_enu = r_wc_enu[indices]
    manifest.stages["keyframes"] = {"count": int(len(indices)), "intrinsics": asdict(intr)}

    # -- 3. Position from the images ------------------------------------------
    sfm_json = out_dir / "sfm.json"
    result = None
    if sfm_json.exists() and not opts.force:
        result = sfm.SfmResult.load(sfm_json)
        log(f"sfm: reusing {sfm_json} ({len(result.cam_from_world)} registered)")
    else:
        try:
            result = sfm.run_sfm(
                images_dir,
                out_dir / "colmap",
                intr,
                exhaustive_below=opts.exhaustive_below,
                log=log,
            )
            result.save(sfm_json)
        except RuntimeError as exc:
            manifest.warnings.append(f"SfM failed: {exc}")
            log(f"sfm: FAILED ({exc}); falling back to sensor orientation only")
    name_to_index = {v: k for k, v in names.items()}
    index_pos = {int(i): p for p, i in enumerate(indices)}

    # -- 4. Sensors fix orientation, gravity, scale ----------------------------
    heading_deg = 0.0
    heading_note = "yaw arbitrary (game rotation vector)"
    if opts.use_magnetic_heading and rec.has_sensor("rotv") and rec.has_sensor("gamerotv"):
        mag_track = rec.orientation_track("rotv")
        game_track = rec.orientation_track("gamerotv")
        r_game = game_track.rotation_at(kf_times)
        r_mag = mag_track.rotation_at(kf_times)
        if sensor == "gamerotv":
            heading_deg, spread = align.heading_offset_deg(r_game, r_mag)
            heading_note = f"magnetic heading offset {heading_deg:.1f}° (spread {spread:.1f}°)"
            if spread > 25.0:
                manifest.warnings.append(
                    f"magnetometer heading unstable (spread {spread:.0f}°); applied median anyway"
                )
        else:
            heading_note = "magnetic rotation vector used directly"
    log(f"heading: {heading_note}")

    bundle_frames: list[depth.BundleFrame] = []
    points = np.zeros((0, 3))
    observations: list[list[str]] = []
    blur_by_name = {}
    if keyframes_json.exists():
        kf_meta = json.loads(keyframes_json.read_text())
        if "blur" in kf_meta:
            blur_by_name = {
                names[int(i)]: b
                for i, b in zip(kf_meta["indices"], kf_meta["blur"], strict=False)
                if int(i) in names
            }

    # SfM is trusted only when it registered most of the keyframes: a capture
    # that only turned on the spot still yields a handful of registered images
    # with invented translation, and that is worse than no translation at all.
    enough = result is not None and len(result.cam_from_world) >= max(5, int(0.4 * len(indices)))
    if result is not None and not enough:
        manifest.warnings.append(
            f"SfM registered only {len(result.cam_from_world)}/{len(indices)} keyframes; "
            "treating the capture as rotation-only"
        )
        log(f"sfm: only {len(result.cam_from_world)}/{len(indices)} registered; using sensors only")
    if enough:
        reg = result.registered
        pos = [index_pos[name_to_index[n]] for n in reg]
        imu_r = kf_r_wc_enu[pos]
        sfm_r = np.stack([result.cam_from_world[n][:3, :3].T for n in reg])
        rot = align.solve_rotation(imu_r, sfm_r)
        log(
            f"align: sensors vs SfM rotation residual median {rot.median_deg:.2f}°, "
            f"p95 {rot.p95_deg:.2f}°"
        )
        if rot.p95_deg > 15.0:
            manifest.warnings.append(f"sensor/SfM orientation disagree (p95 {rot.p95_deg:.1f}°)")
        if rot.median_deg > 10.0:
            manifest.warnings.append(
                f"sensor/SfM orientation disagree at the median ({rot.median_deg:.1f}°); "
                "SfM poses are suspect"
            )

        centres = np.stack([result.camera_centres()[n] for n in reg])
        centres_enu = (rot.rotation @ centres.T).T
        scale_note = ""
        if rec.has_sensor("acc"):
            acc_t, acc_v = rec.imu["acc"]
            r_wd_acc = track.rotation_at(acc_t)
            a_world = imu.world_linear_acceleration(r_wd_acc, acc_v[:, :3])
            scale = align.scale_from_imu(centres_enu, kf_times[pos], acc_t, a_world)
        else:
            scale = align.ScaleEstimate(
                float("nan"), "none", 0, 0, float("nan"), "no accelerometer samples"
            )
        imu_scale = scale
        # Floor-based estimate in the sensor-aligned, Y-up frame (unit scale).
        preview = align.to_metric_frame(result.cam_from_world, result.points, rot.rotation, 1.0)
        floor_scale = align.scale_from_floor(
            preview.points,
            np.stack([t for _, t in preview.poses.values()]),
            camera_height_m=opts.camera_height_m,
        )
        log(
            f"scale: IMU {imu_scale.metres_per_unit:.4f} m/unit "
            f"({imu_scale.inliers}/{imu_scale.triplets} triplets, "
            f"dispersion {imu_scale.dispersion * 100:.0f}%, "
            f"{'usable' if imu_scale.usable else 'rejected'}); "
            f"floor {floor_scale.metres_per_unit:.4f} m/unit ({floor_scale.note})"
        )
        if opts.scale_source == "imu" or (opts.scale_source == "auto" and imu_scale.usable):
            scale = imu_scale
        elif floor_scale.usable or opts.scale_source == "floor":
            scale = floor_scale
        else:
            scale = imu_scale
        if scale.usable:
            metres_per_unit = scale.metres_per_unit
            baseline = (
                np.linalg.norm(centres_enu.max(axis=0) - centres_enu.min(axis=0)) * metres_per_unit
            )
            log(
                f"scale: using {scale.method}: {metres_per_unit:.4f} m/unit; "
                f"camera path spans {baseline:.2f} m"
            )
        else:
            metres_per_unit = 1.0
            scale_note = scale.note or "scale unusable"
            manifest.warnings.append(f"scale not recovered ({scale_note}); output is in SfM units")
            log(f"scale: NOT recovered ({scale_note}); poses left in SfM units")
        metric = align.to_metric_frame(
            result.cam_from_world,
            result.points,
            rot.rotation,
            metres_per_unit,
            heading_deg=heading_deg,
        )
        for n in reg:
            r, t = metric.poses[n]
            idx = name_to_index[n]
            bundle_frames.append(
                depth.BundleFrame(
                    id=f"kf{idx:06d}",
                    image_name=n,
                    timestamp_ns=int(timing.timestamps_ns[idx]),
                    rotation=r,
                    translation=t,
                    blur_score=float(blur_by_name.get(n, 0.0)),
                )
            )
        points, observations = metric.points, result.observations
        # The poses were solved with COLMAP's refined intrinsics; the depth model
        # and fusion must use the same ones, or every map is back-projected with
        # the wrong focal length (measured 5.6% on the reference walk).
        intr = result.intrinsics
        if any(abs(v) > 0 for v in intr.distortion):
            undistorted_dir = out_dir / "images_undistorted"
            intr = frames.undistort_frames(images_dir, undistorted_dir, intr, names.values())
            images_dir = undistorted_dir
            log(
                f"undistort: k1 {result.intrinsics.distortion[0]:+.4f} k2 "
                f"{result.intrinsics.distortion[1]:+.4f} -> pinhole {intr.width}x{intr.height}, "
                f"fx {intr.fx:.1f}"
            )
        manifest.stages["sfm"] = result.stats
        manifest.stages["align"] = {
            "rotation_residual_median_deg": rot.median_deg,
            "rotation_residual_p95_deg": rot.p95_deg,
            "scale": asdict(scale),
            "scale_imu": asdict(imu_scale),
            "scale_floor": asdict(floor_scale),
            "metres_per_unit": metres_per_unit,
            "heading_deg": heading_deg,
            "heading": heading_note,
            "mode": "sfm+imu",
        }
    else:
        # Rotation-only fallback: every keyframe at the origin, oriented by the
        # sensors. Correct for a capture that only turned on the spot; for one
        # that walked, it is what the sensors alone can honestly say.
        manifest.warnings.append("using sensor orientation only; positions unknown (no SfM)")
        yaw = Rotation.from_euler("z", heading_deg, degrees=True).as_matrix()
        for p, idx in enumerate(indices):
            r_enu = yaw @ kf_r_wc_enu[p]
            r_gl = imu.ENU_TO_Y_UP @ r_enu @ imu.CV_TO_GL
            bundle_frames.append(
                depth.BundleFrame(
                    id=f"kf{int(idx):06d}",
                    image_name=names[int(idx)],
                    timestamp_ns=int(timing.timestamps_ns[idx]),
                    rotation=r_gl,
                    translation=np.zeros(3),
                    blur_score=float(blur_by_name.get(names[int(idx)], 0.0)),
                )
            )
        manifest.stages["align"] = {
            "mode": "rotation-only",
            "heading_deg": heading_deg,
            "heading": heading_note,
        }

    bundle_rotation = 0
    if opts.depth_upright and rec.sensor_orientation_deg % 360 != 0:
        bundle_rotation = rec.sensor_orientation_deg % 360
        bundle_frames, intr = depth.upright_for_depth(
            images_dir, out_dir / "images_upright", bundle_frames, intr, bundle_rotation
        )
        images_dir = out_dir / "images_upright"
        log(f"depth input: keyframes rotated {bundle_rotation}° clockwise to upright")
    manifest.stages["bundle_rotation_deg"] = bundle_rotation
    bundle_dir = out_dir / "bundle"
    depth.write_bundle(
        bundle_dir,
        images_dir,
        bundle_frames,
        intr,
        points,
        observations,
        session_id=recording_dir.name,
        device=rec.meta.get("device", {}),
        extra={"manifest_stages": manifest.stages, "warnings": manifest.warnings},
    )
    log(f"bundle: {len(bundle_frames)} keyframes, {len(points)} sparse points -> {bundle_dir}")
    manifest.stages["bundle"] = {
        "keyframes": len(bundle_frames),
        "sparse_points": int(len(points)),
        "path": str(bundle_dir),
    }

    # -- 5. Dense depth, 6. fusion -------------------------------------------------
    depth_npz = out_dir / "depth.npz"
    rotation_only = manifest.stages["align"].get("mode") == "rotation-only"
    if rotation_only and opts.run_depth:
        manifest.warnings.append(
            "depth and fusion skipped: pose-conditioned depth needs camera baseline, and this "
            "capture has sensor orientation only. Re-record walking, or use the image-only "
            "panorama pipeline (docs/image-only-panorama-cubemap-point-cloud.md)."
        )
        log("depth: skipped (rotation-only poses); see manifest warnings")
    if opts.run_depth and not rotation_only:
        if depth_npz.exists() and not opts.force:
            log(f"depth: reusing {depth_npz}")
        else:
            diag = depth.run_da3(
                bundle_dir,
                depth_npz,
                variant=opts.depth_variant,
                anchor_to_sparse_points=len(points) > 0,
                window=opts.depth_window,
                log=log,
            )
            manifest.stages["depth"] = {
                k: v for k, v in diag.items() if k in ("model", "runtime_s", "peak_vram_gb")
            }
        if opts.run_fusion:
            report = fuse.fuse(
                bundle_dir,
                depth_npz,
                out_dir,
                voxel_m=opts.voxel_m,
                depth_trunc_m=opts.depth_trunc_m,
                log=log,
            )
            manifest.stages["fusion"] = asdict(report)
    manifest.stages["runtime_s"] = time.time() - t_start
    manifest.save(out_dir / "manifest.json")
    log(f"done in {time.time() - t_start:.0f}s; manifest at {out_dir / 'manifest.json'}")
    return manifest
