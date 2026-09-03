"""Camera positions from the video, with COLMAP.

The sensors give orientation; they cannot give position (double-integrating a
phone accelerometer drifts by metres within seconds). Position comes from the
images: SIFT features, sequential matching along the video with loop closure
by exhaustive matching on short captures, and COLMAP's incremental mapper,
seeded with the camera's own intrinsics and allowed to refine the focal length.

Output poses are world<-camera in COLMAP's world, which is arbitrary in
orientation and scale. `videorecon.align` fixes both from the sensors.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from videorecon.recording import Intrinsics


@dataclass
class SfmResult:
    #: image name -> 4x4 world-to-camera (OpenCV camera axes), registered images only.
    cam_from_world: dict[str, np.ndarray]
    #: Refined pinhole intrinsics, at the SfM image resolution.
    intrinsics: Intrinsics
    #: (P, 3) sparse points in COLMAP world.
    points: np.ndarray
    #: For each point, the image names that observe it.
    observations: list[list[str]]
    stats: dict = field(default_factory=dict)

    @property
    def registered(self) -> list[str]:
        return sorted(self.cam_from_world)

    def camera_centres(self) -> dict[str, np.ndarray]:
        out = {}
        for name, w2c in self.cam_from_world.items():
            out[name] = -w2c[:3, :3].T @ w2c[:3, 3]
        return out

    def save(self, path: Path) -> None:
        payload = {
            "intrinsics": asdict(self.intrinsics),
            "stats": self.stats,
            "images": {n: m.tolist() for n, m in self.cam_from_world.items()},
            "points": self.points.tolist(),
            "observations": self.observations,
        }
        path.write_text(json.dumps(payload))

    @classmethod
    def load(cls, path: Path) -> SfmResult:
        d = json.loads(path.read_text())
        intr = d["intrinsics"]
        intr["distortion"] = tuple(intr.get("distortion", ()))
        return cls(
            cam_from_world={n: np.asarray(m, dtype=np.float64) for n, m in d["images"].items()},
            intrinsics=Intrinsics(**intr),
            points=np.asarray(d["points"], dtype=np.float64).reshape(-1, 3),
            observations=[list(o) for o in d["observations"]],
            stats=d.get("stats", {}),
        )


def _camera_model_and_params(intr: Intrinsics, fit_distortion: bool) -> tuple[str, str]:
    d = list(intr.distortion) + [0.0] * 4
    k1, k2, p1, p2 = d[:4]
    if any(abs(v) > 0 for v in (p1, p2)):
        # COLMAP OPENCV: fx fy cx cy k1 k2 p1 p2 — only when a tangential prior exists.
        params = [intr.fx, intr.fy, intr.cx, intr.cy, k1, k2, p1, p2]
        return "OPENCV", ",".join(f"{v:.6f}" for v in params)
    if fit_distortion or any(abs(v) > 0 for v in (k1, k2)):
        # COLMAP RADIAL: f cx cy k1 k2. Fitting tangential terms as well on the
        # reference walk made the mapper's bundle adjustment fail to converge
        # (CHOLMOD "not positive definite"); radial-only is what a phone needs.
        f = 0.5 * (intr.fx + intr.fy)
        params = [f, intr.cx, intr.cy, k1, k2]
        return "RADIAL", ",".join(f"{v:.6f}" for v in params)
    return "PINHOLE", ",".join(f"{v:.6f}" for v in (intr.fx, intr.fy, intr.cx, intr.cy))


def run_sfm(
    image_dir: Path,
    work_dir: Path,
    intrinsics: Intrinsics,
    *,
    exhaustive_below: int = 220,
    sequential_overlap: int = 20,
    max_image_size: int = 1600,
    max_features: int = 8192,
    refine_focal: bool = True,
    #: Fit radial and tangential distortion (OPENCV model) instead of a pure pinhole.
    fit_distortion: bool = True,
    num_threads: int = -1,
    #: Lower finds more (weaker) SIFT peaks; rooms are low-texture and need them.
    peak_threshold: float = 0.004,
    #: Affine-shape estimation and domain-size pooling: slower extraction, more
    #: repeatable descriptors on the blank walls a room is mostly made of.
    robust_sift: bool = True,
    log=print,
) -> SfmResult:
    import pycolmap

    work_dir.mkdir(parents=True, exist_ok=True)
    database = work_dir / "database.db"
    sparse_dir = work_dir / "sparse"
    if database.exists():
        database.unlink()
    if sparse_dir.exists():
        shutil.rmtree(sparse_dir)
    sparse_dir.mkdir()

    names = sorted(
        p.name for p in image_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )
    if len(names) < 3:
        raise ValueError(f"SfM needs at least three images, found {len(names)} in {image_dir}")

    model, params = _camera_model_and_params(intrinsics, fit_distortion)
    reader = pycolmap.ImageReaderOptions()
    reader.camera_model = model
    reader.camera_params = params
    extraction = pycolmap.FeatureExtractionOptions()
    extraction.max_image_size = max_image_size
    extraction.num_threads = num_threads
    extraction.sift.max_num_features = max_features
    extraction.sift.peak_threshold = peak_threshold
    extraction.sift.estimate_affine_shape = robust_sift
    extraction.sift.domain_size_pooling = robust_sift
    log(f"sfm: extracting SIFT from {len(names)} images ({model}, prior {params})")
    pycolmap.extract_features(
        database,
        image_dir,
        image_names=names,
        camera_mode=pycolmap.CameraMode.SINGLE,
        reader_options=reader,
        extraction_options=extraction,
        device=pycolmap.Device.cpu,
    )

    matching = pycolmap.FeatureMatchingOptions()
    matching.num_threads = num_threads
    matching.guided_matching = True
    if len(names) <= exhaustive_below:
        log(f"sfm: exhaustive matching ({len(names) * (len(names) - 1) // 2} pairs)")
        pycolmap.match_exhaustive(database, matching_options=matching, device=pycolmap.Device.cpu)
        strategy = "exhaustive"
    else:
        pairing = pycolmap.SequentialPairingOptions()
        pairing.overlap = sequential_overlap
        pairing.quadratic_overlap = True
        pairing.loop_detection = False
        log(f"sfm: sequential matching, overlap {sequential_overlap} (quadratic)")
        pycolmap.match_sequential(
            database, matching_options=matching, pairing_options=pairing, device=pycolmap.Device.cpu
        )
        strategy = f"sequential-{sequential_overlap}"

    options = pycolmap.IncrementalPipelineOptions()
    options.num_threads = num_threads
    options.ba_refine_focal_length = refine_focal
    options.ba_refine_principal_point = False
    options.ba_refine_extra_params = model != "PINHOLE"
    options.mapper.abs_pose_refine_extra_params = model != "PINHOLE"
    options.min_model_size = 5
    options.min_num_matches = 10
    # Rooms are low-texture and a hand-held video has small baselines between
    # neighbours. Measured on a 38-image capture: the defaults registered 14
    # images, these register 25 with the same 0.4 px reprojection error.
    options.mapper.init_min_num_inliers = 50
    options.mapper.abs_pose_min_num_inliers = 15
    options.mapper.abs_pose_min_inlier_ratio = 0.15
    # Snapshots let a viewer watch the sparse model grow while the mapper runs.
    snapshots = work_dir / "snapshots"
    snapshots.mkdir(exist_ok=True)
    options.snapshot_path = str(snapshots)
    options.snapshot_frames_freq = 10
    log("sfm: incremental mapping")
    maps = pycolmap.incremental_mapping(database, image_dir, sparse_dir, options=options)
    if not maps:
        raise RuntimeError("COLMAP could not register any images; is the camera actually moving?")
    best_key = max(maps, key=lambda k: maps[k].num_reg_images())
    rec = maps[best_key]
    log(
        f"sfm: {len(maps)} model(s); best registers {rec.num_reg_images()}/{len(names)} images, "
        f"{rec.num_points3D()} points, mean reprojection error "
        f"{rec.compute_mean_reprojection_error():.2f} px"
    )
    return _extract(rec, intrinsics, strategy, len(names), len(maps))


def _rigid_matrix(image) -> np.ndarray:
    cfw = image.cam_from_world
    if callable(cfw):
        cfw = cfw()
    m = np.asarray(cfw.matrix(), dtype=np.float64)
    out = np.eye(4)
    out[:3, :4] = m[:3, :4]
    return out


def _extract(
    rec, prior: Intrinsics, strategy: str, total_images: int, model_count: int
) -> SfmResult:
    cam_from_world: dict[str, np.ndarray] = {}
    id_to_name: dict[int, str] = {}
    camera_id = None
    for image_id, image in rec.images.items():
        if not image.has_pose:
            continue
        cam_from_world[image.name] = _rigid_matrix(image)
        id_to_name[int(image_id)] = image.name
        camera_id = image.camera_id
    if camera_id is None:
        raise RuntimeError("reconstruction has no registered images")
    cam = rec.cameras[camera_id]
    k = np.asarray(cam.calibration_matrix(), dtype=np.float64)
    params = list(cam.params)
    distortion: tuple[float, ...] = ()
    if cam.model_name in ("FULL_OPENCV", "OPENCV"):
        k1, k2, p1, p2 = (float(v) for v in params[4:8])
        distortion = (k1, k2, p1, p2, 0.0)  # OpenCV order [k1, k2, p1, p2, k3]
    elif cam.model_name == "RADIAL":
        k1, k2 = (float(v) for v in params[3:5])
        distortion = (k1, k2, 0.0, 0.0, 0.0)
    elif cam.model_name == "SIMPLE_RADIAL":
        distortion = (float(params[3]), 0.0, 0.0, 0.0, 0.0)
    intr = Intrinsics(
        fx=float(k[0, 0]),
        fy=float(k[1, 1]),
        cx=float(k[0, 2]),
        cy=float(k[1, 2]),
        width=int(cam.width),
        height=int(cam.height),
        distortion=distortion,
        source=f"colmap:{cam.model_name}",
    )
    points = []
    observations = []
    for p in rec.points3D.values():
        names = sorted(
            {id_to_name[int(e.image_id)] for e in p.track.elements if int(e.image_id) in id_to_name}
        )
        if len(names) < 2:
            continue
        points.append(np.asarray(p.xyz, dtype=np.float64))
        observations.append(names)
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    stats = {
        "strategy": strategy,
        "images_total": total_images,
        "images_registered": len(cam_from_world),
        "models": model_count,
        "points": int(len(pts)),
        "mean_reprojection_error_px": float(rec.compute_mean_reprojection_error()),
        "mean_track_length": float(rec.compute_mean_track_length()),
        "focal_prior_px": [prior.fx, prior.fy],
        "focal_refined_px": [intr.fx, intr.fy],
        "camera_model": cam.model_name,
        "distortion": list(distortion),
    }
    return SfmResult(cam_from_world, intr, pts, observations, stats)


def remap_with_model(
    work_dir: Path,
    image_dir: Path,
    intrinsics: Intrinsics,
    *,
    fit_distortion: bool = True,
    num_threads: int = -1,
    log=print,
) -> SfmResult:
    """Re-run only the mapper on an existing database, with a new camera model.

    Features and verified matches do not depend on the camera model, so a
    finished exhaustive matching (an hour on CPU) can be reused when the model
    changes from PINHOLE to OPENCV. The original database is left untouched.
    """
    import sqlite3

    import pycolmap

    src = work_dir / "database.db"
    dst = work_dir / f"database-{'radial' if fit_distortion else 'pinhole'}.db"
    shutil.copyfile(src, dst)
    model, params = _camera_model_and_params(intrinsics, fit_distortion)
    model_id = {"PINHOLE": 1, "RADIAL": 3, "OPENCV": 4}[model]
    values = np.asarray([float(v) for v in params.split(",")], dtype=np.float64)
    conn = sqlite3.connect(dst)
    conn.execute(
        "update cameras set model=?, params=?, prior_focal_length=1", (model_id, values.tobytes())
    )
    conn.commit()
    conn.close()
    sparse_dir = work_dir / f"sparse-{model.lower()}"
    if sparse_dir.exists():
        shutil.rmtree(sparse_dir)
    sparse_dir.mkdir()
    options = pycolmap.IncrementalPipelineOptions()
    options.num_threads = num_threads
    options.ba_refine_focal_length = True
    options.ba_refine_principal_point = False
    options.ba_refine_extra_params = model != "PINHOLE"
    options.mapper.abs_pose_refine_extra_params = model != "PINHOLE"
    options.min_model_size = 5
    options.min_num_matches = 10
    options.mapper.init_min_num_inliers = 50
    options.mapper.abs_pose_min_num_inliers = 15
    options.mapper.abs_pose_min_inlier_ratio = 0.15
    names = sorted(
        p.name for p in image_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )
    log(f"sfm: re-mapping with the {model} camera model on the existing matches")
    maps = pycolmap.incremental_mapping(dst, image_dir, sparse_dir, options=options)
    if not maps:
        raise RuntimeError("COLMAP could not register any images")
    best_key = max(maps, key=lambda k: maps[k].num_reg_images())
    rec = maps[best_key]
    log(
        f"sfm: {len(maps)} model(s); best registers {rec.num_reg_images()}/{len(names)} images, "
        f"{rec.num_points3D()} points, mean reprojection error "
        f"{rec.compute_mean_reprojection_error():.2f} px"
    )
    return _extract(rec, intrinsics, f"remap-{model.lower()}", len(names), len(maps))
