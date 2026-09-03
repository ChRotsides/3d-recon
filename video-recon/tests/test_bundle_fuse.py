import json

import cv2
import numpy as np
import pytest
from videorecon import depth, fuse
from videorecon.recording import Intrinsics

pytest.importorskip("pydantic")


def _make_images(directory, names, size=(160, 120)):
    directory.mkdir(parents=True, exist_ok=True)
    for i, n in enumerate(names):
        img = np.full((size[1], size[0], 3), 40 + 40 * i, dtype=np.uint8)
        cv2.imwrite(str(directory / n), img)


def test_write_bundle_is_readable_by_the_harness(tmp_path):
    from recon.io.bundle import BundleDir

    names = ["a.jpg", "b.jpg", "c.jpg"]
    _make_images(tmp_path / "images", names)
    intr = Intrinsics(120, 120, 80, 60, 160, 120)
    frames = [
        depth.BundleFrame(
            f"kf{i}", n, 1000 + i, np.eye(3), np.array([0.1 * i, 0, 0]), blur_score=5.0
        )
        for i, n in enumerate(names)
    ]
    points = np.array([[0.0, 0.0, -2.0], [0.5, 0.0, -2.0]])
    obs = [["a.jpg", "b.jpg"], ["a.jpg"]]  # second point seen once: dropped
    out = depth.write_bundle(
        tmp_path / "bundle",
        tmp_path / "images",
        frames,
        intr,
        points,
        obs,
        session_id="s",
        device={"model": "test"},
    )
    bundle = BundleDir.load(out)
    assert [k.id for k in bundle.bundle.keyframes] == ["kf0", "kf1", "kf2"]
    assert len(bundle.bundle.sparse_points) == 1
    assert bundle.bundle.sparse_points[0].observations == ["kf0", "kf1"]
    assert bundle.rgb(bundle.bundle.keyframes[0]).shape == (120, 160, 3)
    raw = json.loads((out / "bundle.json").read_text())
    assert raw["capture_mode"] == "rgb_only"


def test_fuse_synthetic_wall(tmp_path):
    pytest.importorskip("open3d")
    from recon.io.bundle import BundleDir  # noqa: F401

    names = ["a.jpg", "b.jpg"]
    w, h = 160, 120
    _make_images(tmp_path / "images", names, (w, h))
    intr = Intrinsics(150, 150, w / 2, h / 2, w, h)
    # Two cameras at the origin and 20 cm to the right, both looking down -Z at a wall 2 m away.
    frames = [
        depth.BundleFrame("kf0", "a.jpg", 0, np.eye(3), np.zeros(3)),
        depth.BundleFrame("kf1", "b.jpg", 1, np.eye(3), np.array([0.2, 0.0, 0.0])),
    ]
    bundle = depth.write_bundle(
        tmp_path / "bundle",
        tmp_path / "images",
        frames,
        intr,
        np.zeros((0, 3)),
        [],
        session_id="s",
        device={},
    )
    wall = np.full((h, w), 2.0, dtype=np.float32)
    np.savez(tmp_path / "depth.npz", model=np.array("synthetic"), kf0=wall, kf1=wall)
    report = fuse.fuse(
        bundle,
        tmp_path / "depth.npz",
        tmp_path / "out",
        voxel_m=0.05,
        target_triangles=0,
        log=lambda *_: None,
    )
    assert report.frames_used == 2
    assert report.mesh_triangles > 0 and report.cloud_points > 0
    assert (tmp_path / "out" / "mesh.ply").exists()
    meta = json.loads((tmp_path / "out" / "cloud.json").read_text())
    assert meta["count"] == report.cloud_points
    # The wall sits at z = -2 in the repo frame.
    assert abs(meta["centre"][2] + 2.0) < 0.1
    pos = np.fromfile(
        tmp_path / "out" / "cloud.bin", dtype=np.float32, count=3 * meta["count"]
    ).reshape(-1, 3)
    assert np.all(np.isfinite(pos))


def test_rotated_camera_projects_the_same_world_point():
    """A world point must land on the rotated pixel of the rotated image."""
    from videorecon.imu import CV_TO_GL

    rng = np.random.default_rng(1)
    intr = Intrinsics(900, 910, 640, 360, 1280, 720)
    r_gl = np.linalg.qr(rng.normal(size=(3, 3)))[0]
    if np.linalg.det(r_gl) < 0:
        r_gl[:, 0] *= -1
    t = rng.normal(size=3)

    def project(r_gl_, t_, k_, p_world):
        r_cv = r_gl_ @ CV_TO_GL
        pc = r_cv.T @ (p_world - t_)
        return np.array([k_.fx * pc[0] / pc[2] + k_.cx, k_.fy * pc[1] / pc[2] + k_.cy]), pc[2]

    for rot in (90, 180, 270):
        k2 = depth.rotate_intrinsics(intr, rot)
        r2 = depth.rotate_pose(r_gl, rot)
        for _ in range(20):
            local = np.array([rng.uniform(-1, 1), rng.uniform(-0.5, 0.5), rng.uniform(1.5, 4)])
            p_world = t + (r_gl @ CV_TO_GL) @ local
            (u, v), z = project(r_gl, t, intr, p_world)
            (u2, v2), z2 = project(r2, t, k2, p_world)
            if rot == 90:
                exp = ((intr.height - 1) - v, u)
            elif rot == 270:
                exp = (v, (intr.width - 1) - u)
            else:
                exp = ((intr.width - 1) - u, (intr.height - 1) - v)
            np.testing.assert_allclose((u2, v2), exp, atol=1e-9)
            np.testing.assert_allclose(z2, z, atol=1e-9)
        np.testing.assert_allclose(r2 @ r2.T, np.eye(3), atol=1e-12)
        assert np.isclose(np.linalg.det(r2), 1.0)


def test_fuse_consistency_drops_a_surface_only_one_frame_sees(tmp_path):
    pytest.importorskip("open3d")
    names = ["a.jpg", "b.jpg", "c.jpg"]
    w, h = 160, 120
    _make_images(tmp_path / "images", names, (w, h))
    intr = Intrinsics(150, 150, w / 2, h / 2, w, h)
    frames = [
        depth.BundleFrame(f"kf{i}", n, i, np.eye(3), np.array([0.15 * i, 0.0, 0.0]))
        for i, n in enumerate(names)
    ]
    bundle = depth.write_bundle(
        tmp_path / "bundle",
        tmp_path / "images",
        frames,
        intr,
        np.zeros((0, 3)),
        [],
        session_id="s",
        device={},
    )
    wall = np.full((h, w), 2.0, dtype=np.float32)
    ghost = wall.copy()
    ghost[40:80, 60:100] = 1.0  # a phantom block only the middle frame reports
    np.savez(tmp_path / "depth.npz", model=np.array("synthetic"), kf0=wall, kf1=ghost, kf2=wall)
    common = dict(
        voxel_m=0.05,
        target_triangles=0,
        progress_every=0,
        sparse_depth_cap=None,
        support_offsets=(-1, 1),
        log=lambda *_: None,
    )
    loose = fuse.fuse(bundle, tmp_path / "depth.npz", tmp_path / "loose", min_support=0, **common)
    strict = fuse.fuse(bundle, tmp_path / "depth.npz", tmp_path / "strict", min_support=1, **common)
    meta = json.loads((tmp_path / "strict" / "cloud.json").read_text())
    pos = np.fromfile(
        tmp_path / "strict" / "cloud.bin", dtype=np.float32, count=3 * meta["count"]
    ).reshape(-1, 3)
    # Everything left is on the real wall (z = -2); the phantom at z = -1 is gone.
    assert np.all(pos[:, 2] < -1.7)
    assert strict.cloud_points < loose.cloud_points
