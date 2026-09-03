import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from videorecon import frames
from videorecon.recording import Intrinsics, match_video_to_capture_results, video_frame_pts_ns


def test_select_keyframes_by_rotation_then_interval():
    n = 100
    t = np.arange(n, dtype=np.int64) * 33_000_000  # 30 fps
    # Turn 2° per frame for the first half, then hold still.
    yaw = np.concatenate([np.arange(50) * 2.0, np.full(50, 98.0)])
    r = Rotation.from_euler("z", yaw[:, None], degrees=True).as_matrix()
    blur = np.ones(n)
    sel = frames.select_keyframes(
        t, r, blur, min_rotation_deg=8.0, max_interval_s=0.6, sharpest_window=0
    )
    reasons = set(sel.reasons)
    assert "first" in reasons and "rotation" in reasons and "interval" in reasons
    assert np.all(np.diff(sel.indices) > 0)
    # About one keyframe per 4 frames while turning, one per 18 frames while still.
    turning = (sel.indices < 50).sum()
    still = (sel.indices >= 50).sum()
    assert 10 <= turning <= 14
    assert 2 <= still <= 4


def test_select_keyframes_prefers_sharpest_neighbour():
    n = 40
    t = np.arange(n, dtype=np.int64) * 33_000_000
    r = np.repeat(np.eye(3)[None], n, axis=0)
    blur = np.ones(n)
    blur[20] = 50.0  # the sharp frame near the 0.6 s due point (frame 18)
    sel = frames.select_keyframes(
        t, r, blur, min_rotation_deg=90, max_interval_s=0.6, sharpest_window=3
    )
    assert 20 in sel.indices


def _write_video(path, count=12, size=(64, 48), fps=30):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    assert writer.isOpened()
    rng = np.random.default_rng(0)
    for i in range(count):
        frame = rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8)
        cv2.putText(frame, str(i), (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        writer.write(frame)
    writer.release()


def test_video_pts_and_extraction(tmp_path):
    video = tmp_path / "video.mp4"
    _write_video(video)
    pts = video_frame_pts_ns(video)
    assert len(pts) == 12 and pts[0] == 0
    np.testing.assert_allclose(np.diff(pts), 1e9 / 30, rtol=0.02)
    # Capture results with a dropped frame in the encoded stream.
    start = 5_000_000_000
    capture = start + np.round(np.arange(13) * 1e9 / 30).astype(np.int64)
    timing = match_video_to_capture_results(pts, capture)
    assert timing.matched_fraction == 1.0
    assert timing.timestamps_ns[0] == start
    scores = frames.score_all_frames(video, analysis_width=32)
    assert scores.shape == (12,)
    intr = Intrinsics(100, 100, 32, 24, 64, 48)
    out = frames.extract_frames(
        video, np.array([0, 5, 11]), tmp_path / "images", intr, max_long_edge=32
    )
    assert sorted(out.names) == [0, 5, 11]
    assert (out.width, out.height) == (32, 24)
    assert out.intrinsics.fx == 50 and out.intrinsics.cx == 16
    assert all((tmp_path / "images" / n).exists() for n in out.names.values())
    with pytest.raises(ValueError):
        frames.extract_frames(video, np.array([99]), tmp_path / "images2", intr)


def test_undistort_frames_returns_pinhole_and_crops(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    img = np.zeros((120, 160, 3), dtype=np.uint8)
    cv2.rectangle(img, (20, 20), (140, 100), (255, 255, 255), 2)
    cv2.imwrite(str(src / "a.jpg"), img)
    intr = Intrinsics(150, 150, 80, 60, 160, 120, distortion=(0.08, -0.15, 0.0, 0.0))
    out = frames.undistort_frames(src, tmp_path / "dst", intr, ["a.jpg"])
    assert out.distortion == ()
    assert (tmp_path / "dst" / "a.jpg").exists()
    fixed = cv2.imread(str(tmp_path / "dst" / "a.jpg"))
    assert fixed.shape[:2] == (out.height, out.width)
    assert 0 < out.width <= 160 and 0 < out.height <= 120


def test_panorama_stations_and_projection():
    from videorecon import panorama

    pos = np.array(
        [[0, 0, 0], [0.1, 0, 0], [0.2, 0, 0.1], [3, 0, 0], [3.1, 0, 0], [3.2, 0, 0]], float
    )
    runs = panorama.stations(pos, radius_m=0.6, min_frames=2)
    assert runs == [[0, 1, 2], [3, 4, 5]]
    # One image looking down -Z (north) with a bright centre pixel lands at longitude 0, latitude 0.
    img = np.zeros((90, 160, 3), np.uint8)
    img[40:50, 75:85] = 255
    pano, cover = panorama.render_equirect([img], [np.eye(3)], 100, 100, 80, 45, width=360)
    assert pano.shape == (180, 360, 3)
    ys, xs = np.nonzero(pano[..., 0] > 128)
    assert abs(xs.mean() - 180) < 4 and abs(ys.mean() - 90) < 4
    assert 0 < (cover > 0).mean() < 0.5
