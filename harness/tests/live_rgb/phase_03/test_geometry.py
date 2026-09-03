import numpy as np
from recon.live_rgb.draft import (
    FACE_SIZE,
    backproject_direction,
    cubemap_cameras,
    direction_face,
    project_direction,
    render_cubemap,
    rescale_intrinsic,
)


def test_sphere_directions_have_exactly_one_face():
    longitude, latitude = np.meshgrid(
        np.linspace(-np.pi, np.pi, 721, endpoint=False),
        np.linspace(-np.pi / 2, np.pi / 2, 361),
    )
    directions = np.stack(
        [
            np.cos(latitude) * np.sin(longitude),
            np.sin(latitude),
            np.cos(latitude) * np.cos(longitude),
        ],
        axis=-1,
    )
    assigned = direction_face(directions)
    assert assigned.shape == longitude.shape
    assert set(np.unique(assigned)) == set(range(6))


def test_projection_round_trip_and_shared_exact_cameras():
    cameras = cubemap_cameras()
    assert all(camera.rendered_size == (256, 256) for camera in cameras)
    assert all(np.array_equal(camera.extrinsic[:3, 3], np.zeros(3)) for camera in cameras)
    assert all(camera.rendered_intrinsic[0, 0] == FACE_SIZE / 2 for camera in cameras)
    random = np.random.default_rng(3)
    directions = random.normal(size=(20_000, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    faces = direction_face(directions)
    errors = []
    for direction, face in zip(directions, faces, strict=True):
        pixel = project_direction(direction, cameras[int(face)])
        recovered = backproject_direction(pixel, cameras[int(face)])
        errors.append(np.arccos(np.clip(np.dot(direction, recovered), -1.0, 1.0)))
    assert max(errors) < 1e-5


def test_intrinsics_use_exact_processed_resolution_ratio():
    camera = cubemap_cameras(face_size=256, process_res=192)[0]
    expected = rescale_intrinsic(camera.rendered_intrinsic, (256, 256), (192, 192))
    assert np.array_equal(camera.processed_intrinsic, expected)
    assert camera.processed_intrinsic[0, 0] == 96.0
    assert camera.processed_intrinsic[0, 2] == 96.0


def test_low_support_wedge_does_not_change_rgb_or_unrelated_support():
    panorama = np.random.default_rng(4).integers(0, 256, (128, 256, 3), dtype=np.uint8)
    full = np.ones((128, 256), dtype=np.float32)
    wedge = full.copy()
    wedge[:, 120:136] = 0.1
    full_views, full_support, _ = render_cubemap(panorama, full)
    wedge_views, wedge_support, _ = render_cubemap(panorama, wedge)
    assert all(np.array_equal(a, b) for a, b in zip(full_views, wedge_views, strict=True))
    changed = [
        np.count_nonzero(np.abs(a - b) > 1e-6)
        for a, b in zip(full_support, wedge_support, strict=True)
    ]
    assert 0 < sum(changed) < 6 * 256 * 256
    for before, after in zip(full_support, wedge_support, strict=True):
        unchanged = np.abs(before - after) <= 1e-6
        assert np.array_equal(before[unchanged], after[unchanged])
