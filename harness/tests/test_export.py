"""Export tests (PLAN.md §1.1, §3.7).

The exports are the product's actual output, so the properties that matter are
round-tripping and — above all — that the coordinate frame is recorded rather
than assumed. A flipped scene opens fine, looks like a room, and is only caught
by measuring it (plan §7).
"""

from __future__ import annotations

import numpy as np
import pytest
from recon.io.export import (
    CANONICAL_FRAME,
    FrameMismatch,
    GaussianCloud,
    assert_frame,
    read_glb,
    read_spz,
    write_glb,
    write_pdf_floorplan,
    write_spz,
)
from recon.schemas import CoordinateFrame, Handedness, RoomLayout, UpAxis


def _cube() -> tuple[np.ndarray, np.ndarray]:
    v = np.array([[x, y, z] for x in (0.0, 1.0) for y in (0.0, 1.0) for z in (0.0, 1.0)])
    f = np.array([
        [0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5],
        [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6],
        [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3],
    ], dtype=np.uint32)
    return v, f


def _layout() -> RoomLayout:
    return RoomLayout(
        room_id="11111111-2222-3333-4444-555555555555",
        footprint=[[0.0, 0.0], [4.2, 0.0], [4.2, 3.35], [0.0, 3.35]],
        floor_height_m=0.0,
        ceiling_height_m=2.45,
        area_m2=14.07,
        perimeter_m=15.1,
        wall_lengths_m=[4.2, 3.35, 4.2, 3.35],
    )


# -- GLB --------------------------------------------------------------------


def test_glb_round_trips_geometry(tmp_path):
    v, f = _cube()
    write_glb(tmp_path / "m.glb", v, f)
    js, binary = read_glb(tmp_path / "m.glb")

    assert js["asset"]["version"] == "2.0"
    acc = js["accessors"][js["meshes"][0]["primitives"][0]["attributes"]["POSITION"]]
    assert acc["count"] == len(v)
    assert acc["min"] == [0.0, 0.0, 0.0]
    assert acc["max"] == [1.0, 1.0, 1.0]
    idx = js["accessors"][js["meshes"][0]["primitives"][0]["indices"]]
    assert idx["count"] == f.size
    assert len(binary) > 0


def test_glb_records_its_coordinate_frame(tmp_path):
    """Plan §7: an artifact with no recorded frame is how a mirrored scene
    ships. The reader must be able to check rather than trust."""
    v, f = _cube()
    write_glb(tmp_path / "m.glb", v, f)
    js, _ = read_glb(tmp_path / "m.glb")
    frame = CoordinateFrame(**js["asset"]["extras"]["coordinate_frame"])
    assert frame.matches(CANONICAL_FRAME)
    assert_frame(frame)


def test_glb_chunks_are_four_byte_aligned(tmp_path):
    """glTF requires it, and viewers that assume it read garbage otherwise. An
    odd vertex count is what exposes a missing pad."""
    rng = np.random.default_rng(0)
    v = rng.random((7, 3))
    f = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.uint32)
    p = write_glb(tmp_path / "odd.glb", v, f)
    data = p.read_bytes()
    assert len(data) % 4 == 0
    import struct

    total = struct.unpack_from("<I", data, 8)[0]
    assert total == len(data)
    json_len = struct.unpack_from("<I", data, 12)[0]
    assert json_len % 4 == 0


def test_glb_carries_normals_and_colours_when_given(tmp_path):
    v, f = _cube()
    n = np.tile([0.0, 1.0, 0.0], (len(v), 1))
    c = np.tile([0.5, 0.25, 0.75], (len(v), 1))
    write_glb(tmp_path / "m.glb", v, f, normals=n, colors=c)
    js, _ = read_glb(tmp_path / "m.glb")
    attrs = js["meshes"][0]["primitives"][0]["attributes"]
    assert "NORMAL" in attrs and "COLOR_0" in attrs


def test_glb_rejects_out_of_range_faces(tmp_path):
    v, _ = _cube()
    bad = np.array([[0, 1, 99]], dtype=np.uint32)
    with pytest.raises(ValueError):
        write_glb(tmp_path / "m.glb", v, bad)


# -- SPZ --------------------------------------------------------------------


def _cloud(n: int = 500, sh_per_channel: int = 0, seed: int = 0) -> GaussianCloud:
    rng = np.random.default_rng(seed)
    quat = rng.normal(size=(n, 4))
    quat /= np.linalg.norm(quat, axis=1, keepdims=True)
    sh = None
    if sh_per_channel:
        sh = rng.uniform(-0.9, 0.9, (n, sh_per_channel * 3))
    return GaussianCloud(
        positions=rng.uniform(-5.0, 5.0, (n, 3)),
        scales=rng.uniform(-6.0, 0.0, (n, 3)),
        rotations=quat,
        alphas=rng.uniform(0.0, 1.0, n),
        colors=rng.uniform(0.0, 1.0, (n, 3)),
        sh=sh,
    )


def test_spz_positions_survive_the_round_trip(tmp_path):
    """12 fractional bits resolve 0.24 mm — finer than anything a room scan can
    measure, which is what makes the quantisation safe to trade for size."""
    c = _cloud()
    write_spz(tmp_path / "s.spz", c)
    back = read_spz(tmp_path / "s.spz")
    assert len(back.positions) == len(c.positions)
    assert np.abs(back.positions - c.positions).max() < 0.0005


def test_spz_preserves_spherical_harmonics(tmp_path):
    """The reason for choosing SPZ over SPLAT (plan §3.7): SPLAT strips SH and
    view-dependent effects go flat — a mirror stops looking like a mirror."""
    c = _cloud(sh_per_channel=3)
    assert c.sh_degree == 1
    write_spz(tmp_path / "s.spz", c)
    back = read_spz(tmp_path / "s.spz")
    assert back.sh is not None
    assert back.sh.shape == c.sh.shape
    assert np.abs(back.sh - c.sh).max() < 0.02


def test_spz_handles_degree_zero(tmp_path):
    c = _cloud()
    assert c.sh_degree == 0
    write_spz(tmp_path / "s.spz", c)
    assert read_spz(tmp_path / "s.spz").sh is None


def test_spz_colours_and_alphas_round_trip(tmp_path):
    c = _cloud()
    write_spz(tmp_path / "s.spz", c)
    back = read_spz(tmp_path / "s.spz")
    assert np.abs(back.colors - c.colors).max() < 0.005
    assert np.abs(back.alphas - c.alphas).max() < 0.005


def test_spz_rotations_survive_as_rotations(tmp_path):
    """The metric that matters is angular error, not component error.

    Storing xyz and recovering w from the unit-norm constraint is
    ill-conditioned wherever w is small, and measured up to **10.4 degrees** of
    rotation error — enough to visibly reorient a Gaussian. Smallest-three
    encoding drops the largest component instead, which is always the
    well-conditioned one.
    """
    c = _cloud(2000)
    write_spz(tmp_path / "s.spz", c)
    back = read_spz(tmp_path / "s.spz")

    assert np.abs(np.linalg.norm(back.rotations, axis=1) - 1.0).max() < 1e-9
    # Quaternions q and -q are the same rotation, so compare via |dot|.
    dot = np.abs(np.einsum("ij,ij->i", back.rotations, c.rotations))
    angle = np.degrees(2 * np.arccos(np.clip(dot, 0.0, 1.0)))
    assert angle.max() < 0.3, f"worst rotation error {angle.max():.2f} deg"


def test_spz_is_much_smaller_than_the_equivalent_raw_floats(tmp_path):
    """Plan §3.7 claims ~10x over PLY. The comparison here is against the same
    fields as float32, which is what a PLY holds.

    The bar is set against **random** splats, which is the worst case: every
    field is high-entropy, so gzip finds nothing and only the quantisation
    counts (92 bytes/splat down to 29, i.e. 3.2x). Real captures are spatially
    coherent and do far better — the same field sizes laid out on a plane
    compress 41x. Asserting the plan's 10x here would be asserting a property
    of the test data, not of the encoder.
    """
    c = _cloud(20_000, sh_per_channel=3)
    p = write_spz(tmp_path / "s.spz", c)
    raw = sum(a.astype(np.float32).nbytes
              for a in (c.positions, c.scales, c.rotations, c.alphas, c.colors, c.sh))
    ratio = raw / p.stat().st_size
    assert ratio > 3.0, f"only {ratio:.2f}x ({p.stat().st_size} vs raw {raw})"


def test_spz_compresses_coherent_data_far_better(tmp_path):
    """Which is the case that actually matters: a real capture is not noise."""
    n = 20_000
    positions = np.column_stack([
        np.repeat(np.linspace(0.0, 4.0, 200), 100),
        np.tile(np.linspace(0.0, 2.4, 100), 200),
        np.zeros(n),
    ])
    c = GaussianCloud(
        positions=positions,
        scales=np.full((n, 3), -3.0),
        rotations=np.tile([0.0, 0.0, 0.0, 1.0], (n, 1)),
        alphas=np.full(n, 0.8),
        colors=np.tile([0.5, 0.5, 0.5], (n, 1)),
        sh=np.tile(np.linspace(-0.5, 0.5, 9), (n, 1)),
    )
    p = write_spz(tmp_path / "s.spz", c)
    raw = sum(a.astype(np.float32).nbytes
              for a in (c.positions, c.scales, c.rotations, c.alphas, c.colors, c.sh))
    assert raw / p.stat().st_size > 10.0


def test_spz_rejects_mismatched_field_lengths():
    with pytest.raises(ValueError):
        GaussianCloud(
            positions=np.zeros((10, 3)), scales=np.zeros((9, 3)),
            rotations=np.zeros((10, 4)), alphas=np.zeros(10), colors=np.zeros((10, 3)),
        )


# -- PDF --------------------------------------------------------------------


def test_pdf_floorplan_is_a_valid_single_page_document(tmp_path):
    p = write_pdf_floorplan(tmp_path / "plan.pdf", _layout())
    data = p.read_bytes()
    assert data.startswith(b"%PDF-1.4")
    assert data.rstrip().endswith(b"%%EOF")
    assert b"/Type /Page" in data
    assert data.count(b"/Type /Pages") == 1
    assert b"startxref" in data


def test_pdf_states_the_measurements(tmp_path):
    """A printed plan with no numbers on it is a picture, not a deliverable."""
    layout = _layout()
    data = write_pdf_floorplan(tmp_path / "plan.pdf", layout).read_bytes()
    assert b"14.07 m2" in data
    assert b"4.20 m" in data and b"3.35 m" in data
    assert b"ceiling 2.45 m" in data
    assert b"1 m =" in data, "the scale must be stated so the plan can be measured"


def test_pdf_xref_offsets_point_at_their_objects(tmp_path):
    """A wrong xref table still opens in forgiving viewers and fails in strict
    ones, so check the offsets actually land on their object headers."""
    p = write_pdf_floorplan(tmp_path / "plan.pdf", _layout())
    data = p.read_bytes()
    start = int(data.rsplit(b"startxref", 1)[1].split(b"%%EOF")[0].strip())
    lines = data[start:].split(b"\n")
    count = int(lines[1].split()[1])
    for i in range(1, count):
        offset = int(lines[2 + i].split()[0])
        assert data[offset:offset + len(f"{i} 0 obj".encode())] == f"{i} 0 obj".encode()


def test_pdf_refuses_an_empty_footprint(tmp_path):
    layout = _layout()
    layout.footprint = []
    with pytest.raises(ValueError):
        write_pdf_floorplan(tmp_path / "plan.pdf", layout)


# -- frame guard ------------------------------------------------------------


def test_assert_frame_catches_a_left_handed_artifact():
    with pytest.raises(FrameMismatch):
        assert_frame(CoordinateFrame(handedness=Handedness.LEFT))
    with pytest.raises(FrameMismatch):
        assert_frame(CoordinateFrame(up_axis=UpAxis.Z_UP))
    assert_frame(CANONICAL_FRAME)
