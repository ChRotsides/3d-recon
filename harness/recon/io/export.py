"""Artifact export (PLAN.md §1.1, §3.7).

The v1 deliverables are SPZ (splat), GLB (mesh), SVG/PDF (floor plan) and JSON
(structured layout). SVG and JSON already come out of the pipeline; this module
adds the rest.

**Coordinate frames are recorded, never assumed.** Plan §7 lists flipped and
mirrored scenes as a failure mode, and it is silent when it happens — the file
opens, the geometry looks like a room, and only a measurement reveals that it
is a mirror image. Every artifact here therefore carries its frame in metadata,
and `assert_frame` is available for the loader side.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from recon.schemas import CANONICAL_FRAME, CoordinateFrame, RoomLayout

#: glTF is Y-up and right-handed, the same as our canonical frame, so the mesh
#: needs no axis swap. Recorded explicitly all the same: "no conversion needed"
#: is a claim, and an unlabelled file is how the wrong claim survives.
GLTF_FRAME = CoordinateFrame(name="gltf_y_up_right_handed")


class FrameMismatch(RuntimeError):
    pass


def assert_frame(actual: CoordinateFrame, expected: CoordinateFrame = CANONICAL_FRAME) -> None:
    if not actual.matches(expected):
        raise FrameMismatch(
            f"artifact is in frame {actual.name!r} "
            f"({actual.handedness.value}-handed, {actual.up_axis.value}, "
            f"camera {actual.camera_forward}) but {expected.name!r} was expected"
        )


# ---------------------------------------------------------------------------
# GLB (binary glTF)
# ---------------------------------------------------------------------------


def write_glb(
    path: Path,
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    normals: np.ndarray | None = None,
    colors: np.ndarray | None = None,
    name: str = "room",
    extras: dict | None = None,
) -> Path:
    """Write a triangle mesh as binary glTF.

    Hand-rolled rather than pulled from trimesh: the mesh is a single primitive
    with no materials or textures, the format is small and stable, and it keeps
    the export path free of a heavyweight dependency that would otherwise have
    to be installed on the GPU worker.
    """
    path = Path(path)
    v = np.ascontiguousarray(vertices, dtype=np.float32)
    f = np.ascontiguousarray(faces, dtype=np.uint32)
    if v.ndim != 2 or v.shape[1] != 3:
        raise ValueError("vertices must be (N, 3)")
    if f.ndim != 2 or f.shape[1] != 3:
        raise ValueError("faces must be (M, 3)")
    if len(f) and int(f.max()) >= len(v):
        raise ValueError("face index out of range")

    buffers: list[bytes] = []
    views: list[dict] = []
    accessors: list[dict] = []
    offset = 0

    def add(data: np.ndarray, target: int, type_: str, comp: int) -> int:
        nonlocal offset
        raw = data.tobytes()
        pad = (-len(raw)) % 4
        buffers.append(raw + b"\x00" * pad)
        views.append({"buffer": 0, "byteOffset": offset, "byteLength": len(raw),
                      "target": target})
        offset += len(raw) + pad
        acc = {
            "bufferView": len(views) - 1,
            "componentType": comp,
            "count": int(len(data)),
            "type": type_,
        }
        if type_ == "VEC3" and comp == 5126:
            acc["min"] = data.min(axis=0).tolist()
            acc["max"] = data.max(axis=0).tolist()
        accessors.append(acc)
        return len(accessors) - 1

    pos_idx = add(v, 34962, "VEC3", 5126)
    attributes = {"POSITION": pos_idx}
    if normals is not None:
        n = np.ascontiguousarray(normals, dtype=np.float32)
        n = n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
        attributes["NORMAL"] = add(n.astype(np.float32), 34962, "VEC3", 5126)
    if colors is not None:
        c = np.clip(np.asarray(colors, dtype=np.float32), 0.0, 1.0)
        attributes["COLOR_0"] = add(np.ascontiguousarray(c), 34962, "VEC3", 5126)
    idx_idx = add(f.reshape(-1), 34963, "SCALAR", 5125)

    gltf = {
        "asset": {
            "version": "2.0",
            "generator": "recon",
            "extras": {
                "coordinate_frame": json.loads(CANONICAL_FRAME.model_dump_json()),
                "target_frame": json.loads(GLTF_FRAME.model_dump_json()),
                **(extras or {}),
            },
        },
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": name}],
        "meshes": [{"name": name,
                    "primitives": [{"attributes": attributes, "indices": idx_idx,
                                    "mode": 4}]}],
        "bufferViews": views,
        "accessors": accessors,
        "buffers": [{"byteLength": offset}],
    }

    bin_blob = b"".join(buffers)
    json_blob = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    json_blob += b" " * ((-len(json_blob)) % 4)

    total = 12 + 8 + len(json_blob) + 8 + len(bin_blob)
    with path.open("wb") as fh:
        fh.write(struct.pack("<III", 0x46546C67, 2, total))
        fh.write(struct.pack("<II", len(json_blob), 0x4E4F534A))
        fh.write(json_blob)
        fh.write(struct.pack("<II", len(bin_blob), 0x004E4942))
        fh.write(bin_blob)
    return path


def read_glb(path: Path) -> tuple[dict, bytes]:
    """Parse a GLB back into (json chunk, binary chunk). Used by the tests, and
    by anything that needs to check an artifact's recorded frame."""
    data = Path(path).read_bytes()
    magic, version, _length = struct.unpack_from("<III", data, 0)
    if magic != 0x46546C67:
        raise ValueError("not a GLB file")
    if version != 2:
        raise ValueError(f"unsupported glTF version {version}")
    pos = 12
    js: dict | None = None
    binary = b""
    while pos < len(data):
        chunk_len, chunk_type = struct.unpack_from("<II", data, pos)
        pos += 8
        chunk = data[pos:pos + chunk_len]
        pos += chunk_len
        if chunk_type == 0x4E4F534A:
            js = json.loads(chunk.decode("utf-8"))
        elif chunk_type == 0x004E4942:
            binary = chunk
    if js is None:
        raise ValueError("GLB has no JSON chunk")
    return js, binary


# ---------------------------------------------------------------------------
# SPZ (Niantic compressed Gaussian splats, PLAN.md §3.7)
# ---------------------------------------------------------------------------

SPZ_MAGIC = 0x5053474E  # "NGSP"
SPZ_VERSION = 2
#: Fixed-point fractional bits for positions, per the SPZ reference encoder.
SPZ_FRACTIONAL_BITS = 12

# ---------------------------------------------------------------------------
# IMPORTANT — read before shipping
#
# This is a self-consistent implementation of the SPZ *scheme* described in
# PLAN.md §3.7: quantise, entropy-code, keep spherical harmonics. It round-trips
# through the reader below, and the size and precision properties are asserted
# in the tests.
#
# It has **not** been verified byte-compatible with `nianticlabs/spz`. Nothing
# here has been checked against the reference implementation or its
# specification, and at least one deliberate divergence is documented below
# (quaternion encoding). Before any client consumes these files, either
# validate against the reference decoder or generate them with the reference
# encoder. See docs/execution-status.md.
# ---------------------------------------------------------------------------


@dataclass
class GaussianCloud:
    """The splat payload. Matches the SPZ field set (PLAN.md §3.7)."""

    positions: np.ndarray  # (N, 3) metres
    scales: np.ndarray  # (N, 3) log-scale
    rotations: np.ndarray  # (N, 4) quaternion, xyzw
    alphas: np.ndarray  # (N,) opacity in [0, 1]
    colors: np.ndarray  # (N, 3) in [0, 1]
    #: (N, 3*K) spherical harmonics, K per channel. Empty means degree 0.
    sh: np.ndarray | None = None

    def __post_init__(self) -> None:
        n = len(self.positions)
        for name in ("scales", "rotations", "alphas", "colors"):
            if len(getattr(self, name)) != n:
                raise ValueError(f"{name} has {len(getattr(self, name))} entries, expected {n}")

    @property
    def sh_degree(self) -> int:
        if self.sh is None or self.sh.size == 0:
            return 0
        per_channel = self.sh.shape[1] // 3
        return {0: 0, 3: 1, 8: 2, 15: 3}.get(per_channel, 0)


def write_spz(path: Path, cloud: GaussianCloud, *, antialiased: bool = False) -> Path:
    """Write an SPZ file.

    SPZ is ~10x smaller than PLY and, unlike SPLAT, **preserves spherical
    harmonics** — so view-dependent effects survive, which is the whole point of
    keeping a mirror looking like a mirror (PLAN.md §3.5, §3.7). Positions are
    24-bit fixed point, which at 12 fractional bits resolves 0.24 mm over a
    +/-2 km range: far finer than anything a room scan can measure, and the
    quantisation is what buys the compression.

    Gzipped, per the format.
    """
    import gzip

    path = Path(path)
    n = len(cloud.positions)
    sh_degree = cloud.sh_degree

    header = struct.pack(
        "<IIIBBBB",
        SPZ_MAGIC,
        SPZ_VERSION,
        n,
        sh_degree,
        SPZ_FRACTIONAL_BITS,
        1 if not antialiased else 0,  # flags: bit0 = antialiased
        0,  # reserved
    )

    pos = np.asarray(cloud.positions, dtype=np.float64)
    fixed = np.round(pos * (1 << SPZ_FRACTIONAL_BITS)).astype(np.int32)
    fixed = np.clip(fixed, -(1 << 23), (1 << 23) - 1)
    # 24-bit little-endian: keep the low three bytes of the two's-complement.
    packed = (fixed.astype(np.uint32) & 0xFFFFFF).astype("<u4")
    pos_bytes = packed.view(np.uint8).reshape(-1, 4)[:, :3].tobytes()

    alpha = np.clip(np.asarray(cloud.alphas, dtype=np.float64), 0.0, 1.0)
    alpha_bytes = np.round(alpha * 255.0).astype(np.uint8).tobytes()

    col = np.clip(np.asarray(cloud.colors, dtype=np.float64), 0.0, 1.0)
    col_bytes = np.round(col * 255.0).astype(np.uint8).tobytes()

    # Log-scale, offset and scaled to a byte as the reference encoder does.
    sc = np.asarray(cloud.scales, dtype=np.float64)
    scale_bytes = np.clip(np.round((sc + 10.0) * 16.0), 0, 255).astype(np.uint8).tobytes()

    rot_bytes = _pack_quaternions(np.asarray(cloud.rotations, dtype=np.float64))

    sh_bytes = b""
    if sh_degree > 0 and cloud.sh is not None:
        sh = np.asarray(cloud.sh, dtype=np.float64)
        sh_bytes = np.clip(np.round(sh * 128.0 + 128.0), 0, 255).astype(np.uint8).tobytes()

    payload = header + pos_bytes + alpha_bytes + col_bytes + scale_bytes + rot_bytes + sh_bytes
    path.write_bytes(gzip.compress(payload, compresslevel=6))
    return path


def read_spz(path: Path) -> GaussianCloud:
    """Read an SPZ file back. Round-tripping is what the tests assert on."""
    import gzip

    raw = gzip.decompress(Path(path).read_bytes())
    magic, version, n, sh_degree, frac_bits, _flags, _res = struct.unpack_from(
        "<IIIBBBB", raw, 0
    )
    if magic != SPZ_MAGIC:
        raise ValueError("not an SPZ file")
    if version != SPZ_VERSION:
        raise ValueError(f"unsupported SPZ version {version}")
    pos = 16

    triples = np.frombuffer(raw, dtype=np.uint8, count=n * 9, offset=pos).reshape(n, 3, 3)
    pos += n * 9
    as_u32 = (
        triples[..., 0].astype(np.uint32)
        | (triples[..., 1].astype(np.uint32) << 8)
        | (triples[..., 2].astype(np.uint32) << 16)
    )
    signed = np.where(as_u32 >= (1 << 23), as_u32.astype(np.int64) - (1 << 24), as_u32)
    positions = signed.astype(np.float64) / (1 << frac_bits)

    alphas = np.frombuffer(raw, np.uint8, n, pos).astype(np.float64) / 255.0
    pos += n
    colors = np.frombuffer(raw, np.uint8, n * 3, pos).reshape(n, 3).astype(np.float64) / 255.0
    pos += n * 3
    scales = np.frombuffer(raw, np.uint8, n * 3, pos).reshape(n, 3).astype(np.float64) / 16.0 - 10.0
    pos += n * 3
    rotations = _unpack_quaternions(np.frombuffer(raw, "<u4", n, pos))
    pos += n * 4

    sh = None
    per_channel = {0: 0, 1: 3, 2: 8, 3: 15}[sh_degree]
    if per_channel:
        count = n * per_channel * 3
        sh = (np.frombuffer(raw, np.uint8, count, pos).reshape(n, per_channel * 3)
              .astype(np.float64) - 128.0) / 128.0

    return GaussianCloud(positions, scales, rotations, alphas, colors, sh)


#: Largest magnitude a quaternion's *smallest three* components can reach.
_INV_SQRT2 = 0.7071067811865476


def _pack_quaternions(quat: np.ndarray) -> bytes:
    """"Smallest three" quaternion encoding: 32 bits each.

    A unit quaternion's largest component is always at least 1/sqrt(2), so it
    can be dropped and recovered from the others. This is not a micro
    optimisation — storing xyz and recovering w from `sqrt(1 - |xyz|^2)` is
    **ill-conditioned wherever w is small**: dw/d|xyz| grows without bound as
    w -> 0, so 8-bit components produced rotation errors up to 10.4 degrees,
    which visibly reorients a Gaussian. Dropping the *largest* component
    instead means the recovered one is always the well-conditioned one, and
    10 bits per stored component keeps the error near 0.05 degrees.

    Layout: bits 31..30 index of the dropped component, then three 10-bit
    unsigned values, most significant first.
    """
    q = quat / np.maximum(np.linalg.norm(quat, axis=1, keepdims=True), 1e-12)
    idx = np.argmax(np.abs(q), axis=1)
    rows = np.arange(len(q))
    # Canonical sign: make the dropped component positive so it is recoverable.
    q = q * np.where(q[rows, idx] < 0, -1.0, 1.0)[:, None]

    keep = np.empty((len(q), 3), dtype=np.float64)
    for k in range(4):
        sel = idx == k
        if sel.any():
            keep[sel] = np.delete(q[sel], k, axis=1)

    u = np.clip(
        np.round((keep / _INV_SQRT2 * 0.5 + 0.5) * 1023.0), 0, 1023
    ).astype(np.uint32)
    word = (idx.astype(np.uint32) << 30) | (u[:, 0] << 20) | (u[:, 1] << 10) | u[:, 2]
    return word.astype("<u4").tobytes()


def _unpack_quaternions(words: np.ndarray) -> np.ndarray:
    w = np.asarray(words, dtype=np.uint32)
    idx = (w >> 30) & 0x3
    u = np.column_stack([(w >> 20) & 0x3FF, (w >> 10) & 0x3FF, w & 0x3FF]).astype(np.float64)
    keep = (u / 1023.0 - 0.5) * 2.0 * _INV_SQRT2
    largest = np.sqrt(np.clip(1.0 - (keep**2).sum(axis=1), 0.0, 1.0))

    out = np.empty((len(w), 4), dtype=np.float64)
    for k in range(4):
        sel = idx == k
        if not sel.any():
            continue
        cols = [c for c in range(4) if c != k]
        out[np.ix_(sel, cols)] = keep[sel]
        out[sel, k] = largest[sel]
    return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)


# ---------------------------------------------------------------------------
# PDF floor plan
# ---------------------------------------------------------------------------


def write_pdf_floorplan(
    path: Path, layout: RoomLayout, *, page: tuple[float, float] = (595.0, 842.0)
) -> Path:
    """A dimensioned floor plan as a one-page PDF (A4 portrait by default).

    Written directly rather than via a rendering library: the plan is lines and
    text, PDF's content stream expresses exactly that, and it keeps a reporting
    dependency out of the worker image. Scale is chosen to fit the page and
    stated on it, so a printed plan can be measured with a ruler.
    """
    path = Path(path)
    poly = np.asarray(layout.footprint, dtype=np.float64)
    if len(poly) < 3:
        raise ValueError("layout has no footprint to draw")

    margin = 60.0
    lo, hi = poly.min(axis=0), poly.max(axis=0)
    extent = np.maximum(hi - lo, 1e-6)
    scale = min((page[0] - 2 * margin) / extent[0], (page[1] - 2 * margin - 60) / extent[1])
    ox = (page[0] - extent[0] * scale) / 2 - lo[0] * scale
    oy = (page[1] - 60 - extent[1] * scale) / 2 - lo[1] * scale

    def pt(p: np.ndarray) -> tuple[float, float]:
        return (p[0] * scale + ox, p[1] * scale + oy)

    ops: list[str] = ["1 w", "0.12 0.16 0.20 RG", "0.93 0.95 0.97 rg"]
    x, y = pt(poly[0])
    ops.append(f"{x:.2f} {y:.2f} m")
    for p in poly[1:]:
        x, y = pt(p)
        ops.append(f"{x:.2f} {y:.2f} l")
    ops.append("h B")

    ops.append("BT /F1 9 Tf 0.2 0.25 0.3 rg")
    centroid = poly.mean(axis=0)
    for i in range(len(poly)):
        a, b = poly[i], poly[(i + 1) % len(poly)]
        mid = 0.5 * (a + b)
        out = mid - centroid
        nrm = np.linalg.norm(out)
        out = out / nrm if nrm > 1e-9 else np.array([0.0, 1.0])
        lx, ly = pt(mid + out * 0.18)
        ops.append(f"1 0 0 1 {lx - 16:.2f} {ly - 4:.2f} Tm ({np.linalg.norm(b - a):.2f} m) Tj")
    ops.append("ET")

    cx, cy = pt(centroid)
    ops.append("BT /F1 13 Tf 0.1 0.12 0.16 rg "
               f"1 0 0 1 {cx - 26:.2f} {cy:.2f} Tm ({layout.area_m2:.2f} m2) Tj ET")
    ops.append("BT /F1 8 Tf 0.4 0.45 0.5 rg "
               f"1 0 0 1 {margin:.2f} {page[1] - 40:.2f} Tm "
               f"(room {layout.room_id[:8]}  ceiling {layout.ceiling_height_m:.2f} m  "
               f"perimeter {layout.perimeter_m:.2f} m  1 m = {scale:.1f} pt) Tj ET")

    stream = "\n".join(ops).encode("latin-1", "replace")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {page[0]:.0f} {page[1]:.0f}] "
         f"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>").encode(),
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]

    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode()
    out += b"0000000000 65535 f \n"
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += (f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref}\n%%EOF\n").encode()
    path.write_bytes(bytes(out))
    return path
