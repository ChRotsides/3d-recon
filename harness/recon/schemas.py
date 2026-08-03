"""Executable definitions of the wire schemas from PLAN.md §5.

Coordinate convention (normative, PLAN.md §5): right-handed, Y-up. Poses are
world<-camera. The camera looks down -Z with +X right, +Y up (OpenGL / ARKit).
ARCore bundles are converted at the client boundary; `Device.ar_platform` records
where the data came from and `CoordinateFrame` travels with every artifact so the
loader can assert instead of silently shipping a mirrored scene (PLAN.md §7).
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator

Vec3 = Annotated[list[float], Field(min_length=3, max_length=3)]
Vec4 = Annotated[list[float], Field(min_length=4, max_length=4)]
Mat3 = Annotated[list[list[float]], Field(min_length=3, max_length=3)]


class Handedness(str, Enum):
    RIGHT = "right"
    LEFT = "left"


class UpAxis(str, Enum):
    Y_UP = "y_up"
    Z_UP = "z_up"


class CoordinateFrame(BaseModel):
    """Travels with every artifact. Asserted on load. See PLAN.md §7."""

    model_config = ConfigDict(extra="forbid")

    handedness: Handedness = Handedness.RIGHT
    up_axis: UpAxis = UpAxis.Y_UP
    #: Direction the camera looks along in its own frame.
    camera_forward: Literal["-z", "+z"] = "-z"
    name: str = "recon_world_v1"

    def matches(self, other: CoordinateFrame) -> bool:
        return (
            self.handedness == other.handedness
            and self.up_axis == other.up_axis
            and self.camera_forward == other.camera_forward
        )


CANONICAL_FRAME = CoordinateFrame()


class ARPlatform(str, Enum):
    ARKIT = "arkit"
    ARCORE = "arcore"
    SYNTHETIC = "synthetic"


class TrackingState(str, Enum):
    NORMAL = "normal"
    LIMITED = "limited"
    RELOCALIZING = "relocalizing"


class Device(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str
    os: str
    ar_platform: ARPlatform
    ar_version: str = ""


class Intrinsics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    #: OpenCV ordering [k1, k2, p1, p2, k3]. Empty means already undistorted.
    distortion: list[float] = Field(default_factory=list)

    def matrix(self) -> np.ndarray:
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )


class Pose(BaseModel):
    """world<-camera rigid transform: x_world = R @ x_camera + t."""

    model_config = ConfigDict(extra="forbid")

    R: Mat3
    t: Vec3

    @field_validator("R")
    @classmethod
    def _check_rotation(cls, v: list[list[float]]) -> list[list[float]]:
        m = np.asarray(v, dtype=np.float64)
        if not np.allclose(m @ m.T, np.eye(3), atol=1e-4):
            raise ValueError("R is not orthonormal")
        if not np.isclose(np.linalg.det(m), 1.0, atol=1e-4):
            raise ValueError("R has determinant != 1 (mirrored or scaled)")
        return v

    def matrix(self) -> np.ndarray:
        m = np.eye(4, dtype=np.float64)
        m[:3, :3] = np.asarray(self.R, dtype=np.float64)
        m[:3, 3] = np.asarray(self.t, dtype=np.float64)
        return m

    def inverse_matrix(self) -> np.ndarray:
        """camera<-world, the extrinsic matrix fusion and rendering want."""
        r = np.asarray(self.R, dtype=np.float64)
        m = np.eye(4, dtype=np.float64)
        m[:3, :3] = r.T
        m[:3, 3] = -r.T @ np.asarray(self.t, dtype=np.float64)
        return m

    @classmethod
    def from_matrix(cls, m: np.ndarray) -> Pose:
        return cls(R=np.asarray(m)[:3, :3].tolist(), t=np.asarray(m)[:3, 3].tolist())

    @property
    def position(self) -> np.ndarray:
        return np.asarray(self.t, dtype=np.float64)

    @property
    def forward(self) -> np.ndarray:
        """View direction in world space (camera looks down -Z)."""
        return -np.asarray(self.R, dtype=np.float64)[:, 2]


class Keyframe(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    image: str
    timestamp_ns: int
    pose: Pose
    intrinsics: Intrinsics
    tracking_state: TrackingState = TrackingState.NORMAL
    #: Variance of Laplacian; higher is sharper.
    blur_score: float = 0.0
    #: Fraction of pixels at 0 or 255.
    exposure_clip_ratio: float = 0.0
    #: Optional sibling file holding float32 metric depth (synthetic / RGBD rigs).
    depth: str | None = None


class SparsePoint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    xyz: Vec3
    observations: list[str] = Field(default_factory=list)


class MeasurementType(str, Enum):
    USER_TAPE = "user_tape"
    DOOR_PRIOR = "door_prior"
    CEILING_PRIOR = "ceiling_prior"
    CAMERA_HEIGHT = "camera_height"
    VIO_SCALE = "vio_scale"
    MAPANYTHING_SCALE = "mapanything_scale"


class ReferenceMeasurement(BaseModel):
    """One distance constraint feeding the scale solver (PLAN.md §6.2).

    `point_a`/`point_b` are in the *reconstruction* frame; `measured_value_m` is
    the real-world truth. VIO_SCALE / MAPANYTHING_SCALE entries are unitless
    scale priors and carry no points.
    """

    model_config = ConfigDict(extra="forbid")

    type: MeasurementType
    point_a: Vec3 | None = None
    point_b: Vec3 | None = None
    measured_value_m: float
    sigma_m: float

    @field_validator("sigma_m")
    @classmethod
    def _positive_sigma(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("sigma_m must be > 0")
        return v


class Doorway(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    #: Plane in Hessian normal form [nx, ny, nz, d] with n.x + d = 0.
    plane: Vec4
    centroid: Vec3
    width_m: float = 0.0
    height_m: float = 0.0


class SessionBundle(BaseModel):
    """Client -> server payload. One room, one capture session."""

    model_config = ConfigDict(extra="forbid")

    session_id: str
    device: Device
    room_id: str
    #: World-frame down vector (unit, points toward the floor).
    gravity: Vec3 = [0.0, -1.0, 0.0]
    frame: CoordinateFrame = Field(default_factory=CoordinateFrame)
    keyframes: list[Keyframe] = Field(default_factory=list)
    sparse_points: list[SparsePoint] = Field(default_factory=list)
    reference_measurements: list[ReferenceMeasurement] = Field(default_factory=list)
    doorways: list[Doorway] = Field(default_factory=list)
    schema_version: int = 1

    @field_validator("gravity")
    @classmethod
    def _unit_gravity(cls, v: list[float]) -> list[float]:
        n = float(np.linalg.norm(v))
        if not np.isclose(n, 1.0, atol=1e-3):
            raise ValueError(f"gravity must be a unit vector, got norm {n:.4f}")
        return v

    @property
    def up(self) -> np.ndarray:
        return -np.asarray(self.gravity, dtype=np.float64)


# --------------------------------------------------------------------------
# House graph (PLAN.md §5). Multi-floor is schema-only in v1 (PLAN.md §1.2).
# --------------------------------------------------------------------------


class EdgeSource(str, Enum):
    USER_TAGGED = "user_tagged"
    AUTO_REGISTERED = "auto_registered"
    CONTINUOUS_TRACKING = "continuous_tracking"


class SE2Transform(BaseModel):
    model_config = ConfigDict(extra="forbid")

    yaw_rad: float = 0.0
    tx: float = 0.0
    tz: float = 0.0

    def matrix(self) -> np.ndarray:
        """4x4 world transform: yaw about +Y, translation in the floor plane."""
        c, s = np.cos(self.yaw_rad), np.sin(self.yaw_rad)
        m = np.eye(4, dtype=np.float64)
        m[0, 0], m[0, 2] = c, s
        m[2, 0], m[2, 2] = -s, c
        m[0, 3], m[2, 3] = self.tx, self.tz
        return m


class RoomEdge(BaseModel):
    model_config = ConfigDict(extra="forbid")

    from_room: str
    to_room: str
    #: One doorway id per side.
    via_doorway: Annotated[list[str], Field(min_length=2, max_length=2)]
    transform: SE2Transform
    #: 3x3 over (yaw, tx, tz).
    covariance: Mat3
    source: EdgeSource
    icp_rmse_m: float | None = None


class Floor(BaseModel):
    model_config = ConfigDict(extra="forbid")

    level: int = 0
    rooms: list[str] = Field(default_factory=list)
    edges: list[RoomEdge] = Field(default_factory=list)


class HouseGraph(BaseModel):
    model_config = ConfigDict(extra="forbid")

    property_id: str
    floors: list[Floor] = Field(default_factory=list)
    frame: CoordinateFrame = Field(default_factory=CoordinateFrame)
    schema_version: int = 1


# --------------------------------------------------------------------------
# Reconstruction outputs
# --------------------------------------------------------------------------


class PlaneKind(str, Enum):
    FLOOR = "floor"
    CEILING = "ceiling"
    WALL = "wall"
    OTHER = "other"


class Plane(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: PlaneKind
    #: Hessian normal form [nx, ny, nz, d].
    coeffs: Vec4
    inlier_count: int = 0
    #: Axis-aligned extent of the inliers, world frame.
    bbox_min: Vec3 | None = None
    bbox_max: Vec3 | None = None

    @property
    def normal(self) -> np.ndarray:
        return np.asarray(self.coeffs[:3], dtype=np.float64)

    @property
    def offset(self) -> float:
        return float(self.coeffs[3])


class Opening(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["door", "window"]
    wall_index: int
    #: Along-wall start/end in the wall's own 1D parameterisation, metres.
    u_start_m: float
    u_end_m: float
    height_min_m: float
    height_max_m: float


class RoomLayout(BaseModel):
    """The measurable deliverable: a dimensioned floor plan (PLAN.md §6.4)."""

    model_config = ConfigDict(extra="forbid")

    room_id: str
    #: Closed CCW polygon of floor corners, (x, z) metres.
    footprint: list[Annotated[list[float], Field(min_length=2, max_length=2)]]
    floor_height_m: float
    ceiling_height_m: float
    area_m2: float
    perimeter_m: float
    wall_lengths_m: list[float]
    planes: list[Plane] = Field(default_factory=list)
    openings: list[Opening] = Field(default_factory=list)
    scale_applied: float = 1.0
    frame: CoordinateFrame = Field(default_factory=CoordinateFrame)
