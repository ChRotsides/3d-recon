"""Depth Anything 3 adapter — pose-conditioned mode (PLAN.md §3.1).

The whole point of picking DA3 is that ARKit/ARCore hand us poses and intrinsics
for free, which puts us in DA3's strongest configuration. This adapter therefore
*always* feeds the bundle's poses and intrinsics in and never asks DA3 to
re-solve the cameras.

The dev box's GTX TITANs are Kepler and below the oldest architecture current
PyTorch wheels ship kernels for. A local CUDA 11.8 build targeting ``sm_35``
can still run DA3-SMALL, but Kepler cannot execute DA3's FP16/BF16 autocast
path. `_usable_device` accepts the custom build, while `cuda_requires_fp32`
selects full precision and replaces the upstream API's hard-coded autocast
wrapper. Stock wheels still fall back to CPU instead of failing at the first
kernel launch. Import of torch is deferred to `estimate()` so the rest of the
harness stays importable without it.

The call into `DepthAnything3.inference` is written against the installed
package's actual signature, which is not the one this adapter originally
assumed: it takes a *list of numpy images* plus numpy camera matrices and does
its own normalisation and resizing, and pose conditioning is implied by passing
extrinsics rather than requested by a flag.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, distribution
from types import MethodType
from typing import Any

import numpy as np

from recon.depth.anchor import AffineFit, fit_affine_depth
from recon.depth.base import DepthResult
from recon.frames import world_to_camera_cv
from recon.io.bundle import BundleDir
from recon.torch_compat import (
    compatible_dtype,
    configure_cuda_for_legacy_device,
    cuda_requires_fp32,
)

#: Published checkpoints, with their verified weight licences
#: (docs/licensing-audit.md §1.2). The plan names DA3-Large for the server, but
#: the upstream model zoo lists LARGE and GIANT as CC BY-NC 4.0 while the
#: HuggingFace card for LARGE says Apache — an unresolved contradiction. Until
#: ByteDance clarifies, only the Apache rows are usable commercially, so the
#: benchmark grid must include DA3-BASE and DA3METRIC-LARGE as the
#: commercial-path candidates rather than assuming LARGE will be available.
VARIANTS = {
    "small": "depth-anything/DA3-SMALL",
    "base": "depth-anything/DA3-BASE",
    "large": "depth-anything/DA3-LARGE-1.1",
    "giant": "depth-anything/DA3-GIANT-1.1",
    "metric-large": "depth-anything/DA3METRIC-LARGE",
    "mono-large": "depth-anything/DA3MONO-LARGE",
}

#: True when the checkpoint is safe for commercial use. Asserted at load time so
#: a non-commercial model cannot reach production by accident.
#: Fewest sparse points that may set a window's scale. Below this the median is
#: noise, and silently scaling a whole window by noise is worse than not scaling.
_MIN_ANCHOR_POINTS = 40

#: Fewest visible points before a *frame* may set its own scale rather than
#: inherit the window's. Lower than the window bar because the estimate is a
#: median and the fallback is close by, but not so low that a corner of a wall
#: with four features can move a whole frame.
_MIN_FRAME_POINTS = 20

#: How far a *sparsely* measured frame's scale may sit from its window's median
#: before it is distrusted. A frame scaled by an outlier puts a step in a
#: continuous surface, which reads worse than the uniform error it removes.
_MAX_FRAME_DEVIATION = 0.25

#: Above this many visible points, a frame's own estimate is trusted outright:
#: a median over hundreds of measurements that disagrees with the window is
#: information, and clamping it would restore the error being corrected.
_TRUSTED_FRAME_POINTS = 100

#: Sanity bounds on the anchor. A correction outside these means the prediction
#: or the points are wrong in a way a scalar will not fix, and applying it would
#: hide the problem rather than solve it.
_MIN_ANCHOR_SCALE = 0.2
_MAX_ANCHOR_SCALE = 5.0

#: A frame whose inlier ratios still spread by more than this after robust
#: filtering does not support a single scalar depth correction.
_MAX_RATIO_DISPERSION = 0.20

COMMERCIAL_OK = {
    "small": True,
    "base": True,
    "metric-large": True,
    "mono-large": True,
    "large": False,  # contradictory; treat as NC until clarified
    "giant": False,  # CC BY-NC 4.0 in both the model zoo and the card
}


def _installed_source_revision() -> str | None:
    """Git commit recorded by pip for the installed DA3 package."""
    try:
        dist = distribution("depth-anything-3")
    except PackageNotFoundError:
        return None
    direct_url = dist.read_text("direct_url.json")
    if direct_url:
        try:
            return json.loads(direct_url).get("vcs_info", {}).get("commit_id")
        except json.JSONDecodeError:
            return None
    return None


@dataclass
class DA3Config:
    variant: str = "large"
    #: None resolves at load time to CUDA when there is a *usable* GPU, else CPU.
    #: Not decided here, because that would import torch at construction and the
    #: deferred import is what keeps the rest of the harness usable without it.
    device: str | None = None
    #: Frames per multi-view window. DA3 conditions across the window, so this
    #: trades VRAM against cross-frame consistency.
    window: int = 16
    #: Overlap between consecutive windows, in frames.
    window_overlap: int = 4
    #: None resolves with the device: float16 on a GPU, float32 on CPU, where
    #: half precision is either unimplemented or emulated and slower than the
    #: full-precision path it was meant to beat.
    dtype: str | None = None
    #: Long captures: DA3-Streaming's sliding-window inference stays under 12 GB
    #: (PLAN.md §3.1). It is a separate inference wrapper in the upstream repo
    #: (`da3_streaming/`), not a checkpoint, so it is selected by this flag
    #: rather than by `variant`.
    #: The resolution DA3 works at internally; it resizes the inputs itself and
    #: predicts at this size. The dominant cost knob on CPU.
    process_res: int = 504
    streaming_threshold: int = 120
    use_streaming: bool = False
    weights_dir: str | None = None
    #: Exact Hugging Face checkpoint revision; required by the production worker.
    checkpoint_revision: str | None = None
    #: Exact git commit of the installed Depth Anything 3 package.
    source_revision: str | None = None
    #: Refuse to load a checkpoint that cannot ship. Turn off for Phase-0
    #: research benchmarking, which non-commercial licences do permit.
    enforce_commercial_licence: bool = False
    #: Correct each *frame's* depth against the bundle's own sparse points.
    #: On by default because without it the depth carries DA3's camera-pose
    #: error as a scale error that nothing downstream can undo — see
    #: `_sparse_point_scales`. Turn off to measure the raw model, which is what
    #: the Phase-0 benchmark grid wants: the anchor would otherwise hide the
    #: differences between the models it exists to compare.
    anchor_to_sparse_points: bool = True
    #: Fit each frame to the metric depth map the *device* measured, where the
    #: capture carries one (`arcore_depth`). `d ≈ a·d_pred + b`, robustly, over
    #: every measured pixel — against the sparse anchor's few dozen feature
    #: points, and fitting the offset the sparse anchor cannot. Measured on
    #: bundle_e4a5da81: the sparse anchor alone leaves frames disagreeing with
    #: each other by 2.36x (small) and 2.58x (large). See `recon.depth.anchor`.
    #:
    #: Applied *after* the sparse anchor rather than instead of it, which is
    #: not a double correction: the fit is against measured metres and absorbs
    #: whatever scale it is handed. The ordering buys the fallback for free —
    #: a frame whose fit is refused keeps the sparse anchor.
    anchor_to_reference_depth: bool = True
    #: Space the correction is fitted in: "depth" or "disparity". Monocular
    #: models are affine-invariant in inverse depth about as often as in depth,
    #: and which one holds here is a question for measurement.
    reference_anchor_space: str = "depth"


def _usable_device(torch) -> str:
    """`cuda` only when a GPU is present *and* this torch build has kernels for it.

    `torch.cuda.is_available()` is not enough. The dev box has two GTX TITANs,
    which are Kepler (sm_35) and below the oldest architecture current PyTorch
    wheels are compiled for — so a CUDA build reports them as available and then
    fails at the first kernel launch with "no kernel image is available for
    execution on the device", which reads like a broken install rather than an
    obsolete card. Checking the compute capability against the build's own arch
    list turns that into an automatic, explicable fall back to CPU.
    """
    if not torch.cuda.is_available():
        return "cpu"
    try:
        major, minor = torch.cuda.get_device_capability(0)
        supported = {
            int(a.split("_")[1]) for a in torch.cuda.get_arch_list() if a.startswith("sm_")
        }
    except Exception:  # pragma: no cover - depends on the driver
        return "cpu"
    if supported and major * 10 + minor < min(supported):
        return "cpu"
    return "cuda"


def _install_full_precision_forward(api_model, torch) -> None:
    """Replace DA3's unconditional CUDA autocast wrapper on pre-FP16 GPUs.

    The public ``DepthAnything3.forward`` always enters FP16 or BF16 autocast.
    Its body otherwise just delegates to ``api_model.model``, so binding that
    delegation without autocast preserves the upstream inference pipeline while
    keeping every operation in FP32 on Kepler.
    """
    if getattr(api_model, "_recon_full_precision_forward", False):
        return

    def _forward_full_precision(self, *args, **kwargs):
        with torch.inference_mode():
            return self.model(*args, **kwargs)

    api_model.forward = MethodType(_forward_full_precision, api_model)
    api_model._recon_full_precision_forward = True


class DA3Depth:
    """Pose-conditioned Depth Anything 3."""

    def __init__(self, config: DA3Config | None = None) -> None:
        self.config = config or DA3Config()
        if self.config.variant not in VARIANTS:
            raise ValueError(
                f"unknown DA3 variant {self.config.variant!r}; have {sorted(VARIANTS)}"
            )
        if self.config.enforce_commercial_licence and not COMMERCIAL_OK[self.config.variant]:
            raise ValueError(
                f"DA3 variant {self.config.variant!r} is not cleared for commercial use "
                "(see docs/licensing-audit.md §1.2). Use 'base' or 'metric-large', or "
                "set enforce_commercial_licence=False for research benchmarking."
            )
        self.name = f"da3-{self.config.variant}"
        self._model = None
        #: Resolved in `_load`, once torch is importable.
        self._device = self.config.device or "cpu"
        self._dtype = self.config.dtype or "float32"
        self._legacy_cuda = False

    # -- lifecycle ----------------------------------------------------------

    def _load(self):
        if self._model is not None:
            return self._model
        try:
            import torch  # noqa: F401
            from depth_anything_3.api import DepthAnything3  # type: ignore
        except ImportError as exc:  # pragma: no cover - depends on GPU box
            raise RuntimeError(
                "Depth Anything 3 is not installed. On a GPU box:\n"
                "  pip install torch --index-url https://download.pytorch.org/whl/cu124\n"
                "  pip install git+https://github.com/ByteDance-Seed/Depth-Anything-3\n"
                f"(original error: {exc})"
            ) from exc

        if self.config.source_revision is not None:
            actual_revision = _installed_source_revision()
            if actual_revision != self.config.source_revision:
                raise RuntimeError(
                    "Depth Anything 3 source revision mismatch: configured "
                    f"{self.config.source_revision}, installed {actual_revision or 'unknown'}"
                )

        self._device = self.config.device or _usable_device(torch)
        needs_fp32 = cuda_requires_fp32(torch, self._device)
        self._legacy_cuda = configure_cuda_for_legacy_device(torch, self._device)
        requested_dtype = self.config.dtype or (
            "float32" if not self._device.startswith("cuda") else "float16"
        )
        self._dtype = compatible_dtype(torch, self._device, requested_dtype)

        repo = VARIANTS[self.config.variant]
        self._model = DepthAnything3.from_pretrained(
            self.config.weights_dir or repo,
            revision=self.config.checkpoint_revision,
        ).to(self._device)
        # `inference` converts its own inputs, so precision is a property of the
        # weights now rather than of what we hand it.
        if self._dtype == "float16":
            self._model = self._model.half()
        elif self._dtype == "bfloat16":
            self._model = self._model.to(dtype=torch.bfloat16)
        else:
            self._model = self._model.float()
        if needs_fp32:
            _install_full_precision_forward(self._model, torch)
        self._model.eval()
        return self._model

    # -- inference ----------------------------------------------------------

    def estimate(self, bundle: BundleDir) -> DepthResult:
        import torch

        model = self._load()
        kfs = bundle.bundle.keyframes
        cfg = self.config
        variant = cfg.variant
        if len(kfs) > cfg.streaming_threshold and variant != "streaming":
            # Not an error: the caller asked for DA3-Large on a long capture.
            # Windowing already bounds VRAM, so just note it in the model name.
            self.name = f"da3-{variant}-windowed"

        if self._device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(self._device)
        t0 = time.time()

        depth: dict[str, np.ndarray] = {}
        conf: dict[str, np.ndarray] = {}
        # Math attention is the compatible Kepler backend, but its FP32 memory
        # grows quadratically with the number of views. A 16-view DA3-SMALL
        # window exceeds a 6 GB GTX TITAN; four views at the normal 504px
        # resolution stay below 1 GB. Modern GPUs retain the configured window.
        window = min(cfg.window, 4) if self._legacy_cuda else cfg.window
        overlap = min(cfg.window_overlap, 1) if self._legacy_cuda else cfg.window_overlap

        scale_estimate: float | None = None
        anchor_scales: list[float] = []
        anchor_diagnostics: list[dict[str, Any]] = []
        affine_fits: dict[str, AffineFit] = {}
        affine_diagnostics: list[dict[str, Any]] = []
        for start in _window_starts(len(kfs), window, overlap):
            chunk = kfs[start : start + window]
            if not chunk:
                break

            # `inference` takes numpy images and numpy camera matrices and does
            # its own batching, normalisation and resizing to `process_res`.
            # Handing it tensors, or pre-scaling to 0..1, silently produces
            # nonsense rather than an error.
            images = [bundle.rgb(kf) for kf in chunk]
            extrinsics = np.stack([_world_to_camera_cv(kf) for kf in chunk])
            intrinsics = np.stack([kf.intrinsics.matrix() for kf in chunk])

            with torch.inference_mode():
                out = model.inference(
                    image=images,
                    extrinsics=extrinsics,
                    intrinsics=intrinsics,
                    # Passing the poses *is* the pose-conditioned mode of §3.1;
                    # aligning to their scale is what keeps the prediction in
                    # the ARCore metres the rest of the pipeline assumes.
                    align_to_input_ext_scale=_poses_support_scale_alignment(chunk),
                    process_res=cfg.process_res,
                )

            if out.scale_factor is not None:
                scale_estimate = float(out.scale_factor)
            pred = np.asarray(out.depth, dtype=np.float32)

            pred_conf = np.asarray(out.conf, dtype=np.float32) if out.conf is not None else None
            # DA3 predicts at its processing resolution, but everything
            # downstream indexes depth with the keyframe's own intrinsics, so a
            # mismatch here reads as geometry sampled from the wrong pixels.
            pred = np.stack(
                [_resize_to(d, kf) for d, kf in zip(pred, chunk, strict=True)]
            )
            if pred_conf is not None:
                pred_conf = np.stack(
                    [
                        _resize_to(c, kf)
                        for c, kf in zip(pred_conf, chunk, strict=True)
                    ]
                )

            # After the resize, not before: the anchor projects points with the
            # keyframe's own intrinsics, which only index the map once it is at
            # the keyframe's resolution.
            if cfg.anchor_to_sparse_points:
                window_diagnostics: list[dict[str, Any]] = []
                scales = _sparse_point_scales(
                    bundle, chunk, pred, diagnostics=window_diagnostics
                )
                for diagnostic in window_diagnostics:
                    diagnostic["window_start"] = start
                anchor_diagnostics.extend(window_diagnostics)
                if scales is not None:
                    pred = pred * np.asarray(scales, dtype=np.float32)[:, None, None]
                    anchor_scales.extend(scales)

            if cfg.anchor_to_reference_depth:
                for i, kf in enumerate(chunk):
                    reference = bundle.depth(kf)
                    if reference is None:
                        continue
                    fit = fit_affine_depth(
                        pred[i], reference, space=cfg.reference_anchor_space
                    )
                    if fit is None:
                        # Not an error. A frame facing a window has few metric
                        # returns to fit against; it keeps the sparse anchor.
                        affine_diagnostics.append({"keyframe": kf.id, "fitted": False})
                        continue
                    pred[i] = fit.apply(pred[i])
                    affine_fits[kf.id] = fit
                    affine_diagnostics.append(
                        {
                            "keyframe": kf.id,
                            "fitted": True,
                            "a": fit.a,
                            "b": fit.b,
                            "pixels": fit.pixels,
                            "inlier_fraction": fit.inlier_fraction,
                            "residual_m": fit.residual_m,
                            "space": fit.space,
                        }
                    )

            for i, kf in enumerate(chunk):
                if kf.id in depth:
                    # Overlapping window: keep the more confident prediction.
                    if pred_conf is None:
                        continue
                    better = pred_conf[i] > conf[kf.id]
                    depth[kf.id] = np.where(better, pred[i], depth[kf.id])
                    conf[kf.id] = np.maximum(conf[kf.id], pred_conf[i])
                else:
                    depth[kf.id] = pred[i].astype(np.float32)
                    if pred_conf is not None:
                        conf[kf.id] = pred_conf[i].astype(np.float32)

        # Kept for inspection rather than folded into `scale_estimate`: that
        # field feeds plan §6.2's scale solver as an *observation*, and the
        # anchor has already been applied to the depth here. Reporting it there
        # too would apply the same correction twice.
        self.anchor_scales = anchor_scales
        #: Per-keyframe affine corrections, for callers that want to inspect or
        #: report them. Keyed by keyframe id; absent means the frame was left
        #: on its sparse anchor.
        self.affine_fits = affine_fits

        vram = 0.0
        if self._device.startswith("cuda"):
            vram = torch.cuda.max_memory_allocated(self._device) / 1e9

        return DepthResult(
            depth=depth,
            confidence=conf,
            scale_estimate=scale_estimate,
            model_name=self.name,
            runtime_s=time.time() - t0,
            peak_vram_gb=vram,
            diagnostics={
                "sparse_anchor": anchor_diagnostics,
                "reference_anchor": affine_diagnostics,
            },
        )


def _window_starts(frame_count: int, window: int, overlap: int) -> list[int]:
    """Full-window starts plus one back-filled tail, without tiny duplicates."""
    if frame_count <= 0:
        return []
    if window <= 0:
        raise ValueError("DA3 window must be positive")
    overlap = min(max(overlap, 0), window - 1)
    if frame_count <= window:
        return [0]

    last = frame_count - window
    starts = list(range(0, last + 1, window - overlap))
    if starts[-1] != last:
        starts.append(last)
    return starts


def _poses_support_scale_alignment(chunk) -> bool:
    """Whether camera centres constrain DA3's 3-D Umeyama scale alignment."""
    if len(chunk) < 3:
        return False
    centres = np.asarray([kf.pose.t for kf in chunk], dtype=np.float64)
    return bool(np.linalg.matrix_rank(centres - centres.mean(axis=0)) >= 2)


def _world_to_camera_cv(kf) -> np.ndarray:
    """The bundle's pose as the world-to-camera matrix DA3 expects.

    Kept as a name here because the reason it matters is a DA3 result. Handing
    it the unconverted matrix does not fail — it conditions on cameras pointing
    the wrong way and returns depth that is inconsistent *between frames of the
    same window*, which fuses into a collapsed room rather than an obvious
    error. Measured on a real capture: the per-frame scale ratio against the
    bundle's sparse points spread 2.34x unconverted, 1.16x converted.

    The conversion itself lives in `recon.frames`, because the mesher needs the
    identical matrix and two copies of a sign convention is one too many.
    """
    return world_to_camera_cv(kf)


def _frame_ratios(pts: np.ndarray, kf, depth: np.ndarray) -> np.ndarray | None:
    """True/predicted depth at every sparse point visible in one keyframe.

    The bundle's sparse points are ARCore's triangulated features: metric, and
    in the same world frame as the poses. Projecting them into the frame and
    reading the prediction at the same pixel is a direct measurement of how far
    off the model's depth is, in the units the rest of the pipeline works in.
    """
    intr = kf.intrinsics
    ext = kf.pose.inverse_matrix()
    cam = pts @ ext[:3, :3].T + ext[:3, 3]
    z = -cam[:, 2]  # canonical frame: the camera looks down -Z
    ok = z > 0.2
    if not ok.any():
        return None
    u = np.round(intr.fx * (cam[:, 0] / z) + intr.cx).astype(np.int32)
    v = np.round(-intr.fy * (cam[:, 1] / z) + intr.cy).astype(np.int32)
    ok &= (u >= 0) & (u < intr.width) & (v >= 0) & (v < intr.height)
    if not ok.any():
        return None
    p = depth[v[ok], u[ok]]
    good = p > 1e-3
    if not good.any():
        return None
    # Median downstream rather than least squares: a projected point may land on
    # a surface that occludes it, and those outliers are all one-sided.
    return z[ok][good] / p[good]


def _robust_ratios(ratios: np.ndarray) -> tuple[np.ndarray, float]:
    """Reject occlusion/outlier ratios and report robust relative dispersion."""
    ratios = ratios[np.isfinite(ratios) & (ratios > 0)]
    if not len(ratios):
        return ratios, float("inf")
    centre = float(np.median(ratios))
    log_residual = np.abs(np.log(ratios / centre))
    mad = float(np.median(log_residual))
    # Keep at least a 10% band so quantised depth and sparse projections are not
    # rejected merely because the MAD happens to be zero.
    cutoff = max(3.0 * 1.4826 * mad, np.log1p(0.10))
    inliers = ratios[log_residual <= cutoff]
    if not len(inliers):
        return inliers, float("inf")
    dispersion = float(np.median(np.abs(inliers / np.median(inliers) - 1.0)))
    return inliers, dispersion


def _sparse_point_scales(
    bundle: BundleDir,
    chunk,
    pred: np.ndarray,
    *,
    diagnostics: list[dict[str, Any]] | None = None,
) -> list[float] | None:
    """A metric scale **per frame**, from the bundle's own sparse points.

    `align_to_input_ext_scale` brings a prediction into the input poses' scale by
    fitting DA3's *estimated* cameras onto them, so whatever error the model
    makes about its own camera positions lands in the depth as a scale error —
    ~1.5x on a real capture even with the extrinsic convention right. Nothing
    downstream can undo that: plan §6.2's scale stage multiplies poses and depth
    together, so it cannot fix a depth-only error, and every metric the plan is
    judged on is a length.

    Per frame, not per window. One scalar per window leaves the frames inside it
    disagreeing, and the windows disagreeing with each other: measured across a
    real capture the per-window scales ran 0.997 to 1.176, so the same wall
    fused up to 18% apart in depth — half a metre at 3 m — depending on which
    window happened to see it. That is precisely the smearing that makes a
    reconstruction look melted.

    Two guards keep the finer granularity from making things worse:

    * a frame with too few visible points takes the window's median instead of a
      scale estimated from a handful of pixels;
    * a frame whose estimate is wildly off the window's median takes the median
      too, since one frame scaled by an outlier is a step in the surface, which
      reads worse than a uniform error.

    Returns None when the window as a whole has too little to go on, leaving the
    prediction untouched rather than scaled by noise.
    """
    sparse_points = bundle.bundle.sparse_points
    if not sparse_points:
        return None

    per_frame: list[tuple[float, int, float] | None] = []
    frame_diagnostics: list[dict[str, Any]] = []
    pooled: list[np.ndarray] = []
    for d, kf in zip(pred, chunk, strict=True):
        # An AR feature being geometrically projectable does not mean it was
        # observed in this image: another wall can occlude it, or it may belong
        # to a different tracking segment. Observation IDs are the association
        # supplied by the capture system and are therefore mandatory.
        observed = [p for p in sparse_points if kf.id in p.observations]
        entry: dict[str, Any] = {
            "keyframe_id": kf.id,
            "support_count": len(observed),
            "inlier_count": 0,
            "ratio_dispersion": None,
            "raw_scale": None,
            "window_support_count": 0,
            "window_inlier_count": 0,
            "window_ratio_dispersion": None,
            "window_scale": None,
            "chosen_scale": None,
            "strategy": "unsupported",
        }
        frame_diagnostics.append(entry)
        if not observed:
            per_frame.append(None)
            continue
        pts = np.asarray([p.xyz for p in observed], dtype=np.float64)
        r = _frame_ratios(pts, kf, d)
        if r is None:
            per_frame.append(None)
            continue
        inliers, dispersion = _robust_ratios(r)
        entry["inlier_count"] = len(inliers)
        entry["ratio_dispersion"] = dispersion
        if not len(inliers):
            per_frame.append(None)
            continue
        scale = float(np.median(inliers))
        entry["raw_scale"] = scale
        pooled.append(inliers)
        per_frame.append((scale, len(inliers), dispersion))

    if not pooled:
        if diagnostics is not None:
            diagnostics.extend(frame_diagnostics)
        return None
    flat = np.concatenate(pooled)
    window_inliers, window_dispersion = _robust_ratios(flat)
    window = (
        float(np.median(window_inliers)) if len(window_inliers) else float("nan")
    )
    for diagnostic in frame_diagnostics:
        diagnostic["window_support_count"] = len(flat)
        diagnostic["window_inlier_count"] = len(window_inliers)
        diagnostic["window_ratio_dispersion"] = window_dispersion
        diagnostic["window_scale"] = window if np.isfinite(window) else None
    all_frames_strong = all(
        entry is not None
        and entry[1] >= _TRUSTED_FRAME_POINTS
        and entry[2] <= _MAX_RATIO_DISPERSION
        and _MIN_ANCHOR_SCALE < entry[0] < _MAX_ANCHOR_SCALE
        for entry in per_frame
    )
    if len(window_inliers) < _MIN_ANCHOR_POINTS or (
        window_dispersion > _MAX_RATIO_DISPERSION and not all_frames_strong
    ):
        for diagnostic in frame_diagnostics:
            diagnostic["strategy"] = "window_rejected"
        if diagnostics is not None:
            diagnostics.extend(frame_diagnostics)
        return None
    if not _MIN_ANCHOR_SCALE < window < _MAX_ANCHOR_SCALE:
        for diagnostic in frame_diagnostics:
            diagnostic["strategy"] = "window_rejected"
        if diagnostics is not None:
            diagnostics.extend(frame_diagnostics)
        return None

    out = []
    for entry, diagnostic in zip(per_frame, frame_diagnostics, strict=True):
        if entry is None:
            out.append(window)
            diagnostic["chosen_scale"] = window
            diagnostic["strategy"] = "window_fallback"
            continue
        s, n, dispersion = entry
        if (
            not (_MIN_ANCHOR_SCALE < s < _MAX_ANCHOR_SCALE)
            or n < _MIN_FRAME_POINTS
            or dispersion > _MAX_RATIO_DISPERSION
        ):
            out.append(window)
            diagnostic["chosen_scale"] = window
            diagnostic["strategy"] = "window_fallback"
            continue
        # The deviation guard exists to stop a handful of points from moving a
        # whole frame, so it applies to frames with a handful of points. Past
        # `_TRUSTED_FRAME_POINTS` the estimate is a median over hundreds of
        # measurements and disagreeing with the window is information, not
        # noise — clamping it there re-introduces exactly the per-frame error
        # this function exists to remove. Measured: a frame with 138 points
        # needing 1.37x was being pinned to the window's 1.00 and left wrong.
        if n < _TRUSTED_FRAME_POINTS and abs(s / window - 1.0) > _MAX_FRAME_DEVIATION:
            out.append(window)
            diagnostic["chosen_scale"] = window
            diagnostic["strategy"] = "window_fallback"
            continue
        out.append(s)
        diagnostic["chosen_scale"] = s
        diagnostic["strategy"] = "per_frame"
    if diagnostics is not None:
        diagnostics.extend(frame_diagnostics)
    return out


def _resize_to(a: np.ndarray, kf) -> np.ndarray:
    """Nearest-neighbour resample of a depth or confidence map to the keyframe.

    Nearest rather than bilinear on purpose: interpolating *across* a depth
    discontinuity invents a surface that slopes from the foreground object to
    the wall behind it, and the TSDF will happily fuse that invention into
    geometry. A blocky edge is honest; a smooth wrong one is not.
    """
    h, w = kf.intrinsics.height, kf.intrinsics.width
    if a.shape == (h, w):
        return a.astype(np.float32)
    ys = (np.arange(h) * (a.shape[0] / h)).astype(np.int32).clip(0, a.shape[0] - 1)
    xs = (np.arange(w) * (a.shape[1] / w)).astype(np.int32).clip(0, a.shape[1] - 1)
    return a[ys[:, None], xs[None, :]].astype(np.float32)
