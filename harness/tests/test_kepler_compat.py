"""Compatibility behavior for CUDA GPUs predating native FP16 arithmetic."""

from __future__ import annotations

from contextlib import nullcontext

from recon.depth.da3 import (
    _install_full_precision_forward,
    _poses_support_scale_alignment,
    _window_starts,
)
from recon.schemas import Intrinsics, Keyframe, Pose
from recon.torch_compat import (
    compatible_dtype,
    configure_cuda_for_legacy_device,
    cuda_requires_fp32,
)


class _Cuda:
    def __init__(self, capability: tuple[int, int]):
        self.capability = capability

    def get_device_capability(self, _device):
        return self.capability


class _Torch:
    def __init__(self, capability: tuple[int, int]):
        self.cuda = _Cuda(capability)
        self.backends = _Backends()

    @staticmethod
    def inference_mode():
        return nullcontext()


class _CudnnBackend:
    enabled = True


class _CudaBackend:
    def __init__(self):
        self.settings = {}

    def enable_flash_sdp(self, enabled):
        self.settings["flash"] = enabled

    def enable_mem_efficient_sdp(self, enabled):
        self.settings["mem_efficient"] = enabled

    def enable_math_sdp(self, enabled):
        self.settings["math"] = enabled


class _Backends:
    def __init__(self):
        self.cudnn = _CudnnBackend()
        self.cuda = _CudaBackend()


def test_kepler_requires_fp32_but_modern_cuda_does_not():
    assert cuda_requires_fp32(_Torch((3, 5)), "cuda")
    assert not cuda_requires_fp32(_Torch((8, 9)), "cuda")
    assert not cuda_requires_fp32(_Torch((3, 5)), "cpu")


def test_kepler_overrides_half_precision_defaults_only():
    assert compatible_dtype(_Torch((3, 5)), "cuda", "float16") == "float32"
    assert compatible_dtype(_Torch((3, 5)), "cuda", "bfloat16") == "float32"
    assert compatible_dtype(_Torch((8, 9)), "cuda", "float16") == "float16"


def test_kepler_selects_native_convolution_and_math_attention():
    torch = _Torch((3, 5))

    assert configure_cuda_for_legacy_device(torch, "cuda")
    assert not torch.backends.cudnn.enabled
    assert torch.backends.cuda.settings == {
        "flash": False,
        "mem_efficient": False,
        "math": True,
    }


def test_modern_cuda_keeps_default_backends():
    torch = _Torch((8, 9))

    assert not configure_cuda_for_legacy_device(torch, "cuda")
    assert torch.backends.cudnn.enabled
    assert torch.backends.cuda.settings == {}


def test_full_precision_forward_delegates_without_autocast():
    calls = []

    class _Inner:
        def __call__(self, *args, **kwargs):
            calls.append((args, kwargs))
            return "prediction"

    api_model = type("API", (), {"model": _Inner()})()
    _install_full_precision_forward(api_model, _Torch((3, 5)))

    assert api_model.forward("image", extrinsics="poses") == "prediction"
    assert calls == [(('image',), {"extrinsics": "poses"})]
    assert api_model._recon_full_precision_forward is True


def test_window_starts_backfill_a_full_tail_instead_of_a_tiny_window():
    assert _window_starts(16, 4, 1) == [0, 3, 6, 9, 12]
    assert _window_starts(5, 4, 1) == [0, 1]
    assert _window_starts(3, 4, 1) == [0]


def test_pose_scale_alignment_rejects_degenerate_camera_paths():
    intrinsics = Intrinsics(fx=100, fy=100, cx=50, cy=50, width=100, height=100)

    def keyframe(index, translation):
        return Keyframe(
            id=str(index),
            image=f"{index}.png",
            timestamp_ns=index,
            pose=Pose(
                R=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                t=translation,
            ),
            intrinsics=intrinsics,
        )

    assert not _poses_support_scale_alignment([keyframe(0, [0.0, 0.0, 0.0])])
    assert not _poses_support_scale_alignment(
        [keyframe(i, [float(i), 0.0, 0.0]) for i in range(4)]
    )
    assert _poses_support_scale_alignment(
        [
            keyframe(0, [0.0, 0.0, 0.0]),
            keyframe(1, [1.0, 0.0, 0.0]),
            keyframe(2, [0.0, 1.0, 0.0]),
        ]
    )
