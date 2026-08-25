"""Small runtime compatibility decisions shared by optional torch models."""

from __future__ import annotations

import ctypes
from contextlib import suppress
from pathlib import Path


def cuda_requires_fp32(torch, device: str) -> bool:
    """Whether ``device`` predates native half-precision arithmetic.

    CUDA compute capability 5.3 introduced native FP16 arithmetic. Kepler
    devices such as the GK110 Titans are 3.5: half tensors may be representable,
    but model GEMMs in FP16/BF16 are not executable there.
    """
    if not device.startswith("cuda"):
        return False
    try:
        return tuple(torch.cuda.get_device_capability(device)) < (5, 3)
    except Exception:  # pragma: no cover - depends on the driver/device string
        return False


def compatible_dtype(torch, device: str, requested: str) -> str:
    """Resolve an old CUDA device to FP32 while preserving modern defaults."""
    return "float32" if cuda_requires_fp32(torch, device) else requested


def configure_cuda_for_legacy_device(torch, device: str) -> bool:
    """Select PyTorch-native kernels that still execute on legacy CUDA.

    The CUDA 11.8 build can target Kepler, but its contemporary cuDNN runtime
    returns ``CUDNN_STATUS_ARCH_MISMATCH`` there. Likewise, fused SDPA kernels
    require newer architectures. Disabling those optional backends leaves the
    FP32 convolution and math-attention implementations built into PyTorch.

    Returns ``True`` when the legacy settings were applied.
    """
    if not cuda_requires_fp32(torch, device):
        return False

    # NVRTC opens its builtins companion with dlopen rather than declaring it
    # as an ELF dependency. The Kepler build links the CUDA runtime beside
    # libtorch, but Linux does not search that directory for a bare dlopen
    # soname. Loading the linked file once by absolute path makes JIT-fused
    # PyTorch ops available without requiring a process-wide LD_LIBRARY_PATH.
    torch_file = getattr(torch, "__file__", None)
    if torch_file:
        builtins = Path(torch_file).parent / "lib" / "libnvrtc-builtins.so.11.8"
        if builtins.exists():
            with suppress(OSError):
                ctypes.CDLL(str(builtins), mode=ctypes.RTLD_GLOBAL)

    cudnn = getattr(getattr(torch, "backends", None), "cudnn", None)
    if cudnn is not None:
        cudnn.enabled = False

    cuda_backend = getattr(getattr(torch, "backends", None), "cuda", None)
    if cuda_backend is not None:
        for name, enabled in (
            ("enable_flash_sdp", False),
            ("enable_mem_efficient_sdp", False),
            ("enable_math_sdp", True),
        ):
            setter = getattr(cuda_backend, name, None)
            if setter is not None:
                setter(enabled)
    return True
