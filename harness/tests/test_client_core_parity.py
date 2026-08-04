"""The two clients must not disagree with the shared core (ADR-0001).

`core/` is the single source of truth for capture behaviour. Until the native
library is cross-compiled and shipped in each app, both clients still carry
their own copy of the keyframe rule — flagged as a stopgap in both files.

A stopgap with no guard is just duplication waiting to drift, and the failure is
silent: iOS starts accepting keyframes at 12° while Android holds at 15°, both
apps work, and Phase 0 reports a per-platform quality difference nobody can
explain. This test reads the constants out of all three sources and asserts they
agree, which needs no Rust toolchain, no NDK and no device — so it runs on every
commit.

Delete it when the clients call `defaultParams()` over UniFFI instead.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
RUST = ROOT / "core" / "src" / "quality.rs"
KOTLIN = (
    ROOT / "clients" / "android" / "app" / "src" / "main" / "java"
    / "com" / "recon" / "poselogger" / "CaptureRecorder.kt"
)
SWIFT = ROOT / "clients" / "ios" / "PoseLogger" / "CaptureRecorder.swift"


def _rust_defaults() -> dict[str, float]:
    """Parse the `impl Default for QualityParams` block."""
    text = RUST.read_text()
    block = re.search(
        r"impl Default for QualityParams \{.*?QualityParams \{(.*?)\n\s*\}\n\s*\}", text, re.S
    )
    assert block, "could not find QualityParams::default() — has quality.rs been restructured?"
    out: dict[str, float] = {}
    for name, expr in re.findall(r"(\w+):\s*([^,\n]+),", block.group(1)):
        expr = expr.strip()
        deg = re.fullmatch(r"([\d.]+)_f32\.to_radians\(\)", expr)
        if deg:
            out[name] = math.radians(float(deg.group(1)))
            continue
        try:
            out[name] = float(expr.rstrip("f32").rstrip("_"))
        except ValueError:
            continue
    return out


def _kotlin_defaults() -> dict[str, float]:
    text = KOTLIN.read_text()
    block = re.search(r"data class Params\((.*?)\n\s*\)", text, re.S)
    assert block, "could not find CaptureRecorder.Params in the Kotlin client"
    out: dict[str, float] = {}
    for name, expr in re.findall(r"val (\w+):\s*\w+\s*=\s*([^,\n]+)", block.group(1)):
        expr = expr.strip()
        deg = re.fullmatch(r"Math\.toRadians\(([\d.]+)\)\.toFloat\(\)", expr)
        if deg:
            out[name] = math.radians(float(deg.group(1)))
            continue
        try:
            out[name] = float(expr.rstrip("f").rstrip("F"))
        except ValueError:
            continue
    return out


def _swift_defaults() -> dict[str, float]:
    text = SWIFT.read_text()
    block = re.search(r"struct Params \{(.*?)\n\s*\}", text, re.S)
    assert block, "could not find CaptureRecorder.Params in the Swift client"
    out: dict[str, float] = {}
    for name, expr in re.findall(r"var (\w+):\s*\w+\s*=\s*([^\n]+)", block.group(1)):
        expr = expr.strip()
        deg = re.fullmatch(r"([\d.]+)\s*\*\s*\.pi\s*/\s*180(?:\.0)?", expr)
        if deg:
            out[name] = math.radians(float(deg.group(1)))
            continue
        try:
            out[name] = float(expr)
        except ValueError:
            continue
    return out


#: rust name -> (kotlin name, swift name)
SHARED = {
    "min_rotation_rad": ("minRotationRad", "minRotationRad"),
    "min_translation_m": ("minTranslationM", "minTranslationM"),
    "min_blur_score": ("minBlurScore", "minBlurScore"),
    "max_exposure_clip_ratio": ("maxExposureClipRatio", "maxExposureClipRatio"),
    "max_keyframes": ("maxKeyframes", "maxKeyframes"),
}


def test_the_parsers_find_something():
    """A regex that silently matches nothing would make every check below pass."""
    for name, values in (("rust", _rust_defaults()), ("kotlin", _kotlin_defaults()),
                         ("swift", _swift_defaults())):
        assert len(values) >= 4, f"{name} parser extracted only {values}"


@pytest.mark.parametrize("rust_name", sorted(SHARED))
def test_clients_agree_with_the_core(rust_name):
    kotlin_name, swift_name = SHARED[rust_name]
    rust = _rust_defaults()
    kotlin = _kotlin_defaults()
    swift = _swift_defaults()

    assert rust_name in rust, f"{rust_name} vanished from QualityParams::default()"
    assert kotlin_name in kotlin, f"{kotlin_name} missing from the Android client"
    assert swift_name in swift, f"{swift_name} missing from the iOS client"

    assert kotlin[kotlin_name] == pytest.approx(rust[rust_name], rel=1e-4), (
        f"Android {kotlin_name}={kotlin[kotlin_name]} disagrees with core "
        f"{rust_name}={rust[rust_name]}"
    )
    assert swift[swift_name] == pytest.approx(rust[rust_name], rel=1e-4), (
        f"iOS {swift_name}={swift[swift_name]} disagrees with core "
        f"{rust_name}={rust[rust_name]}"
    )


def test_the_keyframe_cap_matches_the_server_and_the_plan():
    """PLAN §7 caps keyframes at 150 as a cost control. The clients, the core
    and the server all enforce it, so all three have to say the same number."""
    from app.main import MAX_KEYFRAMES

    assert _rust_defaults()["max_keyframes"] == 150
    assert _kotlin_defaults()["maxKeyframes"] == 150
    assert _swift_defaults()["maxKeyframes"] == 150
    assert MAX_KEYFRAMES == 150


def test_generated_bindings_are_present_for_both_platforms():
    """UniFFI output is committed so a client build needs no Rust toolchain."""
    kotlin = (ROOT / "clients" / "android" / "app" / "src" / "main" / "java"
              / "uniffi" / "recon_core" / "recon_core.kt")
    swift = ROOT / "clients" / "ios" / "PoseLogger" / "Generated" / "recon_core.swift"
    assert kotlin.exists() and swift.exists()

    # The pieces the app actually drives.
    kt = kotlin.read_text()
    for symbol in ("FfiCaptureSession", "FfiGuidance", "FfiNag", "defaultParams"):
        assert symbol in kt, f"{symbol} missing from the generated Kotlin"
    sw = swift.read_text()
    for symbol in ("FfiCaptureSession", "FfiGuidance", "FfiNag"):
        assert symbol in sw, f"{symbol} missing from the generated Swift"


def test_nag_copy_lives_in_the_core_only():
    """Capture guidance is a product surface. If each client writes its own
    wording the two apps say different things to the same user."""
    core = (ROOT / "core" / "src" / "quality.rs").read_text()
    assert "Include a corner or doorway in the frame" in core
    for client in (KOTLIN, SWIFT):
        text = client.read_text()
        assert "Include a corner or doorway" not in text, (
            f"{client.name} has its own copy of the nag wording"
        )
