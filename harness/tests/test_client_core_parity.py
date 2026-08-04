"""The two clients must not disagree with the shared core (ADR-0001).

`core/` is the single source of truth for capture behaviour.

**Android now calls it.** The native library is cross-compiled into the APK and
`CaptureRecorder` reads its thresholds from `defaultParams()` over UniFFI, so
there is nothing left to drift there — the checks below assert that it really
does, rather than that a copy happens to match.

**iOS does not yet**, because building it needs a Mac. It still carries its own
copy of the keyframe rule, so the parity check remains for that one client. A
stopgap with no guard is duplication waiting to drift, and the failure is
silent: iOS accepting keyframes at 12° while the core says 15°, both apps
working, and Phase 0 reporting a per-platform quality difference nobody can
explain. Parsing the sources needs no Rust toolchain, no NDK and no device, so
it runs on every commit.

Delete the iOS half when that client calls `defaultParams()` too.
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
#: rust name -> swift name. Android is gone from this map because it now reads
#: the values from the core instead of restating them.
SHARED = {
    "min_rotation_rad": "minRotationRad",
    "min_translation_m": "minTranslationM",
    "min_blur_score": "minBlurScore",
    "max_exposure_clip_ratio": "maxExposureClipRatio",
    "max_keyframes": "maxKeyframes",
}


def test_the_parsers_find_something():
    """A regex that silently matches nothing would make every check below pass."""
    for name, values in (("rust", _rust_defaults()), ("swift", _swift_defaults())):
        assert len(values) >= 4, f"{name} parser extracted only {values}"


@pytest.mark.parametrize("rust_name", sorted(SHARED))
def test_ios_agrees_with_the_core(rust_name):
    """iOS still hardcodes these. Until it calls the core, they must match."""
    _rust, swift_name = rust_name, SHARED[rust_name]
    rust = _rust_defaults()
    swift = _swift_defaults()

    assert rust_name in rust, f"{rust_name} vanished from QualityParams::default()"
    assert swift_name in swift, f"{swift_name} missing from the iOS client"
    assert swift[swift_name] == pytest.approx(rust[rust_name], rel=1e-4), (
        f"iOS {swift_name}={swift[swift_name]} disagrees with core "
        f"{rust_name}={rust[rust_name]}"
    )


def test_android_reads_its_thresholds_from_the_core():
    """The stronger property, now that Android calls UniFFI: it must have no
    copy of its own to drift *from*."""
    kt = KOTLIN.read_text()
    assert "defaultParams()" in kt, "Android no longer reads the core's parameters"
    for hardcoded in ("Math.toRadians(15.0)", "0.30f", "maxKeyframes: Int = 150"):
        assert hardcoded not in kt, (
            f"Android has re-introduced a hardcoded {hardcoded!r} instead of "
            "using the core's value"
        )
    assert "core.maxKeyframes" in kt and "core.minTranslationM" in kt


def test_android_ships_the_native_library():
    """Bindings that compile but have no .so behind them throw at runtime, and
    only on a device — the worst place to find out."""
    jni = ROOT / "clients" / "android" / "app" / "src" / "main" / "jniLibs"
    for abi in ("arm64-v8a", "armeabi-v7a"):
        so = jni / abi / "librecon_core.so"
        assert so.exists(), f"missing {abi} native library — run cargo ndk (core/README.md)"
        assert so.stat().st_size > 100_000, f"{abi} library is suspiciously small"


def test_the_keyframe_cap_matches_the_server_and_the_plan():
    """PLAN §7 caps keyframes at 150 as a cost control. The core, iOS and the
    server all enforce it, so all three have to say the same number. Android is
    absent from this list on purpose — it reads the core's value."""
    from app.main import MAX_KEYFRAMES

    assert _rust_defaults()["max_keyframes"] == 150
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
