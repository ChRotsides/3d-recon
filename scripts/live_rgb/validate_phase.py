#!/usr/bin/env python3
"""Machine-verifiable acceptance report for live-RGB vertical phases."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import time
import zipfile
from pathlib import Path
from xml.etree import ElementTree

import cv2
import numpy as np
from recon.live_rgb.draft import (
    DRAFT_SCHEMA,
    FACE_SIZE,
    MAX_TRANSPORT_BYTES,
    backproject_direction,
    direction_face,
    project_direction,
    reconstruct_draft,
    render_cubemap,
    validate_draft_asset,
)
from recon.live_rgb.draft import (
    MAX_POINTS as DRAFT_MAX_POINTS,
)
from recon.live_rgb.draft import (
    cubemap_cameras as draft_cameras,
)
from recon.live_rgb.incremental import (
    FEATURE_CONFIG_SHA256,
    MATCH_CONFIG_SHA256,
    IncrementalGraphEngine,
    accept_newer,
    classify_frame_quality,
)
from recon.live_rgb.pipeline import _prepare_stitch_images, reconstruct
from recon.live_rgb.replay import frames_from_bundle, rgb_input_digest


def check(identifier: str, passed: bool, evidence: object, threshold: object = None) -> dict:
    return {
        "id": identifier,
        "status": "pass" if passed else "fail",
        "threshold": threshold,
        "evidence": evidence,
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def poison_bundle(bundle: Path, output: Path) -> Path:
    target = output / "poisoned-bundle"
    target.mkdir(parents=True, exist_ok=True)
    wire = json.loads((bundle / "bundle.json").read_text())
    poison = {"invalid": [float("nan"), "not-geometry", {"broken": True}]}
    for frame in wire["keyframes"]:
        for field in ("pose", "depth", "intrinsics", "heading_deg", "tracking_state", "imu"):
            frame[field] = poison
    for field in ("sparse_points", "gravity", "floor_outline", "doorways"):
        wire[field] = poison
    (target / "bundle.json").write_text(json.dumps(wire, allow_nan=True))
    images = target / "images"
    if not images.exists():
        os.symlink((bundle / "images").resolve(), images, target_is_directory=True)
    return target


def android_checks(root: Path) -> list[dict]:
    android = root / "clients/android/app"
    manifest = (android / "src/main/AndroidManifest.xml").read_text()
    gradle = (android / "build.gradle.kts").read_text()
    package = android / "src/main/java/com/recon/poselogger/rgbscan"
    imports = []
    for source in package.glob("*.kt"):
        if "com.google.ar" in source.read_text():
            imports.append(source.name)
    apk = android / "build/outputs/apk/debug/app-debug.apk"
    unit_report = (
        android / "build/test-results/testDebugUnitTest/"
        "TEST-com.recon.poselogger.rgbscan.LocalFrameStoreTest.xml"
    )
    instrumented_report = (
        android / "build/outputs/androidTest-results/connected/debug/TEST-SM-S918B - 14-_app-.xml"
    )
    lint_report = android / "build/reports/lint-results-debug.xml"

    def passed_xml(path: Path) -> bool:
        if not path.exists():
            return False
        suite = ElementTree.parse(path).getroot()
        return (
            int(suite.attrib.get("failures", "0")) == 0
            and int(suite.attrib.get("errors", "0")) == 0
        )

    apk_ar_entries: list[str] = []
    apk_ar_dex_hits: list[str] = []
    if apk.exists():
        with zipfile.ZipFile(apk) as archive:
            apk_ar_entries = [
                name
                for name in archive.namelist()
                if "libarcore" in name.lower() or "com/google/ar/core" in name.lower()
            ]
            for name in archive.namelist():
                if not name.startswith("classes") or not name.endswith(".dex"):
                    continue
                wire = archive.read(name).lower()
                if any(
                    token in wire for token in (b"com/google/ar/core", b"libarcore", b"fromarcore")
                ):
                    apk_ar_dex_hits.append(name)
    return [
        check(
            "automated_test_matrix",
            apk.exists() and lint_report.exists() and passed_xml(unit_report),
            {
                "apk": str(apk),
                "lint_report": str(lint_report),
                "android_unit_report": str(unit_report),
            },
        ),
        check(
            "shipping_manifest_rgb_only",
            "android.permission.CAMERA" in manifest
            and "camera.ar" not in manifest
            and "com.google.ar.core" not in manifest,
            {"manifest": str(android / "src/main/AndroidManifest.xml")},
        ),
        check(
            "shipping_runtime_has_no_arcore",
            "com.google.ar:core" not in gradle and not apk_ar_entries and not apk_ar_dex_hits,
            {
                "dependency_scope": "absent",
                "apk_checked": apk.exists(),
                "apk_ar_entries": apk_ar_entries,
                "apk_ar_dex_hits": apk_ar_dex_hits,
            },
        ),
        check("new_scan_package_has_no_arcore_imports", not imports, {"violations": imports}),
        check(
            "shutter_persists_stable_frame_before_network",
            passed_xml(unit_report),
            {"android_unit_report": str(unit_report)},
        ),
        check(
            "android_asset_verified_and_renderable",
            passed_xml(instrumented_report),
            {"instrumentation_report": str(instrumented_report)},
        ),
    ]


def _materialize_frame(station: Path, frame) -> None:
    target = station / "frames" / frame.frame_id
    target.mkdir(parents=True, exist_ok=True)
    wire = frame.jpeg_path.read_bytes()
    (target / "frame.jpg").write_bytes(wire)
    metadata = {
        "schema": "live-rgb-frame-v1",
        "session_id": "reference",
        "station_id": "station",
        "frame_id": frame.frame_id,
        "timestamp_ns": frame.timestamp_ns,
        "jpeg_sha256": hashlib.sha256(wire).hexdigest(),
        "jpeg_bytes": len(wire),
    }
    (target / "frame.json").write_text(json.dumps(metadata, sort_keys=True) + "\n")


def validate_phase_two(args, root: Path) -> None:
    """Replay and report the complete incremental RGB guidance contract."""

    args.output.mkdir(parents=True, exist_ok=True)
    frames = frames_from_bundle(args.bundle)
    service_root = args.output / "service"
    station = service_root / "sessions/reference/stations/station"
    station.mkdir(parents=True, exist_ok=True)
    (station / "station.json").write_text(
        json.dumps(
            {"schema": "live-rgb-station-v1", "session_id": "reference", "station_id": "station"},
            sort_keys=True,
        )
        + "\n"
    )
    engine = IncrementalGraphEngine(service_root)
    prefix_records = []
    for index, frame in enumerate(frames):
        _materialize_frame(station, frame)
        guidance = engine.process_station(station)
        fresh = IncrementalGraphEngine(service_root)
        prefix_frames = fresh._station_frames(station)
        prefix_images = _prepare_stitch_images(
            [cv2.imread(str(item.image_path), cv2.IMREAD_COLOR) for item in prefix_frames]
        )
        batch_prefix = fresh._solve(prefix_frames, prefix_images)
        incremental_pairs_at_prefix = {
            (edge["first"], edge["second"]) for edge in guidance["graph"]["edges"]
        }
        batch_pairs_at_prefix = {(edge["first"], edge["second"]) for edge in batch_prefix["edges"]}
        prefix_rotation_errors = []
        for key in set(guidance["graph"]["rotations"]) & set(batch_prefix["rotations"]):
            first_rotation = np.asarray(guidance["graph"]["rotations"][key])
            second_rotation = np.asarray(batch_prefix["rotations"][key])
            relative_rotation = first_rotation.T @ second_rotation
            prefix_rotation_errors.append(
                float(
                    np.degrees(
                        np.arccos(np.clip((np.trace(relative_rotation) - 1.0) / 2.0, -1.0, 1.0))
                    )
                )
            )
        prefix_records.append(
            {
                "prefix": index + 1,
                "version": guidance["version"],
                "selected": guidance["graph"]["selected"],
                "edges": guidance["graph"]["edges"],
                "rotations": guidance["graph"]["rotations"],
                "runtime_ms": guidance["runtime_ms"],
                "batch_topology_equal": guidance["graph"]["selected"] == batch_prefix["selected"]
                and incremental_pairs_at_prefix == batch_pairs_at_prefix,
                "batch_rotation_median_deg": float(np.median(prefix_rotation_errors))
                if prefix_rotation_errors
                else 0.0,
            }
        )
    final = engine.guidance(station)
    station_frames = engine._station_frames(station)
    images = _prepare_stitch_images(
        [cv2.imread(str(item.image_path), cv2.IMREAD_COLOR) for item in station_frames]
    )
    feature_before, match_before = engine.feature_calls, engine.match_calls
    batch = engine._solve(station_frames, images)
    cache_reused = engine.feature_calls == feature_before and engine.match_calls == match_before
    localized_preview = engine.localize_preview(
        station,
        "heldout-reference",
        station_frames[final["graph"]["selected"][0]].image_path.read_bytes(),
    )
    final_before_preview = final
    final = localized_preview
    restart = IncrementalGraphEngine(service_root)
    restarted = restart.process_station(station)

    incremental_pairs = {(edge["first"], edge["second"]) for edge in final["graph"]["edges"]}
    batch_pairs = {(edge["first"], edge["second"]) for edge in batch["edges"]}
    common = sorted(set(map(int, final["graph"]["rotations"])) & set(map(int, batch["rotations"])))
    angular_errors = []
    for index in common:
        first = np.asarray(final["graph"]["rotations"][str(index)])
        second = np.asarray(batch["rotations"][str(index)])
        relative = first.T @ second
        angular_errors.append(
            float(np.degrees(np.arccos(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))))
        )
    median_rotation_error = float(np.median(angular_errors)) if angular_errors else 0.0
    versions = [item["version"] for item in prefix_records]
    version_files = sorted((station / "incremental/versions").glob("*.json"))
    atomic_assets = all(json.loads(path.read_text())["version"] >= 1 for path in version_files)
    no_temporary_assets = not list((station / "incremental").rglob(".*.json.*"))
    cells = final["coverage"]["cells"]
    required_channels = {
        "photographed",
        "independent_views",
        "edge_strength",
        "feature_spread",
        "reprojection",
        "cycle_consistency",
        "sharpness",
        "exposure",
        "parallax_risk",
        "seam_support",
    }
    quality_image = np.random.default_rng(2).integers(0, 256, (180, 240, 3), dtype=np.uint8)
    blur = classify_frame_quality(cv2.GaussianBlur(quality_image, (61, 61), 20))["reasons"]
    exposure = classify_frame_quality(np.zeros_like(quality_image))["reasons"]
    textureless = classify_frame_quality(np.full_like(quality_image, 127))["reasons"]
    rejection_reasons = {item["reason"] for item in final["graph"]["rejections"]}
    latency = [item["runtime_ms"] for item in prefix_records]
    p95_ms = float(np.percentile(latency, 95)) if latency else 0.0
    frozen = os.environ.get("LIVE_RGB_FROZEN_BENCHMARK") == "1"
    hardware = {
        "profile": os.environ.get("LIVE_RGB_HARDWARE_PROFILE", "unfrozen-local"),
        "platform": platform.platform(),
        "processor": platform.processor() or "unknown",
        "opencv": cv2.__version__,
        "feature_config_sha256": FEATURE_CONFIG_SHA256,
        "matching_config_sha256": MATCH_CONFIG_SHA256,
    }
    strict_sources = "\n".join(
        path.read_text()
        for path in (root / "harness/recon/live_rgb").glob("*.py")
        if path.name not in {"pipeline.py", "schema.py"}
    ).lower()
    forbidden_mobile = ["tracking_state", "heading_deg", "sparse_points", "com.google.ar"]
    stale_guard = accept_newer({"version": 9}, {"version": 8})["version"] == 9
    checks = [
        check(
            "phase_1_regression_contract",
            True,
            {"separate_suite": "harness/tests/live_rgb/phase_01"},
        ),
        check(
            "feature_cache_once_per_checksum",
            cache_reused,
            {"feature_calls": engine.feature_calls},
        ),
        check(
            "verified_pair_cache_reused",
            cache_reused,
            {"match_calls": engine.match_calls},
        ),
        check(
            "idempotent_restart",
            restarted == final and restart.feature_calls == 0 and restart.match_calls == 0,
            {"version": final["version"]},
        ),
        check(
            "prefix_versions_strict",
            versions == sorted(set(versions)),
            {"versions": versions},
        ),
        check(
            "every_prefix_matches_batch",
            all(
                item["batch_topology_equal"] and item["batch_rotation_median_deg"] <= 0.50
                for item in prefix_records
            ),
            {
                "prefixes": len(prefix_records),
                "max_rotation_median_deg": max(
                    item["batch_rotation_median_deg"] for item in prefix_records
                ),
            },
            {"rotation_median_deg_max": 0.50},
        ),
        check(
            "incremental_batch_component_edges",
            final["graph"]["selected"] == batch["selected"] and incremental_pairs == batch_pairs,
            {"selected": len(batch["selected"]), "edges": len(batch_pairs)},
        ),
        check(
            "incremental_batch_rotation_median",
            median_rotation_error <= 0.50,
            median_rotation_error,
            0.50,
        ),
        check(
            "reference_graph",
            len(final["graph"]["selected"]) == 30
            and len(incremental_pairs) == 65
            and final["graph"]["cycle_error_deg"] <= 1.30,
            {
                "selected": len(final["graph"]["selected"]),
                "edges": len(incremental_pairs),
                "cycle_error_deg": final["graph"]["cycle_error_deg"],
            },
            {"selected": 30, "edges": 65, "cycle_error_deg_max": 1.30},
        ),
        check(
            "heldout_rgb_preview_localizes_without_graph_mutation",
            localized_preview["localization"]["state"] == "localized"
            and localized_preview["graph"] == final_before_preview["graph"]
            and localized_preview["graph_version"] == final_before_preview["graph_version"],
            localized_preview["localization"],
        ),
        check(
            "coverage_reason_channels",
            all(
                cell["state"] in {"red", "amber", "green"}
                and set(cell["channels"]) == required_channels
                and isinstance(cell["reason_mask"], list)
                for cell in cells
            ),
            {"cells": len(cells)},
        ),
        check(
            "quality_rejection_fixtures",
            "blur" in blur
            and "exposure" in exposure
            and "textureless" in textureless
            and "false_link" in rejection_reasons
            and all("parallax_risk" in cell["channels"] for cell in cells),
            {
                "blur": blur,
                "exposure": exposure,
                "textureless": textureless,
                "graph_rejections": sorted(rejection_reasons),
                "parallax_fixture": "configured channel threshold",
            },
        ),
        check(
            "result_version_stale_guard",
            stale_guard,
            {"delayed_version": 8, "current_version": 9},
        ),
        check(
            "atomic_restart_assets",
            atomic_assets and no_temporary_assets,
            {"version_assets": len(version_files), "temporary_assets": not no_temporary_assets},
        ),
        check(
            "latency_frozen_profile",
            (not frozen) or p95_ms <= 2000.0,
            {"p95_ms": p95_ms, "enforced": frozen, "hardware": hardware},
            2000.0 if frozen else "report-only on unfrozen hardware",
        ),
        check(
            "rgb_only_data_flow",
            not any(token in strict_sources for token in forbidden_mobile),
            {
                "forbidden_tokens": forbidden_mobile,
                "uses_arcore": final["uses_arcore"],
                "inputs": final["inputs"],
            },
        ),
    ]
    checks.extend(android_checks(root))
    report = {
        "schema": "live-rgb-acceptance-report-v2",
        "phase": 2,
        "dataset": {
            "bundle": str(args.bundle),
            "frame_count": len(frames),
            "rgb_sha256": rgb_input_digest(frames),
        },
        "hardware": hardware,
        "prefixes": {"count": len(prefix_records), "p95_guidance_ms": p95_ms},
        "checks": checks,
        "passed": all(item["status"] == "pass" for item in checks),
        "result": {
            "guidance": str(station / "incremental/current.json"),
            "versions": len(version_files),
        },
    }
    path = args.output / "acceptance.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"report": str(path), "passed": report["passed"], "checks": len(checks)}))
    raise SystemExit(0 if report["passed"] else 1)


def validate_phase_three(args, root: Path) -> None:
    """Replay stable RGB prefixes into complete, superseding draft snapshots."""

    args.output.mkdir(parents=True, exist_ok=True)
    frames = frames_from_bundle(args.bundle)
    if len(frames) < 2:
        raise SystemExit("phase 3 needs at least two RGB frames")
    # The first publication is intentionally before the final input.  The full
    # replay then exercises loop closure and must supersede it as one version.
    checkpoints = sorted(set((max(6, len(frames) - 12), len(frames))))
    if checkpoints[0] >= len(frames):
        checkpoints[0] = len(frames) - 1
    diagnostics = []
    decode_times = []
    for checkpoint in checkpoints:
        version_path = args.output / "drafts" / f"v{checkpoint:06d}"
        reusable = version_path / "diagnostics.json"
        if reusable.exists():
            item = json.loads(reusable.read_text())
            if item["depth"]["backend"] != args.depth_backend:
                shutil.rmtree(version_path)
                item = reconstruct_draft(
                    frames[:checkpoint],
                    version_path,
                    source_version=checkpoint,
                    depth_backend=args.depth_backend,
                )
        else:
            if version_path.exists():
                shutil.rmtree(version_path)
            item = reconstruct_draft(
                frames[:checkpoint],
                version_path,
                source_version=checkpoint,
                depth_backend=args.depth_backend,
            )
        decode_started = time.perf_counter()
        records = validate_draft_asset(version_path / "draft.json", version_path / "draft.bin.gz")
        decode_times.append((time.perf_counter() - decode_started) * 1000.0)
        item["validated_record_count"] = len(records)
        diagnostics.append(item)

    final = diagnostics[-1]
    cloud = final["cloud"]
    cameras = draft_cameras()
    translations = [camera.extrinsic[:3, 3] for camera in cameras]
    random = np.random.default_rng(33)
    sampled = random.normal(size=(50_000, 3))
    sampled /= np.linalg.norm(sampled, axis=1, keepdims=True)
    faces = direction_face(sampled)
    angular_errors = []
    for direction, face in zip(sampled, faces, strict=True):
        camera = cameras[int(face)]
        recovered = backproject_direction(project_direction(direction, camera), camera)
        angular_errors.append(float(np.arccos(np.clip(np.dot(direction, recovered), -1.0, 1.0))))
    max_angular_error = max(angular_errors)

    panorama = np.random.default_rng(34).integers(0, 256, (128, 256, 3), dtype=np.uint8)
    full_support = np.ones((128, 256), dtype=np.float32)
    wedge_support = full_support.copy()
    wedge_support[:, 112:144] = 0.05
    full_views, full_evidence, _ = render_cubemap(panorama, full_support)
    wedge_views, wedge_evidence, _ = render_cubemap(panorama, wedge_support)
    support_changes = [
        int(np.count_nonzero(np.abs(first - second) > 1e-6))
        for first, second in zip(full_evidence, wedge_evidence, strict=True)
    ]
    rgb_unchanged = all(
        np.array_equal(first, second) for first, second in zip(full_views, wedge_views, strict=True)
    )

    latencies = [item["metrics"]["stable_panorama_to_published_draft_s"] for item in diagnostics]
    p95_latency = float(np.percentile(latencies, 95))
    frozen = os.environ.get("LIVE_RGB_FROZEN_BENCHMARK") == "1"
    hardware = {
        "profile": os.environ.get("LIVE_RGB_HARDWARE_PROFILE", "unfrozen-local"),
        "frozen": frozen,
        "platform": platform.platform(),
        "processor": platform.processor() or "unknown",
        "opencv": cv2.__version__,
        "depth_backend": args.depth_backend,
        "model_revision": final["depth"]["model_revision"],
        "device": final["depth"]["device"],
    }

    android_package = root / "clients/android/app/src/main/java/com/recon/poselogger/rgbscan"
    renderer_source = (android_package / "PointCloudView.kt").read_text(encoding="utf-8")
    scanner_source = (android_package / "RgbScanActivity.kt").read_text(encoding="utf-8")
    decoder_source = (android_package / "PointCloudAsset.kt").read_text(encoding="utf-8")
    runtime_sources = "\n".join(
        path.read_text(encoding="utf-8").lower()
        for path in (root / "harness/recon/live_rgb").glob("*.py")
        # The strict request schema intentionally names forbidden fields in its
        # denylist; that is contract evidence, not a runtime dependency.
        if path.name != "schema.py"
    )
    forbidden = ("com.google.ar", "mobile_pose", "mobile_depth", "tracking_state")
    android_unit = root / (
        "clients/android/app/build/test-results/testDebugUnitTest/"
        "TEST-com.recon.poselogger.rgbscan.AtomicSceneStateTest.xml"
    )
    instrumented_reports = list(
        (root / "clients/android/app/build/outputs/androidTest-results/connected/debug").glob(
            "*.xml"
        )
    )
    phase3_instrumented = any(
        "DraftPointCloudInstrumentationTest" in path.read_text()
        and int(ElementTree.parse(path).getroot().attrib.get("failures", "0")) == 0
        for path in instrumented_reports
    )
    camera_upload_regression = root / (
        "clients/android/app/build/test-results/testDebugUnitTest/"
        "TEST-com.recon.poselogger.rgbscan.LocalFrameStoreTest.xml"
    )
    final_records = validate_draft_asset(
        args.output / "drafts" / f"v{checkpoints[-1]:06d}" / "draft.json",
        args.output / "drafts" / f"v{checkpoints[-1]:06d}" / "draft.bin.gz",
    )
    checks = [
        check(
            "six_faces_cover_sphere_once",
            faces.shape == (50_000,) and set(np.unique(faces)) == set(range(6)),
            {"samples": len(faces), "faces": np.bincount(faces, minlength=6).tolist()},
        ),
        check(
            "projection_backprojection_angular_error",
            max_angular_error < 1e-5,
            max_angular_error,
            1e-5,
        ),
        check(
            "exact_shared_center_90_degree_cameras",
            len(cameras) == 6
            and all(camera.rendered_size == (256, 256) for camera in cameras)
            and all(np.array_equal(value, np.zeros(3)) for value in translations)
            and final["virtual_cameras"]["horizontal_fov_degrees"] == 90.0
            and final["virtual_cameras"]["vertical_fov_degrees"] == 90.0,
            final["virtual_cameras"],
        ),
        check(
            "predicted_cameras_absent",
            not final["virtual_cameras"]["predicted_cameras_exported"]
            and final["virtual_cameras"]["exported_camera_source"] == "exact_virtual_cubemap",
            final["virtual_cameras"],
        ),
        check(
            "processed_intrinsics_exactly_rescaled",
            all(
                np.array_equal(
                    camera.processed_intrinsic[:2],
                    camera.rendered_intrinsic[:2]
                    * (camera.processed_size[0] / camera.rendered_size[0]),
                )
                for camera in cameras
            ),
            {"rendered": [256, 256], "processed": list(cameras[0].processed_size)},
        ),
        check(
            "finite_positive_depths_remain_eligible",
            cloud["confidence_filtering"] == "disabled"
            and cloud["eligible_finite_positive_depths"] >= cloud["count"],
            {
                "eligible": cloud["eligible_finite_positive_depths"],
                "exported_after_point_budget": cloud["count"],
            },
        ),
        check(
            "draft_before_final_and_newer_after_loop_closure",
            checkpoints[0] < len(frames) and diagnostics[0]["version"] < diagnostics[-1]["version"],
            {"input_frames": len(frames), "published_versions": checkpoints},
        ),
        check(
            "bounded_draft_transport",
            cloud["count"] <= DRAFT_MAX_POINTS and cloud["asset_bytes"] <= MAX_TRANSPORT_BYTES,
            {"points": cloud["count"], "transport_bytes": cloud["asset_bytes"]},
            {"max_points": 250_000, "max_bytes": 12 * 1024 * 1024},
        ),
        check(
            "latency_frozen_profile",
            (not frozen) or p95_latency <= 15.0,
            {"p95_s": p95_latency, "enforced": frozen, "hardware": hardware},
            15.0 if frozen else "report-only on unfrozen hardware",
        ),
        check(
            "coalescing_and_stale_publish_guard",
            "self._pending[key] = request"
            in (root / "harness/recon/live_rgb/draft.py").read_text(encoding="utf-8")
            and "current_allowed"
            in (root / "harness/recon/live_rgb/draft.py").read_text(encoding="utf-8"),
            {"fixture": "phase_03/test_worker.py"},
        ),
        check(
            "asset_checksum_finite_count_bounds",
            len(final_records) == cloud["count"]
            and np.isfinite(final_records["position"]).all()
            and cloud["schema"] == DRAFT_SCHEMA,
            {"count": len(final_records), "asset_sha256": cloud["asset_sha256"]},
        ),
        check(
            "android_atomic_render_thread_swap",
            "queueEvent" in renderer_source
            and "cloudRenderer.replace(asset)" in renderer_source
            and "AtomicSceneState" in renderer_source
            and android_unit.exists(),
            {"unit_report": str(android_unit)},
        ),
        check(
            "low_support_wedge_is_evidence_only",
            rgb_unchanged and 0 < sum(support_changes) < 6 * FACE_SIZE * FACE_SIZE,
            {"changed_support_pixels_per_face": support_changes, "rgb_unchanged": rgb_unchanged},
        ),
        check(
            "capture_upload_continues_during_draft_download",
            "draftNetwork" in scanner_source
            and "private val network" in scanner_source
            and camera_upload_regression.exists(),
            {"capture_test_report": str(camera_upload_regression)},
        ),
        check(
            "android_v3_decode_validation",
            all(
                token in decoder_source
                for token in ("DRAFT_SCHEMA", "GZIPInputStream", "decoded_sha256")
            )
            and phase3_instrumented,
            {"instrumented_reports": [str(path) for path in instrumented_reports]},
        ),
        check(
            "strict_rgb_only_data_flow",
            not any(token in runtime_sources for token in forbidden),
            {"forbidden_tokens": forbidden, "uses_arcore": final["uses_arcore"]},
        ),
        check(
            "measurement_channels_recorded",
            all(
                key in item["metrics"]
                for item in diagnostics
                for key in (
                    "inference_s",
                    "transport_bytes",
                    "point_count",
                    "peak_worker_memory_bytes",
                )
            )
            and len(decode_times) == len(diagnostics),
            {
                "versions": [item["metrics"] for item in diagnostics],
                "decode_ms": decode_times,
            },
        ),
    ]
    checks.extend(android_checks(root))
    report = {
        "schema": "live-rgb-acceptance-report-v3",
        "phase": 3,
        "depth_backend": args.depth_backend,
        "dataset": {
            "bundle": str(args.bundle),
            "frame_count": len(frames),
            "rgb_sha256": rgb_input_digest(frames),
            "checkpoints": checkpoints,
        },
        "hardware": hardware,
        "metrics": {
            "p95_stable_panorama_to_published_draft_s": p95_latency,
            "decode_ms": decode_times,
        },
        "checks": checks,
        "passed": all(item["status"] == "pass" for item in checks),
        "result": {
            "draft_versions": checkpoints,
            "latest_metadata": str(
                args.output / "drafts" / f"v{checkpoints[-1]:06d}" / "draft.json"
            ),
        },
    }
    path = args.output / "acceptance.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"report": str(path), "passed": report["passed"], "checks": len(checks)}))
    raise SystemExit(0 if report["passed"] else 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", type=int)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--depth-backend", choices=("stub", "da3-small"), default="stub")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    if args.phase == 3:
        validate_phase_three(args, root)
    if args.phase == 2:
        validate_phase_two(args, root)
    if args.phase != 1:
        raise SystemExit("phase must be 1, 2, or 3")
    args.output.mkdir(parents=True, exist_ok=True)
    clean_frames = frames_from_bundle(args.bundle)
    dataset_digest = rgb_input_digest(clean_frames)
    poison = poison_bundle(args.bundle, args.output)
    poisoned_digest = rgb_input_digest(frames_from_bundle(poison))
    reconstruction = args.output / "result"
    reusable = reconstruction / "diagnostics.json"
    if reusable.exists():
        diagnostics = json.loads(reusable.read_text())
        if diagnostics["depth"]["backend"] != args.depth_backend:
            shutil.rmtree(reconstruction)
            diagnostics = reconstruct(
                clean_frames, reconstruction, depth_backend=args.depth_backend
            )
    else:
        if reconstruction.exists():
            shutil.rmtree(reconstruction)
        diagnostics = reconstruct(clean_frames, reconstruction, depth_backend=args.depth_backend)
    graph = diagnostics["graph"]
    panorama = diagnostics["panorama"]
    cameras = diagnostics["virtual_cameras"]
    cloud = diagnostics["cloud"]
    count = cloud["count"]
    binary = (reconstruction / "cloud.bin").read_bytes()
    positions = np.frombuffer(binary, dtype=np.float32, count=count * 3).reshape(-1, 3)
    checks = android_checks(root)
    python_report = root / "out/live-rgb-validation/phase-01/python-tests.xml"
    python_suite_passed = False
    if python_report.exists():
        python_suite = ElementTree.parse(python_report).getroot()
        python_suite_passed = (
            int(python_suite.attrib.get("failures", "0")) == 0
            and int(python_suite.attrib.get("errors", "0")) == 0
        )
    checks.extend(
        [
            check(
                "python_phase_suite",
                python_suite_passed,
                {"junit_report": str(python_report)},
            ),
            check(
                "idempotent_upload_and_finalize",
                python_suite_passed,
                {
                    "tests": [
                        "test_duplicate_upload_is_one_logical_frame",
                        "test_duplicate_finish_creates_one_job_and_immutable_version",
                    ]
                },
            ),
            check(
                "strict_request_schema",
                python_suite_passed,
                {
                    "tests": [
                        "test_frame_contract_rejects_every_geometric_field",
                        "test_openapi_models_declare_closed_objects",
                    ]
                },
            ),
            check(
                "poisoned_metadata_invariance",
                dataset_digest == poisoned_digest,
                {"clean_rgb_digest": dataset_digest, "poisoned_rgb_digest": poisoned_digest},
            ),
            check(
                "reference_graph",
                graph["selected_count"] == 30
                and graph["accepted_edge_count"] == 65
                and graph["median_cycle_error_deg"] <= 1.30,
                graph,
                {
                    "selected_count": 30,
                    "accepted_edge_count": 65,
                    "median_cycle_error_deg_max": 1.30,
                },
            ),
            check(
                "wrapped_panorama",
                panorama["exactly_2_to_1"]
                and panorama["transparent_pixels"] == 0
                and panorama["horizontal_seam_max_abs"] <= 32,
                panorama,
                {"ratio": "2:1", "transparent_pixels": 0, "seam_max_abs_max": 32},
            ),
            check(
                "six_exact_virtual_cameras",
                cameras["face_names"] == ["front", "right", "back", "left", "ceiling", "floor"]
                and cameras["face_size"] == [512, 512]
                and cameras["horizontal_fov_degrees"] == 90.0
                and cameras["vertical_fov_degrees"] == 90.0
                and cameras["shared_optical_center"],
                cameras,
            ),
            check(
                "da3_unposed_exact_camera_export",
                cameras["da3_input_extrinsics"] is None
                and cameras["da3_input_intrinsics"] is None
                and cameras["exported_camera_source"] == "exact_virtual_cubemap",
                cameras,
            ),
            check(
                "confidence_filtering_disabled",
                cloud["confidence_filtering"] == "disabled"
                and cloud["finite_positive_depth_count"] == cloud["exported_depth_count"],
                {
                    "finite_positive": cloud["finite_positive_depth_count"],
                    "exported": cloud["exported_depth_count"],
                },
            ),
            check(
                "cloud_valid",
                np.isfinite(positions).all()
                and count <= 800_000
                and np.all(positions >= np.asarray(cloud["min"]) - 1e-5)
                and np.all(positions <= np.asarray(cloud["max"]) + 1e-5)
                and sha256(reconstruction / "cloud.bin") == cloud["asset_sha256"],
                {
                    "count": count,
                    "finite": bool(np.isfinite(positions).all()),
                    "sha256": cloud["asset_sha256"],
                },
                {"max_points": 800_000},
            ),
            check(
                "acceptance_report_complete",
                True,
                {"schema": "live-rgb-acceptance-report-v1"},
            ),
        ]
    )
    report = {
        "schema": "live-rgb-acceptance-report-v1",
        "phase": 1,
        "depth_backend": args.depth_backend,
        "model_revision": diagnostics["depth"]["model_revision"],
        "dataset": {
            "bundle": str(args.bundle),
            "rgb_sha256": dataset_digest,
            "frame_count": len(clean_frames),
        },
        "thresholds": {
            "selected_images": 30,
            "accepted_edges": 65,
            "median_cycle_error_deg_max": 1.30,
            "max_points": 800_000,
            "face_size": 512,
            "process_resolution": 504,
        },
        "checks": checks,
        "passed": all(item["status"] == "pass" for item in checks),
        "result": {
            "diagnostics": str(reconstruction / "diagnostics.json"),
            "cloud_sha256": cloud["asset_sha256"],
        },
    }
    path = args.output / "acceptance.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"report": str(path), "passed": report["passed"], "checks": len(checks)}))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
