//  CaptureRecorder.swift
//  The throwaway pose-logging capture app from plan task 3.
//
//  Scope, deliberately: log ARKit poses + intrinsics + gravity, select
//  keyframes, score blur and exposure, and write a plan §5 SessionBundle that
//  the Phase-0 harness can reconstruct. No guidance overlay — that is Phase 3,
//  and it lives in the shared Rust core, not here.

import ARKit
import Accelerate
import CoreImage
import Foundation
import UIKit

@MainActor
final class CaptureRecorder: NSObject, ARSessionDelegate {

    // Plan §6.1 keyframe rule. Kept in sync with recon-core's QualityParams;
    // once the Rust core is wired in via UniFFI these constants go away and the
    // core decides, so the two platforms cannot drift.
    struct Params {
        var minRotationRad: Float = 15.0 * .pi / 180.0
        var minTranslationM: Float = 0.30
        var minBlurScore: Double = 100.0
        var maxExposureClipRatio: Double = 0.25
        var maxKeyframes: Int = 150
        /// Downscale before scoring and saving. Full-resolution frames are
        /// wasted bytes at 150 keyframes and slow the blur score down.
        var targetWidth: Int = 960
    }

    private(set) var params = Params()
    private(set) var keyframes: [KeyframeWire] = []
    private(set) var isRecording = false

    private let session: ARSession
    private let roomId: String
    private let sessionId = UUID().uuidString
    private var bundleDir: URL!
    private var lastPose: simd_float4x4?
    private var sparsePoints: [SparsePointWire] = []
    private var measurements: [ReferenceMeasurementWire] = []
    private let ciContext = CIContext(options: [.useSoftwareRenderer: false])

    /// Surfaced in the UI so a capture that is quietly rejecting everything is
    /// visible rather than mysterious.
    private(set) var rejected: [String: Int] = [:]

    init(session: ARSession, roomId: String = UUID().uuidString) {
        self.session = session
        self.roomId = roomId
        super.init()
        session.delegate = self
    }

    // MARK: - Lifecycle

    func start() throws {
        let config = ARWorldTrackingConfiguration()
        config.worldAlignment = .gravity
        config.planeDetection = [.horizontal, .vertical]
        config.isAutoFocusEnabled = true
        config.frameSemantics = []

        bundleDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("bundle_\(sessionId)", isDirectory: true)
        try FileManager.default.createDirectory(
            at: bundleDir.appendingPathComponent("images"),
            withIntermediateDirectories: true)

        keyframes.removeAll()
        sparsePoints.removeAll()
        rejected.removeAll()
        lastPose = nil
        isRecording = true
        session.run(config, options: [.resetTracking, .removeExistingAnchors])
    }

    func stop() throws -> URL {
        isRecording = false
        session.pause()
        return try writeBundle()
    }

    // MARK: - ARSessionDelegate

    nonisolated func session(_ session: ARSession, didUpdate frame: ARFrame) {
        // ARKit hands frames over on its own queue; hop to the main actor and
        // release the ARFrame promptly — holding them starves the session.
        let transform = frame.camera.transform
        let intrinsics = frame.camera.intrinsics
        let resolution = frame.camera.imageResolution
        let trackingState = frame.camera.trackingState
        let timestamp = frame.timestamp
        let features = frame.rawFeaturePoints?.points
        let lightEstimate = frame.lightEstimate?.ambientIntensity ?? 0
        let pixelBuffer = frame.capturedImage

        Task { @MainActor in
            self.handle(
                transform: transform, intrinsics: intrinsics, resolution: resolution,
                trackingState: trackingState, timestamp: timestamp,
                features: features, ambientIntensity: lightEstimate,
                pixelBuffer: pixelBuffer)
        }
    }

    private func handle(
        transform: simd_float4x4,
        intrinsics: simd_float3x3,
        resolution: CGSize,
        trackingState: ARCamera.TrackingState,
        timestamp: TimeInterval,
        features: [simd_float3]?,
        ambientIntensity: CGFloat,
        pixelBuffer: CVPixelBuffer
    ) {
        guard isRecording, keyframes.count < params.maxKeyframes else {
            if isRecording { bump("cap_reached") }
            return
        }

        // Hard gates first, so a rejected frame never becomes the baseline the
        // next frame is measured against.
        guard case .normal = trackingState else {
            bump("tracking")
            return
        }

        if let prev = lastPose {
            let moved = simd_distance(prev.columns.3, transform.columns.3)
            let rotated = angleBetween(prev, transform)
            guard moved >= params.minTranslationM || rotated >= params.minRotationRad else {
                return  // too similar; not an error, just not a keyframe
            }
        }

        guard let image = downscaled(pixelBuffer) else {
            bump("decode")
            return
        }
        let blur = varianceOfLaplacian(image)
        guard blur >= params.minBlurScore else {
            bump("blur")
            return
        }
        let clip = exposureClipRatio(image)
        guard clip <= params.maxExposureClipRatio else {
            bump("exposure")
            return
        }

        let id = String(format: "kf_%04d", keyframes.count)
        guard let jpeg = UIImage(cgImage: image).jpegData(compressionQuality: 0.92) else {
            bump("encode")
            return
        }
        let rel = "images/\(id).jpg"
        try? jpeg.write(to: bundleDir.appendingPathComponent(rel))

        // Intrinsics are reported for the full capture resolution; rescale to
        // the resolution actually written or every projection downstream is
        // wrong by the downscale factor.
        let scale = Double(image.width) / Double(resolution.width)
        let wire = IntrinsicsWire(
            fx: Double(intrinsics[0][0]) * scale,
            fy: Double(intrinsics[1][1]) * scale,
            cx: Double(intrinsics[2][0]) * scale,
            cy: Double(intrinsics[2][1]) * scale,
            width: image.width,
            height: image.height)

        keyframes.append(KeyframeWire(
            id: id,
            image: rel,
            timestampNs: Int64(timestamp * 1e9),
            pose: FrameConversion.poseFromARKit(transform),
            intrinsics: wire,
            trackingState: .normal,
            blurScore: blur,
            exposureClipRatio: clip,
            depth: nil))
        lastPose = transform

        if let features, !features.isEmpty {
            // A thinned sample is plenty: these exist to gate keyframe
            // acceptance on texture, not to reconstruct anything.
            for p in stride(from: 0, to: features.count, by: max(1, features.count / 20)) {
                sparsePoints.append(SparsePointWire(
                    xyz: [Double(features[p].x), Double(features[p].y), Double(features[p].z)],
                    observations: [id]))
            }
        }

        if ambientIntensity > 0, ambientIntensity < 60 {
            bump("dark")  // plan §7 lux gate; surfaced, not enforced, in the test app
        }
    }

    // MARK: - Reference measurements (plan §6.2)

    /// One tape measure per room. `a`/`b` are AR-picked world points.
    func addTapeMeasure(from a: simd_float3, to b: simd_float3, measuredMeters: Double) {
        measurements.append(ReferenceMeasurementWire(
            type: "user_tape",
            pointA: [Double(a.x), Double(a.y), Double(a.z)],
            pointB: [Double(b.x), Double(b.y), Double(b.z)],
            measuredValueM: measuredMeters,
            sigmaM: 0.005))
    }

    /// The VIO scale prior is never omitted — it is the anchor that keeps one
    /// bad tape entry from rescaling the room (plan §6.2).
    private func defaultMeasurements() -> [ReferenceMeasurementWire] {
        [ReferenceMeasurementWire(
            type: "vio_scale", pointA: nil, pointB: nil,
            measuredValueM: 1.0, sigmaM: 0.03)]
    }

    // MARK: - Writing

    private func writeBundle() throws -> URL {
        let gravity = FrameConversion.gravityVector(alignment: .gravity)
        guard gravity.count == 3 else {
            throw NSError(domain: "recon", code: 1, userInfo: [
                NSLocalizedDescriptionKey:
                    "world alignment has no gravity; refusing to write an unoriented bundle"
            ])
        }

        let bundle = SessionBundleWire(
            sessionId: sessionId,
            device: DeviceInfo(
                model: UIDevice.current.modelIdentifier,
                os: "\(UIDevice.current.systemName) \(UIDevice.current.systemVersion)",
                arPlatform: .arkit,
                arVersion: "ARKit/\(ProcessInfo.processInfo.operatingSystemVersionString)"),
            roomId: roomId,
            gravity: gravity,
            keyframes: keyframes,
            sparsePoints: sparsePoints,
            referenceMeasurements: defaultMeasurements() + measurements)

        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(bundle)
        let url = bundleDir.appendingPathComponent("bundle.json")
        try data.write(to: url)
        return bundleDir
    }

    private func bump(_ reason: String) {
        rejected[reason, default: 0] += 1
    }

    // MARK: - Image quality

    private func downscaled(_ buffer: CVPixelBuffer) -> CGImage? {
        let ci = CIImage(cvPixelBuffer: buffer)
        let scale = CGFloat(params.targetWidth) / ci.extent.width
        guard scale < 1.0 else { return ciContext.createCGImage(ci, from: ci.extent) }
        let scaled = ci.transformed(by: CGAffineTransform(scaleX: scale, y: scale))
        return ciContext.createCGImage(scaled, from: scaled.extent)
    }

    /// Variance of the Laplacian — the plan §4/Phase 1 blur metric.
    private func varianceOfLaplacian(_ image: CGImage) -> Double {
        guard let gray = grayscaleBytes(image) else { return 0 }
        let (w, h, pixels) = gray
        guard w > 2, h > 2 else { return 0 }

        var sum = 0.0
        var sumSq = 0.0
        var n = 0.0
        for y in 1..<(h - 1) {
            for x in 1..<(w - 1) {
                let i = y * w + x
                let lap = -4.0 * Double(pixels[i])
                    + Double(pixels[i - 1]) + Double(pixels[i + 1])
                    + Double(pixels[i - w]) + Double(pixels[i + w])
                sum += lap
                sumSq += lap * lap
                n += 1
            }
        }
        guard n > 0 else { return 0 }
        let mean = sum / n
        return sumSq / n - mean * mean
    }

    private func exposureClipRatio(_ image: CGImage) -> Double {
        guard let gray = grayscaleBytes(image) else { return 0 }
        let (_, _, pixels) = gray
        guard !pixels.isEmpty else { return 0 }
        var clipped = 0
        for p in pixels where p == 0 || p == 255 { clipped += 1 }
        return Double(clipped) / Double(pixels.count)
    }

    private func grayscaleBytes(_ image: CGImage) -> (Int, Int, [UInt8])? {
        let w = image.width, h = image.height
        var buf = [UInt8](repeating: 0, count: w * h)
        guard let ctx = CGContext(
            data: &buf, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w,
            space: CGColorSpaceCreateDeviceGray(),
            bitmapInfo: CGImageAlphaInfo.none.rawValue)
        else { return nil }
        ctx.draw(image, in: CGRect(x: 0, y: 0, width: w, height: h))
        return (w, h, buf)
    }

    private func angleBetween(_ a: simd_float4x4, _ b: simd_float4x4) -> Float {
        let ra = simd_float3x3(a.columns.0.xyz, a.columns.1.xyz, a.columns.2.xyz)
        let rb = simd_float3x3(b.columns.0.xyz, b.columns.1.xyz, b.columns.2.xyz)
        let rel = ra.transpose * rb
        let trace = rel[0][0] + rel[1][1] + rel[2][2]
        return acos(min(max((trace - 1) * 0.5, -1), 1))
    }
}

private extension simd_float4 {
    var xyz: simd_float3 { simd_float3(x, y, z) }
}

extension UIDevice {
    /// "iPhone16,1" — the identifier the harness records, since `model` only
    /// ever says "iPhone".
    var modelIdentifier: String {
        var info = utsname()
        uname(&info)
        return withUnsafePointer(to: &info.machine) {
            $0.withMemoryRebound(to: CChar.self, capacity: 1) { String(cString: $0) }
        }
    }
}
