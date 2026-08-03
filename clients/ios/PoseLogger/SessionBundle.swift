//  SessionBundle.swift
//  Wire types for the plan §5 SessionBundle, and the ARKit -> canonical frame
//  conversion.
//
//  ARKit is already right-handed, Y-up, with the camera looking down -Z, which
//  is the canonical frame (plan §5). So the conversion here is the identity —
//  but it is written out explicitly rather than skipped, because the Android
//  side is *not* the identity and the two must stay visibly symmetric. A silent
//  identity on one platform and a conversion on the other is how a mirrored
//  scene ships (plan §7).

import ARKit
import Foundation
import simd

enum ARPlatform: String, Codable {
    case arkit, arcore, synthetic
}

enum TrackingStateWire: String, Codable {
    case normal, limited, relocalizing
}

struct CoordinateFrame: Codable {
    var handedness: String = "right"
    var upAxis: String = "y_up"
    var cameraForward: String = "-z"
    var name: String = "recon_world_v1"

    enum CodingKeys: String, CodingKey {
        case handedness
        case upAxis = "up_axis"
        case cameraForward = "camera_forward"
        case name
    }
}

struct DeviceInfo: Codable {
    var model: String
    var os: String
    var arPlatform: ARPlatform
    var arVersion: String

    enum CodingKeys: String, CodingKey {
        case model, os
        case arPlatform = "ar_platform"
        case arVersion = "ar_version"
    }
}

struct IntrinsicsWire: Codable {
    var fx: Double
    var fy: Double
    var cx: Double
    var cy: Double
    var width: Int
    var height: Int
    var distortion: [Double] = []
}

struct PoseWire: Codable {
    /// Row-major 3x3, world<-camera.
    var R: [[Double]]
    var t: [Double]
}

struct KeyframeWire: Codable {
    var id: String
    var image: String
    var timestampNs: Int64
    var pose: PoseWire
    var intrinsics: IntrinsicsWire
    var trackingState: TrackingStateWire
    var blurScore: Double
    var exposureClipRatio: Double
    var depth: String?

    enum CodingKeys: String, CodingKey {
        case id, image, pose, intrinsics, depth
        case timestampNs = "timestamp_ns"
        case trackingState = "tracking_state"
        case blurScore = "blur_score"
        case exposureClipRatio = "exposure_clip_ratio"
    }
}

struct SparsePointWire: Codable {
    var xyz: [Double]
    var observations: [String] = []
}

struct ReferenceMeasurementWire: Codable {
    var type: String
    var pointA: [Double]?
    var pointB: [Double]?
    var measuredValueM: Double
    var sigmaM: Double

    enum CodingKeys: String, CodingKey {
        case type
        case pointA = "point_a"
        case pointB = "point_b"
        case measuredValueM = "measured_value_m"
        case sigmaM = "sigma_m"
    }
}

struct DoorwayWire: Codable {
    var id: String
    var plane: [Double]
    var centroid: [Double]
    var widthM: Double
    var heightM: Double

    enum CodingKeys: String, CodingKey {
        case id, plane, centroid
        case widthM = "width_m"
        case heightM = "height_m"
    }
}

struct SessionBundleWire: Codable {
    var sessionId: String
    var device: DeviceInfo
    var roomId: String
    var gravity: [Double]
    var frame = CoordinateFrame()
    var keyframes: [KeyframeWire] = []
    var sparsePoints: [SparsePointWire] = []
    var referenceMeasurements: [ReferenceMeasurementWire] = []
    var doorways: [DoorwayWire] = []
    var schemaVersion = 1

    enum CodingKeys: String, CodingKey {
        case device, gravity, frame, keyframes, doorways
        case sessionId = "session_id"
        case roomId = "room_id"
        case sparsePoints = "sparse_points"
        case referenceMeasurements = "reference_measurements"
        case schemaVersion = "schema_version"
    }
}

// MARK: - Frame conversion

enum FrameConversion {
    /// ARKit's world transform is already in the canonical frame.
    ///
    /// `ARCamera.transform` is world<-camera, right-handed, with -Z forward and
    /// +Y up. `ARSession` with `.gravity` alignment puts +Y along world up. So
    /// this is the identity; it exists so that both clients have exactly one
    /// named place where the frame is established.
    static func poseFromARKit(_ transform: simd_float4x4) -> PoseWire {
        // simd_float4x4 is column-major: columns.0 is the first basis vector.
        let c0 = transform.columns.0
        let c1 = transform.columns.1
        let c2 = transform.columns.2
        let c3 = transform.columns.3
        return PoseWire(
            R: [
                [Double(c0.x), Double(c1.x), Double(c2.x)],
                [Double(c0.y), Double(c1.y), Double(c2.y)],
                [Double(c0.z), Double(c1.z), Double(c2.z)],
            ],
            t: [Double(c3.x), Double(c3.y), Double(c3.z)]
        )
    }

    /// World-frame down vector. With gravity alignment this is exactly -Y; it is
    /// read from the frame rather than hardcoded so a future non-gravity-aligned
    /// configuration cannot silently produce an upside-down room.
    static func gravityVector(alignment: ARConfiguration.WorldAlignment) -> [Double] {
        switch alignment {
        case .gravity, .gravityAndHeading:
            return [0.0, -1.0, 0.0]
        default:
            // .camera alignment has no world up; refuse rather than guess.
            return []
        }
    }
}
