package com.recon.poselogger

import com.google.ar.core.Camera
import com.google.ar.core.Pose as ArPose
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable

/**
 * Wire types for the plan §5 SessionBundle, and the ARCore -> canonical frame
 * conversion.
 *
 * Plan §7 lists "flipped/mirrored scenes" as a failure mode and attributes it to
 * "Apple right-handed vs Google left-handed". That framing is worth being
 * precise about, because getting it wrong in either direction ships a mirrored
 * house:
 *
 *  - **ARCore's world/camera frame is right-handed**, +Y up (with the default
 *    gravity-aligned session) and -Z forward — the same convention as ARKit and
 *    the same as our canonical frame. [FrameConversion.fromArCore] is therefore
 *    an honest identity, and [FrameConversionTest] asserts it.
 *  - The left-handed convention people hit is **Unity's**, which AR Foundation
 *    converts into. Since this client is native Kotlin talking to ARCore
 *    directly, we never enter that frame. If anyone ever moves this client to
 *    Unity (plan §3.8's rejected alternative), the conversion belongs *here*
 *    and the tests below are what will catch its absence.
 *
 * The determinant check in [FrameConversion.fromArCore] is the backstop: any
 * handedness flip makes det(R) = -1, and we refuse rather than upload.
 */

@Serializable
data class CoordinateFrame(
    val handedness: String = "right",
    @SerialName("up_axis") val upAxis: String = "y_up",
    @SerialName("camera_forward") val cameraForward: String = "-z",
    val name: String = "recon_world_v1",
)

@Serializable
data class DeviceInfo(
    val model: String,
    val os: String,
    @SerialName("ar_platform") val arPlatform: String,
    @SerialName("ar_version") val arVersion: String,
)

@Serializable
data class IntrinsicsWire(
    val fx: Double,
    val fy: Double,
    val cx: Double,
    val cy: Double,
    val width: Int,
    val height: Int,
    val distortion: List<Double> = emptyList(),
)

/** Row-major 3x3 rotation and a translation, world<-camera. */
@Serializable
data class PoseWire(val R: List<List<Double>>, val t: List<Double>)

@Serializable
data class KeyframeWire(
    val id: String,
    val image: String,
    @SerialName("timestamp_ns") val timestampNs: Long,
    val pose: PoseWire,
    val intrinsics: IntrinsicsWire,
    @SerialName("tracking_state") val trackingState: String,
    @SerialName("blur_score") val blurScore: Double,
    @SerialName("exposure_clip_ratio") val exposureClipRatio: Double,
    val depth: String? = null,
)

@Serializable
data class SparsePointWire(
    val xyz: List<Double>,
    val observations: List<String> = emptyList(),
)

@Serializable
data class ReferenceMeasurementWire(
    val type: String,
    @SerialName("point_a") val pointA: List<Double>? = null,
    @SerialName("point_b") val pointB: List<Double>? = null,
    @SerialName("measured_value_m") val measuredValueM: Double,
    @SerialName("sigma_m") val sigmaM: Double,
)

@Serializable
data class DoorwayWire(
    val id: String,
    val plane: List<Double>,
    val centroid: List<Double>,
    @SerialName("width_m") val widthM: Double,
    @SerialName("height_m") val heightM: Double,
)

@Serializable
data class SessionBundleWire(
    @SerialName("session_id") val sessionId: String,
    val device: DeviceInfo,
    @SerialName("room_id") val roomId: String,
    val gravity: List<Double>,
    val frame: CoordinateFrame = CoordinateFrame(),
    val keyframes: List<KeyframeWire> = emptyList(),
    @SerialName("sparse_points") val sparsePoints: List<SparsePointWire> = emptyList(),
    @SerialName("reference_measurements")
    val referenceMeasurements: List<ReferenceMeasurementWire> = emptyList(),
    val doorways: List<DoorwayWire> = emptyList(),
    @SerialName("schema_version") val schemaVersion: Int = 1,
)

class FrameConversionException(message: String) : RuntimeException(message)

object FrameConversion {

    /**
     * Convert an ARCore camera pose into the canonical frame.
     *
     * ARCore returns the pose as a column-major 4x4 (OpenGL layout) via
     * [ArPose.toMatrix]. Column-major means element (row, col) lives at
     * `m[col * 4 + row]` — transposing this by accident is precisely how a
     * mirrored scene gets shipped, so the indexing is written out explicitly.
     */
    fun fromArCore(pose: ArPose): PoseWire {
        val m = FloatArray(16)
        pose.toMatrix(m, 0)

        val r = List(3) { row -> List(3) { col -> m[col * 4 + row].toDouble() } }
        val t = listOf(m[12].toDouble(), m[13].toDouble(), m[14].toDouble())

        val det = determinant3(r)
        if (det < 0.0) {
            throw FrameConversionException(
                "ARCore pose has determinant $det (mirrored). Refusing to upload a " +
                    "bundle that would reconstruct as a mirror image. See plan §7."
            )
        }
        if (kotlin.math.abs(det - 1.0) > 1e-3) {
            throw FrameConversionException(
                "ARCore pose is not a pure rotation (det=$det); the pose carries scale."
            )
        }
        return PoseWire(r, t)
    }

    private fun determinant3(r: List<List<Double>>): Double =
        r[0][0] * (r[1][1] * r[2][2] - r[1][2] * r[2][1]) -
            r[0][1] * (r[1][0] * r[2][2] - r[1][2] * r[2][0]) +
            r[0][2] * (r[1][0] * r[2][1] - r[1][1] * r[2][0])

    /**
     * World-frame down vector. ARCore's default session is gravity-aligned with
     * +Y up, so this is -Y. Read from a constant here, but asserted against the
     * device's own gravity reading at session start by [CaptureRecorder].
     */
    val GRAVITY_DOWN = listOf(0.0, -1.0, 0.0)

    /**
     * ARCore reports intrinsics for the CPU image at its own resolution. If we
     * downscale before saving, the intrinsics must be scaled to match or every
     * back-projection downstream is off by the scale factor.
     */
    fun intrinsicsFor(camera: Camera, savedWidth: Int, savedHeight: Int): IntrinsicsWire {
        val intr = camera.imageIntrinsics
        val dims = intr.imageDimensions
        val focal = intr.focalLength
        val principal = intr.principalPoint
        val sx = savedWidth.toDouble() / dims[0].toDouble()
        val sy = savedHeight.toDouble() / dims[1].toDouble()
        return IntrinsicsWire(
            fx = focal[0].toDouble() * sx,
            fy = focal[1].toDouble() * sy,
            cx = principal[0].toDouble() * sx,
            cy = principal[1].toDouble() * sy,
            width = savedWidth,
            height = savedHeight,
        )
    }
}
