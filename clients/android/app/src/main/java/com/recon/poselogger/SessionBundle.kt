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
    /**
     * Device heading when this frame was taken, degrees clockwise from magnetic
     * north, or null if the device has no rotation-vector sensor or had not yet
     * reported one.
     *
     * Recorded per keyframe rather than once per session because the value of a
     * compass here is *relative*: indoors the field is bent tens of degrees by
     * steel and wiring, so no single reading is a true bearing, but two readings
     * in the same room share most of that distortion and their difference is
     * meaningful. A per-session heading would throw away exactly that.
     *
     * Nothing consumes this yet. It is recorded because it cannot be recovered
     * afterwards — a capture without it is a capture that can never be used to
     * test whether it helps.
     */
    @SerialName("heading_deg") val headingDeg: Double? = null,
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

/**
 * The footprint the user tapped out corner by corner, in capture-frame metres.
 *
 * `closed` is not decoration: an open outline is a partial trace the user
 * abandoned, and the harness must not treat it as a room boundary.
 */
@Serializable
data class FloorOutlineWire(
    val corners: List<List<Double>> = emptyList(),
    /** Stable identities parallel to `corners`, used by adjustment records. */
    @SerialName("corner_ids") val cornerIds: List<Long> = emptyList(),
    val closed: Boolean = false,
    /**
     * User-driven moves of footprint corners made after recording began.
     *
     * The final `corners` remain the asserted room boundary.  This history is
     * the correction signal: it says which stable reference point moved, from
     * where to where, and during which part of the capture it happened.
     */
    @SerialName("reference_point_adjustments")
    val referencePointAdjustments: List<ReferencePointAdjustmentWire> = emptyList(),
)

@Serializable
data class ReferencePointAdjustmentWire(
    @SerialName("corner_id") val cornerId: Long,
    @SerialName("from") val fromPoint: List<Double>,
    @SerialName("to") val toPoint: List<Double>,
    val delta: List<Double>,
    @SerialName("started_at_ns") val startedAtNs: Long,
    @SerialName("ended_at_ns") val endedAtNs: Long,
)

@Serializable
data class DoorwayWire(
    val id: String,
    val plane: List<Double>,
    val centroid: List<Double>,
    @SerialName("width_m") val widthM: Double,
    @SerialName("height_m") val heightM: Double,
)

/**
 * A shot the user reported as unobtainable.
 *
 * Recorded so the harness can tell a direction nobody could reach from one
 * the script never asked for. Without it a capture that skipped its way to
 * the end is indistinguishable from a complete one, and the first thing the
 * validator would do with that is trust it.
 *
 * Yaw is degrees relative to where the capture started, matching the core's
 * own convention — the room's orientation is not known when the script is
 * built, and converting to world angles later would invent precision.
 */
@Serializable
data class SkippedShotWire(
    @SerialName("yaw_deg") val yawDeg: Double,
    /** "bottom" | "middle" | "top" — the floor, wall or ceiling junction. */
    val height: String,
    @SerialName("at_station") val atStation: Boolean,
)

@Serializable
data class SessionBundleWire(
    @SerialName("session_id") val sessionId: String,
    val device: DeviceInfo,
    @SerialName("room_id") val roomId: String,
    val gravity: List<Double>,
    /** Explicitly separates production depth captures from the RGB-only fallback. */
    @SerialName("capture_mode") val captureMode: String = "unknown",
    /** Startup proof that ARCore returned a real depth image for this session. */
    @SerialName("depth_self_test") val depthSelfTest: String? = null,
    @SerialName("depth_frame_count") val depthFrameCount: Int = 0,
    val frame: CoordinateFrame = CoordinateFrame(),
    val keyframes: List<KeyframeWire> = emptyList(),
    @SerialName("sparse_points") val sparsePoints: List<SparsePointWire> = emptyList(),
    @SerialName("reference_measurements")
    val referenceMeasurements: List<ReferenceMeasurementWire> = emptyList(),
    val doorways: List<DoorwayWire> = emptyList(),
    /** User-tapped room footprint; null when the user never traced one. */
    @SerialName("floor_outline") val floorOutline: FloorOutlineWire? = null,
    /** Shots the user could not obtain; empty on a capture that got everything. */
    @SerialName("skipped_shots") val skippedShots: List<SkippedShotWire> = emptyList(),
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
    fun fromArCore(
        pose: ArPose,
        clockwiseImageRotationDegrees: Int = 0,
    ): PoseWire {
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
        return rotateCameraAxes(
            PoseWire(r, t),
            clockwiseImageRotationDegrees,
        )
    }

    /**
     * Rotate the camera basis with display-upright pixels.
     *
     * Rotating an image is not just an intrinsics change: the saved image's +X
     * and +Y axes are new camera axes. Post-multiplying world<-camera by this
     * Z rotation keeps every unprojected ray on the same world-space ray.
     */
    fun rotateCameraAxes(
        pose: PoseWire,
        clockwiseImageRotationDegrees: Int,
    ): PoseWire {
        val rotation = ((clockwiseImageRotationDegrees % 360) + 360) % 360
        require(rotation % 90 == 0) { "image rotation must be a multiple of 90, got $rotation" }
        val radians = Math.toRadians(rotation.toDouble())
        val c = kotlin.math.cos(radians)
        val s = kotlin.math.sin(radians)
        val oldFromNew = listOf(
            listOf(c, -s, 0.0),
            listOf(s, c, 0.0),
            listOf(0.0, 0.0, 1.0),
        )
        val rotated = List(3) { row ->
            List(3) { col ->
                (0 until 3).sumOf { k -> pose.R[row][k] * oldFromNew[k][col] }
            }
        }
        return pose.copy(R = rotated)
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
     * ARCore reports unrotated intrinsics for the CPU sensor image. Captured
     * pixels are rotated into display-upright orientation before saving, so the
     * principal point and focal axes must undergo the exact same transform
     * before they are scaled to the saved resolution.
     */
    fun intrinsicsFor(
        camera: Camera,
        savedWidth: Int,
        savedHeight: Int,
        clockwiseRotationDegrees: Int,
    ): IntrinsicsWire {
        val intr = camera.imageIntrinsics
        val dims = intr.imageDimensions
        val focal = intr.focalLength
        val principal = intr.principalPoint
        return rotateAndScaleIntrinsics(
            IntrinsicsWire(
                fx = focal[0].toDouble(),
                fy = focal[1].toDouble(),
                cx = principal[0].toDouble(),
                cy = principal[1].toDouble(),
                width = dims[0],
                height = dims[1],
            ),
            clockwiseRotationDegrees,
            savedWidth,
            savedHeight,
        )
    }

    /** Pure form of [intrinsicsFor], kept device-free for reprojection tests. */
    fun rotateAndScaleIntrinsics(
        source: IntrinsicsWire,
        clockwiseRotationDegrees: Int,
        savedWidth: Int,
        savedHeight: Int,
    ): IntrinsicsWire {
        val rotation = ((clockwiseRotationDegrees % 360) + 360) % 360
        require(rotation % 90 == 0) { "image rotation must be a multiple of 90, got $rotation" }
        require(savedWidth > 0 && savedHeight > 0) { "saved image dimensions must be positive" }

        val rotated = when (rotation) {
            0 -> source
            90 -> IntrinsicsWire(
                fx = source.fy,
                fy = source.fx,
                cx = source.height - 1.0 - source.cy,
                cy = source.cx,
                width = source.height,
                height = source.width,
                distortion = source.distortion,
            )
            180 -> IntrinsicsWire(
                fx = source.fx,
                fy = source.fy,
                cx = source.width - 1.0 - source.cx,
                cy = source.height - 1.0 - source.cy,
                width = source.width,
                height = source.height,
                distortion = source.distortion,
            )
            270 -> IntrinsicsWire(
                fx = source.fy,
                fy = source.fx,
                cx = source.cy,
                cy = source.width - 1.0 - source.cx,
                width = source.height,
                height = source.width,
                distortion = source.distortion,
            )
            else -> error("unreachable rotation $rotation")
        }
        val sx = savedWidth.toDouble() / rotated.width.toDouble()
        val sy = savedHeight.toDouble() / rotated.height.toDouble()
        return rotated.copy(
            fx = rotated.fx * sx,
            fy = rotated.fy * sy,
            cx = rotated.cx * sx,
            cy = rotated.cy * sy,
            width = savedWidth,
            height = savedHeight,
        )
    }
}
