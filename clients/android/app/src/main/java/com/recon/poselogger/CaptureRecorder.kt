package com.recon.poselogger

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.media.Image
import android.os.Build
import com.google.ar.core.Frame
import com.google.ar.core.Session
import com.google.ar.core.TrackingState
import com.google.ar.core.exceptions.NotYetAvailableException
import java.io.ByteArrayOutputStream
import java.io.File
import java.util.UUID
import kotlin.math.acos
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import uniffi.recon_core.FfiCaptureSession
import uniffi.recon_core.FfiDecision
import uniffi.recon_core.FfiFrame
import uniffi.recon_core.FfiGuidance
import uniffi.recon_core.FfiParams
import uniffi.recon_core.FfiPose
import uniffi.recon_core.FfiTracking
import uniffi.recon_core.FfiVec3
import uniffi.recon_core.defaultParams

/**
 * The throwaway pose-logging capture app from plan task 3, Android side.
 *
 * Deliberately mirrors [com.recon.poselogger] iOS `CaptureRecorder` decision for
 * decision: same keyframe rule, same blur metric, same rejection accounting. Any
 * behavioural difference between the two clients would show up as a per-platform
 * quality difference in Phase 0 that nobody could explain — which is exactly why
 * plan §3.8 puts this logic in a shared Rust core from Phase 3 onward. Until the
 * UniFFI bindings land, keeping these two files symmetric is the stopgap.
 */
class CaptureRecorder(
    private val session: Session,
    private val outputRoot: File,
    private val roomId: String = UUID.randomUUID().toString(),
) {

    /**
     * Capture thresholds, read from the shared Rust core (ADR-0001).
     *
     * These used to be hand-written constants duplicated here and in the iOS
     * client, flagged as a stopgap in both. They now come from
     * [uniffi.recon_core.defaultParams], so there is exactly one place where
     * "15 degrees" is written down and the two platforms cannot disagree.
     *
     * `targetWidth` stays local: it is a client-side encoding choice about how
     * much image to upload, not capture behaviour the core has an opinion on.
     */
    val core: FfiParams = defaultParams()
    val targetWidth: Int = 960

    val maxKeyframes: Int get() = core.maxKeyframes.toInt()
    private val minRotationRad: Float get() = Math.toRadians(core.minRotationDeg.toDouble()).toFloat()
    private val minTranslationM: Float get() = core.minTranslationM

    /**
     * The core's own capture-session state machine, which owns the keyframe
     * rule, the nag priorities and their cooldowns.
     *
     * Points and normals are passed empty for now: the on-device depth pass
     * (DA3-Small via ExecuTorch, plan §3.6) does not exist yet, so there is no
     * surface to integrate. The core handles that — with no geometry it still
     * runs the quality gates and returns a decision, and simply reports no
     * coverage and no next station.
     */
    private var guidance: FfiCaptureSession? = null

    /** The most recent guidance, for the UI to draw. */
    @Volatile var lastGuidance: FfiGuidance? = null
        private set
    private val sessionId = UUID.randomUUID().toString()
    private val keyframes = mutableListOf<KeyframeWire>()
    private val sparsePoints = mutableListOf<SparsePointWire>()
    private val measurements = mutableListOf<ReferenceMeasurementWire>()

    /** Rejection counts, surfaced in the UI so a silently failing capture is visible. */
    val rejected = mutableMapOf<String, Int>()

    private var lastPose: FloatArray? = null
    private lateinit var bundleDir: File
    private var recording = false

    val keyframeCount: Int get() = keyframes.size

    fun start() {
        bundleDir = File(outputRoot, "bundle_$sessionId").apply { mkdirs() }
        File(bundleDir, "images").mkdirs()
        keyframes.clear()
        sparsePoints.clear()
        rejected.clear()
        lastPose = null
        lastGuidance = null
        // Gravity as the world-frame down vector, per plan §5. ARCore is
        // gravity-aligned, so this is -Y.
        guidance = FfiCaptureSession(FfiVec3(0.0f, -1.0f, 0.0f))
        recording = true
    }

    /** Call once per ARCore frame. Cheap gates run before any image work. */
    fun onFrame(frame: Frame) {
        if (!recording) return
        if (keyframes.size >= maxKeyframes) {
            bump("cap_reached")
            return
        }
        val camera = frame.camera
        if (camera.trackingState != TrackingState.TRACKING) {
            bump("tracking")
            return
        }

        val matrix = FloatArray(16)
        camera.pose.toMatrix(matrix, 0)
        // Cheap pre-filter before decoding an image. The core makes the real
        // decision below; this only avoids the cost of a YUV copy for frames
        // that obviously cannot qualify.
        lastPose?.let { prev ->
            val moved = translationDistance(prev, matrix)
            val rotated = rotationAngle(prev, matrix)
            if (moved < minTranslationM && rotated < minRotationRad) return
        }

        val image: Image = try {
            frame.acquireCameraImage()
        } catch (e: NotYetAvailableException) {
            bump("image_unavailable")
            return
        }

        try {
            val gray = luminancePlane(image) ?: run { bump("decode"); return }
            val blur = varianceOfLaplacian(gray.pixels, gray.width, gray.height)
            val clip = exposureClipRatio(gray.pixels)

            // The core decides. Blur and exposure are measured here because
            // only the client has the pixels, but the thresholds and the
            // priority ordering live in one place for both platforms.
            val pose = FrameConversion.fromArCore(camera.pose)
            val update = guidance?.observe(
                FfiFrame(
                    timestampS = frame.timestamp / 1e9,
                    pose = FfiPose(
                        r = pose.R.flatten().map { it.toFloat() },
                        t = FfiVec3(
                            pose.t[0].toFloat(), pose.t[1].toFloat(), pose.t[2].toFloat()
                        ),
                    ),
                    tracking = FfiTracking.NORMAL,
                    blurScore = blur.toFloat(),
                    exposureClipRatio = clip.toFloat(),
                    // No on-device depth yet, so no median-depth reading. The
                    // core treats 0 as "unknown" and skips the TOO_CLOSE nag.
                    medianDepthM = 0.0f,
                    angularVelocityRadS = lastPose?.let { rotationAngle(it, matrix) } ?: 0.0f,
                    sparsePointCount = sparsePointsInView(frame),
                    lux = frame.lightEstimate?.pixelIntensity?.times(1000f) ?: 200.0f,
                ),
                emptyList(),
                emptyList(),
            )
            lastGuidance = update

            if (update != null && update.decision != FfiDecision.ACCEPT) {
                bump(update.decision.name.removePrefix("REJECT_").lowercase())
                return
            }

            val bitmap = toBitmap(image, targetWidth) ?: run { bump("encode"); return }
            val id = "kf_%04d".format(keyframes.size)
            val rel = "images/$id.jpg"
            File(bundleDir, rel).outputStream().use {
                bitmap.compress(Bitmap.CompressFormat.JPEG, 92, it)
            }

            keyframes += KeyframeWire(
                id = id,
                image = rel,
                timestampNs = frame.timestamp,
                pose = pose,
                intrinsics = FrameConversion.intrinsicsFor(camera, bitmap.width, bitmap.height),
                trackingState = "normal",
                blurScore = blur,
                exposureClipRatio = clip,
            )
            lastPose = matrix

            frame.acquirePointCloud().use { cloud ->
                val buf = cloud.points
                val n = buf.remaining() / 4
                val stride = max(1, n / 20)
                var i = 0
                while (i < n) {
                    val base = i * 4
                    // The 4th component is a confidence, not a coordinate.
                    if (buf.get(base + 3) > 0.3f) {
                        sparsePoints += SparsePointWire(
                            xyz = listOf(
                                buf.get(base).toDouble(),
                                buf.get(base + 1).toDouble(),
                                buf.get(base + 2).toDouble(),
                            ),
                            observations = listOf(id),
                        )
                    }
                    i += stride
                }
            }
        } finally {
            image.close()
        }
    }

    /** How many AR feature points fall in view — the core's texture signal. */
    private fun sparsePointsInView(frame: Frame): UInt =
        try {
            frame.acquirePointCloud().use { (it.points.remaining() / 4).toUInt() }
        } catch (e: Exception) {
            0u
        }

    /** One tape measure per room (plan §6.2). */
    fun addTapeMeasure(a: FloatArray, b: FloatArray, measuredMeters: Double) {
        measurements += ReferenceMeasurementWire(
            type = "user_tape",
            pointA = a.map { it.toDouble() },
            pointB = b.map { it.toDouble() },
            measuredValueM = measuredMeters,
            sigmaM = 0.005,
        )
    }

    fun stop(): File {
        recording = false
        guidance?.destroy()
        guidance = null
        val bundle = SessionBundleWire(
            sessionId = sessionId,
            device = DeviceInfo(
                model = "${Build.MANUFACTURER} ${Build.MODEL}",
                os = "Android ${Build.VERSION.RELEASE} (API ${Build.VERSION.SDK_INT})",
                arPlatform = "arcore",
                arVersion = "ARCore",
            ),
            roomId = roomId,
            gravity = FrameConversion.GRAVITY_DOWN,
            keyframes = keyframes.toList(),
            sparsePoints = sparsePoints.toList(),
            // The VIO scale prior is always present: it is the anchor that stops
            // one bad tape entry from rescaling the room (plan §6.2).
            referenceMeasurements = listOf(
                ReferenceMeasurementWire(
                    type = "vio_scale", measuredValueM = 1.0, sigmaM = 0.03,
                )
            ) + measurements,
        )
        val json = Json { prettyPrint = true; encodeDefaults = true }
        File(bundleDir, "bundle.json").writeText(json.encodeToString(bundle))
        return bundleDir
    }

    // -- image quality ------------------------------------------------------

    private class Gray(val pixels: ByteArray, val width: Int, val height: Int)

    /**
     * ARCore hands back YUV_420_888; the Y plane *is* the luminance image, so
     * blur and exposure are scored without ever decoding colour.
     */
    private fun luminancePlane(image: Image): Gray? {
        if (image.format != android.graphics.ImageFormat.YUV_420_888) return null
        val plane = image.planes[0]
        val buffer = plane.buffer
        val rowStride = plane.rowStride
        val w = image.width
        val h = image.height
        val out = ByteArray(w * h)
        val row = ByteArray(rowStride)
        var pos = 0
        for (y in 0 until h) {
            buffer.position(y * rowStride)
            val len = min(rowStride, buffer.remaining())
            buffer.get(row, 0, len)
            System.arraycopy(row, 0, out, pos, w)
            pos += w
        }
        return Gray(out, w, h)
    }

    /** Variance of the Laplacian — same metric as the iOS client. */
    private fun varianceOfLaplacian(pixels: ByteArray, w: Int, h: Int): Double {
        if (w < 3 || h < 3) return 0.0
        var sum = 0.0
        var sumSq = 0.0
        var n = 0.0
        fun at(i: Int) = (pixels[i].toInt() and 0xFF).toDouble()
        for (y in 1 until h - 1) {
            for (x in 1 until w - 1) {
                val i = y * w + x
                val lap = -4.0 * at(i) + at(i - 1) + at(i + 1) + at(i - w) + at(i + w)
                sum += lap
                sumSq += lap * lap
                n += 1.0
            }
        }
        if (n == 0.0) return 0.0
        val mean = sum / n
        return sumSq / n - mean * mean
    }

    private fun exposureClipRatio(pixels: ByteArray): Double {
        if (pixels.isEmpty()) return 0.0
        var clipped = 0
        for (p in pixels) {
            val v = p.toInt() and 0xFF
            if (v == 0 || v == 255) clipped++
        }
        return clipped.toDouble() / pixels.size
    }

    private fun toBitmap(image: Image, targetWidth: Int): Bitmap? {
        val yuv = android.graphics.YuvImage(
            yuvBytes(image), android.graphics.ImageFormat.NV21,
            image.width, image.height, null,
        )
        val out = ByteArrayOutputStream()
        yuv.compressToJpeg(
            android.graphics.Rect(0, 0, image.width, image.height), 95, out
        )
        val full = BitmapFactory.decodeByteArray(out.toByteArray(), 0, out.size()) ?: return null
        if (full.width <= targetWidth) return full
        val h = (full.height.toLong() * targetWidth / full.width).toInt()
        return Bitmap.createScaledBitmap(full, targetWidth, h, true)
    }

    private fun yuvBytes(image: Image): ByteArray {
        val y = image.planes[0].buffer
        val u = image.planes[1].buffer
        val v = image.planes[2].buffer
        val out = ByteArray(y.remaining() + u.remaining() + v.remaining())
        y.get(out, 0, y.remaining())
        val vStart = out.size - v.remaining() - u.remaining()
        v.get(out, vStart, v.remaining())
        u.get(out, vStart + v.remaining(), u.remaining())
        return out
    }

    // -- pose helpers -------------------------------------------------------

    private fun translationDistance(a: FloatArray, b: FloatArray): Float {
        val dx = a[12] - b[12]
        val dy = a[13] - b[13]
        val dz = a[14] - b[14]
        return sqrt(dx * dx + dy * dy + dz * dz)
    }

    /** Relative rotation magnitude via trace(Ra^T Rb); column-major indexing. */
    private fun rotationAngle(a: FloatArray, b: FloatArray): Float {
        var trace = 0.0f
        for (col in 0 until 3) {
            for (row in 0 until 3) {
                trace += a[col * 4 + row] * b[col * 4 + row]
            }
        }
        return acos(max(-1.0f, min(1.0f, (trace - 1.0f) * 0.5f)))
    }

    private fun bump(reason: String) {
        rejected[reason] = (rejected[reason] ?: 0) + 1
    }
}
