package com.recon.poselogger

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.media.Image
import android.os.Build
import com.google.ar.core.Frame
import com.google.ar.core.Session
import com.google.ar.core.TrackingFailureReason
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
import uniffi.recon_core.FfiCoverageCloud
import uniffi.recon_core.FfiDecision
import uniffi.recon_core.FfiFrame
import uniffi.recon_core.FfiGuidance
import uniffi.recon_core.FfiParams
import uniffi.recon_core.FfiPose
import uniffi.recon_core.FfiShot
import uniffi.recon_core.FfiShotState
import uniffi.recon_core.FfiSkippedShot
import uniffi.recon_core.FfiStation
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
    /** False when the device cannot do depth, so the UI can say why bars are empty. */
    val depthAvailable: Boolean = true,
    /** A real startup depth image; production depth capture is forbidden without it. */
    private val verifiedDepth: DepthSampler.Samples? = null,
    /** ARCore SDK/runtime and selected camera dimensions, persisted for diagnosis. */
    private val arCoreVersion: String = "ARCore",
    /** Clockwise sensor-buffer rotation required to make saved pixels display-upright. */
    val imageRotationDegrees: Int = 0,
) {
    init {
        require(imageRotationDegrees in 0 until 360 && imageRotationDegrees % 90 == 0) {
            "imageRotationDegrees must be one of 0, 90, 180 or 270"
        }
        require(!depthAvailable || verifiedDepth != null) {
            "depth capture requires a successful startup depth self-test"
        }
    }

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
     * Surface geometry comes from [DepthSampler] — ARCore's depth map rather
     * than the planned on-device DA3-Small pass (§3.6), which does not exist
     * yet. It feeds the coverage grid and accepted keyframes retain the same
     * depth as 16-bit PNG evidence. Where the device has no depth support the
     * core still runs every quality gate and simply reports no coverage
     * and no next station — which the UI says out loud rather than drawing as
     * zeros.
     */
    private var guidance: FfiCaptureSession? = null
    private val depth = DepthSampler()

    /** The most recent guidance, for the UI to draw. */
    @Volatile var lastGuidance: FfiGuidance? = null
        private set
    private val sessionId = UUID.randomUUID().toString()
    private val keyframes = mutableListOf<KeyframeWire>()
    private val sparsePoints = mutableListOf<SparsePointWire>()
    private val measurements = mutableListOf<ReferenceMeasurementWire>()

    /** Rejection counts, surfaced in the UI so a silently failing capture is visible. */
    val rejected = mutableMapOf<String, Int>()

    /**
     * The verdict on the most recent frame offered to the core, and how long
     * it has stood. Held separately from [rejected] because a running total
     * answers "is this capture working" while the user needs "what happened
     * just now, and what do I do about it" — a tally cannot say that.
     *
     * The timestamp is what lets the UI hold a verdict on screen long enough
     * to read: frames arrive at 30 Hz, and a label rewritten every 33 ms is
     * not a message, it is a flicker.
     */
    @Volatile var lastDecision: FfiDecision? = null
        private set
    @Volatile var lastDecisionAtMs: Long = 0L
        private set

    /** Frames the core turned into keyframes, against every frame it judged. */
    @Volatile var acceptedFrames: Int = 0
        private set
    @Volatile var judgedFrames: Int = 0
        private set

    /**
     * Shots the user reported as unobtainable, written into the bundle.
     *
     * The harness cannot otherwise tell a direction nobody could reach from
     * one the script never asked for, and those mean different things about
     * a room: the first is a fact about the furniture, the second a bug.
     */
    val skippedShots = mutableListOf<FfiSkippedShot>()

    /**
     * Give up on the shot being asked for and move to the next.
     *
     * Returns what was abandoned, or null if the script had already finished.
     */
    fun skipCurrentShot(): FfiSkippedShot? {
        val skipped = guidance?.skipShot() ?: return null
        skippedShots.add(skipped)
        return skipped
    }

    /**
     * Hand the core the current compass heading, degrees from magnetic north.
     *
     * Pushed rather than pulled because the sensor and the camera run on
     * different clocks; the core keeps the latest and stamps it onto whichever
     * frame is accepted next.
     */
    fun setHeading(degrees: Float) {
        guidance?.setHeading(degrees)
    }

    /** Where to go to get tracking back, or null before the first keyframe. */
    fun recovery(): uniffi.recon_core.FfiRecovery? = guidance?.recovery()

    /**
     * Take the picture the user just asked for.
     *
     * Writes the best frame of the last moment — which is a frame or two old
     * by the time the finger lands, and deliberately so: that removes shutter
     * lag and the smear that comes with tapping a screen.
     *
     * Always takes one. The core keeps the most recent frame as a candidate
     * whatever its quality, so there is nothing to report and nothing to
     * refuse.
     */
    fun takePicture() {
        guidance?.captureNow()
        writeHeldKeyframe()
    }

    /** Whether a frame is ready for the shutter, for the button's state. */
    fun readyToShoot(): Boolean = guidance?.hasCandidate() ?: false

    /**
     * The best frame of the current viewpoint, encoded and waiting.
     *
     * The core decides *which* frame to keep but never sees pixels, and an
     * ARCore image is recycled the instant the frame callback returns — so the
     * moment the core says "hold", the bitmap has to be made and kept. Exactly
     * one is retained at a time: a better frame of the same view replaces it,
     * and moving to a new view writes it out.
     */
    private data class HeldFrame(
        val bitmap: Bitmap,
        val pose: PoseWire,
        val matrix: FloatArray,
        val timestampNs: Long,
        val blur: Double,
        val clip: Double,
        val samples: DepthSampler.Samples?,
        val rotationDegrees: Int,
        val intrinsics: IntrinsicsWire,
        val shot: FfiShot?,
        val sparse: List<List<Double>>,
    )

    private var held: HeldFrame? = null

    private var lastOfferAtMs: Long = 0
    private var lastPose: FloatArray? = null
    private var lastKeyframeTimeNs: Long = 0
    /**
     * The previous *frame's* pose and time — not the previous keyframe's.
     *
     * Angular velocity has to be measured between consecutive frames. Measuring
     * it against the last keyframe reports the angle the user has turned since
     * that shot, which after a 90 degree turn to the next target reads ~1.57 rad
     * and never falls, so "hold still" could never be satisfied however still
     * the phone was. It also made the TOO_FAST nag fire on turning rather than
     * on speed.
     */
    private var prevFramePose: FloatArray? = null
    private var prevFrameTimeNs: Long = 0
    private lateinit var bundleDir: File
    private var recording = false

    /**
     * How tracking was last lost, or null while tracking.
     *
     * The *kind* matters, not just the fact. Only a relocalisation rebases
     * ARCore's world; `LIMITED` — too few features, too dark, moving too fast —
     * is a degradation that keeps the frame intact. Treating both as breaks
     * declared a capture unjoinable the first time the user pointed at a blank
     * ceiling, which the script asks for on every third shot.
     */
    private var lostAs: FfiTracking? = null

    /** Set when the next keyframe begins a new world frame. */
    private var tagNextKeyframe = false

    /**
     * Relocalisations seen this capture. Surfaced because a bundle with breaks
     * in it cannot be fused as one room, and the user is the only one who can
     * decide to walk the room again.
     */
    var trackingBreaks: Int = 0
        private set

    /**
     * Surface samples the last keyframe contributed. Zero here with a healthy
     * keyframe count means the depth pass is failing silently, which otherwise
     * looks identical to "the room is not covered yet".
     */
    @Volatile var lastDepthSamples: Int = 0
        private set

    private var depthFrameCount = 0
    private val floorSectors = mutableSetOf<Int>()
    private val wallSectors = mutableSetOf<Int>()
    private val ceilingSectors = mutableSetOf<Int>()

    val supportedSurfaceSectors: Triple<Int, Int, Int>
        get() = Triple(floorSectors.size, wallSectors.size, ceilingSectors.size)

    val hasUsefulBaseline: Boolean
        get() {
            var best2 = 0.0
            for (i in keyframes.indices) for (j in i + 1 until keyframes.size) {
                val a = keyframes[i].pose.t
                val b = keyframes[j].pose.t
                val dx = a[0] - b[0]
                val dz = a[2] - b[2]
                best2 = maxOf(best2, dx * dx + dz * dz)
            }
            return sqrt(best2) >= MIN_CAPTURE_BASELINE_M.toDouble()
        }

    val completionReady: Boolean
        get() {
            val sectorsReady = floorSectors.size >= MIN_FLOOR_SECTORS &&
                wallSectors.size >= MIN_WALL_SECTORS &&
                ceilingSectors.size >= MIN_CEILING_SECTORS
            val modeReady = if (depthAvailable) {
                lastGuidance?.shot?.state == FfiShotState.FINISHED &&
                    lastGuidance?.complete == true && depthFrameCount == keyframes.size
            } else {
                // The core cannot declare surface coverage without depth and
                // therefore keeps adding station passes. RGB-only deliberately
                // stops after the full centre+station visual script instead.
                keyframes.size >= MIN_RGB_ONLY_KEYFRAMES
            }
            return keyframes.size >= MIN_COMPLETION_KEYFRAMES &&
                hasUsefulBaseline && sectorsReady && modeReady && trackingBreaks == 0
        }

    val keyframeCount: Int get() = keyframes.size

    fun start() {
        bundleDir = File(outputRoot, "bundle_$sessionId").apply { mkdirs() }
        File(bundleDir, "images").mkdirs()
        keyframes.clear()
        sparsePoints.clear()
        rejected.clear()
        skippedShots.clear()
        held?.bitmap?.recycle()
        held = null
        lastDecision = null
        lastDecisionAtMs = 0
        acceptedFrames = 0
        judgedFrames = 0
        lastPose = null
        lastKeyframeTimeNs = 0
        prevFramePose = null
        prevFrameTimeNs = 0
        lostAs = null
        tagNextKeyframe = false
        trackingBreaks = 0
        depthFrameCount = 0
        floorSectors.clear()
        wallSectors.clear()
        ceilingSectors.clear()
        lastGuidance = null
        if (depthAvailable) {
            File(bundleDir, "depth").mkdirs()
            DepthPng.write(File(bundleDir, DEPTH_SELF_TEST_PATH), verifiedDepth!!)
        }
        // Gravity as the world-frame down vector, per plan §5. ARCore is
        // gravity-aligned, so this is -Y.
        guidance = FfiCaptureSession(FfiVec3(0.0f, -1.0f, 0.0f))
        // Scripted capture (aim, hold, shoot). The next frame's heading becomes
        // the script's yaw origin, so this must happen with the user already
        // standing where they mean to start.
        guidance?.startScript()
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
        val tracking = TrackingWatch.classify(
            tracking = camera.trackingState == TrackingState.TRACKING,
            badState = camera.trackingFailureReason == TrackingFailureReason.BAD_STATE ||
                camera.trackingFailureReason == TrackingFailureReason.CAMERA_UNAVAILABLE,
        )
        if (tracking != FfiTracking.NORMAL) {
            // Report it rather than drop it. The core rejects the frame anyway,
            // but only by *seeing* it can it raise TRACKING_LOST — and a capture
            // that says nothing while ARCore rebases the world under it is how
            // an unusable bundle gets carried home.
            reportLostTracking(frame, tracking)
            bump(if (tracking == FfiTracking.RELOCALIZING) "relocalizing" else "limited")
            // Keep the more serious kind: a relocalisation during an outage
            // taints it however it ends.
            if (lostAs != FfiTracking.RELOCALIZING) lostAs = tracking
            return
        }
        lostAs?.let { how ->
            // The baseline is reset either way — frames across any gap should
            // not be compared — but only a relocalisation means the world moved
            // under us, and only that makes the bundle unjoinable.
            guidance?.onRelocalised()
            if (how == FfiTracking.RELOCALIZING) noteBreak() else lastPose = null
            lostAs = null
        }

        val matrix = FloatArray(16)
        camera.pose.toMatrix(matrix, 0)

        // Updated before any early return below, or the "previous frame" drifts
        // to whenever the last *interesting* frame was and the velocity is wrong
        // again in a subtler way.
        val angularVelocity = prevFramePose?.let { prev ->
            val dt = (frame.timestamp - prevFrameTimeNs) / 1e9
            if (dt > 1e-4) (rotationAngle(prev, matrix) / dt).toFloat() else 0.0f
        } ?: 0.0f
        prevFramePose = matrix.copyOf()
        prevFrameTimeNs = frame.timestamp
        // Cheap pre-filter before decoding an image. The core makes the real
        // decision below; this only avoids the cost of a YUV copy for frames
        // that obviously cannot qualify.
        val pending = lastGuidance?.shot
        // No aim or baseline pre-filter any more: the candidate the shutter
        // takes must be a picture of what the user is looking at *now*, and a
        // filter that skips frames while they turn leaves a stale one in hand.
        //
        // A rate limit stays, because something has to stand between 60 fps and
        // a full-frame Laplacian plus a depth pass — without one the camera HAL
        // was starved into flushing its session and the app died mid-capture
        // with a tombstone in camera.qcom.so. This throttles on time alone, so
        // it never depends on where the phone is pointed.
        val nowMs = System.currentTimeMillis()
        if (nowMs - lastOfferAtMs < OFFER_INTERVAL_MS) return
        lastOfferAtMs = nowMs

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

            // The surface samples the coverage grid integrates. Taken here, on
            // the frame the core is about to judge, so the geometry and the pose
            // it is credited to are the same frame — sampling on a later one
            // smears coverage across the movement in between.
            val samples = if (depthAvailable) depth.sample(frame, camera) else null
            lastDepthSamples = (samples?.points?.size ?: 0) / 3
            // A depth-mode target cannot advance without depth support. This is
            // the failure that previously allowed 36 green targets with 0% wall
            // and ceiling coverage on the S23 Ultra.
            if (depthAvailable && lastDepthSamples < MIN_DEPTH_SAMPLES) {
                bump("depth_missing")
                return
            }

            // The core decides. Blur and exposure are measured here because
            // only the client has the pixels, but the thresholds and the
            // priority ordering live in one place for both platforms.
            val pose = FrameConversion.fromArCore(camera.pose, imageRotationDegrees)
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
                    // 0 means "unknown" to the core, which then skips the
                    // TOO_CLOSE nag rather than guessing.
                    medianDepthM = samples?.medianDepthM ?: 0.0f,
                    angularVelocityRadS = angularVelocity,
                    sparsePointCount = sparsePointsInView(frame),
                    lux = frame.lightEstimate?.pixelIntensity?.times(1000f) ?: 200.0f,
                ),
                samples?.points ?: emptyList(),
                samples?.normals ?: emptyList(),
            )
            lastGuidance = update

            if (update != null) {
                lastDecision = update.decision
                lastDecisionAtMs = System.currentTimeMillis()
                judgedFrames++
                if (update.decision == FfiDecision.ACCEPT) acceptedFrames++
                if (update.decision != FfiDecision.ACCEPT) {
                    bump(update.decision.name.removePrefix("REJECT_").lowercase())
                }
            }

            // Nothing is written here any more — the shutter does that. This
            // only keeps the best recent frame ready for it.
            if (update?.hold != true) return

            // This frame is the best of its viewpoint so far, so it has to be
            // encoded now — the ARCore image is recycled the moment this
            // function returns, and the decision to keep it comes later.
            val bitmap = toBitmap(image, targetWidth) ?: run { bump("encode"); return }
            held?.bitmap?.recycle()
            held = HeldFrame(
                bitmap = bitmap,
                pose = pose,
                matrix = matrix.copyOf(),
                timestampNs = frame.timestamp,
                blur = blur,
                clip = clip,
                samples = samples,
                rotationDegrees = imageRotationDegrees,
                intrinsics = FrameConversion.intrinsicsFor(
                    camera, bitmap.width, bitmap.height, imageRotationDegrees
                ),
                shot = pending,
                sparse = sparseSnapshot(frame),
            )
        } finally {
            image.close()
        }
    }

    /**
     * Write the retained candidate as a keyframe.
     *
     * Everything here was captured at the moment the frame was held, not now:
     * the camera has already moved on, and crediting this geometry to the
     * current pose is exactly the smearing the selector exists to avoid.
     */
    private fun writeHeldKeyframe() {
        val h = held ?: return
        held = null
        val id = "kf_%04d".format(keyframes.size)
        val rel = "images/$id.jpg"
        File(bundleDir, rel).outputStream().use {
            h.bitmap.compress(Bitmap.CompressFormat.JPEG, 92, it)
        }
        val depthRel = h.samples?.let {
            val path = "depth/$id.png"
            val rotatedWidth = if (h.rotationDegrees % 180 == 0) it.width else it.height
            val rotatedHeight = if (h.rotationDegrees % 180 == 0) it.height else it.width
            DepthPng.write(
                File(bundleDir, path), it, rotatedWidth, rotatedHeight, h.rotationDegrees
            )
            depthFrameCount += 1
            path
        }

        // A relocalisation ARCore never announced still shows up as a teleport
        // between consecutive keyframes, so the pose is checked as well as the
        // reported state.
        lastPose?.let { prev ->
            val dt = (h.timestampNs - lastKeyframeTimeNs) / 1e9
            if (TrackingWatch.isTeleport(translationDistance(prev, h.matrix), dt)) {
                noteBreak()
            }
        }

        keyframes += KeyframeWire(
            id = id,
            image = rel,
            timestampNs = h.timestampNs,
            pose = h.pose,
            intrinsics = h.intrinsics,
            trackingState = if (tagNextKeyframe) "relocalizing" else "normal",
            blurScore = h.blur,
            exposureClipRatio = h.clip,
            depth = depthRel,
        )
        h.shot?.let { recordSurfaceSector(it, h.samples != null) }
        tagNextKeyframe = false
        lastPose = h.matrix
        lastKeyframeTimeNs = h.timestampNs
        for (xyz in h.sparse) {
            sparsePoints += SparsePointWire(xyz = xyz, observations = listOf(id))
        }
        h.bitmap.recycle()
    }

    /** Sparse points visible now, sampled while the frame is still alive. */
    private fun sparseSnapshot(frame: Frame): List<List<Double>> {
        val out = mutableListOf<List<Double>>()
        try {
            frame.acquirePointCloud().use { cloud ->
                val buf = cloud.points
                val n = buf.remaining() / 4
                val stride = max(1, n / 20)
                var i = 0
                while (i < n) {
                    val base = i * 4
                    // The 4th component is a confidence, not a coordinate.
                    if (buf.get(base + 3) > 0.3f) {
                        out += listOf(
                            buf.get(base).toDouble(),
                            buf.get(base + 1).toDouble(),
                            buf.get(base + 2).toDouble(),
                        )
                    }
                    i += stride
                }
            }
        } catch (e: Exception) {
            // A point cloud is evidence, not a requirement.
        }
        return out
    }

    /**
     * ARCore's floor plane, which the core prefers over inferring the floor from
     * the lowest up-facing surface it has seen — early in a capture that is
     * usually a tabletop, and a station ring floating at table height is worse
     * than none.
     */
    fun setFloorHeight(y: Float) {
        guidance?.setFloorHeight(y)
    }

    /**
     * The coverage grid as drawable points, or null before recording starts.
     *
     * `target` is a drawing budget the core honours approximately; it subsamples
     * by a hash of the voxel key so the same voxels are kept from call to call
     * and the overlay does not boil. Callers should throttle: this walks the
     * whole grid, which is thousands of voxels by the end of a room.
     */
    fun coverageCloud(target: Int): FfiCoverageCloud? =
        guidance?.coverageCloud(target.toUInt())

    /** Where the core wants the user to stand next, if anywhere. */
    val station: FfiStation? get() = lastGuidance?.station

    /** The shot the script is asking for, if a script is running. */
    val shot: FfiShot? get() = lastGuidance?.shot

    /**
     * Push a frame the core will reject, purely so it can nag.
     *
     * No image work: the pose is not trustworthy and nothing here can become a
     * keyframe, so decoding a YUV plane to score blur would be wasted on every
     * frame of an outage. The core's gate rejects on tracking before it reads
     * the pose, and its nag only needs the tracking state, the light and the
     * feature count.
     */
    private fun reportLostTracking(frame: Frame, tracking: FfiTracking) {
        val pose = try {
            FrameConversion.fromArCore(frame.camera.pose)
        } catch (e: FrameConversionException) {
            // A lost camera can report a degenerate pose. It is never used —
            // the frame is rejected — but it still has to satisfy the FFI.
            IDENTITY_POSE
        }
        try {
            lastGuidance = guidance?.observe(
                FfiFrame(
                    timestampS = frame.timestamp / 1e9,
                    pose = FfiPose(
                        r = pose.R.flatten().map { it.toFloat() },
                        t = FfiVec3(
                            pose.t[0].toFloat(), pose.t[1].toFloat(), pose.t[2].toFloat()
                        ),
                    ),
                    tracking = tracking,
                    // Zeros mean "unmeasured" to the core, which skips the nags
                    // that depend on them rather than inventing a reading.
                    blurScore = 0.0f,
                    exposureClipRatio = 0.0f,
                    medianDepthM = 0.0f,
                    angularVelocityRadS = 0.0f,
                    sparsePointCount = sparsePointsInView(frame),
                    lux = frame.lightEstimate?.pixelIntensity?.times(1000f) ?: 200.0f,
                ),
                emptyList(),
                emptyList(),
            )
        } catch (e: Exception) {
            // Guidance is not worth failing a capture over.
        }
    }

    /**
     * Record that the world frame may have changed here.
     *
     * The next keyframe is tagged `relocalizing` in the bundle. It was captured
     * while tracking normally, so the tag is not about *that* frame's quality —
     * it marks the discontinuity, which is the thing a reconstruction has to
     * split on. Without it the break is only recoverable by guessing from pose
     * jumps after the fact.
     */
    private fun noteBreak() {
        trackingBreaks += 1
        tagNextKeyframe = true
        lastPose = null // the previous keyframe is in a different world frame
        // The core discards its held candidate across a relocalisation, since
        // its pose is expressed in a frame that no longer exists. The pixels
        // have to go with it or they would be written against the next commit.
        held?.bitmap?.recycle()
        held = null
    }

    /**
     * A cheap pose-only test for "worth decoding an image over".
     *
     * Deliberately **looser** than the core's own aim and steadiness tests, and
     * deliberately not a copy of them: this may only discard frames the core
     * would certainly reject anyway. The core stays the one that decides what a
     * shot is — this just declines to do a frame's worth of image work for a
     * camera pointed at the opposite wall.
     */
    private fun roughlyOnTarget(matrix: FloatArray, shot: FfiShot, angularVelocity: Float): Boolean {
        if (angularVelocity > LOOSE_STEADY_RAD_S) return false
        // Column-major; the camera looks down -Z (plan §5).
        val fx = -matrix[8]
        val fy = -matrix[9]
        val fz = -matrix[10]
        val dot = fx * shot.direction.x + fy * shot.direction.y + fz * shot.direction.z
        return dot >= LOOSE_AIM_COS
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
        // The last viewpoint is still held, waiting for a move that will never
        // come now. Without this every capture silently loses its final frame.
        if (guidance?.flush() == true) {
            writeHeldKeyframe()
        }
        held?.bitmap?.recycle()
        held = null
        guidance?.destroy()
        guidance = null
        val bundle = SessionBundleWire(
            sessionId = sessionId,
            device = DeviceInfo(
                model = "${Build.MANUFACTURER} ${Build.MODEL}",
                os = "Android ${Build.VERSION.RELEASE} (API ${Build.VERSION.SDK_INT})",
                arPlatform = "arcore",
                arVersion = arCoreVersion,
            ),
            roomId = roomId,
            gravity = FrameConversion.GRAVITY_DOWN,
            captureMode = if (depthAvailable) "arcore_depth" else "rgb_only",
            depthSelfTest = if (depthAvailable) DEPTH_SELF_TEST_PATH else null,
            depthFrameCount = depthFrameCount,
            keyframes = keyframes.toList(),
            sparsePoints = sparsePoints.toList(),
            // The VIO scale prior is always present: it is the anchor that stops
            // one bad tape entry from rescaling the room (plan §6.2).
            referenceMeasurements = listOf(
                ReferenceMeasurementWire(
                    type = "vio_scale", measuredValueM = 1.0, sigmaM = 0.03,
                )
            ) + measurements,
            skippedShots = skippedShots.map {
                SkippedShotWire(
                    yawDeg = it.yawDeg.toDouble(),
                    height = it.height.name.lowercase(),
                    atStation = it.atStation,
                )
            },
        )
        val json = Json { prettyPrint = true; encodeDefaults = true }
        File(bundleDir, "bundle.json").writeText(json.encodeToString(bundle))
        return bundleDir
    }

    private fun recordSurfaceSector(shot: FfiShot, supported: Boolean) {
        // In depth mode "supported" means a real depth return existed for the
        // shot. RGB-only has separate criteria and records guided visual sectors.
        if (depthAvailable && !supported) return
        val yaw = kotlin.math.atan2(shot.direction.x, -shot.direction.z)
        val sector = (((yaw + Math.PI).toFloat() / (2f * Math.PI.toFloat()) * YAW_SECTORS)
            .toInt()).floorMod(YAW_SECTORS)
        when (shot.height) {
            uniffi.recon_core.FfiShotHeight.BOTTOM -> floorSectors += sector
            uniffi.recon_core.FfiShotHeight.MIDDLE -> wallSectors += sector
            uniffi.recon_core.FfiShotHeight.TOP -> ceilingSectors += sector
        }
    }

    private fun Int.floorMod(divisor: Int): Int = ((this % divisor) + divisor) % divisor

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
        val sensor = BitmapFactory.decodeByteArray(out.toByteArray(), 0, out.size()) ?: return null
        val upright = if (imageRotationDegrees == 0) {
            sensor
        } else {
            Bitmap.createBitmap(
                sensor,
                0,
                0,
                sensor.width,
                sensor.height,
                Matrix().apply { postRotate(imageRotationDegrees.toFloat()) },
                true,
            )
        }
        if (upright.width <= targetWidth) return upright
        val h = (upright.height.toLong() * targetWidth / upright.width).toInt()
        return Bitmap.createScaledBitmap(upright, targetWidth, h, true)
    }

    private fun yuvBytes(image: Image): ByteArray {
        require(image.planes.size == 3) { "YUV_420_888 image must have three planes" }
        val width = image.width
        val height = image.height
        val chromaWidth = (width + 1) / 2
        val chromaHeight = (height + 1) / 2
        val out = ByteArray(width * height + 2 * chromaWidth * chromaHeight)

        fun sample(planeIndex: Int, row: Int, col: Int): Byte {
            val plane = image.planes[planeIndex]
            val buffer = plane.buffer
            val offset = buffer.position() + row * plane.rowStride + col * plane.pixelStride
            require(offset < buffer.limit()) {
                "YUV plane $planeIndex is truncated at row=$row col=$col"
            }
            return buffer.get(offset)
        }

        var dst = 0
        for (row in 0 until height) {
            for (col in 0 until width) {
                out[dst++] = sample(0, row, col)
            }
        }
        // YuvImage expects NV21: one tightly packed Y plane followed by
        // interleaved VU chroma, irrespective of the source plane strides.
        for (row in 0 until chromaHeight) {
            for (col in 0 until chromaWidth) {
                out[dst++] = sample(2, row, col)
                out[dst++] = sample(1, row, col)
            }
        }
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

    private companion object {
        /** Frame budget for the scoring pass: ~10 Hz, well inside the HAL's tolerance. */
        const val OFFER_INTERVAL_MS = 100L
        const val DEPTH_SELF_TEST_PATH = "depth/self_test.png"
        const val MIN_DEPTH_SAMPLES = 100
        const val MIN_COMPLETION_KEYFRAMES = 24
        const val MIN_RGB_ONLY_KEYFRAMES = 36
        const val MIN_CAPTURE_BASELINE_M = 0.65f
        const val YAW_SECTORS = 12
        const val MIN_FLOOR_SECTORS = 4
        const val MIN_WALL_SECTORS = 8
        const val MIN_CEILING_SECTORS = 4
        /** ~35 deg, against the core's ~17: loose enough never to pre-empt it. */
        const val LOOSE_AIM_COS = 0.82f
        /** ~3x the core's steadiness bar, for the same reason. */
        const val LOOSE_STEADY_RAD_S = 1.0f

        /** Stand-in for a pose the tracker could not give us. Never integrated. */
        val IDENTITY_POSE = PoseWire(
            R = listOf(
                listOf(1.0, 0.0, 0.0),
                listOf(0.0, 1.0, 0.0),
                listOf(0.0, 0.0, 1.0),
            ),
            t = listOf(0.0, 0.0, 0.0),
        )
    }
}
