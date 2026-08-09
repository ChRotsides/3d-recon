package com.recon.poselogger

import android.Manifest
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraManager
import android.content.pm.PackageManager
import android.opengl.GLSurfaceView
import android.opengl.Matrix
import android.os.Bundle
import android.os.SystemClock
import android.view.Surface
import android.view.View
import android.widget.Button
import android.widget.Toast
import android.util.Log
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.google.ar.core.ArCoreApk
import com.google.ar.core.Config
import com.google.ar.core.Plane
import com.google.ar.core.Session
import com.google.ar.core.TrackingState
import com.google.ar.core.exceptions.CameraNotAvailableException
import com.google.ar.core.exceptions.UnavailableException
import uniffi.recon_core.decisionAccepted
import uniffi.recon_core.decisionMessage
import uniffi.recon_core.defaultParams
import uniffi.recon_core.nagMessage
import uniffi.recon_core.recoveryMessage
import java.io.File
import javax.microedition.khronos.egl.EGLConfig
import javax.microedition.khronos.opengles.GL10
import kotlin.math.abs
import kotlin.math.sqrt

/**
 * The capture screen: camera preview, the shared core's guidance drawn over it,
 * and one button.
 *
 * Guidance is deliberately shown two ways, because they answer different
 * questions. [CoverageRenderer] paints the core's coverage grid onto the real
 * surfaces — *which wall still needs work* — and [GuidanceHud] reports the
 * numbers and the nag — *how close am I to done*. [StationRenderer] draws the
 * §6.1 floor ring for where to stand next, and the HUD grows an arrow when that
 * ring is behind the user.
 *
 * All of it reads one `FfiGuidance` per frame from the core (plan §3.8): the
 * Activity decides how to draw, never what is true.
 */
class MainActivity : AppCompatActivity(), GLSurfaceView.Renderer {

    private enum class DepthReadiness { CHECKING, READY, UNSUPPORTED, FAILED }

    private lateinit var surfaceView: GLSurfaceView
    private lateinit var hud: GuidanceHud
    private lateinit var actionButton: Button
    private lateinit var skipButton: Button
    private lateinit var shutterButton: Button

    /**
     * Heading, used only to say which way to turn after tracking is lost.
     * Never fed to the reconstruction — see [Compass].
     */
    private val compass by lazy { Compass(this) }

    private var session: Session? = null
    private var recorder: CaptureRecorder? = null
    private val background = BackgroundRenderer()
    private val coverage = CoverageRenderer()
    private val stationRing = StationRenderer()
    private val planes = PlaneRenderer()
    private val featurePoints = FeaturePointRenderer()
    private val shotTarget = ShotTargetRenderer()
    private var stationAnchor: com.google.ar.core.Anchor? = null
    private var sessionConfigured = false
    private var depthSupported = false
    private val depthProbe = DepthSampler()
    @Volatile private var depthReadiness = DepthReadiness.CHECKING
    @Volatile private var verifiedDepth: DepthSampler.Samples? = null
    private var depthProbeStartedMs = 0L
    private var depthProbeLastAtMs = 0L
    private var depthProbeSuccesses = 0
    private var depthProbeFirstSuccessMs = 0L
    private var depthFailure: String? = null

    /**
     * How far the phone has travelled since the self-test began.
     *
     * The test is not really "does this device support depth" — it is "can
     * ARCore produce depth *here, now*". On a phone with `depthSensor=DO_NOT_USE`
     * that is motion stereo, which derives disparity from the camera moving
     * between frames. A stationary phone has no baseline, so it produces no
     * depth however long it is given, and timing out against a stopwatch tests
     * the operator's fidgeting rather than the device.
     */
    private var depthProbeMovedM = 0f
    private var depthProbeLastPos: FloatArray? = null
    private var arCoreDiagnostics = "ARCore SDK 1.49.0"

    // Reused rather than allocated per frame: this runs at 60 Hz, and a GC pause
    // in the render loop is visible as a stutter in the preview.
    private val viewMatrix = FloatArray(16)
    private val projMatrix = FloatArray(16)
    private val viewProjection = FloatArray(16)
    private val worldPoint = FloatArray(4)
    private val clipPoint = FloatArray(4)
    private val cameraPos = FloatArray(3)
    private var viewportHeight = 1

    private var lastCoverageAtMs = 0L
    private var lastHudAtMs = 0L
    private var mappedVoxels = 0
    private var planeCount = 0
    private var featurePointCount = 0
    private var floorHeight: Float? = null
    /** When the core first said the room was complete, or 0 while it is not. */
    private var completeSinceMs = 0L

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        surfaceView = findViewById(R.id.surface)
        hud = findViewById(R.id.hud)
        actionButton = findViewById(R.id.action)
        skipButton = findViewById(R.id.skip)
        shutterButton = findViewById(R.id.shutter)

        surfaceView.preserveEGLContextOnPause = true
        surfaceView.setEGLContextClientVersion(3)
        surfaceView.setEGLConfigChooser(8, 8, 8, 8, 16, 0)
        surfaceView.setRenderer(this)
        surfaceView.renderMode = GLSurfaceView.RENDERMODE_CONTINUOUSLY

        coverage.voxelM = defaultParams().voxelM

        actionButton.setOnClickListener { toggleRecording() }
        skipButton.setOnClickListener { skipCurrentShot() }
        shutterButton.setOnClickListener { takePicture() }
    }

    /**
     * The user has lined the marker up and pressed the shutter.
     *
     * The app takes no pictures on its own, and nothing here second-guesses
     * the press. There is no failure path and no message: a shutter that
     * argues is worse than a picture that needs work later, and the pipeline
     * has techniques for a poor frame but none for a frame never taken.
     */
    private fun takePicture() {
        recorder?.takePicture()
    }

    /**
     * The user says the shot being asked for cannot be got.
     *
     * Taken at face value. The obstruction — a wardrobe, a wall too close to
     * back away from, a window onto someone else's garden — is precisely the
     * thing the device cannot see, so there is nothing to verify it against.
     * The alternative is a user stuck on a shot that will never be satisfied,
     * who abandons the whole capture instead of one direction of it.
     */
    private fun skipCurrentShot() {
        val skipped = recorder?.skipCurrentShot()
        val message = if (skipped == null) {
            getString(R.string.skip_nothing_left)
        } else {
            getString(R.string.skipped_shot)
        }
        Toast.makeText(this, message, Toast.LENGTH_SHORT).show()
    }

    // -- lifecycle ----------------------------------------------------------

    override fun onResume() {
        super.onResume()
        if (!hasCameraPermission()) {
            ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.CAMERA), 1)
            return
        }
        if (session == null && !createSession()) return
        try {
            session?.resume()
        } catch (e: CameraNotAvailableException) {
            setStatus("camera unavailable: ${e.message}")
            session = null
            return
        }
        surfaceView.onResume()
        compass.displayRotation = surfaceView.display?.rotation ?: Surface.ROTATION_0
        compass.start()
    }

    override fun onPause() {
        super.onPause()
        surfaceView.onPause()
        session?.pause()
        compass.stop()
    }

    override fun onRequestPermissionsResult(
        requestCode: Int, permissions: Array<out String>, grantResults: IntArray,
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (!hasCameraPermission()) {
            Toast.makeText(this, "Camera access is required to capture", Toast.LENGTH_LONG).show()
            finish()
        }
    }

    private fun hasCameraPermission() =
        ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) ==
            PackageManager.PERMISSION_GRANTED

    private fun createSession(): Boolean {
        try {
            when (ArCoreApk.getInstance().requestInstall(this, true)) {
                ArCoreApk.InstallStatus.INSTALL_REQUESTED -> return false
                ArCoreApk.InstallStatus.INSTALLED -> Unit
            }
            val s = Session(this)
            // Depth is what feeds the coverage grid. Not every ARCore device has
            // it, and asking for an unsupported mode throws, so it is a query
            // rather than an assumption — without it the capture still works and
            // the HUD says coverage is unmeasurable instead of showing zeros.
            depthSupported = DepthSampler.supportedBy(s)
            s.configure(
                Config(s).apply {
                    // Plan §5: the bundle frame assumes a gravity-aligned world.
                    // ARCore is gravity-aligned by default; the plane finding is
                    // what gives us the floor height for free.
                    planeFindingMode = Config.PlaneFindingMode.HORIZONTAL_AND_VERTICAL
                    focusMode = Config.FocusMode.AUTO
                    lightEstimationMode = Config.LightEstimationMode.AMBIENT_INTENSITY
                    updateMode = Config.UpdateMode.LATEST_CAMERA_IMAGE
                    if (depthSupported) depthMode = Config.DepthMode.AUTOMATIC
                }
            )
            val camera = s.cameraConfig
            val runtime = try {
                packageManager.getPackageInfo("com.google.ar.core", 0).versionName ?: "unknown"
            } catch (_: Exception) {
                "unknown"
            }
            arCoreDiagnostics = buildString {
                append("SDK 1.49.0 runtime ").append(runtime)
                append(" camera=").append(camera.cameraId)
                append(" cpu=").append(camera.imageSize.width).append('x').append(camera.imageSize.height)
                append(" texture=").append(camera.textureSize.width).append('x').append(camera.textureSize.height)
                append(" fps=").append(camera.fpsRange)
                append(" depthSensor=").append(camera.depthSensorUsage)
            }
            Log.i("PoseLoggerDepth", arCoreDiagnostics)
            depthReadiness = if (depthSupported) DepthReadiness.CHECKING else DepthReadiness.UNSUPPORTED
            verifiedDepth = null
            depthProbeStartedMs = 0L
            depthProbeSuccesses = 0
            depthProbeFirstSuccessMs = 0L
            depthFailure = null
            if (!depthSupported) actionButton.text = getString(R.string.start_rgb_only)
            session = s
            return true
        } catch (e: UnavailableException) {
            setStatus("ARCore unavailable: ${e.message}")
            return false
        }
    }

    // -- recording ----------------------------------------------------------

    private fun toggleRecording() {
        val s = session ?: return
        val rec = recorder
        if (rec == null) {
            when (depthReadiness) {
                DepthReadiness.READY -> startRecording(s, useDepth = true)
                DepthReadiness.CHECKING -> Toast.makeText(
                    this, "Waiting for a real ARCore depth frame", Toast.LENGTH_SHORT
                ).show()
                DepthReadiness.UNSUPPORTED, DepthReadiness.FAILED -> confirmRgbOnly(s)
            }
        } else {
            val dir = rec.stop()
            recorder = null
            stationAnchor?.detach()
            stationAnchor = null
            surfaceView.queueEvent { coverage.clear() }
            actionButton.text = getString(R.string.start)
            val shots = rec.lastGuidance?.shot
            val why = when {
                rec.completionReady && !rec.depthAvailable -> "RGB-only criteria complete"
                shots?.state == uniffi.recon_core.FfiShotState.FINISHED ->
                    "all ${shots.total} shots taken"
                rec.lastGuidance?.complete == true -> "room complete"
                else -> "stopped"
            }
            Toast.makeText(
                this, "$why — ${rec.keyframeCount} keyframes -> $dir", Toast.LENGTH_LONG
            ).show()
        }
    }

    private fun startRecording(s: Session, useDepth: Boolean) {
        val r = CaptureRecorder(
            s,
            File(getExternalFilesDir(null), "captures"),
            depthAvailable = useDepth,
            verifiedDepth = if (useDepth) verifiedDepth else null,
            arCoreVersion = arCoreDiagnostics,
            imageRotationDegrees = cameraImageRotationDegrees(s),
        )
        r.start()
        floorHeight?.let { r.setFloorHeight(it) }
        recorder = r
        actionButton.text = getString(R.string.stop)
    }

    private fun confirmRgbOnly(s: Session) {
        val why = depthFailure ?: if (depthSupported) {
            "ARCore did not return a depth frame"
        } else {
            "This camera configuration does not support ARCore depth"
        }
        val builder = AlertDialog.Builder(this)
            .setTitle("Use RGB-only capture?")
            .setMessage(
                "$why. Coverage cannot be measured and the bundle will be marked RGB-only. " +
                    "Use this only for the separate RGB-only evaluation path."
            )
            .setNegativeButton("Cancel", null)
            .setPositiveButton("Start RGB-only") { _, _ -> startRecording(s, useDepth = false) }
        // Offered before the RGB-only button is reached for, because a failed
        // test usually means the room or the standing still, not the phone —
        // and RGB-only is a decision that costs the capture its coverage.
        if (depthSupported) {
            builder.setNeutralButton("Retry depth test") { _, _ -> retryDepthSelfTest() }
        }
        builder.show()
    }

    /**
     * Clockwise rotation from ARCore's unrotated CPU image into the portrait
     * display orientation. ARCore's camera id is a Camera2 id, so the standard
     * back-camera `sensorOrientation - displayRotation` rule applies.
     */
    private fun cameraImageRotationDegrees(session: Session): Int {
        val manager = getSystemService(CameraManager::class.java)
        val cameraId = session.cameraConfig.cameraId
        val characteristics = manager.getCameraCharacteristics(cameraId)
        val sensor = characteristics.get(CameraCharacteristics.SENSOR_ORIENTATION) ?: 0
        val displayDegrees = when (surfaceView.display?.rotation ?: Surface.ROTATION_0) {
            Surface.ROTATION_0 -> 0
            Surface.ROTATION_90 -> 90
            Surface.ROTATION_180 -> 180
            Surface.ROTATION_270 -> 270
            else -> 0
        }
        val facing = characteristics.get(CameraCharacteristics.LENS_FACING)
        return if (facing == CameraCharacteristics.LENS_FACING_FRONT) {
            (sensor + displayDegrees) % 360
        } else {
            (sensor - displayDegrees + 360) % 360
        }
    }

    // -- GL -----------------------------------------------------------------

    override fun onSurfaceCreated(gl: GL10?, config: EGLConfig?) {
        background.createOnGlThread()
        coverage.createOnGlThread()
        stationRing.createOnGlThread()
        planes.createOnGlThread()
        featurePoints.createOnGlThread()
        shotTarget.createOnGlThread()
    }

    override fun onSurfaceChanged(gl: GL10?, width: Int, height: Int) {
        android.opengl.GLES30.glViewport(0, 0, width, height)
        viewportHeight = height
        session?.setDisplayGeometry(display?.rotation ?: 0, width, height)
    }

    override fun onDrawFrame(gl: GL10?) {
        android.opengl.GLES30.glClear(
            android.opengl.GLES30.GL_COLOR_BUFFER_BIT or
                android.opengl.GLES30.GL_DEPTH_BUFFER_BIT
        )
        val s = session ?: return
        if (!sessionConfigured) {
            s.setCameraTextureName(background.textureId)
            sessionConfigured = true
        }
        try {
            val frame = s.update()
            background.draw(frame)

            val camera = frame.camera
            val tracking = camera.trackingState
            val rec = recorder

            if (rec == null) updateDepthSelfTest(frame, camera, tracking)
            rec?.onFrame(frame)
            rec?.let { stopIfComplete(it) }
            updateFloorHeight(s, rec)

            if (tracking == TrackingState.TRACKING) {
                camera.getViewMatrix(viewMatrix, 0)
                camera.getProjectionMatrix(projMatrix, 0, NEAR_M, FAR_M)
                Matrix.multiplyMM(viewProjection, 0, projMatrix, 0, viewMatrix, 0)
                // Before the coverage paint, so the paint reads on top of them,
                // and outside the `rec != null` check because deciding whether a
                // room is trackable happens *before* pressing record.
                val tracked = s.getAllTrackables(Plane::class.java)
                planeCount = tracked.count { it.trackingState == TrackingState.TRACKING }
                planes.draw(tracked, viewProjection)
                frame.acquirePointCloud().use {
                    featurePointCount = it.points.remaining() / 4
                    featurePoints.draw(it, viewProjection)
                }
                if (rec != null) drawGuidance(rec)
            }

            cameraPos[0] = camera.pose.tx()
            cameraPos[1] = camera.pose.ty()
            cameraPos[2] = camera.pose.tz()
            publishHud(rec, tracking, cameraPos)
        } catch (e: CameraNotAvailableException) {
            setStatus("camera lost: ${e.message}")
        } catch (e: Throwable) {
            setStatus("error: ${e.message}")
        }
    }

    private fun updateDepthSelfTest(
        frame: com.google.ar.core.Frame,
        camera: com.google.ar.core.Camera,
        tracking: TrackingState,
    ) {
        if (depthReadiness != DepthReadiness.CHECKING || tracking != TrackingState.TRACKING) return
        val now = SystemClock.uptimeMillis()
        if (depthProbeStartedMs == 0L) depthProbeStartedMs = now
        if (now - depthProbeLastAtMs < DEPTH_PROBE_INTERVAL_MS) return
        depthProbeLastAtMs = now

        // Path length, not displacement: a user who sways back to where they
        // started has still given motion stereo the baseline it needs, and
        // displacement would score that as standing still.
        val pos = floatArrayOf(camera.pose.tx(), camera.pose.ty(), camera.pose.tz())
        depthProbeLastPos?.let { prev ->
            val dx = pos[0] - prev[0]
            val dy = pos[1] - prev[1]
            val dz = pos[2] - prev[2]
            depthProbeMovedM += sqrt(dx * dx + dy * dy + dz * dz)
        }
        depthProbeLastPos = pos
        when (val result = depthProbe.probe(frame, camera)) {
            is DepthSampler.Probe.Ready -> {
                if (result.samples.points.size / 3 >= MIN_SELF_TEST_DEPTH_SAMPLES) {
                    verifiedDepth = result.samples
                    depthProbeSuccesses += 1
                    if (depthProbeFirstSuccessMs == 0L) depthProbeFirstSuccessMs = now
                    Log.i(
                        "PoseLoggerDepth",
                        "self-test depth frame $depthProbeSuccesses/$MIN_SELF_TEST_DEPTH_FRAMES " +
                            "${result.samples.width}x${result.samples.height} " +
                            "samples=${result.samples.points.size / 3}",
                    )
                    if (depthProbeSuccesses >= MIN_SELF_TEST_DEPTH_FRAMES &&
                        now - depthProbeFirstSuccessMs >= MIN_SELF_TEST_SPAN_MS
                    ) {
                        depthReadiness = DepthReadiness.READY
                        depthFailure = null
                        Log.i("PoseLoggerDepth", "self-test passed; $arCoreDiagnostics")
                        runOnUiThread { actionButton.text = getString(R.string.start) }
                    }
                }
            }
            is DepthSampler.Probe.Failed -> {
                depthFailure = result.reason
                Log.w("PoseLoggerDepth", "depth probe failed: ${result.reason}; $arCoreDiagnostics")
            }
            DepthSampler.Probe.Waiting -> Unit
        }
        if (depthReadiness != DepthReadiness.CHECKING) return

        // Only give up once the device has been given a fair chance: enough
        // time *and* enough movement to have produced a stereo baseline.
        // Failing a stationary phone at ten seconds condemns the whole session
        // to RGB-only over a test that was never actually run — which is what
        // happened twice on an S23 Ultra that had captured depth fine an hour
        // earlier, in the same room.
        val elapsed = now - depthProbeStartedMs
        val moved = depthProbeMovedM >= MIN_SELF_TEST_MOTION_M
        val verdict = when {
            elapsed >= DEPTH_SELF_TEST_TIMEOUT_MS && moved ->
                "no depth after ${elapsed / 1000}s and ${"%.2f".format(depthProbeMovedM)} m " +
                    "of movement; this device cannot supply depth here"
            // The stopwatch alone must never fail the test, but it cannot run
            // forever either: past the cap the honest answer is that the test
            // was inconclusive, and it says so rather than blaming the device.
            elapsed >= DEPTH_SELF_TEST_CAP_MS ->
                "no depth after ${elapsed / 1000}s, but the phone moved only " +
                    "${"%.2f".format(depthProbeMovedM)} m — depth needs the phone " +
                    "carried around, not held still"
            else -> null
        } ?: return

        depthReadiness = DepthReadiness.FAILED
        depthFailure = depthFailure ?: verdict
        Log.e("PoseLoggerDepth", "self-test failed: $depthFailure; $arCoreDiagnostics")
        runOnUiThread { actionButton.text = getString(R.string.start_rgb_only) }
    }

    /**
     * Run the depth self-test again.
     *
     * A failure is about the moment, not the hardware: the phone was held
     * still, or aimed at a blank wall. Making that verdict permanent for the
     * session means the fix — walk a few steps and try again — is unavailable
     * precisely when the user has worked out what to do.
     */
    private fun retryDepthSelfTest() {
        if (!depthSupported || recorder != null) return
        depthReadiness = DepthReadiness.CHECKING
        depthProbeStartedMs = 0L
        depthProbeLastAtMs = 0L
        depthProbeSuccesses = 0
        depthProbeFirstSuccessMs = 0L
        depthProbeMovedM = 0f
        depthProbeLastPos = null
        depthFailure = null
        verifiedDepth = null
        runOnUiThread { actionButton.text = getString(R.string.starting) }
        Toast.makeText(this, R.string.depth_retry, Toast.LENGTH_SHORT).show()
    }

    /**
     * The coverage snapshot is refreshed on a timer rather than per frame: it
     * walks the whole grid and crosses the FFI boundary, and at 60 Hz that is
     * both wasted work and a visibly stuttering preview. Three times a second is
     * faster than a user can walk into new geometry.
     */
    private fun drawGuidance(rec: CaptureRecorder) {
        val now = SystemClock.uptimeMillis()
        if (now - lastCoverageAtMs >= COVERAGE_INTERVAL_MS) {
            rec.coverageCloud(COVERAGE_BUDGET)?.let {
                coverage.update(it.points, it.fill)
                mappedVoxels = it.total.toInt()
            }
            lastCoverageAtMs = now
        }
        coverage.draw(viewProjection, projMatrix, viewportHeight)

        rec.shot?.let { shot ->
            worldPoint[0] = shot.direction.x
            worldPoint[1] = shot.direction.y
            worldPoint[2] = shot.direction.z
            val state = when (shot.state) {
                uniffi.recon_core.FfiShotState.TAKE -> 2
                uniffi.recon_core.FfiShotState.NOT_STEADY -> 1
                else -> 0
            }
            shotTarget.draw(
                viewProjection, worldPoint, cameraPos, UP_AXIS, state,
                (now % PULSE_MS).toFloat() / PULSE_MS,
            )
        }

        rec.station?.let { station ->
            // Drawn at the *anchor's* pose, not the raw solver output. An anchor
            // is ARCore's promise to keep a point attached to the same physical
            // spot across a relocalisation — and this capture relocalises. An
            // unanchored ring silently slides to a different part of the room
            // when that happens, sending the user to the wrong place.
            val p = anchorFor(station) ?: station.position.let {
                worldPoint[0] = it.x; worldPoint[1] = it.y; worldPoint[2] = it.z; worldPoint
            }
            worldPoint[3] = 1f
            val phase = (now % PULSE_MS).toFloat() / PULSE_MS
            stationRing.draw(viewProjection, p, phase)
            stationRing.drawArrow(
                viewProjection, cameraPos, p, floorHeight ?: p[1],
                (now % ARROW_MS).toFloat() / ARROW_MS,
            )
        }
    }

    /**
     * An ARCore anchor at the suggested station, recreated only when the solver
     * moves it meaningfully. Anchors cost tracking work, so one that is
     * re-made every frame is worse than none.
     */
    private fun anchorFor(station: uniffi.recon_core.FfiStation): FloatArray? {
        val s = session ?: return null
        val moved = stationAnchor?.let {
            val p = it.pose
            val dx = p.tx() - station.position.x
            val dy = p.ty() - station.position.y
            val dz = p.tz() - station.position.z
            sqrt(dx * dx + dy * dy + dz * dz) > STATION_MOVE_M
        } ?: true
        if (moved) {
            stationAnchor?.detach()
            stationAnchor = try {
                s.createAnchor(
                    com.google.ar.core.Pose.makeTranslation(
                        station.position.x, station.position.y, station.position.z
                    )
                )
            } catch (e: Exception) {
                null
            }
        }
        val a = stationAnchor ?: return null
        if (a.trackingState != TrackingState.TRACKING) return null
        worldPoint[0] = a.pose.tx()
        worldPoint[1] = a.pose.ty()
        worldPoint[2] = a.pose.tz()
        return worldPoint
    }

    /**
     * ARCore's own floor plane, handed to the core so the station solver puts
     * the ring on the floor rather than on the lowest thing it has seen — which
     * early in a capture is usually a tabletop.
     */
    private fun updateFloorHeight(s: Session, rec: CaptureRecorder?) {
        if (floorHeight != null && rec == null) return
        var lowest: Float? = null
        for (plane in s.getAllTrackables(Plane::class.java)) {
            if (plane.trackingState != TrackingState.TRACKING) continue
            if (plane.type != Plane.Type.HORIZONTAL_UPWARD_FACING) continue
            val y = plane.centerPose.ty()
            if (lowest == null || y < lowest) lowest = y
        }
        val y = lowest ?: return
        if (floorHeight == null || abs(y - floorHeight!!) > FLOOR_EPSILON_M) {
            floorHeight = y
            rec?.setFloorHeight(y)
        }
    }

    // -- HUD ----------------------------------------------------------------

    /**
     * End the capture once the core says the room is done (plan §6.1).
     *
     * The HUD has been saying "you can stop" for a while without anything
     * stopping, which leaves the 150-keyframe cap to end captures — a cost
     * ceiling doing a job the completeness rule was written for. §1 puts it
     * plainly: guide "until coverage is complete".
     *
     * Held for a moment first. Completeness depends on a coverage ratio that
     * can cross its threshold and fall back as new geometry comes into view,
     * and ending someone's capture on a flicker is worse than a second's delay.
     */
    private fun stopIfComplete(rec: CaptureRecorder) {
        // The recorder combines supported floor/wall/ceiling coverage, per-class
        // yaw sectors, a translated baseline, tracking continuity and the
        // 24-keyframe minimum. A finished target list alone is never completion.
        if (!rec.completionReady) {
            completeSinceMs = 0L
            return
        }
        val now = SystemClock.uptimeMillis()
        if (completeSinceMs == 0L) {
            completeSinceMs = now
            return
        }
        if (now - completeSinceMs < COMPLETE_HOLD_MS) return
        completeSinceMs = 0L
        runOnUiThread { if (recorder === rec) toggleRecording() }
    }

    /**
     * Rebuilt at 10 Hz, not per frame. None of this changes faster than a person
     * can read it, and posting a fresh state to the UI thread 60 times a second
     * is pure allocation.
     */
    private fun publishHud(rec: CaptureRecorder?, tracking: TrackingState, cameraPos: FloatArray) {
        val now = SystemClock.uptimeMillis()
        if (now - lastHudAtMs < HUD_INTERVAL_MS) return
        lastHudAtMs = now

        val trackingWord = when (tracking) {
            TrackingState.TRACKING -> "tracking"
            TrackingState.PAUSED -> "tracking paused"
            else -> "tracking lost"
        }
        val guidance = rec?.lastGuidance
        val params = rec?.core
        compass.headingDeg?.let { rec?.setHeading(it) }

        // Only while tracking is actually lost. Shown at any other time it is
        // noise, and worse, it teaches the user to ignore the one banner that
        // means the capture has stopped collecting anything.
        val recoveryLine = if (rec != null && tracking != TrackingState.TRACKING) {
            rec.recovery()?.let { recoveryMessage(it) }
        } else {
            null
        }

        val statusLine = buildString {
            append(if (rec == null) {
                when (depthReadiness) {
                    // The instruction, not the diagnosis: a user watching
                    // "depth self-test" has no idea it is waiting on them.
                    DepthReadiness.CHECKING ->
                        if (depthProbeMovedM < MIN_SELF_TEST_MOTION_M) {
                            "depth self-test · walk a step or two"
                        } else {
                            "depth self-test · %.2f m moved".format(depthProbeMovedM)
                        }
                    DepthReadiness.READY -> "ready · depth verified"
                    DepthReadiness.UNSUPPORTED -> "ready · RGB-only (depth unsupported)"
                    DepthReadiness.FAILED -> "ready · RGB-only (depth failed)"
                }
            } else if (rec.depthAvailable) "recording · depth" else "recording · RGB-only")
            append("  ·  ")
            append(trackingWord)
            if (rec != null && rec.rejected.isNotEmpty()) {
                // Only the commonest reason, and only as a word. The full
                // breakdown used to be appended here and ran off the right
                // edge of the screen; the per-frame verdict below now carries
                // the detail, which is where a user can act on it anyway.
                val worst = rec.rejected.maxByOrNull { it.value }
                if (worst != null) {
                    append("  ·  mostly ")
                    append(worst.key)
                }
            }
        }

        var bearing: Float? = null
        var distance: Float? = null
        if (rec != null && tracking == TrackingState.TRACKING) {
            rec.shot?.let { shot ->
            worldPoint[0] = shot.direction.x
            worldPoint[1] = shot.direction.y
            worldPoint[2] = shot.direction.z
            val state = when (shot.state) {
                uniffi.recon_core.FfiShotState.TAKE -> 2
                uniffi.recon_core.FfiShotState.NOT_STEADY -> 1
                else -> 0
            }
            shotTarget.draw(
                viewProjection, worldPoint, cameraPos, UP_AXIS, state,
                (now % PULSE_MS).toFloat() / PULSE_MS,
            )
        }

        rec.station?.let { station ->
                val dx = station.position.x - cameraPos[0]
                val dy = station.position.y - cameraPos[1]
                val dz = station.position.z - cameraPos[2]
                distance = sqrt(dx * dx + dy * dy + dz * dz)
                bearing = offScreenBearing(station.position.x, station.position.y, station.position.z)
                if (bearing == null) distance = null
            }
        }

        val sectors = rec?.supportedSurfaceSectors
        val state = GuidanceHud.State(
            statusLine = statusLine,
            recording = rec != null,
            floor = guidance?.floorCoverage ?: 0f,
            walls = guidance?.wallCoverage ?: 0f,
            ceiling = guidance?.ceilingCoverage ?: 0f,
            floorTarget = params?.floorTarget ?: 0.9f,
            wallTarget = params?.wallTarget ?: 0.9f,
            ceilingTarget = params?.ceilingTarget ?: 0.7f,
            keyframes = guidance?.keyframes?.toInt() ?: 0,
            maxKeyframes = rec?.maxKeyframes ?: 0,
            mappedVoxels = mappedVoxels,
            // The wording is the core's, so the two clients cannot say
            // different things to the same user (plan §6.1).
            nag = guidance?.nag?.let { nagMessage(it) },
            recovery = recoveryLine,
            complete = guidance?.complete ?: false,
            shotTaken = guidance?.shot?.taken?.toInt() ?: 0,
            shotTotal = guidance?.shot?.total?.toInt() ?: 0,
            shotSkipped = guidance?.shot?.skipped?.toInt() ?: 0,
            shotInstruction = guidance?.shot?.let { shotInstruction(it) },
            trackingBreaks = rec?.trackingBreaks ?: 0,
            noDepth = rec != null && !rec.depthAvailable,
            planeCount = planeCount,
            featurePoints = featurePointCount,
            depthSamples = rec?.lastDepthSamples ?: 0,
            floorSectors = sectors?.first ?: 0,
            wallSectors = sectors?.second ?: 0,
            ceilingSectors = sectors?.third ?: 0,
            stationBearingRad = bearing,
            stationDistanceM = distance,
        )
        // Only offered while a shot is actually being asked for. A skip button
        // with nothing pending is an invitation to press something that does
        // nothing, and on the free-form path there is no script to skip within.
        val canSkip = rec != null &&
            guidance?.shot != null &&
            guidance.shot?.state != uniffi.recon_core.FfiShotState.FINISHED

        // The shutter is live whenever a capture is running: it is the only
        // way to take a picture, so hiding it would strand the user.
        val shooting = rec != null
        runOnUiThread {
            hud.state = state
            skipButton.visibility = if (canSkip) View.VISIBLE else View.GONE
            shutterButton.visibility = if (shooting) View.VISIBLE else View.GONE
        }
    }

    /**
     * What to tell the user about the requested shot.
     *
     * Named heights rather than angles, and the two not-yet states worded as
     * different actions: "turn" and "hold still" need different things done,
     * and a single "not ready" would leave the user guessing which.
     */
    private fun shotInstruction(shot: uniffi.recon_core.FfiShot): String {
        val where = when (shot.height) {
            uniffi.recon_core.FfiShotHeight.BOTTOM -> "floor"
            uniffi.recon_core.FfiShotHeight.MIDDLE -> "wall"
            uniffi.recon_core.FfiShotHeight.TOP -> "ceiling"
        }
        return when (shot.state) {
            uniffi.recon_core.FfiShotState.NOT_AIMED ->
                if (shot.atStation) "Walk to the ring, then frame the $where"
                else "Turn to the square · $where"
            uniffi.recon_core.FfiShotState.NOT_STEADY -> "Hold still"
            // The one instruction that cannot be satisfied by turning: the
            // second phase exists to give the reconstruction a baseline.
            uniffi.recon_core.FfiShotState.NOT_MOVED -> "Walk to the ring first"
            uniffi.recon_core.FfiShotState.TAKE -> "Captured"
            uniffi.recon_core.FfiShotState.FINISHED -> "All shots taken"
        }
    }

    /** Project the station and ask [StationArrow] which way to point, if at all. */
    private fun offScreenBearing(x: Float, y: Float, z: Float): Float? {
        worldPoint[0] = x
        worldPoint[1] = y
        worldPoint[2] = z
        worldPoint[3] = 1f
        Matrix.multiplyMV(clipPoint, 0, viewProjection, 0, worldPoint, 0)
        return StationArrow.bearing(clipPoint[0], clipPoint[1], clipPoint[3])
    }

    private fun setStatus(text: String) {
        runOnUiThread { hud.state = hud.state.copy(statusLine = text) }
    }

    private companion object {
        const val DEPTH_PROBE_INTERVAL_MS = 250L
        /** Time the device gets *once it has been moved enough to be testable*. */
        const val DEPTH_SELF_TEST_TIMEOUT_MS = 10_000L
        /**
         * Movement before a depth failure can be blamed on the device.
         *
         * Roughly a hand's sway. Motion stereo needs a real baseline between
         * the frames it compares; below this the phone has effectively rotated
         * in place, which produces no disparity at any distance.
         */
        const val MIN_SELF_TEST_MOTION_M = 0.20f
        /** Upper bound on an inconclusive test, so it cannot check forever. */
        const val DEPTH_SELF_TEST_CAP_MS = 45_000L
        const val MIN_SELF_TEST_DEPTH_SAMPLES = 100
        const val MIN_SELF_TEST_DEPTH_FRAMES = 3
        const val MIN_SELF_TEST_SPAN_MS = 1_000L
        const val NEAR_M = 0.1f
        const val FAR_M = 100f
        /** Refresh interval for the coverage overlay. */
        const val COVERAGE_INTERVAL_MS = 330L
        const val HUD_INTERVAL_MS = 100L
        /**
         * How long a per-frame verdict stays on screen after the frame that
         * produced it. Frames are judged at camera rate; a label rewritten
         * every 33 ms cannot be read, so the last one is held long enough to
         * register and then allowed to lapse rather than going stale on screen.
         */
        const val VERDICT_HOLD_MS = 1200L
        /** How long room-complete must hold before the capture is ended for the user. */
        const val COMPLETE_HOLD_MS = 1500L
        /** World up, per plan §5's gravity-aligned frame. */
        val UP_AXIS = floatArrayOf(0f, 1f, 0f)
        /** Drawing budget in points; the core subsamples to about this many. */
        const val COVERAGE_BUDGET = 3500
        const val PULSE_MS = 1400L
        /** Period of the marching chevrons. */
        const val ARROW_MS = 1100L
        /** How far the solver must move a station before the anchor is remade. */
        const val STATION_MOVE_M = 0.25f
        /** Floor moves smaller than this are plane-tracking jitter, not a new floor. */
        const val FLOOR_EPSILON_M = 0.05f
    }
}
