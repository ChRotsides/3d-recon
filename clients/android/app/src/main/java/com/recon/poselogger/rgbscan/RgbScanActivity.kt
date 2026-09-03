package com.recon.poselogger.rgbscan

import android.Manifest
import android.content.res.ColorStateList
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Color
import android.graphics.SurfaceTexture
import android.hardware.camera2.CameraCaptureSession
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraDevice
import android.hardware.camera2.CameraManager
import android.hardware.camera2.CaptureRequest
import android.media.ImageReader
import android.os.Bundle
import android.os.Handler
import android.os.HandlerThread
import android.view.Surface
import android.view.TextureView
import android.view.View
import android.view.ViewGroup
import android.widget.Button
import android.widget.EditText
import android.widget.FrameLayout
import android.widget.LinearLayout
import android.widget.TextView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import java.io.File
import java.util.UUID
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean

/** Camera2-only RGB capture and server-derived guidance. No mobile pose or AR state is observed. */
class RgbScanActivity : AppCompatActivity(), TextureView.SurfaceTextureListener {
    private lateinit var preview: TextureView
    private lateinit var server: EditText
    private lateinit var status: TextView
    private lateinit var start: Button
    private lateinit var shutter: Button
    private lateinit var finish: Button
    private lateinit var openLast: Button
    private lateinit var guidance: SphericalGuidanceView
    private lateinit var directionTracker: SensorDirectionTracker
    private lateinit var draftSection: LinearLayout
    private lateinit var draftContainer: FrameLayout
    private lateinit var draftLabel: TextView
    private lateinit var cameraViewButton: Button
    private lateinit var draftViewButton: Button
    private lateinit var fitDraftButton: Button
    private lateinit var startActions: LinearLayout
    private lateinit var captureActions: LinearLayout
    private var draftView: PointCloudView? = null
    private var showingDraft = false
    private val network = Executors.newSingleThreadExecutor()
    // A draft can be several MiB. It never occupies the capture/upload executor.
    private val draftNetwork = Executors.newSingleThreadExecutor()
    private val mainHandler = Handler(android.os.Looper.getMainLooper())
    private val guidanceStore = GuidanceStateStore()
    private val pollInFlight = AtomicBoolean(false)
    private val draftPollInFlight = AtomicBoolean(false)
    private val cameraThread = HandlerThread("rgb-camera").apply { start() }
    private val cameraHandler = Handler(cameraThread.looper)
    private var camera: CameraDevice? = null
    private var cameraOpening = false
    private var cameraLifecycleActive = false
    private var captureSession: CameraCaptureSession? = null
    private var reader: ImageReader? = null
    private var capturePending = false
    private var sessionId: String? = null
    private var stationId: String? = null
    private var protocol: RgbProtocol? = null
    @Volatile private var stationFinishing = false
    private var uploadedFrames = 0
    @Volatile private var latestDraftVersion = 0
    private lateinit var frameStore: LocalFrameStore
    private val guidancePoll = object : Runnable {
        override fun run() {
            pollGuidance()
            if (!isFinishing && !isDestroyed && stationId != null && !stationFinishing) {
                mainHandler.postDelayed(this, GUIDANCE_POLL_MS)
            }
        }
    }
    private val draftPoll = object : Runnable {
        override fun run() {
            pollDraft()
            if (!isFinishing && !isDestroyed && stationId != null && !stationFinishing) {
                mainHandler.postDelayed(this, DRAFT_POLL_MS)
            }
        }
    }

    private val permission = registerForActivityResult(ActivityResultContracts.RequestPermission()) {
        if (it) openCamera() else setStatus("Camera permission is required")
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        frameStore = LocalFrameStore(File(filesDir, "live-rgb/frames"))
        preview = TextureView(this).apply {
            surfaceTextureListener = this@RgbScanActivity
            keepScreenOn = true
        }
        server = EditText(this).apply {
            setText(getPreferences(MODE_PRIVATE).getString("server", "http://192.168.0.226:8765"))
            hint = "Reconstruction service"
        }
        status = TextView(this).apply { text = "Point at the room, then start a station" }
        guidance = SphericalGuidanceView(this).apply {
            minimumHeight = (resources.displayMetrics.density * 96).toInt()
            visibility = View.GONE
        }
        directionTracker = SensorDirectionTracker(this) { yaw, pitch ->
            guidance.showSensorDirection(yaw, pitch)
        }
        draftLabel = TextView(this).apply {
            text = "Live 3D builds after 6 connected photos"
            setPadding(16, 8, 16, 4)
        }
        draftContainer = FrameLayout(this).apply { visibility = View.GONE }
        cameraViewButton = Button(this).apply {
            text = "CAMERA"
            setOnClickListener { showCameraViewport() }
        }
        draftViewButton = Button(this).apply {
            text = "LIVE 3D"
            isEnabled = false
            setOnClickListener { showDraftViewport() }
        }
        fitDraftButton = Button(this).apply {
            text = "FIT"
            isEnabled = false
            setOnClickListener { draftView?.fitCloud() }
        }
        val draftModes = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            addView(cameraViewButton, LinearLayout.LayoutParams(
                0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f,
            ))
            addView(draftViewButton, LinearLayout.LayoutParams(
                0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f,
            ))
            addView(fitDraftButton, LinearLayout.LayoutParams(
                0, ViewGroup.LayoutParams.WRAP_CONTENT, 0.55f,
            ))
        }
        draftSection = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            visibility = View.GONE
            addView(draftLabel)
            addView(draftModes)
        }
        start = Button(this).apply { text = "Start station"; setOnClickListener { startStation() } }
        shutter = Button(this).apply { text = "Capture"; isEnabled = false; setOnClickListener { takePhoto() } }
        finish = Button(this).apply { text = "Finish 3D"; isEnabled = false; setOnClickListener { finishStation() } }
        openLast = Button(this).apply {
            text = "Open last 3D"
            isEnabled = latestCloudFiles() != null
            setOnClickListener { openLatestCloud() }
        }
        startActions = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            addView(start, LinearLayout.LayoutParams(
                0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f,
            ))
            addView(openLast, LinearLayout.LayoutParams(
                0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f,
            ))
        }
        captureActions = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            visibility = View.GONE
            addView(shutter, LinearLayout.LayoutParams(
                0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f,
            ))
            addView(finish, LinearLayout.LayoutParams(
                0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f,
            ))
        }
        val viewport = FrameLayout(this).apply {
            addView(preview, FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.MATCH_PARENT,
            ))
            addView(draftContainer, FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.MATCH_PARENT,
            ))
        }
        val controls = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            addView(server)
            addView(draftSection)
            addView(guidance, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT,
                (resources.displayMetrics.density * 96).toInt()))
            addView(status)
            addView(startActions)
            addView(captureActions)
        }
        setContentView(LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            addView(viewport, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, 0, 1f))
            addView(controls)
        })
        showCameraViewport()
        if (checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            if (preview.isAvailable) openCamera()
        } else permission.launch(Manifest.permission.CAMERA)
    }

    private fun startStation() {
        val endpoint = server.text.toString().trim()
        if (endpoint.isEmpty()) return
        server.isEnabled = false
        start.isEnabled = false
        getPreferences(MODE_PRIVATE).edit().putString("server", endpoint).apply()
        val newSession = "session-${UUID.randomUUID()}"
        val newStation = "station-${UUID.randomUUID()}"
        val client = RgbProtocol(endpoint)
        network.execute {
            try {
                client.createSession(newSession)
                client.createStation(newSession, newStation)
                sessionId = newSession
                stationId = newStation
                protocol = client
                stationFinishing = false
                uploadedFrames = 0
                latestDraftVersion = 0
                val initialGuidance = client.guidance(newSession, newStation)
                runOnUiThread {
                    applyGuidance(initialGuidance)
                    server.visibility = View.GONE
                    startActions.visibility = View.GONE
                    draftSection.visibility = View.VISIBLE
                    guidance.visibility = View.VISIBLE
                    captureActions.visibility = View.VISIBLE
                    shutter.isEnabled = true
                    finish.isEnabled = true
                    setStatus("Ready · 0 RGB photos")
                    mainHandler.removeCallbacks(guidancePoll)
                    mainHandler.postDelayed(guidancePoll, GUIDANCE_POLL_MS)
                    mainHandler.removeCallbacks(draftPoll)
                    mainHandler.postDelayed(draftPoll, DRAFT_POLL_MS)
                }
            } catch (error: Exception) {
                runOnUiThread {
                    server.visibility = View.VISIBLE
                    startActions.visibility = View.VISIBLE
                    draftSection.visibility = View.GONE
                    guidance.visibility = View.GONE
                    captureActions.visibility = View.GONE
                    server.isEnabled = true
                    start.isEnabled = true
                    setStatus("Start failed: ${error.message}")
                }
            }
        }
    }

    private fun takePhoto() {
        val activeCamera = camera ?: return
        val output = reader?.surface ?: return
        if (capturePending) return
        capturePending = true
        shutter.isEnabled = false
        val request = activeCamera.createCaptureRequest(CameraDevice.TEMPLATE_STILL_CAPTURE).apply {
            addTarget(output)
            set(CaptureRequest.CONTROL_AF_MODE, CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_PICTURE)
        }
        captureSession?.capture(request.build(), object : CameraCaptureSession.CaptureCallback() {}, cameraHandler)
    }

    private fun finishStation() {
        val session = sessionId ?: return
        val station = stationId ?: return
        val client = protocol ?: return
        shutter.isEnabled = false
        finish.isEnabled = false
        stationFinishing = true
        mainHandler.removeCallbacks(guidancePoll)
        mainHandler.removeCallbacks(draftPoll)
        setStatus("Building panorama and 3D…")
        network.execute {
            try {
                val job = client.finalize(session, station)
                check(job.getString("state") == "complete") { "job did not complete" }
                val version = job.getInt("result_version")
                val result = File(filesDir, "live-rgb/results/$session/$station/v$version")
                val metadata = File(result, "cloud.json")
                val binary = File(result, "cloud.bin")
                client.download(session, station, version, "cloud.json", metadata)
                client.download(session, station, version, "cloud.bin", binary)
                // Decode before launching: corrupt/incompatible assets never reach rendering state.
                PointCloudDecoder.decode(metadata, binary)
                runOnUiThread {
                    openLast.isEnabled = true
                    startActivity(Intent(this, CloudViewerActivity::class.java).apply {
                        putExtra(CloudViewerActivity.EXTRA_METADATA, metadata.absolutePath)
                        putExtra(CloudViewerActivity.EXTRA_BINARY, binary.absolutePath)
                    })
                    setStatus("3D ready")
                }
            } catch (error: Exception) {
                stationFinishing = false
                runOnUiThread {
                    finish.isEnabled = true
                    mainHandler.postDelayed(guidancePoll, GUIDANCE_POLL_MS)
                    mainHandler.postDelayed(draftPoll, DRAFT_POLL_MS)
                    setStatus("Build failed: ${error.message}")
                }
            }
        }
    }

    private fun latestCloudFiles(): Pair<File, File>? {
        val root = File(filesDir, "live-rgb")
        val metadata = root.walkTopDown()
            .filter { it.isFile && it.name in setOf("cloud.json", "draft.json") }
            .maxByOrNull { it.lastModified() }
            ?: return null
        val binary = File(
            metadata.parentFile,
            if (metadata.name == "draft.json") "draft.bin.gz" else "cloud.bin",
        )
        return if (binary.isFile) metadata to binary else null
    }

    private fun openLatestCloud() {
        val (metadata, binary) = latestCloudFiles() ?: run {
            openLast.isEnabled = false
            setStatus("No saved 3D result")
            return
        }
        try {
            PointCloudDecoder.decode(metadata, binary)
            startActivity(Intent(this, CloudViewerActivity::class.java).apply {
                putExtra(CloudViewerActivity.EXTRA_METADATA, metadata.absolutePath)
                putExtra(CloudViewerActivity.EXTRA_BINARY, binary.absolutePath)
            })
        } catch (error: Exception) {
            setStatus("Saved 3D rejected: ${error.message}")
        }
    }

    private fun openCamera() {
        if (
            !preview.isAvailable ||
            !cameraLifecycleActive ||
            camera != null ||
            cameraOpening ||
            checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED
        ) return
        cameraOpening = true
        val manager = getSystemService(CameraManager::class.java)
        val cameraId = manager.cameraIdList.firstOrNull { id ->
            manager.getCameraCharacteristics(id).get(CameraCharacteristics.LENS_FACING) == CameraCharacteristics.LENS_FACING_BACK
        } ?: manager.cameraIdList.first()
        val map = manager.getCameraCharacteristics(cameraId)
            .get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP)
        if (map == null) {
            cameraOpening = false
            setStatus("Camera configuration unavailable")
            return
        }
        val jpegSize = map.getOutputSizes(android.graphics.ImageFormat.JPEG).maxBy { it.width * it.height }
        reader = ImageReader.newInstance(jpegSize.width, jpegSize.height, android.graphics.ImageFormat.JPEG, 2).apply {
            setOnImageAvailableListener({ source ->
                source.acquireNextImage().use { image ->
                    val buffer = image.planes[0].buffer
                    val jpeg = ByteArray(buffer.remaining()).also { buffer.get(it) }
                    val local = frameStore.persist(jpeg) // stable ID and durable JPEG precede network I/O
                    val session = sessionId
                    val station = stationId
                    val client = protocol
                    if (session != null && station != null && client != null) {
                        val sensorAtCapture = directionTracker.captureOrientation()
                        network.execute {
                            try {
                                val uploaded = client.upload(session, station, local.id, local.timestampNs, local.jpeg)
                                guidanceFrom(uploaded)?.let(::publishGuidance)
                                // Reuse the captured RGB JPEG as a localization preview. It never adds
                                // a frame or geometric field to the graph.
                                val previewId = "preview-${UUID.randomUUID()}"
                                val localized = client.uploadPreview(
                                    session, station, previewId,
                                    local.timestampNs, local.jpeg,
                                )
                                val localizedSnapshot = guidanceFrom(localized)?.let(::publishGuidance)
                                localizedSnapshot?.localization?.let { localization ->
                                    if (
                                        localization.state == LocalizationState.LOCALIZED &&
                                        localization.yawDegrees != null &&
                                        localization.pitchDegrees != null
                                    ) {
                                        directionTracker.anchor(
                                            sensorAtCapture,
                                            localization.yawDegrees,
                                            localization.pitchDegrees,
                                        )
                                    }
                                }
                                uploadedFrames += 1
                                runOnUiThread {
                                    setStatus("Ready · $uploadedFrames RGB photos · guidance v${guidanceStore.current().version}")
                                }
                            } catch (error: Exception) {
                                runOnUiThread { setStatus("Saved locally; upload failed: ${error.message}") }
                            }
                        }
                    }
                }
                capturePending = false
                runOnUiThread { shutter.isEnabled = sessionId != null }
            }, cameraHandler)
        }
        try {
            manager.openCamera(cameraId, object : CameraDevice.StateCallback() {
                override fun onOpened(device: CameraDevice) {
                    cameraOpening = false
                    if (!cameraLifecycleActive || reader == null) {
                        device.close()
                        return
                    }
                    camera = device
                    createPreview()
                }

                override fun onDisconnected(device: CameraDevice) {
                    cameraOpening = false
                    device.close()
                    camera = null
                }

                override fun onError(device: CameraDevice, error: Int) {
                    cameraOpening = false
                    device.close()
                    camera = null
                    setStatus("Camera error $error")
                }
            }, cameraHandler)
        } catch (error: Exception) {
            cameraOpening = false
            reader?.close()
            reader = null
            setStatus("Camera failed: ${error.message}")
        }
    }

    private fun createPreview() {
        val device = camera ?: return
        val texture = preview.surfaceTexture ?: return
        texture.setDefaultBufferSize(preview.width, preview.height)
        val surface = Surface(texture)
        val targets = listOf(surface, reader!!.surface)
        device.createCaptureSession(targets, object : CameraCaptureSession.StateCallback() {
            override fun onConfigured(session: CameraCaptureSession) {
                captureSession = session
                val request = device.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW).apply { addTarget(surface) }
                session.setRepeatingRequest(request.build(), null, cameraHandler)
            }
            override fun onConfigureFailed(session: CameraCaptureSession) { setStatus("Camera preview failed") }
        }, cameraHandler)
    }

    private fun setStatus(message: String) = runOnUiThread { status.text = message }

    private fun pollGuidance() {
        val session = sessionId ?: return
        val station = stationId ?: return
        val client = protocol ?: return
        if (!pollInFlight.compareAndSet(false, true)) return
        network.execute {
            try {
                publishGuidance(client.guidance(session, station))
            } catch (_: Exception) {
                // Capture/upload status is more actionable than a transient poll failure.
            } finally {
                pollInFlight.set(false)
            }
        }
    }

    private fun pollDraft() {
        val session = sessionId ?: return
        val station = stationId ?: return
        val client = protocol ?: return
        if (!draftPollInFlight.compareAndSet(false, true)) return
        draftNetwork.execute {
            try {
                val event = client.latestDraft(session, station)
                if (event.optString("state") != "ready") return@execute
                val version = event.getInt("version")
                if (version <= latestDraftVersion) return@execute
                val result = File(filesDir, "live-rgb/drafts/$session/$station/v$version")
                val metadata = File(result, event.getString("metadata_name"))
                val binary = File(result, event.getString("asset_name"))
                client.downloadDraft(session, station, version, metadata.name, metadata)
                client.downloadDraft(session, station, version, binary.name, binary)
                val decodeStarted = System.nanoTime()
                val asset = PointCloudDecoder.decode(metadata, binary)
                check(asset.version == version) { "draft event/asset version mismatch" }
                if (version <= latestDraftVersion) return@execute
                latestDraftVersion = version
                val decodeMs = (System.nanoTime() - decodeStarted) / 1_000_000
                runOnUiThread { showDraft(asset, decodeMs) }
            } catch (_: Exception) {
                // A later poll retries an incomplete/temporarily unavailable version.
            } finally {
                draftPollInFlight.set(false)
            }
        }
    }

    private fun showDraft(asset: PointCloudAsset, decodeMs: Long) {
        val existing = draftView
        if (existing == null) {
            PointCloudView(this, asset).also { viewer ->
                draftView = viewer
                draftContainer.removeAllViews()
                draftContainer.addView(viewer, FrameLayout.LayoutParams(
                    ViewGroup.LayoutParams.MATCH_PARENT,
                    ViewGroup.LayoutParams.MATCH_PARENT,
                ))
                if (cameraLifecycleActive) viewer.onResume()
            }
        } else {
            existing.replaceScene(asset)
        }
        draftViewButton.isEnabled = true
        fitDraftButton.isEnabled = true
        draftLabel.text = if (showingDraft) {
            "3D v${asset.version} · ${asset.count} points · 1 finger orbit · 2 fingers move/pinch"
        } else {
            "3D v${asset.version} ready · tap LIVE 3D · decoded ${decodeMs} ms"
        }
    }

    private fun showCameraViewport() {
        showingDraft = false
        if (::draftContainer.isInitialized) draftContainer.visibility = View.GONE
        updateViewportButtons()
    }

    private fun showDraftViewport() {
        val viewer = draftView ?: return
        showingDraft = true
        draftContainer.visibility = View.VISIBLE
        draftLabel.text = "Live 3D · 1 finger orbit · 2 fingers move/pinch · double-tap to fit"
        if (cameraLifecycleActive) viewer.onResume()
        viewer.requestRender()
        updateViewportButtons()
    }

    private fun updateViewportButtons() {
        if (!::cameraViewButton.isInitialized) return
        val selected = ColorStateList.valueOf(Color.rgb(30, 125, 210))
        val normal = ColorStateList.valueOf(Color.rgb(75, 75, 82))
        cameraViewButton.backgroundTintList = if (showingDraft) normal else selected
        draftViewButton.backgroundTintList = if (showingDraft) selected else normal
    }

    private fun guidanceFrom(response: org.json.JSONObject): org.json.JSONObject? = when {
        response.optString("schema") == GuidanceStateStore.SCHEMA -> response
        response.optJSONObject("guidance") != null -> response.getJSONObject("guidance")
        else -> null
    }

    private fun publishGuidance(payload: org.json.JSONObject): GuidanceSnapshot? {
        val accepted = try {
            guidanceStore.accept(payload)
        } catch (_: Exception) {
            null
        } ?: return null
        runOnUiThread { guidance.show(accepted) }
        return accepted
    }

    private fun applyGuidance(payload: org.json.JSONObject) = publishGuidance(payload)

    override fun onSurfaceTextureAvailable(surface: SurfaceTexture, width: Int, height: Int) = openCamera()
    override fun onSurfaceTextureSizeChanged(surface: SurfaceTexture, width: Int, height: Int) = Unit
    override fun onSurfaceTextureDestroyed(surface: SurfaceTexture): Boolean = true
    override fun onSurfaceTextureUpdated(surface: SurfaceTexture) = Unit

    override fun onResume() {
        super.onResume()
        if (::openLast.isInitialized) openLast.isEnabled = latestCloudFiles() != null
        cameraLifecycleActive = true
        directionTracker.start()
        draftView?.onResume()
        if (preview.isAvailable) openCamera()
    }

    override fun onPause() {
        cameraLifecycleActive = false
        directionTracker.stop()
        draftView?.onPause()
        closeCamera()
        super.onPause()
    }

    private fun closeCamera() {
        captureSession?.close()
        captureSession = null
        camera?.close()
        camera = null
        reader?.close()
        reader = null
        cameraOpening = false
    }

    override fun onDestroy() {
        mainHandler.removeCallbacks(guidancePoll)
        mainHandler.removeCallbacks(draftPoll)
        closeCamera(); cameraThread.quitSafely(); network.shutdown(); draftNetwork.shutdown()
        super.onDestroy()
    }

    companion object {
        private const val GUIDANCE_POLL_MS = 1_500L
        private const val DRAFT_POLL_MS = 1_500L
    }
}
