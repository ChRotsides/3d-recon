package com.recon.videoscan.live

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.ImageFormat
import android.graphics.SurfaceTexture
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.hardware.camera2.CameraCaptureSession
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraDevice
import android.hardware.camera2.CameraManager
import android.hardware.camera2.CaptureRequest
import android.hardware.camera2.CaptureResult
import android.hardware.camera2.TotalCaptureResult
import android.media.Image
import android.media.ImageReader
import android.opengl.GLSurfaceView
import android.os.Bundle
import android.os.Handler
import android.os.HandlerThread
import android.os.Looper
import android.os.SystemClock
import android.util.Log
import android.util.Size
import android.view.MotionEvent
import android.view.Surface
import android.view.TextureView
import android.view.WindowManager
import android.widget.Button
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicLong
import java.util.concurrent.locks.ReentrantReadWriteLock
import kotlin.concurrent.read
import kotlin.concurrent.write
import kotlin.math.abs
import kotlin.math.sqrt
import kotlin.math.roundToInt

/**
 * Everything on the phone: Camera2 frames + IMU -> OpenVINS poses (JNI) -> DA3-Small depth (onnxruntime)
 * scaled by the VIO landmarks -> voxel-hash TSDF (JNI) -> two GL views (camera POV, orbit), like the web page.
 */
class LiveActivity : AppCompatActivity(), SensorEventListener {
    companion object { private const val ENABLE_EXPERIMENTAL_POSE = false }
    private lateinit var status: TextView
    private lateinit var preview: TextureView
    private lateinit var featureOverlay: FeatureOverlayView
    private lateinit var explorationView: ExplorationMapView
    private val exploration = ExplorationMap()
    private var featureRotation = 90
    private lateinit var povView: GLSurfaceView
    private lateinit var orbitView: GLSurfaceView
    private val povRenderer = CloudRenderer(pov = true)
    private val orbitRenderer = CloudRenderer(pov = false)
    private var handle = 0L
    // VIO and depth may run together; reset/destruction waits for both to finish.
    private val sessionGate = ReentrantReadWriteLock(true)
    private val generation = AtomicLong(0)
    private var nativeGeneration = -1L
    @Volatile private var active = false
    @Volatile private var timingError: String? = null
    private var depth: DepthRunner? = null
    @Volatile private var xfeat: XFeatRunner? = null
    @Volatile private var poseTransformer: PoseTransformerRunner? = null
    @Volatile private var poseSource = "waiting"
    @Volatile private var lastXFeatAt = Double.NEGATIVE_INFINITY
    @Volatile private var xfeatIntervalSec = 0.2
    @Volatile private var xfeatRuns = 0
    private var fusion = true
    // camera
    private var camera: CameraDevice? = null; private var session: CameraCaptureSession? = null; private var reader: ImageReader? = null
    private val cameraLock = Any()
    private val cameraGeneration = AtomicLong(0)
    private val camThread = HandlerThread("camera").apply { start() }; private val camHandler = Handler(camThread.looper)
    private val vioThread = HandlerThread("vio").apply { start() }; private val vioHandler = Handler(vioThread.looper)
    private val depthThread = HandlerThread("depth").apply { start() }; private val depthHandler = Handler(depthThread.looper)
    // Optional learned descriptors run independently of the camera/VIO stream.
    // Native corners replenish tracking on every eligible frame.
    private val xfeatThread = HandlerThread("xfeat").apply { start() }; private val xfeatHandler = Handler(xfeatThread.looper)
    private data class DescriptorFrame(val image: FrameData, val localPose: FloatArray,
        val landmarks: FloatArray, val calibration: DoubleArray, val permit: TrackingHealth.Permit)
    private val stagedFrames = FrameMailbox<DescriptorFrame>()
    private val sensorThread = HandlerThread("imu").apply { start() }; private val sensorHandler = Handler(sensorThread.looper)
    private val streamSize = Size(960, 540)
    private var imageSampler: TrackingImageSampler? = null
    private var captureMsAvg = 0.0
    private lateinit var sensors: SensorManager
    private val inputs = VioInputQueue<FrameData>()
    // Adaptive frame decimation: feed VIO every k-th camera frame so each step fits
    // the camera interval. Regular gaps replace random queue drops, which starved the
    // stationary detector and the tracker on 2026-09-07 (45% of frames dropped).
    @Volatile private var vioDecimation = 1
    private var cameraFrameIndex = 0L; @Volatile private var skippedFrames = 0L
    private var lastCameraArrival = Double.NaN; private var cameraIntervalMsAvg = 33.3
    private var rgbColumns: Pair<ImageCrop, IntArray>? = null
    private val trackingHealth = TrackingHealth()
    private val startupProgress = StartupProgress()
    private val warmStart = BiasWarmStart()
    private val recoveryPending = AtomicBoolean(false)
    @Volatile private var recoveryNotice = ""
    @Volatile private var diagnostics: VioDiagnostics? = null
    @Volatile private var recorder: DiagnosticRecorder? = null
    @Volatile private var recordingStatus = "not started"
    @Volatile private var depthPauseReason = "waiting for reliable tracking"
    private var lastHealthLog = Double.NEGATIVE_INFINITY
    private var lastHealthState: TrackingHealth.State? = null
    private val testPhases = listOf("STILL", "TILT", "HOLD", "TRANSLATE", "STOP", "DONE")
    private var phaseIndex = 0
    private val pumpScheduled = AtomicBoolean(false)
    private var lastFedImu = Double.NEGATIVE_INFINITY
    @Volatile private var imuCount = 0L
    @Volatile private var cameraImuOffset = 0.0
    // state
    @Volatile private var lastPose: FloatArray? = null; @Volatile private var lastPoseTime = 0.0; private val landmarks = FloatArray(600); @Volatile private var landmarkCount = 0
    private val depthBusy = AtomicBoolean(false)
    private var vioMsAvg = 0.0; private var depthMs = 0L; private var tsdfStats = FloatArray(6); private var frames = 0; private var depthFrames = 0; private var lastDepthAt = 0.0; private var lastDepthPose: FloatArray? = null
    private val pathPts = ArrayList<Float>(); private var initTime = -1.0; private var scaleUsed = 0f
    private val ui = Handler(Looper.getMainLooper())
    private val ticker = object : Runnable { override fun run() { updateStatus(); ui.postDelayed(this, 300) } }

    class FrameData(val t: Double, val w: Int, val h: Int, val gray: ByteArray, val rgba: ByteArray, val rw: Int, val rh: Int, val generation: Long) {
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(com.recon.videoscan.R.layout.activity_live)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        status = findViewById(com.recon.videoscan.R.id.status); preview = findViewById(com.recon.videoscan.R.id.preview)
        featureOverlay = findViewById(com.recon.videoscan.R.id.features)
        explorationView = findViewById(com.recon.videoscan.R.id.exploration_map)
        povView = findViewById(com.recon.videoscan.R.id.pov); orbitView = findViewById(com.recon.videoscan.R.id.orbit)
        for ((v, r) in listOf(povView to povRenderer, orbitView to orbitRenderer)) { v.setEGLContextClientVersion(2); v.setRenderer(r); v.renderMode = GLSurfaceView.RENDERMODE_WHEN_DIRTY }
        var lx = 0f; var ly = 0f
        orbitView.setOnTouchListener { _, e -> when (e.actionMasked) {
            MotionEvent.ACTION_DOWN -> { lx = e.x; ly = e.y }
            MotionEvent.ACTION_MOVE -> { if (e.pointerCount == 1) { orbitRenderer.yaw += (e.x - lx) * 0.01f; orbitRenderer.pitch = (orbitRenderer.pitch + (e.y - ly) * 0.01f).coerceIn(-1.4f, 1.5f); orbitView.requestRender() } ; lx = e.x; ly = e.y } }; true }
        findViewById<Button>(com.recon.videoscan.R.id.reset).setOnClickListener { resetMap() }
        findViewById<Button>(com.recon.videoscan.R.id.save_scan).setOnClickListener { saveScan() }
        findViewById<Button>(com.recon.videoscan.R.id.phase).setOnClickListener { view ->
            val button = view as Button
            val capture = recorder
            if (capture == null || capture.state != "recording" || phaseIndex >= testPhases.size) {
                button.text = "Press RESET to start a new test"; return@setOnClickListener
            }
            val phase = testPhases[phaseIndex++]
            capture.marker(SystemClock.elapsedRealtimeNanos() * 1e-9, phase)
            Log.i("livescan", "test phase=$phase")
            if (phase == "DONE") { capture.close(); button.text = "Test saved — RESET to repeat" }
            else button.text = "MARK ${testPhases[phaseIndex]} (next phase)"
        }
        findViewById<Button>(com.recon.videoscan.R.id.model).setOnClickListener { val dr = depth ?: return@setOnClickListener
            if (dr.available.isEmpty()) return@setOnClickListener
            val button = it as Button
            button.isEnabled = false; button.text = "Preparing depth…"
            val nextModel = (modelIndex + 1) % dr.available.size
            depthHandler.post {
                try {
                    val ms = dr.select(nextModel); modelIndex = nextModel
                    getPreferences(MODE_PRIVATE).edit().putString("depth_model", dr.spec?.file).apply()
                    ui.post { (it as Button).text = "${dr.label} ($ms ms)"; resetMap() }
                } catch (e: Exception) {
                    recoveryNotice = "Model unavailable; current model retained: ${e.message}"
                    Log.e("livescan", "depth model switch failed", e)
                } finally { ui.post { button.isEnabled = true; button.text = dr.label } }
            }
        }
        findViewById<Button>(com.recon.videoscan.R.id.fuse).setOnClickListener { fusion = !fusion; (it as Button).text = if (fusion) "fusion: on" else "fusion: off"; resetMap() }
        sensors = getSystemService(Context.SENSOR_SERVICE) as SensorManager
        val files = getExternalFilesDir(null)!!
        val cfg = File(files, "vio/estimator_config.yaml")
        if (!cfg.exists()) copyAssets(files)
        val model = File(files, "models")
        if (ENABLE_EXPERIMENTAL_POSE) depthHandler.post { try {
            val modelFile = copyBundledModel(model, PoseTransformerRunner.MODEL_FILE)
            val started = System.nanoTime(); poseTransformer = PoseTransformerRunner(modelFile)
            Log.i("livescan", "Pose transformer v2 ready model=${modelFile.name} init=${"%.1f".format((System.nanoTime() - started) / 1e6)}ms context=${PoseTransformerRunner.CONTEXT}")
        } catch (e: Throwable) { Log.e("livescan", "Pose transformer unavailable", e); poseSource = "model unavailable" } }
        depthHandler.post { try {
            val modelFile = copyBundledXFeat(model)
            val started = System.nanoTime(); xfeat = XFeatRunner(modelFile)
            Log.i("livescan", "XFeat ready model=${modelFile.name} init=${"%.1f".format((System.nanoTime() - started) / 1e6)}ms input=${XFeatRunner.INPUT_WIDTH}x${XFeatRunner.INPUT_HEIGHT} threads=2")
        } catch (e: Throwable) { Log.e("livescan", "XFeat unavailable; native detector retained", e) } }
        depthHandler.post {
            try {
                if (!File(model, DepthRunner.DEFAULT_MODEL).isFile || File(model, DepthRunner.DEFAULT_MODEL).length() == 0L)
                    copyBundledModel(model, DepthRunner.DEFAULT_MODEL)
                val preferred = intent.getStringExtra("depth_model") ?: getPreferences(MODE_PRIVATE).getString("depth_model", null)
                if (preferred == QnnDepthSession.MAIN_MODEL)
                    depthPauseReason = "Preparing accelerated depth; first setup can take a few minutes"
                val loaded = DepthRunner(model, preferred, cacheDir=noBackupFilesDir, nativeLibDir=applicationInfo.nativeLibraryDir)
                depth = loaded; modelIndex = loaded.available.indexOf(loaded.spec).coerceAtLeast(0)
                getPreferences(MODE_PRIVATE).edit().putString("depth_model", loaded.spec?.file).apply()
                ui.post { intent.removeExtra("depth_model"); findViewById<Button>(com.recon.videoscan.R.id.model).text = loaded.label }
            } catch (e: Exception) {
                depthPauseReason = "Depth model could not load: ${e.message}"
                Log.e("livescan", "depth model failed", e)
            }
        }
    }

    private fun copyAssets(files: File) {
        val dir = File(files, "vio"); dir.mkdirs()
        for (n in assets.list("vio") ?: emptyArray()) assets.open("vio/$n").use { i -> File(dir, n).outputStream().use { i.copyTo(it) } }
    }
    private fun saveScan() {
        vioHandler.post {
            sessionGate.write {
                if (handle == 0L) return@write
                try {
                    val saved = CloudSegmentStore(File(getExternalFilesDir(null)!!, "segments")).save { file ->
                        if (rawCloud.isNotEmpty()) CloudSegmentStore.writeRaw(file, rawCloud)
                        else Native.archiveSurface(handle, file.absolutePath)
                    }
                    recoveryNotice = if (saved != null) "Saved ${saved.name}" else "No confirmed surface to save yet"
                    if (saved != null) Log.i("livescan", "cloud saved: ${saved.absolutePath}")
                } catch (e: Exception) {
                    recoveryNotice = "Save failed; scan remains in memory: ${e.message}"
                    Log.e("livescan", "cloud save failed", e)
                }
            }
        }
    }
    private fun copyBundledXFeat(models: File): File {
        return copyBundledModel(models, XFeatRunner.MODEL_FILE)
    }
    private fun copyBundledModel(models: File, name: String): File {
        models.mkdirs()
        val destination = File(models, name)
        val pending = File(models, "$name.pending")
        try {
            assets.open("models/$name").use { input ->
                pending.outputStream().use { input.copyTo(it) }
            }
            check(pending.renameTo(destination)) { "Could not install $name" }
        } finally {
            pending.delete()
        }
        return destination
    }
    private fun resetMap(recovering: Boolean = false) {
        featureOverlay.clear()
        if (recovering) poseTransformer?.breakVisualTracks() else poseTransformer?.reset()
        startupProgress.start(SystemClock.elapsedRealtimeNanos() * 1e-9, recovering)
        val next = generation.incrementAndGet()
        inputs.reset(SystemClock.elapsedRealtimeNanos() * 1e-9); stagedFrames.clear()
        vioHandler.post {
            sessionGate.write {
                if (!active || next != generation.get()) return@write
                // Holding the write gate drains old depth work. Save BEFORE
                // destruction; on failure retain the old native map and viewer.
                if (handle != 0L && !recovering) {
                    try {
                        val store = CloudSegmentStore(File(getExternalFilesDir(null)!!, "segments"))
                        val saved = store.save { file ->
                            if (rawCloud.isNotEmpty()) CloudSegmentStore.writeRaw(file, rawCloud)
                            else Native.archiveSurface(handle, file.absolutePath)
                        }
                        recoveryNotice = if (saved != null) "Saved ${saved.name}; new segment is separate" else if (recovering) "Restarting tracking; no cloud points to save" else ""
                        if (saved != null) Log.i("livescan", "cloud segment saved: ${saved.absolutePath}")
                    } catch (e: Exception) {
                        timingError = "Cloud save failed; old map retained. Free storage and press RESET: ${e.message}"
                        Log.e("livescan", "recovery save failed; map NOT destroyed", e)
                        return@write
                    }
                }
                timingError = null
                if (!recovering) {
                    depthEvidence?.close(); depthEvidence = null
                    val root = File(getExternalFilesDir(null)!!, "depth-evidence")
                    if (root.walkTopDown().filter { it.isFile }.sumOf { it.length() } < 1024L * 1024 * 1024)
                        depthEvidence = DepthEvidenceRecorder(File(root, "map-${System.currentTimeMillis()}-$next"))
                }
                inputs.reset(SystemClock.elapsedRealtimeNanos() * 1e-9); stagedFrames.clear()
                warmStart.resetEvidence()
                trackingHealth.reset(); diagnostics = null
                recorder?.close(); recorder = null
                val logRoot = File(getExternalFilesDir(null)!!, "diagnostics")
                // Never delete a user's diagnostic captures. Stop creating new ones at quota.
                if (logRoot.walkTopDown().filter { it.isFile }.sumOf { it.length() } < 2048L * 1024 * 1024) {
                    val logDir = File(logRoot, "scan-${System.currentTimeMillis()}-$next")
                    recorder = DiagnosticRecorder(logDir, SystemClock.elapsedRealtimeNanos() * 1e-9)
                    recordingStatus = logDir.name
                    Log.i("livescan", "diagnostic capture: ${logDir.absolutePath}")
                } else recordingStatus = "diagnostic storage quota reached"
                lastHealthLog = Double.NEGATIVE_INFINITY; lastHealthState = null
                ui.post { phaseIndex = 0; findViewById<Button>(com.recon.videoscan.R.id.phase).text = "MARK STILL (start test phase)" }
                depthPauseReason = "waiting for reliable tracking"
                val biasPrior = warmStart.prior(SystemClock.elapsedRealtimeNanos() * 1e-9)
                val replacement = Native.init(File(getExternalFilesDir(null)!!, "vio/estimator_config.yaml").absolutePath, biasPrior)
                if (replacement == 0L) { timingError = "VIO configuration failed; previous map retained"; return@write }
                if (recovering && handle != 0L) Native.retainMap(handle, replacement)
                if (handle != 0L) Native.destroy(handle)
                handle = replacement
                mapAligned = Native.mapFrame(handle, 0.0, false, false, FloatArray(7), FloatArray(0), 0, null, null)[0] == 1.0
                if (recovering && mapAligned) recoveryNotice = "Restarted before the first map; building a new reference"
                mapRevision.incrementAndGet(); lastDescriptorAt = Double.NEGATIVE_INFINITY
                depthQuality.reset()
                Log.i("livescan", "${if (recovering) "automatic recovery" else "session start"}: generation=$next warm_bias=${biasPrior.isNotEmpty()}")
                nativeGeneration = next; lastFedImu = Double.NEGATIVE_INFINITY; imuCount = 0
                if (!recovering || mapAligned) lastPose = null
                initTime = -1.0; scaleUsed = 0f; fitA = 0f; fitB = 0f
                lastXFeatAt = Double.NEGATIVE_INFINITY; xfeatIntervalSec = 0.2; xfeatRuns = 0
                lastDepthPose = null; lastDepthAt = 0.0
                if (!recovering || mapAligned) {
                    exploration.clear(); explorationView.clear()
                    orbitTargetSet = false
                    synchronized(pathPts) { pathPts.clear(); orbitRenderer.path = FloatArray(0) }; rawCloud.clear()
                    povRenderer.points = null; orbitRenderer.points = null
                    povView.requestRender(); orbitView.requestRender()
                    depthFrames = 0
                    tsdfStats = FloatArray(6); landmarkCount = 0; landmarksUsed = 0; lastPoseTime = 0.0
                }
                if (handle == 0L) timingError = "VIO configuration failed to load"
            }
            recoveryPending.set(false)
            scheduleVio()
        }
    }

    override fun onResume() {
        super.onResume(); povView.onResume(); orbitView.onResume()
        active = true; timingError = null; resetMap()
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) startCamera()
        else ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.CAMERA), 1)
        for (t in intArrayOf(Sensor.TYPE_GYROSCOPE, Sensor.TYPE_ACCELEROMETER)) {
            val sensor = sensors.getDefaultSensor(t)
            if (sensor == null || !sensors.registerListener(this, sensor, SensorManager.SENSOR_DELAY_FASTEST, sensorHandler))
                timingError = "Live scan requires working accelerometer and gyroscope sensors"
        }
        ui.post(ticker)
    }
    override fun onPause() {
        featureOverlay.clear()
        poseTransformer?.reset()
        active = false; generation.incrementAndGet(); inputs.reset(SystemClock.elapsedRealtimeNanos() * 1e-9); stagedFrames.clear()
        trackingHealth.reset(); recorder?.close(); depthEvidence?.close()
        ui.removeCallbacks(ticker); sensors.unregisterListener(this)
        saveScan()
        synchronized(cameraLock) {
            cameraGeneration.incrementAndGet()
            session?.close(); camera?.close(); reader?.close()
            session = null; camera = null; reader = null
        }
        povView.onPause(); orbitView.onPause(); super.onPause()
    }
    override fun onRequestPermissionsResult(rc: Int, p: Array<out String>, g: IntArray) { super.onRequestPermissionsResult(rc, p, g); if (g.isNotEmpty() && g[0] == PackageManager.PERMISSION_GRANTED) startCamera() }
    override fun onDestroy() {
        active = false; generation.incrementAndGet()
        trackingHealth.reset(); recorder?.close(); depthEvidence?.close()
        vioHandler.post {
            sessionGate.write { if (handle != 0L) Native.destroy(handle); handle = 0L }
            vioThread.quitSafely(); depthHandler.post { poseTransformer?.close(); poseTransformer = null; depth?.close(); depth = null; depthThread.quitSafely() }; xfeatHandler.post { xfeat?.close(); xfeat = null; xfeatThread.quitSafely() }; sensorThread.quitSafely(); camThread.quitSafely()
        }
        super.onDestroy()
    }

    // ---- IMU
    override fun onSensorChanged(e: SensorEvent) {
        if (!active) return
        if (e.sensor.type == Sensor.TYPE_GYROSCOPE || e.sensor.type == Sensor.TYPE_ACCELEROMETER)
            recorder?.imu(e.timestamp * 1e-9, e.sensor.type, e.values[0], e.values[1], e.values[2])
        val sample = VioInputQueue.Sample(e.timestamp * 1e-9, e.values[0].toDouble(), e.values[1].toDouble(), e.values[2].toDouble())
        when (e.sensor.type) {
            Sensor.TYPE_GYROSCOPE -> inputs.addGyro(sample)
            Sensor.TYPE_ACCELEROMETER -> inputs.addAcceleration(sample)
            else -> return
        }
        scheduleVio()
    }
    override fun onAccuracyChanged(s: Sensor?, a: Int) = Unit

    private fun scheduleVio() {
        if (!active || !pumpScheduled.compareAndSet(false, true)) return
        vioHandler.post {
            // At most one additional pump can queue while this one runs; images live in
            // the bounded input queue rather than accumulating Handler closures.
            pumpScheduled.set(false)
            sessionGate.read {
                if (!active || handle == 0L || nativeGeneration != generation.get()) return@read
                val imuBatch = inputs.drainImu()
                inputs.takeDiscontinuity()?.let {
                    val health = trackingHealth.inputDiscontinuity(it)
                    recoverIfNeeded(health)
                    return@read
                }
                if (trackingHealth.current(SystemClock.elapsedRealtimeNanos() * 1e-9).state == TrackingHealth.State.RESTARTING) return@read
                for (imu in imuBatch) {
                    val a = imu.acceleration; val g = imu.gyro
                    poseTransformer?.addImu(imu)
                    Native.feedImu(handle, a.t, g.x, g.y, g.z, a.x, a.y, a.z)
                    lastFedImu = a.t; imuCount++
                }
                while (active && nativeGeneration == generation.get()) {
                    cameraImuOffset = Native.cameraImuOffset(handle)
                    val fd = inputs.pollFrame(lastFedImu, cameraImuOffset) ?: break
                    if (fd.generation == nativeGeneration) runVio(fd)
                }
            }
        }
    }

    // ---- camera
    private fun startCamera() {
        if (!active) return
        if (!preview.isAvailable) { preview.surfaceTextureListener = object : TextureView.SurfaceTextureListener {
            override fun onSurfaceTextureAvailable(st: SurfaceTexture, w: Int, h: Int) = startCamera()
            override fun onSurfaceTextureSizeChanged(st: SurfaceTexture, w: Int, h: Int) = Unit
            override fun onSurfaceTextureDestroyed(st: SurfaceTexture) = true
            override fun onSurfaceTextureUpdated(st: SurfaceTexture) = Unit }; return }
        val mgr = getSystemService(Context.CAMERA_SERVICE) as CameraManager
        val id = mgr.cameraIdList.first { mgr.getCameraCharacteristics(it).get(CameraCharacteristics.LENS_FACING) == CameraCharacteristics.LENS_FACING_BACK }
        if (mgr.getCameraCharacteristics(id).get(CameraCharacteristics.SENSOR_INFO_TIMESTAMP_SOURCE) != CameraCharacteristics.SENSOR_INFO_TIMESTAMP_SOURCE_REALTIME) {
            timingError = "Live scan unavailable: camera and motion sensors do not share a verified clock"
            return
        }
        featureRotation = mgr.getCameraCharacteristics(id).get(CameraCharacteristics.SENSOR_ORIENTATION) ?: 90
        val streamMap = mgr.getCameraCharacteristics(id).get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP)!!
        val fpsRange = mgr.getCameraCharacteristics(id).get(CameraCharacteristics.CONTROL_AE_AVAILABLE_TARGET_FPS_RANGES)
            ?.filter { it.upper == 30 }?.maxByOrNull { it.lower }
        val previews = streamMap.getOutputSizes(SurfaceTexture::class.java).map { CaptureSize(it.width, it.height) }.toSet()
        val supported = streamMap.getOutputSizes(ImageFormat.YUV_420_888).map { CaptureSize(it.width, it.height) }.filter { it in previews }
        val captureSize = selectCaptureSize(supported, CaptureSize(streamSize.width, streamSize.height))
        if (captureSize == null) { timingError = "No camera stream matches the calibrated image geometry"; return }
        val crop = trackingCrop(captureSize, CaptureSize(streamSize.width, streamSize.height))!!
        imageSampler = TrackingImageSampler(crop, streamSize.width, streamSize.height)
        Log.i("livescan", "capture ${captureSize.width}x${captureSize.height} crop=$crop -> VIO ${streamSize.width}x${streamSize.height}")
        val cameraEpoch = cameraGeneration.incrementAndGet()
        val captureReader = ImageReader.newInstance(captureSize.width, captureSize.height, ImageFormat.YUV_420_888, 3)
        reader = captureReader
        captureReader.setOnImageAvailableListener({ r -> synchronized(cameraLock) {
            if (active && cameraEpoch == cameraGeneration.get()) r.acquireLatestImage()?.let { img -> onFrame(img) }
        } }, camHandler)
        if (ActivityCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) return
        mgr.openCamera(id, object : CameraDevice.StateCallback() {
            override fun onOpened(cam: CameraDevice) {
                synchronized(cameraLock) {
                if (!active || cameraEpoch != cameraGeneration.get()) { cam.close(); return }
                camera = cam
                val st = preview.surfaceTexture!!; st.setDefaultBufferSize(captureSize.width, captureSize.height); val ps = Surface(st)
                val req = cam.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW).apply {
                    addTarget(ps); addTarget(captureReader.surface)
                    fpsRange?.let { set(CaptureRequest.CONTROL_AE_TARGET_FPS_RANGE, it) }
                    set(CaptureRequest.CONTROL_AF_MODE, CaptureRequest.CONTROL_AF_MODE_OFF); set(CaptureRequest.LENS_FOCUS_DISTANCE, 0.4f)
                    set(CaptureRequest.CONTROL_VIDEO_STABILIZATION_MODE, CaptureRequest.CONTROL_VIDEO_STABILIZATION_MODE_OFF)
                    set(CaptureRequest.LENS_OPTICAL_STABILIZATION_MODE, CaptureRequest.LENS_OPTICAL_STABILIZATION_MODE_OFF) }
                @Suppress("DEPRECATION") cam.createCaptureSession(listOf(ps, captureReader.surface), object : CameraCaptureSession.StateCallback() {
                    override fun onConfigured(s: CameraCaptureSession) { synchronized(cameraLock) {
                        if (!active || cameraEpoch != cameraGeneration.get() || camera !== cam) { s.close(); return }
                        session = s
                        s.setRepeatingRequest(req.build(), object : CameraCaptureSession.CaptureCallback() {
                            private var loggedAt = 0L
                            override fun onCaptureCompleted(session: CameraCaptureSession, request: CaptureRequest, result: TotalCaptureResult) {
                                val timestamp = result.get(CaptureResult.SENSOR_TIMESTAMP) ?: return
                                if (timestamp - loggedAt < 1_000_000_000L) return
                                loggedAt = timestamp
                                Log.i("livescan", "camera t=$timestamp exposure_ns=${result.get(CaptureResult.SENSOR_EXPOSURE_TIME)} frame_ns=${result.get(CaptureResult.SENSOR_FRAME_DURATION)} rolling_skew_ns=${result.get(CaptureResult.SENSOR_ROLLING_SHUTTER_SKEW)} fps=$fpsRange")
                            }
                        }, camHandler)
                    } }
                    override fun onConfigureFailed(s: CameraCaptureSession) { ui.post { status.text = "camera session failed" } } }, camHandler)
                }
            }
            override fun onDisconnected(cam: CameraDevice) { synchronized(cameraLock) {
                if (camera === cam) camera = null
                cam.close()
            } }
            override fun onError(cam: CameraDevice, error: Int) { synchronized(cameraLock) {
                if (camera === cam) camera = null
                cam.close(); ui.post { status.text = "camera error $error" }
            } }
        }, camHandler)
    }

    private fun onFrame(img: Image) {
        if (!active) { img.close(); return }
        val captureStart = System.nanoTime()
        try {
        val frameGeneration = generation.get()
        val t = img.timestamp * 1e-9; val rawWidth = img.width; val rawHeight = img.height
        cameraFrameIndex++
        val previousArrival = lastCameraArrival; lastCameraArrival = t
        if (previousArrival.isFinite() && t - previousArrival in 0.005..0.5) cameraIntervalMsAvg = 0.9 * cameraIntervalMsAvg + 0.1 * (t - previousArrival) * 1000
        if (vioDecimation > 1 && cameraFrameIndex % vioDecimation != 0L) { skippedFrames++; return }
        val w = streamSize.width; val h = streamSize.height
        val crop = trackingCrop(CaptureSize(rawWidth, rawHeight), CaptureSize(w, h))
        if (crop == null || crop != imageSampler?.crop) {
            timingError = "Camera delivered unexpected geometry ${rawWidth}x${rawHeight}"; return
        }
        val yp = img.planes[0]; val yb = yp.buffer; val gray = imageSampler!!.sample(yb, yp.rowStride, yp.pixelStride)
        // RGB uses exactly the same crop as VIO, never the uncropped capture field of view.
        val dw = 322; val dh = 182; val rgba = ByteArray(dw * dh * 4)
        val up = img.planes[1]; val vp = img.planes[2]; val ub = up.buffer; val vb = vp.buffer; val cs = up.pixelStride; val crs = up.rowStride
        val vps = vp.pixelStride; val vrs = vp.rowStride; val yps = yp.pixelStride; val yrs = yp.rowStride
        val sxs = rgbColumns?.takeIf { it.first == crop }?.second ?: IntArray(dw) { x -> crop.left + pixelCenterIndex(x, crop.width, dw) }.also { rgbColumns = crop to it }
        val lastColumn = sxs.last()
        val yRow = ByteArray(lastColumn * yps + 1); val uRow = ByteArray((lastColumn / 2) * cs + 1); val vRow = ByteArray((lastColumn / 2) * vps + 1)
        val yd = yb.duplicate(); val ud = ub.duplicate(); val vd = vb.duplicate()
        val yo = yb.position(); val uo = ub.position(); val vo = vb.position()
        for (y in 0 until dh) { val sy = crop.top + pixelCenterIndex(y, crop.height, dh); val suy = sy / 2
            yd.position(yo + sy * yrs); yd.get(yRow, 0, yRow.size)
            ud.position(uo + suy * crs); ud.get(uRow, 0, uRow.size)
            vd.position(vo + suy * vrs); vd.get(vRow, 0, vRow.size)
            for (x in 0 until dw) { val sx = sxs[x]; val Y = yRow[sx * yps].toInt() and 0xff
                val U = (uRow[(sx / 2) * cs].toInt() and 0xff) - 128; val V = (vRow[(sx / 2) * vps].toInt() and 0xff) - 128
                val o = (y * dw + x) * 4
                rgba[o] = (Y + 1.402f * V).toInt().coerceIn(0, 255).toByte(); rgba[o + 1] = (Y - 0.344f * U - 0.714f * V).toInt().coerceIn(0, 255).toByte(); rgba[o + 2] = (Y + 1.772f * U).toInt().coerceIn(0, 255).toByte(); rgba[o + 3] = -1 } }
        if (!active || frameGeneration != generation.get()) return
        val fd = FrameData(t, w, h, gray, rgba, dw, dh, frameGeneration); frames++
        inputs.addFrame(fd.t, fd); scheduleVio()
        captureMsAvg = 0.9 * captureMsAvg + 0.1 * (System.nanoTime() - captureStart) / 1e6
        if (frames % 120 == 0) Log.i("livescan", "capture preprocessing=${"%.1f".format(captureMsAvg)}ms vio=${"%.1f".format(vioMsAvg)}ms camera=${"%.1f".format(cameraIntervalMsAvg)}ms decimation=$vioDecimation skipped=$skippedFrames dropped=${inputs.droppedFrames}")
        } catch (e: Exception) {
            timingError = "Camera preprocessing failed: ${e.message}"; Log.e("livescan", "camera preprocessing", e)
        } finally { img.close() }
    }

    private fun requestDescriptors(fd: FrameData, pose: FloatArray, lm: FloatArray, calibration: DoubleArray) {
        if (xfeat == null || fd.t - lastDescriptorAt < if (mapAligned) 1.0 else 0.4) return
        val permit = trackingHealth.permit(fd.t) ?: return
        lastDescriptorAt = fd.t
        val snapshot = DescriptorFrame(fd, pose.copyOf(), lm.copyOf(), calibration.copyOf(), permit)
        if (stagedFrames.offer(snapshot)) xfeatHandler.post {
            while (true) {
                val next = stagedFrames.poll() ?: break
                describeFrame(next)
            }
        }
    }

    private fun describeFrame(snapshot: DescriptorFrame) {
        val fd = snapshot.image
        if (!active || fd.generation != generation.get()) return
        val detector = xfeat ?: return
        try {
            val detection = detector.detect(fd.gray, fd.w, fd.h, true)
            if (!active || fd.generation != generation.get()) return
            ui.post {
                if (active && fd.generation == generation.get()) featureOverlay.submit(
                    fd.rgba, fd.rw, fd.rh, fd.w, fd.h, detection.candidates, fd.t, featureRotation)
            }
            sessionGate.read {
                if (!active || handle == 0L || fd.generation != nativeGeneration || fd.generation != generation.get()) return@read
                trackingHealth.commit(snapshot.permit, SystemClock.elapsedRealtimeNanos() * 1e-9) {
                    // The pixels, local pose, landmarks and camera model all come
                    // from the same frame, even if VIO is already several frames ahead.
                    Native.mapFrame(handle, fd.t, true, true, snapshot.localPose, snapshot.landmarks,
                        snapshot.landmarks.size / 3, detection.candidates, detection.descriptorMap,
                        false, snapshot.calibration)
                }
            }
        } catch (e: Exception) {
            Log.e("livescan", "Map descriptors unavailable; native tracking continues", e)
        }
    }

    private fun runVio(fd: FrameData) {
        val pose = FloatArray(7); val calibrationValues = DoubleArray(11); val diagnosticValues = DoubleArray(70); val t0 = System.nanoTime()
        val n = Native.feedImage(handle, fd.t, fd.w, fd.h, fd.gray, pose, landmarks, calibrationValues, diagnosticValues)
        vioMsAvg = 0.9 * vioMsAvg + 0.1 * (System.nanoTime() - t0) / 1e6
        updateDecimation()
        val d = VioDiagnostics.fromNative(diagnosticValues); diagnostics = d
        val health = trackingHealth.update(fd.t, if (n > 0) pose else null, d)
        startupProgress.observe(health.poseReliable, SystemClock.elapsedRealtimeNanos()*1e-9)?.let {
            Log.i("livescan", "startup first reliable pose elapsed=${it}s health=${health.state} fusion=${health.fusionAllowed}")
        }
        warmStart.observe(fd.t, health.state == TrackingHealth.State.STATIONARY,
            diagnosticValues[25] == 1.0, d?.speed ?: Double.NaN) { Native.biasPrior(handle) }
        recorder?.frame(fd.t, health, if (n > 0) pose else null, d, cameraImuOffset, calibrationValues, diagnosticValues.copyOfRange(10, 70))
        val featureSamples = Native.featureSamples(handle)
        recorder?.featureSamples(fd.t, featureSamples)
        if (n > 0 && health.poseReliable) requestDescriptors(fd, pose, landmarks.copyOf((n - 1) * 3), calibrationValues)
        val map = Native.mapFrame(handle, fd.t, health.poseReliable, n > 0, pose, landmarks,
            maxOf(0, n - 1), null, null,
            health.state == TrackingHealth.State.RESTARTING)   // a discontinuity must reacquire the saved map frame
        val aligned = map[0] == 1.0
        if (aligned != mapAligned) { mapAligned = aligned; mapRevision.incrementAndGet() }
        val accurate = n > 0 && health.poseReliable && mapAligned
        poseTransformer?.offer(fd.t, fd.gray, fd.w, fd.h, featureSamples, calibrationValues,
            cameraImuOffset, if (accurate) pose else null)
        mapNotice = if (aligned) "Map aligned • ${map[1].toInt()} saved views" else
            if (map[1] == 0.0) "No saved reference: use RESET to start a new map" else
            "Cloud held: revisit a saved view and move sideways • ${map[2].toInt()} matches • ${map[3].toInt()} confirmations"
        if (health.state != lastHealthState || fd.t - lastHealthLog >= 1.0) {
            Log.i("livescan", "map aligned=$mapAligned references=${map[1].toInt()} matches=${map[2].toInt()} confirmations=${map[3].toInt()} depth_frames=$depthFrames pause=$depthPauseReason")
            Log.i("livescan", "health=${health.state} reason=${health.reason} t=${fd.t} tracks=${d?.tracks} msckf=${d?.msckfAccepted}/${d?.msckfAttempted} slam=${d?.slamAccepted}/${d?.slamAttempted} zupt=${d?.stationary} sigma=${d?.positionSigma} speed=${d?.speed} fusion=${health.fusionAllowed}")
            lastHealthLog = fd.t; lastHealthState = health.state
        }
        if (!health.fusionAllowed) depthPauseReason = health.reason
        else if (!mapAligned) depthPauseReason = "matching a saved view"
        recoverIfNeeded(health)
        if (!accurate) { publishProvisionalPose(fd.t); return }
        val calibration = CameraCalibration.fromNative(calibrationValues)
        if (initTime < 0) initTime = fd.t
        landmarkCount = n - 1; publishPose(pose, fd.t, true, "VIO + map")
        // A five-centimetre step can supply the independent view required by
        // confidence (4 cm), while keeping the single depth worker bounded.
        val lp = lastDepthPose; val moved = lp == null || dist(lp, pose) > 0.05f || quatAngle(lp, pose) > 0.087f || fd.t - lastDepthAt > 0.7
        // A recovery reference is useful for a future loss, not a prerequisite
        // for observing the first depth in this already-aligned local map.
        val permit = trackingHealth.permit(fd.t)
        if (moved && permit != null && depth != null && depthBusy.compareAndSet(false, true)) { lastDepthAt = fd.t; lastDepthPose = pose.copyOf(); val lm = landmarks.copyOf(landmarkCount * 3); val revision = mapRevision.get(); depthHandler.post { runDepth(fd, pose.copyOf(), lm, calibration, permit, revision) } }
    }

    /** Smallest k with k * camera interval > VIO time plus margin; step down only with slack. */
    private fun updateDecimation() {
        val interval = cameraIntervalMsAvg.coerceIn(10.0, 200.0)
        var wanted = 1; while (wanted < 4 && wanted * interval < vioMsAvg + 6.0) wanted++
        val current = vioDecimation
        val next = if (wanted > current) wanted else if (wanted < current && (current - 1) * interval > vioMsAvg + 12.0) current - 1 else current
        if (next != current) { vioDecimation = next; Log.i("livescan", "vio decimation ${current} -> ${next} (vio ${"%.1f".format(vioMsAvg)} ms, camera ${"%.1f".format(interval)} ms)") }
    }

    private fun publishProvisionalPose(now: Double) {
        val estimate = poseTransformer?.estimate() ?: return
        if (now - estimate.timestamp !in -0.1..0.75 || estimate.timestamp <= lastPoseTime) return
        publishPose(estimate.pose, estimate.timestamp, false,
            "transformer provisional ±${"%.2f".format(estimate.sigmaMeters)} m, ${estimate.features} tracks, ${"%.0f".format(estimate.inferenceMs)} ms")
    }

    private fun publishPose(pose: FloatArray, timestamp: Double, recordPath: Boolean, source: String) {
        lastPose = pose.copyOf(); lastPoseTime = timestamp; poseSource = source
        explorationView.position(pose[0], pose[2])
        povRenderer.pose = pose; orbitRenderer.pose = pose
        if (recordPath) synchronized(pathPts) {
            if (pathPts.size < 3 || abs(pathPts[pathPts.size - 3] - pose[0]) + abs(pathPts[pathPts.size - 2] - pose[1]) + abs(pathPts[pathPts.size - 1] - pose[2]) > 0.02f) {
                if (pathPts.size >= 60_000) {
                    val reduced = ArrayList<Float>(pathPts.size / 2)
                    for (i in 0 until pathPts.size step 6) for (axis in 0..2) reduced.add(pathPts[i + axis])
                    pathPts.clear(); pathPts.addAll(reduced)
                }
                pathPts.add(pose[0]); pathPts.add(pose[1]); pathPts.add(pose[2]); orbitRenderer.path = pathPts.toFloatArray()
            }
        }
        if (!orbitTargetSet) {
            orbitRenderer.target[0] = pose[0]; orbitRenderer.target[1] = pose[1]; orbitRenderer.target[2] = pose[2]; orbitTargetSet = true
            val r0 = quatToRot(pose[3], pose[4], pose[5], pose[6]); val fx0 = -r0[2]; val fz0 = -r0[8]
            orbitRenderer.yaw = kotlin.math.atan2(-fx0, -fz0); orbitRenderer.pitch = 0.6f; orbitRenderer.dist = 4f
        }
        povView.requestRender(); orbitView.requestRender()
    }
    private var orbitTargetSet = false; private var modelIndex = 0
    private fun dist(a: FloatArray, b: FloatArray) = sqrt((a[0] - b[0]) * (a[0] - b[0]) + (a[1] - b[1]) * (a[1] - b[1]) + (a[2] - b[2]) * (a[2] - b[2]))
    private fun quatAngle(a: FloatArray, b: FloatArray): Float { val d = abs(a[3] * b[3] + a[4] * b[4] + a[5] * b[5] + a[6] * b[6]).coerceAtMost(1f); return 2f * kotlin.math.acos(d) }

    private val mapRevision = AtomicLong(0)
    @Volatile private var mapAligned = true
    @Volatile private var mapNotice = ""
    private var lastDescriptorAt = Double.NEGATIVE_INFINITY
    private var lastRecoveryAt = Double.NEGATIVE_INFINITY
    private val depthQuality = DepthQuality()
    private var depthReferenceKey: Triple<Long, Long, String>? = null
    @Volatile private var depthAttempts = 0
    @Volatile private var supportedDepthPixels = 0
    @Volatile private var depthEvidence: DepthEvidenceRecorder? = null

    private fun runDepth(fd: FrameData, pose: FloatArray, lm: FloatArray, calibration: CameraCalibration, permit: TrackingHealth.Permit, revision: Long) {
        sessionGate.read {
        val evidenceWriter = depthEvidence
        var evidence: DepthEvidenceRecorder.Frame? = null
        try {
            if (!active || fd.generation != generation.get() || fd.generation != nativeGeneration || handle == 0L || !mapAligned || revision != mapRevision.get()) return
            if (!trackingHealth.commit(permit, SystemClock.elapsedRealtimeNanos() * 1e-9) {}) return
            val dr = depth ?: return
            depthAttempts++
            supportedDepthPixels = 0
            // Every supported depth family requires at least eight same-frame
            // landmarks. Avoid running a costly network on an impossible fit,
            // especially during stationary startup and tracking recovery.
            if (lm.size / 3 < 8) {
                landmarksUsed = lm.size / 3
                depthPauseReason = "need at least 8 tracked depth anchors"
                return
            }
            val selectedBefore = dr.name
            val prediction = dr.run(fd.rgba, fd.rw, fd.rh, calibration); depthMs = prediction.ms
            if (dr.name != selectedBefore) {
                modelIndex = dr.available.indexOf(dr.spec).coerceAtLeast(0)
                getPreferences(MODE_PRIVATE).edit().putString("depth_model",dr.spec?.file).apply()
                recoveryNotice = "Using ${dr.label} after acceleration failed"
                ui.post { findViewById<Button>(com.recon.videoscan.R.id.model).text = dr.label }
            }
            val referenceKey = Triple(fd.generation, revision, dr.name)
            if (referenceKey != depthReferenceKey) {
                depthQuality.reset(); depthReferenceKey = referenceKey
            }
            if (!active || fd.generation != generation.get()) return
            val output = prediction.frame; val mw = output.camera.width; val mh = output.camera.height
            val k = output.camera.intrinsics()
            val fx = k[0]; val fy = k[1]; val cx = k[2]; val cy = k[3]; val rel = prediction.depth
            val cam = calibration.camera
            evidence = DepthEvidenceRecorder.Frame(fd.t, fd.generation, revision, "${dr.name}:${dr.kind}", "processing", false,
                mw, mh, fd.w, fd.h, pose.copyOf(), k.copyOf(),
                doubleArrayOf(cam.fx, cam.fy, cam.cx, cam.cy, calibration.d0, calibration.d1, calibration.d2, calibration.d3, cam.width.toDouble(), cam.height.toDouble(), calibration.model.ordinal.toDouble()),
                lm.copyOf(), rel, ShortArray(0), ByteArray(0), output.rgba, fd.gray)
            // world<-camera rotation: pose quaternion is R_gl (camera looks -Z); OpenCV axes = R_gl * diag(1,-1,-1)
            val q = pose; val r = quatToRot(q[3], q[4], q[5], q[6]); val rcv = floatArrayOf(r[0], -r[1], -r[2], r[3], -r[4], -r[5], r[6], -r[7], -r[8])
            // Calibrate relative depth against VIO landmarks from this exact frame.
            val zs = ArrayList<Float>(); val ds = ArrayList<Float>(); val anchorCells = ArrayList<Int>()
            var i = 0; while (i + 2 < lm.size) { val px = lm[i] - q[0]; val py = lm[i + 1] - q[1]; val pz = lm[i + 2] - q[2]
                val zc = rcv[2] * px + rcv[5] * py + rcv[8] * pz
                if (zc > 0.2f && zc < 8f) { val xc = rcv[0] * px + rcv[3] * py + rcv[6] * pz; val yc = rcv[1] * px + rcv[4] * py + rcv[7] * pz
                    val uf = fx * xc / zc + cx; val vf = fy * yc / zc + cy
                    if (uf >= 0 && uf <= mw - 1 && vf >= 0 && vf <= mh - 1) {
                        val d = rel[vf.roundToInt() * mw + uf.roundToInt()]
                        if (d.isFinite() && d > 1e-4f) { zs.add(zc); ds.add(d); anchorCells.add((uf * 4 / mw).toInt().coerceIn(0, 3) + 4 * (vf * 3 / mh).toInt().coerceIn(0, 2)) }
                    } }
                i += 3 }
            landmarksUsed = zs.size
            // Metric models also have scene-dependent scale bias. Fit their one
            // scale parameter to the same geometry instead of accepting raw scale.
            val calibrationFit = depthQuality.calibrate(ds, zs, anchorCells, dr.kind == DepthRunner.Kind.RELATIVE_INVERSE)
            val fit = calibrationFit.fit
            if (fit == null) { depthPauseReason = calibrationFit.reason; return }
            fitA = fit.a; fitB = fit.b; scaleUsed = fitA
            val mm = ShortArray(rel.size) { index ->
                val z = fit.depth(rel[index])
                if (rel[index].isFinite() && z.isFinite() && z in .2f..6f) (z * 1000f).roundToInt().toShort() else 0
            }
            val centre = floatArrayOf(q[0], q[1], q[2])
            val confidence = depthQuality.confidence(mm, mw, mh, k, rcv, centre, fd.t)
            supportedDepthPixels = confidence.count { (it.toInt() and 255) == 255 }
            evidence = evidence.copy(depth = mm, confidence = confidence)
            var integrated = false
            val commitNow = SystemClock.elapsedRealtimeNanos() * 1e-9
            val maxDepthAge = (depthMs / 1000.0 + .75).coerceIn(1.5, 3.0)
            val committed = trackingHealth.commit(permit, commitNow, maxDepthAge) {
            if (!active || fd.generation != generation.get() || !mapAligned || revision != mapRevision.get()) {
                depthPauseReason = "map alignment changed while depth was processing"; return@commit
            }
            if (fusion) {
                tsdfStats = Native.tsdfIntegrate(handle, mm, mw, mh, k, rcv, centre, output.rgba, mw, mh, if (mw >= 300) 2 else 1, confidence)
                if (tsdfStats[5] != 0f) {
                    depthPauseReason = if (tsdfStats[5] == 1f) "Map memory limit reached; save scan and start a new segment" else "Invalid depth geometry rejected"
                    return@commit
                }
            } else {
                // raw back-projection of this frame appended to the cloud (comparison mode)
                val pts = ArrayList<Float>(); var y = 1
                while (y < mh) { var x = 1; while (x < mw) { val z = mm[y * mw + x].toInt() and 0xffff; if (z in 1..6000) { val zf = z / 1000f
                    val X = (x - cx) / fx * zf; val Y = (y - cy) / fy * zf
                    pts.add(q[0] + rcv[0] * X + rcv[1] * Y + rcv[2] * zf); pts.add(q[1] + rcv[3] * X + rcv[4] * Y + rcv[5] * zf); pts.add(q[2] + rcv[6] * X + rcv[7] * Y + rcv[8] * zf)
                    val o = (y * mw + x) * 4; pts.add((output.rgba[o].toInt() and 0xff) / 255f); pts.add((output.rgba[o + 1].toInt() and 0xff) / 255f); pts.add((output.rgba[o + 2].toInt() and 0xff) / 255f) }; x += 3 }; y += 3 }
                val remaining = (300_000 * 6 - rawCloud.size).coerceAtLeast(0)
                rawCloud.addAll(pts.take(remaining))
            }
            integrated = true
            }
            if (!committed || !integrated) {
                if (!committed) depthPauseReason = if (commitNow - permit.timestamp > maxDepthAge)
                    "depth processing too slow; skipped this frame" else "tracking changed while depth was processing"
                return
            }
            // Rendering and exploration are not estimator mutations. Keep their
            // map-wide work outside the health lock so VIO can process the next image.
            if (fusion) publish(Native.tsdfSurface(handle, 100_000)) else publish(rawCloud.toFloatArray())
            depthFrames++
            evidence = evidence?.copy(fused = fusion, reason = if (fusion) "integrated" else "raw comparison")
            depthQuality.remember(mm, mw, mh, k, rcv, centre, fd.t, confidence)
            exploration.visit(q[0],q[2])
            exploration.observe(mm,confidence,mw,mh,k,rcv,centre)
            explorationView.submit(exploration.snapshot())
            depthPauseReason = if (supportedDepthPixels == 0) "depth provisional; move sideways for another view" else "none"
            if (depthFrames % 20 == 1) Log.i("livescan", "calibrated depth frame=$depthFrames t=${fd.t} model=${dr.name} ${mw}x${mh} K=${k.contentToString()} sourceK=${calibration.camera} distortion=${calibration.model} anchors=$landmarksUsed points=${tsdfStats[4].toInt()} ms=$depthMs")
        } catch (e: Throwable) { depthPauseReason = "depth error: ${e.message}"; Log.e("livescan", "depth: $e", e) }
        finally {
            if (evidence != null) Log.i("livescan", "depth result t=${fd.t} attempt=$depthAttempts integrated=${evidence?.fused} supported=$supportedDepthPixels anchors=$landmarksUsed reason=$depthPauseReason")
            evidence?.let { evidenceWriter?.offer(if (it.reason != "processing") it else it.copy(reason = depthPauseReason)) }
            depthBusy.set(false)
        }
        }
    }
    private val rawCloud = ArrayList<Float>(); private var fitA = 0f; private var fitB = 0f; @Volatile private var landmarksUsed = 0
    private fun publish(surf: FloatArray) {
        val buf = ByteBuffer.allocateDirect(surf.size * 4).order(ByteOrder.nativeOrder()).asFloatBuffer().put(surf); buf.position(0)
        povRenderer.points = buf.asReadOnlyBuffer(); orbitRenderer.points = buf.asReadOnlyBuffer()
        povView.requestRender(); orbitView.requestRender()
    }
    private fun quatToRot(x: Float, y: Float, z: Float, w: Float) = floatArrayOf(   // row-major 3x3
        1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
        2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y))

    private fun updateStatus() {
        timingError?.let { status.text = it; return }
        val now = SystemClock.elapsedRealtimeNanos() * 1e-9
        val health = trackingHealth.current(now)
        recoverIfNeeded(health)
        if (!health.poseReliable || !mapAligned) publishProvisionalPose(now)
        val d = diagnostics
        val provisional = poseSource.startsWith("transformer") && now - lastPoseTime < 1.0
        val mapState = if (provisional) "ESTIMATING" else if (health.poseReliable && !mapAligned) "RELOCALIZING" else health.state.toString()
        explorationView.tracking = health.poseReliable && mapAligned
        val trackingText = if (provisional) "transformer motion; finding accurate saved-view pose in background" else
            if (!mapAligned) "waiting for map alignment" else health.reason
        val banner = startupProgress.message(now) +
            "$mapState: $trackingText\n" +
            "tracks ${d?.tracks ?: 0} accepted ${d?.accepted ?: 0}/${d?.attempted ?: 0} sigma ${"%.2f".format(d?.positionSigma ?: 0.0)} m | fusion ${if (health.fusionAllowed && mapAligned) depthPauseReason else "PAUSED"}\n"
        val p = lastPose
        val details = if (p == null) "frames $frames   imu $imuCount   vio ${"%.0f".format(vioMsAvg)} ms (1/$vioDecimation frames)   depth ${depth?.backend ?: "loading"}"
        else "pose ${"%.2f %.2f %.2f".format(p[0], p[1], p[2])} [$poseSource]\nlandmarks $landmarkCount (in frame $landmarksUsed)   vio ${"%.0f".format(vioMsAvg)} ms (1/$vioDecimation frames)\n${depth?.name} ${depth?.backend} $depthMs ms   scale ${"%.2f".format(scaleUsed)}\n" +
            "depth attempts $depthAttempts integrated $depthFrames supported pixels $supportedDepthPixels\n" +
            "tsdf integrate ${"%.0f".format(tsdfStats[0])} ms extract ${"%.0f".format(tsdfStats[1])} ms   blocks ${tsdfStats[3].toInt()}   points ${(tsdfStats[4] / 1000).toInt()} k\n" +
            "camera–IMU offset ${"%.1f".format(cameraImuOffset * 1000)} ms   camera frames dropped ${inputs.droppedFrames}"
        status.text = banner + details + "\n$mapNotice\n$recoveryNotice\nlogs: $recordingStatus (${recorder?.state ?: "off"}); depth evidence ${depthEvidence?.state ?: "off/quota"}"
    }
    private fun recoverIfNeeded(health: TrackingHealth.Status) {
        val now = SystemClock.elapsedRealtimeNanos() * 1e-9
        if (active && timingError == null && health.state == TrackingHealth.State.RESTARTING && now - lastRecoveryAt >= 3.0 && recoveryPending.compareAndSet(false, true)) {
            lastRecoveryAt = now
            recoveryNotice = "Cloud retained; restarting tracking, then matching saved views"
            Log.i("livescan", "automatic recovery requested: ${health.reason}")
            resetMap(recovering = true)
        }
    }
}
