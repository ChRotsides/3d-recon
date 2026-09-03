package com.recon.videoscan

import android.Manifest
import android.annotation.SuppressLint
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Matrix
import android.graphics.SurfaceTexture
import android.hardware.camera2.CameraCaptureSession
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraDevice
import android.hardware.camera2.CameraManager
import android.hardware.camera2.CameraMetadata
import android.hardware.camera2.CaptureRequest
import android.hardware.camera2.CaptureResult
import android.hardware.camera2.TotalCaptureResult
import android.hardware.camera2.params.OutputConfiguration
import android.hardware.camera2.params.SessionConfiguration
import android.media.MediaRecorder
import android.os.Build
import android.os.Handler
import android.os.HandlerThread
import android.os.SystemClock
import android.util.Log
import android.util.Range
import android.util.Size
import android.view.Surface
import android.view.TextureView
import androidx.core.content.ContextCompat
import java.io.BufferedWriter
import java.io.File
import java.io.FileWriter
import java.util.concurrent.Executor
import kotlin.math.abs

/**
 * Camera2 video recording with a per-frame sensor-timestamp log.
 *
 * Deliberate choices, all of them for the reconstruction rather than the video:
 *  - **No stabilisation.** OIS moves the lens and EIS warps the frame; both make
 *    the intrinsics lie. Both are switched off where the device allows.
 *  - **Fixed focus.** Focus breathing changes the focal length; a hyperfocal
 *    setting keeps a room sharp from ~1.2 m out with one focal length.
 *  - **Sensor-native orientation.** No orientation hint goes into the MP4, so
 *    the frames come out exactly as the sensor saw them and the intrinsics
 *    apply unchanged.
 *  - **Every frame is logged** from the capture result, not the encoder: the
 *    sensor timestamp is the clock the IMU uses, so a frame can be paired with
 *    its orientation to within a sample.
 */
class CameraRecorder(private val context: Context, private val textureView: TextureView) {
    interface Listener {
        fun onStatus(text: String)
    }

    var listener: Listener? = null

    private val manager = context.getSystemService(Context.CAMERA_SERVICE) as CameraManager
    private val thread = HandlerThread("camera").apply { start() }
    private val handler = Handler(thread.looper)
    private val executor = Executor { handler.post(it) }

    val cameraId: String
    val characteristics: CameraCharacteristics
    val videoSize: Size
    private val previewSize: Size
    private val fpsRange: Range<Int>

    private var device: CameraDevice? = null
    private var session: CameraCaptureSession? = null
    private var previewSurface: Surface? = null
    private var recorder: MediaRecorder? = null
    private var recorderSurface: Surface? = null
    private var framesLog: BufferedWriter? = null
    private var recordingDir: File? = null
    private var startElapsedNs = 0L
    private var lastTimestampNs = 0L

    @Volatile var isRecording = false
        private set

    @Volatile var frameCount = 0L
        private set

    init {
        cameraId = pickBackCamera()
        characteristics = manager.getCameraCharacteristics(cameraId)
        val map = characteristics.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP)!!
        videoSize = pickVideoSize(map.getOutputSizes(MediaRecorder::class.java).toList())
        previewSize = pickPreviewSize(map.getOutputSizes(SurfaceTexture::class.java).toList(), videoSize)
        fpsRange = pickFpsRange(characteristics.get(CameraCharacteristics.CONTROL_AE_AVAILABLE_TARGET_FPS_RANGES))
        Log.i(TAG, "camera $cameraId video $videoSize preview $previewSize fps $fpsRange " +
            "orientation ${sensorOrientation()} clock ${timestampSource()}")
    }

    private fun status(text: String) {
        listener?.onStatus(text)
    }

    // -- lifecycle ------------------------------------------------------------

    fun open() {
        if (textureView.isAvailable) {
            openCamera()
        } else {
            textureView.surfaceTextureListener = object : TextureView.SurfaceTextureListener {
                override fun onSurfaceTextureAvailable(st: SurfaceTexture, w: Int, h: Int) = openCamera()
                override fun onSurfaceTextureSizeChanged(st: SurfaceTexture, w: Int, h: Int) = configureTransform(w, h)
                override fun onSurfaceTextureDestroyed(st: SurfaceTexture) = true
                override fun onSurfaceTextureUpdated(st: SurfaceTexture) = Unit
            }
        }
    }

    fun close() {
        if (isRecording) stopRecording()
        handler.post {
            session?.close(); session = null
            device?.close(); device = null
            previewSurface?.release(); previewSurface = null
        }
        thread.quitSafely()
    }

    @SuppressLint("MissingPermission")
    private fun openCamera() {
        if (ContextCompat.checkSelfPermission(context, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            status("camera permission missing")
            return
        }
        val texture = textureView.surfaceTexture ?: return
        texture.setDefaultBufferSize(previewSize.width, previewSize.height)
        previewSurface = Surface(texture)
        configureTransform(textureView.width, textureView.height)
        manager.openCamera(cameraId, object : CameraDevice.StateCallback() {
            override fun onOpened(camera: CameraDevice) {
                device = camera
                startPreviewSession()
            }
            override fun onDisconnected(camera: CameraDevice) { camera.close(); device = null }
            override fun onError(camera: CameraDevice, error: Int) {
                camera.close(); device = null
                status("camera error $error")
            }
        }, handler)
    }

    private fun startPreviewSession() {
        val cam = device ?: return
        val surface = previewSurface ?: return
        createSession(cam, listOf(surface)) { s ->
            session = s
            val request = cam.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW).apply {
                addTarget(surface)
                applyGeometryLocks(this)
            }
            s.setRepeatingRequest(request.build(), null, handler)
            status(context.getString(R.string.ready))
        }
    }

    private fun createSession(cam: CameraDevice, surfaces: List<Surface>, onReady: (CameraCaptureSession) -> Unit) {
        val configs = surfaces.map { OutputConfiguration(it) }
        val config = SessionConfiguration(
            SessionConfiguration.SESSION_REGULAR, configs, executor,
            object : CameraCaptureSession.StateCallback() {
                override fun onConfigured(s: CameraCaptureSession) = onReady(s)
                override fun onConfigureFailed(s: CameraCaptureSession) = status("capture session failed")
            },
        )
        cam.createCaptureSession(config)
    }

    // -- recording --------------------------------------------------------------

    fun startRecording(dir: File) {
        val cam = device ?: run { status("camera not open"); return }
        val preview = previewSurface ?: return
        handler.post {
            dir.mkdirs()
            val rec = MediaRecorder(context).apply {
                setVideoSource(MediaRecorder.VideoSource.SURFACE)
                setOutputFormat(MediaRecorder.OutputFormat.MPEG_4)
                setOutputFile(File(dir, "video.mp4").absolutePath)
                setVideoEncodingBitRate(BIT_RATE)
                setVideoFrameRate(fpsRange.upper)
                setVideoSize(videoSize.width, videoSize.height)
                setVideoEncoder(MediaRecorder.VideoEncoder.H264)
                // No setOrientationHint: frames stay in the sensor's orientation.
                prepare()
            }
            recorder = rec
            recorderSurface = rec.surface
            framesLog = BufferedWriter(FileWriter(File(dir, "frames.csv")), 1 shl 16).apply {
                write(FrameLog.HEADER); newLine()
            }
            recordingDir = dir
            frameCount = 0
            lastTimestampNs = 0
            session?.close()
            createSession(cam, listOf(preview, rec.surface)) { s ->
                session = s
                val request = cam.createCaptureRequest(CameraDevice.TEMPLATE_RECORD).apply {
                    addTarget(preview)
                    addTarget(rec.surface)
                    applyGeometryLocks(this)
                }
                startElapsedNs = SystemClock.elapsedRealtimeNanos()
                File(dir, "meta.json").writeText(buildMeta(null, null))
                s.setRepeatingRequest(request.build(), captureCallback, handler)
                rec.start()
                isRecording = true
                status("Recording ${videoSize.width}x${videoSize.height}")
            }
        }
    }

    fun stopRecording() {
        if (!isRecording) return
        isRecording = false
        handler.post {
            try { session?.stopRepeating() } catch (e: Exception) { Log.w(TAG, "stopRepeating", e) }
            val rec = recorder
            try { rec?.stop() } catch (e: RuntimeException) { Log.w(TAG, "recorder.stop", e) }
            rec?.release()
            recorder = null
            recorderSurface?.release(); recorderSurface = null
            framesLog?.flush(); framesLog?.close(); framesLog = null
            val dir = recordingDir
            if (dir != null) {
                val duration = if (lastTimestampNs > 0) lastTimestampNs - firstTimestampNs else 0L
                File(dir, "meta.json").writeText(buildMeta(frameCount, duration))
            }
            session?.close(); session = null
            startPreviewSession()
            status("Saved ${dir?.name} ($frameCount frames)")
        }
    }

    private var firstTimestampNs = 0L

    private val captureCallback = object : CameraCaptureSession.CaptureCallback() {
        override fun onCaptureCompleted(s: CameraCaptureSession, request: CaptureRequest, result: TotalCaptureResult) {
            val log = framesLog ?: return
            val ts = result.get(CaptureResult.SENSOR_TIMESTAMP) ?: return
            if (frameCount == 0L) firstTimestampNs = ts
            lastTimestampNs = ts
            log.write(
                FrameLog.line(
                    result.frameNumber, ts,
                    result.get(CaptureResult.SENSOR_EXPOSURE_TIME),
                    result.get(CaptureResult.LENS_FOCUS_DISTANCE),
                    result.get(CaptureResult.SENSOR_SENSITIVITY),
                ),
            )
            log.newLine()
            frameCount++
        }
    }

    // -- request settings -------------------------------------------------------

    private fun applyGeometryLocks(b: CaptureRequest.Builder) {
        b.set(CaptureRequest.CONTROL_MODE, CameraMetadata.CONTROL_MODE_AUTO)
        b.set(CaptureRequest.CONTROL_AE_TARGET_FPS_RANGE, fpsRange)
        b.set(CaptureRequest.CONTROL_VIDEO_STABILIZATION_MODE, CameraMetadata.CONTROL_VIDEO_STABILIZATION_MODE_OFF)
        val ois = characteristics.get(CameraCharacteristics.LENS_INFO_AVAILABLE_OPTICAL_STABILIZATION)
        if (ois != null && ois.contains(CameraMetadata.LENS_OPTICAL_STABILIZATION_MODE_OFF)) {
            b.set(CaptureRequest.LENS_OPTICAL_STABILIZATION_MODE, CameraMetadata.LENS_OPTICAL_STABILIZATION_MODE_OFF)
        }
        b.set(CaptureRequest.CONTROL_ZOOM_RATIO, 1.0f)
        val focus = fixedFocusDiopters()
        if (focus != null) {
            b.set(CaptureRequest.CONTROL_AF_MODE, CameraMetadata.CONTROL_AF_MODE_OFF)
            b.set(CaptureRequest.LENS_FOCUS_DISTANCE, focus)
        } else {
            b.set(CaptureRequest.CONTROL_AF_MODE, CameraMetadata.CONTROL_AF_MODE_CONTINUOUS_VIDEO)
        }
    }

    /** Hyperfocal-ish focus in diopters when the lens can be driven; null on fixed-focus hardware. */
    private fun fixedFocusDiopters(): Float? {
        val min = characteristics.get(CameraCharacteristics.LENS_INFO_MINIMUM_FOCUS_DISTANCE) ?: return null
        if (min <= 0f) return null
        val modes = characteristics.get(CameraCharacteristics.CONTROL_AF_AVAILABLE_MODES) ?: return null
        if (!modes.contains(CameraMetadata.CONTROL_AF_MODE_OFF)) return null
        return FOCUS_DIOPTERS.coerceAtMost(min)
    }

    // -- metadata -----------------------------------------------------------------

    fun sensorOrientation(): Int = characteristics.get(CameraCharacteristics.SENSOR_ORIENTATION) ?: 0

    fun timestampSource(): String =
        when (characteristics.get(CameraCharacteristics.SENSOR_INFO_TIMESTAMP_SOURCE)) {
            CameraMetadata.SENSOR_INFO_TIMESTAMP_SOURCE_REALTIME -> "REALTIME"
            CameraMetadata.SENSOR_INFO_TIMESTAMP_SOURCE_UNKNOWN -> "UNKNOWN"
            else -> "UNSPECIFIED"
        }

    fun hasIntrinsicCalibration(): Boolean =
        characteristics.get(CameraCharacteristics.LENS_INTRINSIC_CALIBRATION)?.let { it.size >= 4 && it[0] > 0f } == true

    private fun buildMeta(frames: Long?, durationNs: Long?): String {
        val active = characteristics.get(CameraCharacteristics.SENSOR_INFO_ACTIVE_ARRAY_SIZE)!!
        val pre = characteristics.get(CameraCharacteristics.SENSOR_INFO_PRE_CORRECTION_ACTIVE_ARRAY_SIZE)
        val physical = characteristics.get(CameraCharacteristics.SENSOR_INFO_PHYSICAL_SIZE)
        val focal = characteristics.get(CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS)?.firstOrNull() ?: 0f
        val level = when (characteristics.get(CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL)) {
            CameraMetadata.INFO_SUPPORTED_HARDWARE_LEVEL_LEGACY -> "LEGACY"
            CameraMetadata.INFO_SUPPORTED_HARDWARE_LEVEL_LIMITED -> "LIMITED"
            CameraMetadata.INFO_SUPPORTED_HARDWARE_LEVEL_FULL -> "FULL"
            CameraMetadata.INFO_SUPPORTED_HARDWARE_LEVEL_3 -> "LEVEL_3"
            else -> "UNKNOWN"
        }
        val focus = fixedFocusDiopters()
        val ois = characteristics.get(CameraCharacteristics.LENS_INFO_AVAILABLE_OPTICAL_STABILIZATION)
        val camera = RecordingMeta.Camera(
            id = cameraId,
            sensorOrientationDeg = sensorOrientation(),
            timestampSource = timestampSource(),
            activeArray = RecordingMeta.Size(active.width(), active.height()),
            preCorrectionActiveArray = pre?.let { RecordingMeta.Size(it.width(), it.height()) },
            intrinsicCalibration = characteristics.get(CameraCharacteristics.LENS_INTRINSIC_CALIBRATION),
            lensDistortionAndroid = characteristics.get(CameraCharacteristics.LENS_DISTORTION),
            focalLengthMm = focal,
            physicalSizeMm = Pair(physical?.width ?: 0f, physical?.height ?: 0f),
            hardwareLevel = level,
            focusMode = if (focus != null) "fixed" else "continuous_video",
            focusDiopters = focus,
            videoStabilization = "off",
            opticalStabilization = if (ois != null && ois.contains(CameraMetadata.LENS_OPTICAL_STABILIZATION_MODE_OFF)) "off" else "unavailable",
        )
        val device = mapOf(
            "manufacturer" to Build.MANUFACTURER,
            "model" to Build.MODEL,
            "os" to "android ${Build.VERSION.RELEASE} (sdk ${Build.VERSION.SDK_INT})",
            "app" to "videoscan 0.1",
        )
        return RecordingMeta.build(
            device, camera,
            RecordingMeta.Video(videoSize.width, videoSize.height, fpsRange.upper, BIT_RATE),
            startElapsedNs, System.currentTimeMillis(), frames, durationNs,
        )
    }

    // -- selection helpers ------------------------------------------------------

    private fun pickBackCamera(): String {
        val backs = manager.cameraIdList.filter {
            manager.getCameraCharacteristics(it).get(CameraCharacteristics.LENS_FACING) == CameraCharacteristics.LENS_FACING_BACK
        }
        require(backs.isNotEmpty()) { "no back camera" }
        // Prefer a camera that publishes its intrinsic calibration.
        return backs.firstOrNull {
            manager.getCameraCharacteristics(it).get(CameraCharacteristics.LENS_INTRINSIC_CALIBRATION)?.let { c -> c.size >= 4 && c[0] > 0f } == true
        } ?: backs.first()
    }

    private fun configureTransform(viewWidth: Int, viewHeight: Int) {
        // Portrait activity: the framework already rotates the sensor buffer for
        // a TextureView consumer, so only the aspect ratio needs correcting.
        // Scale about the centre so the preview fills the view without stretching.
        if (viewWidth == 0 || viewHeight == 0) return
        val contentAspect = previewSize.height.toFloat() / previewSize.width
        val viewAspect = viewWidth.toFloat() / viewHeight
        val matrix = Matrix()
        if (viewAspect < contentAspect) {
            matrix.setScale(contentAspect / viewAspect, 1f, viewWidth / 2f, viewHeight / 2f)
        } else {
            matrix.setScale(1f, viewAspect / contentAspect, viewWidth / 2f, viewHeight / 2f)
        }
        textureView.post { textureView.setTransform(matrix) }
    }

    companion object {
        private const val TAG = "VideoScan"
        const val BIT_RATE = 24_000_000
        /** 0.4 dioptres = focus at 2.5 m; sharp from roughly 1.2 m to infinity on a phone lens. */
        const val FOCUS_DIOPTERS = 0.4f

        fun pickVideoSize(sizes: List<Size>): Size {
            val wide = sizes.filter { abs(it.width * 9 - it.height * 16) < 16 && it.height <= 1080 }
            return wide.maxByOrNull { it.width * it.height }
                ?: sizes.filter { it.height <= 1080 }.maxByOrNull { it.width * it.height }
                ?: sizes.minByOrNull { it.width * it.height }!!
        }

        fun pickPreviewSize(sizes: List<Size>, video: Size): Size {
            val sameAspect = sizes.filter { it.width * video.height == it.height * video.width && it.width <= 1280 }
            return sameAspect.maxByOrNull { it.width * it.height } ?: video
        }

        fun pickFpsRange(ranges: Array<Range<Int>>?): Range<Int> {
            if (ranges == null || ranges.isEmpty()) return Range(30, 30)
            return ranges.firstOrNull { it.lower == 30 && it.upper == 30 }
                ?: ranges.filter { it.upper <= 30 }.maxByOrNull { it.lower } ?: ranges.first()
        }
    }
}
