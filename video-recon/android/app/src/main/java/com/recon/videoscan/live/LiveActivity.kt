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
import kotlin.math.abs
import kotlin.math.sqrt

/**
 * Everything on the phone: Camera2 frames + IMU -> OpenVINS poses (JNI) -> DA3-Small depth (onnxruntime)
 * scaled by the VIO landmarks -> voxel-hash TSDF (JNI) -> two GL views (camera POV, orbit), like the web page.
 */
class LiveActivity : AppCompatActivity(), SensorEventListener {
    private lateinit var status: TextView
    private lateinit var preview: TextureView
    private lateinit var povView: GLSurfaceView
    private lateinit var orbitView: GLSurfaceView
    private val povRenderer = CloudRenderer(pov = true)
    private val orbitRenderer = CloudRenderer(pov = false)
    private var handle = 0L
    private var depth: DepthRunner? = null
    private var fusion = true
    // camera
    private var camera: CameraDevice? = null; private var session: CameraCaptureSession? = null; private var reader: ImageReader? = null
    private val camThread = HandlerThread("camera").apply { start() }; private val camHandler = Handler(camThread.looper)
    private val vioThread = HandlerThread("vio").apply { start() }; private val vioHandler = Handler(vioThread.looper)
    private val depthThread = HandlerThread("depth").apply { start() }; private val depthHandler = Handler(depthThread.looper)
    private val streamSize = Size(960, 540)
    // sensors: gyro and accelerometer arrive separately; each accelerometer sample is paired with the latest gyro
    private lateinit var sensors: SensorManager; private val gyro = DoubleArray(3); private var haveGyro = false; private var imuCount = 0L
    // state
    @Volatile private var lastPose: FloatArray? = null; @Volatile private var lastPoseTime = 0.0; private val landmarks = FloatArray(600); @Volatile private var landmarkCount = 0
    @Volatile private var latestFrame: FrameData? = null; private val depthBusy = AtomicBoolean(false)
    private var vioMsAvg = 0.0; private var depthMs = 0L; private var tsdfStats = FloatArray(5); private var frames = 0; private var depthFrames = 0; private var lastDepthAt = 0.0; private var lastDepthPose: FloatArray? = null
    private val pathPts = ArrayList<Float>(); private var initTime = -1.0; private var scaleUsed = 0f
    private val ui = Handler(Looper.getMainLooper())
    private val ticker = object : Runnable { override fun run() { updateStatus(); ui.postDelayed(this, 300) } }

    class FrameData(val t: Double, val w: Int, val h: Int, val gray: ByteArray, val rgba: ByteArray, val rw: Int, val rh: Int)

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(com.recon.videoscan.R.layout.activity_live)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        status = findViewById(com.recon.videoscan.R.id.status); preview = findViewById(com.recon.videoscan.R.id.preview)
        povView = findViewById(com.recon.videoscan.R.id.pov); orbitView = findViewById(com.recon.videoscan.R.id.orbit)
        for ((v, r) in listOf(povView to povRenderer, orbitView to orbitRenderer)) { v.setEGLContextClientVersion(2); v.setRenderer(r); v.renderMode = GLSurfaceView.RENDERMODE_CONTINUOUSLY }
        var lx = 0f; var ly = 0f
        orbitView.setOnTouchListener { _, e -> when (e.actionMasked) {
            MotionEvent.ACTION_DOWN -> { lx = e.x; ly = e.y }
            MotionEvent.ACTION_MOVE -> { if (e.pointerCount == 1) { orbitRenderer.yaw += (e.x - lx) * 0.01f; orbitRenderer.pitch = (orbitRenderer.pitch + (e.y - ly) * 0.01f).coerceIn(-1.4f, 1.5f) } ; lx = e.x; ly = e.y } }; true }
        findViewById<Button>(com.recon.videoscan.R.id.reset).setOnClickListener { resetMap() }
        findViewById<Button>(com.recon.videoscan.R.id.fuse).setOnClickListener { fusion = !fusion; (it as Button).text = if (fusion) "fusion: on" else "fusion: off"; resetMap() }
        sensors = getSystemService(Context.SENSOR_SERVICE) as SensorManager
        val files = getExternalFilesDir(null)!!
        val cfg = File(files, "vio/estimator_config.yaml")
        if (!cfg.exists()) copyAssets(files)
        handle = Native.init(cfg.absolutePath)
        val model = File(files, "models")
        depthHandler.post { try { depth = DepthRunner(model) } catch (e: Throwable) { Log.e("livescan", "depth model: $e"); ui.post { status.text = "depth model failed: $e" } } }
    }

    private fun copyAssets(files: File) {
        val dir = File(files, "vio"); dir.mkdirs()
        for (n in assets.list("vio") ?: emptyArray()) assets.open("vio/$n").use { i -> File(dir, n).outputStream().use { i.copyTo(it) } }
    }
    private fun resetMap() {
        // restart the VIO too: it may have initialised on a featureless view and drifted
        vioHandler.post { val old = handle; handle = Native.init(File(getExternalFilesDir(null)!!, "vio/estimator_config.yaml").absolutePath); if (old != 0L) Native.destroy(old); lastPose = null; initTime = -1.0; scaleUsed = 0f; lastDepthPose = null; orbitTargetSet = false }
        Native.tsdfReset(handle, 0.03f); synchronized(pathPts) { pathPts.clear(); orbitRenderer.path = FloatArray(0) }; rawCloud.clear()
        povRenderer.points = null; orbitRenderer.points = null; povRenderer.pointCount = 0; orbitRenderer.pointCount = 0; depthFrames = 0
    }

    override fun onResume() {
        super.onResume(); povView.onResume(); orbitView.onResume()
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) startCamera()
        else ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.CAMERA), 1)
        for (t in intArrayOf(Sensor.TYPE_GYROSCOPE, Sensor.TYPE_ACCELEROMETER)) sensors.getDefaultSensor(t)?.let { sensors.registerListener(this, it, SensorManager.SENSOR_DELAY_FASTEST, camHandler) }
        ui.post(ticker)
    }
    override fun onPause() { ui.removeCallbacks(ticker); sensors.unregisterListener(this); session?.close(); camera?.close(); reader?.close(); povView.onPause(); orbitView.onPause(); super.onPause() }
    override fun onRequestPermissionsResult(rc: Int, p: Array<out String>, g: IntArray) { super.onRequestPermissionsResult(rc, p, g); if (g.isNotEmpty() && g[0] == PackageManager.PERMISSION_GRANTED) startCamera() }
    override fun onDestroy() { if (handle != 0L) Native.destroy(handle); super.onDestroy() }

    // ---- IMU
    override fun onSensorChanged(e: SensorEvent) {
        if (handle == 0L) return
        if (e.sensor.type == Sensor.TYPE_GYROSCOPE) { gyro[0] = e.values[0].toDouble(); gyro[1] = e.values[1].toDouble(); gyro[2] = e.values[2].toDouble(); haveGyro = true }
        else if (haveGyro) { Native.feedImu(handle, e.timestamp * 1e-9, gyro[0], gyro[1], gyro[2], e.values[0].toDouble(), e.values[1].toDouble(), e.values[2].toDouble()); imuCount++ }
    }
    override fun onAccuracyChanged(s: Sensor?, a: Int) = Unit

    // ---- camera
    private fun startCamera() {
        if (!preview.isAvailable) { preview.surfaceTextureListener = object : TextureView.SurfaceTextureListener {
            override fun onSurfaceTextureAvailable(st: SurfaceTexture, w: Int, h: Int) = startCamera()
            override fun onSurfaceTextureSizeChanged(st: SurfaceTexture, w: Int, h: Int) = Unit
            override fun onSurfaceTextureDestroyed(st: SurfaceTexture) = true
            override fun onSurfaceTextureUpdated(st: SurfaceTexture) = Unit }; return }
        val mgr = getSystemService(Context.CAMERA_SERVICE) as CameraManager
        val id = mgr.cameraIdList.first { mgr.getCameraCharacteristics(it).get(CameraCharacteristics.LENS_FACING) == CameraCharacteristics.LENS_FACING_BACK }
        reader = ImageReader.newInstance(streamSize.width, streamSize.height, ImageFormat.YUV_420_888, 3).also { it.setOnImageAvailableListener({ r -> r.acquireLatestImage()?.let { img -> onFrame(img) } }, camHandler) }
        if (ActivityCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) return
        mgr.openCamera(id, object : CameraDevice.StateCallback() {
            override fun onOpened(cam: CameraDevice) {
                camera = cam
                val st = preview.surfaceTexture!!; st.setDefaultBufferSize(streamSize.width, streamSize.height); val ps = Surface(st)
                val req = cam.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW).apply {
                    addTarget(ps); addTarget(reader!!.surface)
                    set(CaptureRequest.CONTROL_AF_MODE, CaptureRequest.CONTROL_AF_MODE_OFF); set(CaptureRequest.LENS_FOCUS_DISTANCE, 0.4f)
                    set(CaptureRequest.CONTROL_VIDEO_STABILIZATION_MODE, CaptureRequest.CONTROL_VIDEO_STABILIZATION_MODE_OFF)
                    set(CaptureRequest.LENS_OPTICAL_STABILIZATION_MODE, CaptureRequest.LENS_OPTICAL_STABILIZATION_MODE_OFF) }
                @Suppress("DEPRECATION") cam.createCaptureSession(listOf(ps, reader!!.surface), object : CameraCaptureSession.StateCallback() {
                    override fun onConfigured(s: CameraCaptureSession) { session = s; s.setRepeatingRequest(req.build(), null, camHandler) }
                    override fun onConfigureFailed(s: CameraCaptureSession) { ui.post { status.text = "camera session failed" } } }, camHandler)
            }
            override fun onDisconnected(cam: CameraDevice) { cam.close() }
            override fun onError(cam: CameraDevice, error: Int) { cam.close(); ui.post { status.text = "camera error $error" } }
        }, camHandler)
    }

    private fun onFrame(img: Image) {
        val t = img.timestamp * 1e-9; val w = img.width; val h = img.height
        val yp = img.planes[0]; val yb = yp.buffer; val gray = ByteArray(w * h)
        if (yp.rowStride == w) { yb.position(0); yb.get(gray, 0, w * h) } else for (r in 0 until h) { yb.position(r * yp.rowStride); yb.get(gray, r * w, w) }
        // RGBA at the depth model's resolution (322x182) from the YUV planes, nearest neighbour
        val dw = 168; val dh = 98; val rgba = ByteArray(dw * dh * 4)
        val up = img.planes[1]; val vp = img.planes[2]; val ub = up.buffer; val vb = vp.buffer; val cs = up.pixelStride; val crs = up.rowStride
        for (y in 0 until dh) { val sy = y * h / dh; val suy = sy / 2
            for (x in 0 until dw) { val sx = x * w / dw; val Y = gray[sy * w + sx].toInt() and 0xff; val ci = suy * crs + (sx / 2) * cs
                val U = (ub.get(ci).toInt() and 0xff) - 128; val V = (vb.get(ci).toInt() and 0xff) - 128
                val o = (y * dw + x) * 4
                rgba[o] = (Y + 1.402f * V).toInt().coerceIn(0, 255).toByte(); rgba[o + 1] = (Y - 0.344f * U - 0.714f * V).toInt().coerceIn(0, 255).toByte(); rgba[o + 2] = (Y + 1.772f * U).toInt().coerceIn(0, 255).toByte(); rgba[o + 3] = -1 } }
        img.close()
        val fd = FrameData(t, w, h, gray, rgba, dw, dh); latestFrame = fd; frames++
        vioHandler.post { runVio(fd) }
    }

    private fun runVio(fd: FrameData) {
        val pose = FloatArray(7); val t0 = System.nanoTime()
        val n = Native.feedImage(handle, fd.t, fd.w, fd.h, fd.gray, pose, landmarks)
        vioMsAvg = 0.9 * vioMsAvg + 0.1 * (System.nanoTime() - t0) / 1e6
        if (n <= 0) return
        if (initTime < 0) initTime = fd.t
        landmarkCount = n - 1; lastPose = pose; lastPoseTime = fd.t
        povRenderer.pose = pose; orbitRenderer.pose = pose
        synchronized(pathPts) { if (pathPts.size < 3 || abs(pathPts[pathPts.size - 3] - pose[0]) + abs(pathPts[pathPts.size - 2] - pose[1]) + abs(pathPts[pathPts.size - 1] - pose[2]) > 0.02f) { pathPts.add(pose[0]); pathPts.add(pose[1]); pathPts.add(pose[2]); orbitRenderer.path = pathPts.toFloatArray() } }
        if (!orbitTargetSet) { orbitRenderer.target[0] = pose[0]; orbitRenderer.target[1] = pose[1]; orbitRenderer.target[2] = pose[2]; orbitTargetSet = true
            // orbit camera behind the phone's initial heading (camera looks along -Z of its frame)
            val r0 = quatToRot(pose[3], pose[4], pose[5], pose[6]); val fx0 = -r0[2]; val fz0 = -r0[8]; orbitRenderer.yaw = kotlin.math.atan2(-fx0, -fz0); orbitRenderer.pitch = 0.6f; orbitRenderer.dist = 4f }
        // keyframe rule for depth: moved 10 cm or turned, or 0.7 s elapsed, and the depth thread is free
        val lp = lastDepthPose; val moved = lp == null || dist(lp, pose) > 0.10f || quatAngle(lp, pose) > 0.087f || fd.t - lastDepthAt > 0.7
        if (moved && depth != null && depthBusy.compareAndSet(false, true)) { lastDepthAt = fd.t; lastDepthPose = pose.copyOf(); val lm = landmarks.copyOf(landmarkCount * 3); depthHandler.post { runDepth(fd, pose.copyOf(), lm) } }
    }
    private var orbitTargetSet = false
    private fun dist(a: FloatArray, b: FloatArray) = sqrt((a[0] - b[0]) * (a[0] - b[0]) + (a[1] - b[1]) * (a[1] - b[1]) + (a[2] - b[2]) * (a[2] - b[2]))
    private fun quatAngle(a: FloatArray, b: FloatArray): Float { val d = abs(a[3] * b[3] + a[4] * b[4] + a[5] * b[5] + a[6] * b[6]).coerceAtMost(1f); return 2f * kotlin.math.acos(d) }

    private fun runDepth(fd: FrameData, pose: FloatArray, lm: FloatArray) {
        try {
            val dr = depth ?: return
            val (rel, ms) = dr.run(fd.rgba); depthMs = ms
            // intrinsics at the depth resolution: factory calibration is for 1920 px wide frames (see recording meta)
            val fx = 1278.4f * fd.rw / 1920f; val fy = 1278.6f * fd.rw / 1920f; val cx = 962.1f * fd.rw / 1920f; val cy = 536.9f * fd.rh / 1080f
            // world<-camera rotation: pose quaternion is R_gl (camera looks -Z); OpenCV axes = R_gl * diag(1,-1,-1)
            val q = pose; val r = quatToRot(q[3], q[4], q[5], q[6]); val rcv = floatArrayOf(r[0], -r[1], -r[2], r[3], -r[4], -r[5], r[6], -r[7], -r[8])
            // scale the relative depth with the VIO landmarks that project into this frame (median ratio)
            val zs = ArrayList<Float>(); val ds = ArrayList<Float>()
            var i = 0; while (i + 2 < lm.size) { val px = lm[i] - q[0]; val py = lm[i + 1] - q[1]; val pz = lm[i + 2] - q[2]
                val zc = rcv[2] * px + rcv[5] * py + rcv[8] * pz
                if (zc > 0.2f && zc < 8f) { val xc = rcv[0] * px + rcv[3] * py + rcv[6] * pz; val yc = rcv[1] * px + rcv[4] * py + rcv[7] * pz
                    val u = (fx * xc / zc + cx).toInt(); val v = (fy * yc / zc + cy).toInt()
                    if (u in 0 until fd.rw && v in 0 until fd.rh) { val d = rel[v * fd.rw + u]; if (d > 1e-4f) { zs.add(zc); ds.add(d) } } }
                i += 3 }
            landmarksUsed = zs.size
            val mm: ShortArray
            if (dr.inverse) {
                // relative inverse depth: 1/z = a*d + b, least squares over the landmarks, kept from the last good fit
                if (zs.size >= 4) { var sd = 0.0; var sy = 0.0; var sdd = 0.0; var sdy = 0.0; val n = zs.size
                    for (k in 0 until n) { val d = ds[k].toDouble(); val y = 1.0 / zs[k]; sd += d; sy += y; sdd += d * d; sdy += d * y }
                    val den = n * sdd - sd * sd; if (abs(den) > 1e-9) { val a = (n * sdy - sd * sy) / den; val b = (sy - a * sd) / n; if (a > 0) { fitA = a.toFloat(); fitB = b.toFloat(); scaleUsed = fitA } } }
                if (fitA <= 0f) return
                mm = ShortArray(rel.size) { k -> val inv = fitA * rel[k] + fitB; if (inv > 0.05f) (1000f / inv).toInt().coerceIn(0, 65535).toShort() else 0 }
            } else {
                if (zs.size >= 3) { val ratios = FloatArray(zs.size) { k -> zs[k] / ds[k] }; ratios.sort(); scaleUsed = ratios[ratios.size / 2] }
                if (scaleUsed <= 0f) return
                mm = ShortArray(rel.size) { k -> (rel[k] * scaleUsed * 1000f).toInt().coerceIn(0, 65535).toShort() }
            }
            if (fusion) {
                tsdfStats = Native.tsdfIntegrate(handle, mm, fd.rw, fd.rh, floatArrayOf(fx, fy, cx, cy), rcv, floatArrayOf(q[0], q[1], q[2]), fd.rgba, fd.rw, fd.rh, 2)
                val surf = Native.tsdfSurface(handle, 1_500_000); publish(surf)
            } else {
                // raw back-projection of this frame appended to the cloud (comparison mode)
                val pts = ArrayList<Float>(); var y = 1
                while (y < fd.rh) { var x = 1; while (x < fd.rw) { val z = mm[y * fd.rw + x].toInt() and 0xffff; if (z in 1..6000) { val zf = z / 1000f
                    val X = (x - cx) / fx * zf; val Y = (y - cy) / fy * zf
                    pts.add(q[0] + rcv[0] * X + rcv[1] * Y + rcv[2] * zf); pts.add(q[1] + rcv[3] * X + rcv[4] * Y + rcv[5] * zf); pts.add(q[2] + rcv[6] * X + rcv[7] * Y + rcv[8] * zf)
                    val o = (y * fd.rw + x) * 4; pts.add((fd.rgba[o].toInt() and 0xff) / 255f); pts.add((fd.rgba[o + 1].toInt() and 0xff) / 255f); pts.add((fd.rgba[o + 2].toInt() and 0xff) / 255f) }; x += 3 }; y += 3 }
                rawCloud.addAll(pts); publish(rawCloud.toFloatArray())
            }
            depthFrames++
        } catch (e: Throwable) { Log.e("livescan", "depth: $e", e) } finally { depthBusy.set(false) }
    }
    private val rawCloud = ArrayList<Float>(); private var fitA = 0f; private var fitB = 0f; @Volatile private var landmarksUsed = 0
    private fun publish(surf: FloatArray) {
        val n = surf.size / 6; val buf = ByteBuffer.allocateDirect(surf.size * 4).order(ByteOrder.nativeOrder()).asFloatBuffer().put(surf); buf.position(0)
        povRenderer.points = buf; povRenderer.pointCount = n; orbitRenderer.points = buf; orbitRenderer.pointCount = n
    }
    private fun quatToRot(x: Float, y: Float, z: Float, w: Float) = floatArrayOf(   // row-major 3x3
        1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
        2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y))

    private fun updateStatus() {
        val p = lastPose
        status.text = if (p == null) "VIO: initialising (${if (initTime < 0) "hold still, then bounce" else "…"})   frames $frames   imu $imuCount   vio ${"%.0f".format(vioMsAvg)} ms   depth ${depth?.backend ?: "loading"}"
        else "pose ${"%.2f %.2f %.2f".format(p[0], p[1], p[2])}   landmarks $landmarkCount (in frame $landmarksUsed)   vio ${"%.0f".format(vioMsAvg)} ms\n${depth?.name} ${depth?.backend} $depthMs ms   scale ${"%.2f".format(scaleUsed)}\n" +
            "depth frames $depthFrames   tsdf integrate ${"%.0f".format(tsdfStats[0])} ms extract ${"%.0f".format(tsdfStats[1])} ms   blocks ${tsdfStats[3].toInt()}   points ${(tsdfStats[4] / 1000).toInt()} k"
    }
}
