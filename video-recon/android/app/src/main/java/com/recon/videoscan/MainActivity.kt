package com.recon.videoscan

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.view.TextureView
import android.view.WindowManager
import android.widget.Button
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

class MainActivity : AppCompatActivity(), CameraRecorder.Listener {
    private lateinit var preview: TextureView
    private lateinit var status: TextView
    private lateinit var stats: TextView
    private lateinit var button: Button
    private var camera: CameraRecorder? = null
    private var imu: ImuLogger? = null
    private var recordingStartedMs = 0L
    private val ui = Handler(Looper.getMainLooper())
    private val ticker = object : Runnable {
        override fun run() {
            updateStats()
            ui.postDelayed(this, 500)
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        preview = findViewById(R.id.preview)
        status = findViewById(R.id.status)
        stats = findViewById(R.id.stats)
        button = findViewById(R.id.record)
        button.setOnClickListener { toggle() }
    }

    override fun onResume() {
        super.onResume()
        if (hasCameraPermission()) openCamera() else
            ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.CAMERA), 1)
        ui.post(ticker)
    }

    override fun onPause() {
        ui.removeCallbacks(ticker)
        if (camera?.isRecording == true) stopRecording()
        camera?.close()
        camera = null
        super.onPause()
    }

    override fun onRequestPermissionsResult(code: Int, permissions: Array<out String>, results: IntArray) {
        super.onRequestPermissionsResult(code, permissions, results)
        if (hasCameraPermission()) openCamera() else status.text = getString(R.string.camera_permission_needed)
    }

    private fun hasCameraPermission() =
        ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED

    private fun openCamera() {
        if (camera != null) return
        try {
            camera = CameraRecorder(this, preview).also {
                it.listener = this
                it.open()
            }
        } catch (e: Exception) {
            status.text = "Camera unavailable: ${e.message}"
        }
    }

    private fun toggle() {
        val cam = camera ?: return
        if (cam.isRecording) stopRecording() else startRecording(cam)
    }

    private fun startRecording(cam: CameraRecorder) {
        if (cam.timestampSource() != "REALTIME") {
            Toast.makeText(this, "Camera clock is ${cam.timestampSource()}; frames cannot be matched to the IMU", Toast.LENGTH_LONG).show()
        }
        val stamp = SimpleDateFormat("yyyyMMdd-HHmmss", Locale.US).format(Date())
        val dir = File(getExternalFilesDir(null), "recordings/rec_$stamp")
        dir.mkdirs()
        imu = ImuLogger(this, File(dir, "imu.csv")).also { it.start() }
        cam.startRecording(dir)
        recordingStartedMs = System.currentTimeMillis()
        button.text = getString(R.string.stop)
    }

    private fun stopRecording() {
        camera?.stopRecording()
        imu?.stop()
        imu = null
        button.text = getString(R.string.start)
    }

    override fun onStatus(text: String) {
        ui.post { status.text = text }
    }

    private fun updateStats() {
        val cam = camera ?: return
        val sb = StringBuilder()
        sb.append("cam ").append(cam.cameraId).append("  ").append(cam.videoSize.width).append('x').append(cam.videoSize.height)
            .append("  orient ").append(cam.sensorOrientation()).append("°  clock ").append(cam.timestampSource())
            .append("  intrinsics ").append(if (cam.hasIntrinsicCalibration()) "yes" else "fallback").append('\n')
        if (cam.isRecording) {
            val s = (System.currentTimeMillis() - recordingStartedMs) / 1000
            sb.append(String.format(Locale.US, "REC %d:%02d  frames %d", s / 60, s % 60, cam.frameCount))
            val counts = imu?.sampleCounts()
            if (counts != null && s > 0) {
                sb.append('\n')
                for ((name, n) in counts.toSortedMap()) sb.append(name).append(' ').append(n / s).append("Hz ")
            }
        } else {
            sb.append("sensors: ").append(imuAvailable())
        }
        stats.text = sb.toString()
    }

    private fun imuAvailable(): String {
        val m = getSystemService(SENSOR_SERVICE) as android.hardware.SensorManager
        return ImuLogger.SENSORS.filter { m.getDefaultSensor(it.first) != null }.joinToString(" ") { it.second }
    }
}
