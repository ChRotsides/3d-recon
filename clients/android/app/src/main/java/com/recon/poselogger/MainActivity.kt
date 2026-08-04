package com.recon.poselogger

import android.Manifest
import android.content.pm.PackageManager
import android.opengl.GLSurfaceView
import android.os.Bundle
import android.widget.Button
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.google.ar.core.ArCoreApk
import com.google.ar.core.Config
import com.google.ar.core.Session
import com.google.ar.core.TrackingState
import com.google.ar.core.exceptions.CameraNotAvailableException
import com.google.ar.core.exceptions.UnavailableException
import uniffi.recon_core.nagMessage
import java.io.File
import javax.microedition.khronos.egl.EGLConfig
import javax.microedition.khronos.opengles.GL10

/**
 * The throwaway pose-logging capture screen (plan task 3).
 *
 * Deliberately minimal: start, walk around, stop, and a bundle lands on disk.
 * No guidance overlay and no next-best-station ring — that is Phase 3 and it
 * belongs in the shared Rust core (plan §3.8), not in an Activity.
 *
 * The camera preview is real rather than a black screen because a pose logger
 * whose operator cannot see what they are pointing at produces useless captures.
 */
class MainActivity : AppCompatActivity(), GLSurfaceView.Renderer {

    private lateinit var surfaceView: GLSurfaceView
    private lateinit var statusView: TextView
    private lateinit var actionButton: Button

    private var session: Session? = null
    private var recorder: CaptureRecorder? = null
    private val background = BackgroundRenderer()
    private var sessionConfigured = false

    @Volatile private var status: String = "starting…"

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        surfaceView = findViewById(R.id.surface)
        statusView = findViewById(R.id.status)
        actionButton = findViewById(R.id.action)

        surfaceView.preserveEGLContextOnPause = true
        surfaceView.setEGLContextClientVersion(3)
        surfaceView.setEGLConfigChooser(8, 8, 8, 8, 16, 0)
        surfaceView.setRenderer(this)
        surfaceView.renderMode = GLSurfaceView.RENDERMODE_CONTINUOUSLY

        actionButton.setOnClickListener { toggleRecording() }
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
            status = "camera unavailable: ${e.message}"
            session = null
            return
        }
        surfaceView.onResume()
    }

    override fun onPause() {
        super.onPause()
        surfaceView.onPause()
        session?.pause()
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
            s.configure(
                Config(s).apply {
                    // Plan §5: the bundle frame assumes a gravity-aligned world.
                    // ARCore is gravity-aligned by default; the plane finding is
                    // what gives us the floor height for free.
                    planeFindingMode = Config.PlaneFindingMode.HORIZONTAL_AND_VERTICAL
                    focusMode = Config.FocusMode.AUTO
                    lightEstimationMode = Config.LightEstimationMode.AMBIENT_INTENSITY
                    updateMode = Config.UpdateMode.LATEST_CAMERA_IMAGE
                }
            )
            session = s
            return true
        } catch (e: UnavailableException) {
            status = "ARCore unavailable: ${e.message}"
            return false
        }
    }

    // -- recording ----------------------------------------------------------

    private fun toggleRecording() {
        val s = session ?: return
        val rec = recorder
        if (rec == null) {
            val r = CaptureRecorder(s, File(getExternalFilesDir(null), "captures"))
            r.start()
            recorder = r
            actionButton.text = getString(R.string.stop)
        } else {
            val dir = rec.stop()
            recorder = null
            actionButton.text = getString(R.string.start)
            Toast.makeText(
                this, "${rec.keyframeCount} keyframes -> $dir", Toast.LENGTH_LONG
            ).show()
        }
    }

    // -- GL -----------------------------------------------------------------

    override fun onSurfaceCreated(gl: GL10?, config: EGLConfig?) {
        background.createOnGlThread()
    }

    override fun onSurfaceChanged(gl: GL10?, width: Int, height: Int) {
        android.opengl.GLES30.glViewport(0, 0, width, height)
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

            recorder?.onFrame(frame)

            val tracking = frame.camera.trackingState
            val rec = recorder
            status = buildString {
                append(if (rec == null) "ready" else "recording")
                append("  ·  ")
                append(
                    when (tracking) {
                        TrackingState.TRACKING -> "tracking"
                        TrackingState.PAUSED -> "tracking paused"
                        else -> "tracking lost"
                    }
                )
                if (rec != null) {
                    append("  ·  ${rec.keyframeCount}/${rec.maxKeyframes} keyframes")
                    // Guidance copy comes from the core so both platforms say
                    // the same thing to the same user (plan §6.1).
                    rec.lastGuidance?.nag?.let { append("\n${nagMessage(it)}") }
                    if (rec.rejected.isNotEmpty()) {
                        // Surfaced rather than hidden: a capture that is quietly
                        // rejecting every frame should look wrong, not idle.
                        append("\nrejected: ")
                        append(rec.rejected.entries.joinToString(" ") { "${it.key}=${it.value}" })
                    }
                }
            }
        } catch (e: CameraNotAvailableException) {
            status = "camera lost: ${e.message}"
        } catch (e: Throwable) {
            status = "error: ${e.message}"
        }
        runOnUiThread { statusView.text = status }
    }
}
