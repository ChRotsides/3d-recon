package com.recon.poselogger.rgbscan

import android.content.Context
import android.opengl.GLES20
import android.opengl.GLSurfaceView
import android.opengl.Matrix
import android.view.GestureDetector
import android.view.MotionEvent
import android.view.ScaleGestureDetector
import android.view.ViewConfiguration
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import javax.microedition.khronos.egl.EGLConfig
import javax.microedition.khronos.opengles.GL10
import kotlin.math.max
import kotlin.math.hypot
import kotlin.math.tan

internal class ViewerCameraState(private val initialDistance: Float = DEFAULT_DISTANCE) {
    @Volatile var yaw = 0f
    @Volatile var pitch = 0f
    @Volatile var distance = initialDistance
    @Volatile var panX = 0f
    @Volatile var panY = 0f

    fun orbit(deltaX: Float, deltaY: Float) {
        yaw += deltaX * ORBIT_DEGREES_PER_PIXEL
        pitch = (pitch + deltaY * ORBIT_DEGREES_PER_PIXEL).coerceIn(-85f, 85f)
    }

    fun orbitByViewport(deltaX: Float, deltaY: Float, width: Int, height: Int) {
        yaw += deltaX / max(1, width) * HORIZONTAL_DEGREES_PER_SCREEN
        pitch = (pitch + deltaY / max(1, height) * VERTICAL_DEGREES_PER_SCREEN)
            .coerceIn(-85f, 85f)
    }

    fun zoomBy(scale: Float) {
        if (scale.isFinite() && scale > 0f) {
            distance = (distance * scale).coerceIn(MIN_DISTANCE, MAX_DISTANCE)
        }
    }

    fun panByPixels(deltaX: Float, deltaY: Float, viewportHeight: Int) {
        val halfFovRadians = Math.toRadians((FIELD_OF_VIEW_DEGREES / 2f).toDouble())
        val worldPerPixel = 2f * distance * tan(halfFovRadians).toFloat() / max(1, viewportHeight)
        panX = (panX + deltaX * worldPerPixel).coerceIn(-MAX_PAN, MAX_PAN)
        panY = (panY - deltaY * worldPerPixel).coerceIn(-MAX_PAN, MAX_PAN)
    }

    fun reset() {
        yaw = 0f
        pitch = 0f
        distance = initialDistance
        panX = 0f
        panY = 0f
    }

    companion object {
        const val FIELD_OF_VIEW_DEGREES = 55f
        const val DEFAULT_DISTANCE = 4f
        const val MIN_DISTANCE = 0.25f
        const val MAX_DISTANCE = 30f
        private const val MAX_PAN = 20f
        private const val ORBIT_DEGREES_PER_PIXEL = 0.2f
        private const val HORIZONTAL_DEGREES_PER_SCREEN = 180f
        private const val VERTICAL_DEGREES_PER_SCREEN = 150f
    }
}

class PointCloudView(context: Context, asset: PointCloudAsset) : GLSurfaceView(context) {
    private val cloudBounds = CloudBounds.from(asset.positions)
    private val cameraState = ViewerCameraState(cloudBounds.fitDistance)
    internal val sceneState = AtomicSceneState()
    private val cloudRenderer = CloudRenderer(asset, cameraState, sceneState)
    private var lastX = 0f
    private var lastY = 0f
    private var lastFocusX = 0f
    private var lastFocusY = 0f
    private var singleOriginX = 0f
    private var singleOriginY = 0f
    private var singleDragStarted = false
    private var multiTouchGesture = false
    private val touchSlop = ViewConfiguration.get(context).scaledTouchSlop.toFloat()
    private val scaleDetector = ScaleGestureDetector(context,
        object : ScaleGestureDetector.SimpleOnScaleGestureListener() {
            override fun onScale(detector: ScaleGestureDetector): Boolean {
                cameraState.zoomBy(1f / detector.scaleFactor)
                requestRender()
                return true
            }
        })
    private val gestureDetector = GestureDetector(context,
        object : GestureDetector.SimpleOnGestureListener() {
            override fun onDown(event: MotionEvent) = true
            override fun onDoubleTap(event: MotionEvent): Boolean {
                cameraState.reset()
                requestRender()
                return true
            }
        })

    init {
        setEGLContextClientVersion(2)
        setRenderer(cloudRenderer)
        renderMode = RENDERMODE_WHEN_DIRTY
        contentDescription = "Interactive 3D point cloud"
    }

    override fun onTouchEvent(event: MotionEvent): Boolean {
        gestureDetector.onTouchEvent(event)
        scaleDetector.onTouchEvent(event)
        when (event.actionMasked) {
            MotionEvent.ACTION_DOWN -> {
                lastX = event.x
                lastY = event.y
                singleOriginX = event.x
                singleOriginY = event.y
                singleDragStarted = false
                multiTouchGesture = false
            }
            MotionEvent.ACTION_POINTER_DOWN -> {
                if (event.pointerCount >= 2) {
                    lastFocusX = focusX(event)
                    lastFocusY = focusY(event)
                    multiTouchGesture = true
                }
            }
            MotionEvent.ACTION_MOVE -> {
                if (event.pointerCount >= 2) {
                    val focusX = focusX(event)
                    val focusY = focusY(event)
                    cameraState.panByPixels(focusX - lastFocusX, focusY - lastFocusY, height)
                    lastFocusX = focusX
                    lastFocusY = focusY
                    requestRender()
                } else if (!multiTouchGesture && !scaleDetector.isInProgress) {
                    if (!singleDragStarted) {
                        singleDragStarted = hypot(
                            event.x - singleOriginX,
                            event.y - singleOriginY,
                        ) >= touchSlop
                    }
                    if (singleDragStarted) {
                        cameraState.orbitByViewport(
                            event.x - lastX,
                            event.y - lastY,
                            width,
                            height,
                        )
                        requestRender()
                    }
                }
                lastX = event.x
                lastY = event.y
            }
            MotionEvent.ACTION_POINTER_UP -> {
                // Keep one-finger orbit suppressed until every finger is lifted. This avoids
                // the jump Android warns about when the active pointer changes mid-gesture.
                multiTouchGesture = true
            }
            MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL -> {
                multiTouchGesture = false
                singleDragStarted = false
            }
        }
        return true
    }

    private fun focusX(event: MotionEvent): Float =
        (0 until event.pointerCount).sumOf { event.getX(it).toDouble() }.toFloat() /
            event.pointerCount

    private fun focusY(event: MotionEvent): Float =
        (0 until event.pointerCount).sumOf { event.getY(it).toDouble() }.toFloat() /
            event.pointerCount

    fun zoomIn() {
        cameraState.zoomBy(0.8f)
        requestRender()
    }

    fun zoomOut() {
        cameraState.zoomBy(1.25f)
        requestRender()
    }

    fun panScreen(deltaX: Float, deltaY: Float) {
        cameraState.panByPixels(deltaX * width, deltaY * height, height)
        requestRender()
    }

    fun fitCloud() {
        cameraState.reset()
        requestRender()
    }

    /** Queue one complete immutable version; buffer replacement happens only on the GL thread. */
    fun replaceScene(asset: PointCloudAsset) {
        queueEvent {
            cloudRenderer.replace(asset)
            requestRender()
        }
    }
}

internal data class RenderedSceneSnapshot(val version: Int, val pointCount: Int)

internal class AtomicSceneState {
    @Volatile private var current = RenderedSceneSnapshot(-1, 0)

    fun commit(version: Int, pointCount: Int) {
        current = RenderedSceneSnapshot(version, pointCount)
    }

    fun snapshot(): RenderedSceneSnapshot = current
}

private data class CloudBounds(
    val centreX: Float,
    val centreY: Float,
    val centreZ: Float,
    val fitDistance: Float,
) {
    companion object {
        fun from(positions: FloatArray): CloudBounds {
            var minX = Float.POSITIVE_INFINITY
            var minY = Float.POSITIVE_INFINITY
            var minZ = Float.POSITIVE_INFINITY
            var maxX = Float.NEGATIVE_INFINITY
            var maxY = Float.NEGATIVE_INFINITY
            var maxZ = Float.NEGATIVE_INFINITY
            positions.indices.step(3).forEach { index ->
                minX = minOf(minX, positions[index])
                minY = minOf(minY, positions[index + 1])
                minZ = minOf(minZ, positions[index + 2])
                maxX = maxOf(maxX, positions[index])
                maxY = maxOf(maxY, positions[index + 1])
                maxZ = maxOf(maxZ, positions[index + 2])
            }
            val centreX = (minX + maxX) / 2f
            val centreY = (minY + maxY) / 2f
            val centreZ = (minZ + maxZ) / 2f
            val radius = maxOf(maxX - minX, maxY - minY, maxZ - minZ) / 2f
            val halfFov = Math.toRadians((ViewerCameraState.FIELD_OF_VIEW_DEGREES / 2f).toDouble())
            val distance = (radius * 1.25f / tan(halfFov).toFloat())
                .coerceIn(ViewerCameraState.MIN_DISTANCE, ViewerCameraState.MAX_DISTANCE)
            return CloudBounds(centreX, centreY, centreZ, distance)
        }
    }
}

private class CloudRenderer(
    asset: PointCloudAsset,
    private val camera: ViewerCameraState,
    private val sceneState: AtomicSceneState,
) : GLSurfaceView.Renderer {
    private data class SceneBuffers(
        val positions: FloatBuffer,
        val colours: ByteBuffer,
        val count: Int,
        val version: Int,
        val bounds: CloudBounds,
    ) {
        companion object {
            fun from(asset: PointCloudAsset) = SceneBuffers(
                positions = ByteBuffer.allocateDirect(asset.positions.size * 4)
                    .order(ByteOrder.nativeOrder()).asFloatBuffer().apply {
                        put(asset.positions); position(0)
                    },
                colours = ByteBuffer.allocateDirect(asset.colours.size).apply {
                    put(asset.colours); position(0)
                },
                count = asset.count,
                version = asset.version,
                bounds = CloudBounds.from(asset.positions),
            )
        }
    }
    private var scene = SceneBuffers.from(asset)
    private val projection = FloatArray(16)
    private val view = FloatArray(16)
    private val model = FloatArray(16)
    private val viewModel = FloatArray(16)
    private val mvp = FloatArray(16)
    private var program = 0

    override fun onSurfaceCreated(gl: GL10?, config: EGLConfig?) {
        program = link(VERTEX, FRAGMENT)
        GLES20.glClearColor(0.02f, 0.02f, 0.025f, 1f)
        GLES20.glEnable(GLES20.GL_DEPTH_TEST)
        sceneState.commit(scene.version, scene.count)
    }

    override fun onSurfaceChanged(gl: GL10?, width: Int, height: Int) {
        GLES20.glViewport(0, 0, width, height)
        Matrix.perspectiveM(projection, 0, ViewerCameraState.FIELD_OF_VIEW_DEGREES,
            width.toFloat() / max(1, height), 0.01f, 100f)
    }

    override fun onDrawFrame(gl: GL10?) {
        val drawing = scene
        GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT or GLES20.GL_DEPTH_BUFFER_BIT)
        Matrix.setLookAtM(view, 0, 0f, 0f, camera.distance, 0f, 0f, 0f, 0f, 1f, 0f)
        Matrix.translateM(view, 0, camera.panX, camera.panY, 0f)
        Matrix.setIdentityM(model, 0)
        Matrix.rotateM(model, 0, camera.pitch, 1f, 0f, 0f)
        Matrix.rotateM(model, 0, camera.yaw, 0f, 1f, 0f)
        Matrix.translateM(model, 0, -drawing.bounds.centreX, -drawing.bounds.centreY, -drawing.bounds.centreZ)
        Matrix.multiplyMM(viewModel, 0, view, 0, model, 0)
        Matrix.multiplyMM(mvp, 0, projection, 0, viewModel, 0)
        GLES20.glUseProgram(program)
        GLES20.glUniformMatrix4fv(GLES20.glGetUniformLocation(program, "uMvp"), 1, false, mvp, 0)
        val p = GLES20.glGetAttribLocation(program, "aPosition")
        val c = GLES20.glGetAttribLocation(program, "aColour")
        GLES20.glEnableVertexAttribArray(p)
        GLES20.glEnableVertexAttribArray(c)
        GLES20.glVertexAttribPointer(p, 3, GLES20.GL_FLOAT, false, 0, drawing.positions)
        GLES20.glVertexAttribPointer(c, 3, GLES20.GL_UNSIGNED_BYTE, true, 0, drawing.colours)
        GLES20.glDrawArrays(GLES20.GL_POINTS, 0, drawing.count)
    }

    /** Called exclusively through GLSurfaceView.queueEvent. */
    fun replace(asset: PointCloudAsset) {
        val replacement = SceneBuffers.from(asset)
        scene = replacement
        sceneState.commit(replacement.version, replacement.count)
    }

    private fun link(vertex: String, fragment: String): Int {
        fun compile(type: Int, source: String): Int = GLES20.glCreateShader(type).also {
            GLES20.glShaderSource(it, source); GLES20.glCompileShader(it)
        }
        return GLES20.glCreateProgram().also {
            GLES20.glAttachShader(it, compile(GLES20.GL_VERTEX_SHADER, vertex))
            GLES20.glAttachShader(it, compile(GLES20.GL_FRAGMENT_SHADER, fragment))
            GLES20.glLinkProgram(it)
        }
    }

    companion object {
        private const val VERTEX = "attribute vec3 aPosition; attribute vec3 aColour; uniform mat4 uMvp; varying vec3 vColour; void main(){ gl_Position=uMvp*vec4(aPosition,1.0); gl_PointSize=2.0; vColour=aColour; }"
        private const val FRAGMENT = "precision mediump float; varying vec3 vColour; void main(){ gl_FragColor=vec4(vColour,1.0); }"
    }
}
