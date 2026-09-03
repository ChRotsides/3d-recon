package com.recon.videoscan.live

import android.opengl.GLES20
import android.opengl.GLSurfaceView
import android.opengl.Matrix
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import javax.microedition.khronos.egl.EGLConfig
import javax.microedition.khronos.opengles.GL10
import kotlin.math.cos
import kotlin.math.sin

/**
 * Point-cloud renderer shared by the two 3D views. Points are x,y,z,r,g,b floats in the Y-up world.
 * mode POV: camera placed at the live pose (upright, video field of view). mode ORBIT: touch orbit around a target.
 */
class CloudRenderer(private val pov: Boolean) : GLSurfaceView.Renderer {
    @Volatile var points: FloatBuffer? = null; @Volatile var pointCount = 0
    @Volatile var path = FloatArray(0)                 // trajectory x,y,z
    @Volatile var pose = floatArrayOf(0f, 0f, 0f, 0f, 0f, 0f, 1f)   // px,py,pz,qx,qy,qz,qw camera->world
    @Volatile var fovDeg = 60f
    var yaw = 0.6f; var pitch = 0.9f; var dist = 5f; val target = floatArrayOf(0f, 0f, 0f)
    private var program = 0; private var aPos = 0; private var aCol = 0; private var uMvp = 0; private var uSize = 0; private var uColor = 0; private var uUseColor = 0
    private val proj = FloatArray(16); private val view = FloatArray(16); private val mvp = FloatArray(16)
    private var aspect = 1f; private var vbo = IntArray(1); private var uploaded = -1
    private val frustum = FloatArray(16 * 3)

    override fun onSurfaceCreated(gl: GL10?, config: EGLConfig?) {
        val vs = """attribute vec3 aPos; attribute vec3 aCol; uniform mat4 uMvp; uniform float uSize; varying vec3 vCol;
            void main(){ gl_Position = uMvp * vec4(aPos,1.0); gl_PointSize = min(uSize / max(gl_Position.w, 0.2), 9.0); vCol = aCol; }"""
        val fs = """precision mediump float; varying vec3 vCol; uniform vec3 uColor; uniform float uUseColor;
            void main(){ gl_FragColor = vec4(mix(uColor, vCol, uUseColor), 1.0); }"""
        program = GLES20.glCreateProgram()
        for ((type, src) in listOf(GLES20.GL_VERTEX_SHADER to vs, GLES20.GL_FRAGMENT_SHADER to fs)) {
            val sh = GLES20.glCreateShader(type); GLES20.glShaderSource(sh, src); GLES20.glCompileShader(sh); GLES20.glAttachShader(program, sh)
        }
        GLES20.glLinkProgram(program)
        aPos = GLES20.glGetAttribLocation(program, "aPos"); aCol = GLES20.glGetAttribLocation(program, "aCol")
        uMvp = GLES20.glGetUniformLocation(program, "uMvp"); uSize = GLES20.glGetUniformLocation(program, "uSize")
        uColor = GLES20.glGetUniformLocation(program, "uColor"); uUseColor = GLES20.glGetUniformLocation(program, "uUseColor")
        GLES20.glGenBuffers(1, vbo, 0); GLES20.glClearColor(0.09f, 0.09f, 0.09f, 1f); GLES20.glEnable(GLES20.GL_DEPTH_TEST)
    }
    override fun onSurfaceChanged(gl: GL10?, width: Int, height: Int) { GLES20.glViewport(0, 0, width, height); aspect = width.toFloat() / height }

    override fun onDrawFrame(gl: GL10?) {
        GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT or GLES20.GL_DEPTH_BUFFER_BIT)
        GLES20.glUseProgram(program)
        val p = pose
        if (pov) {
            // camera at the pose; upright: the sensor-native frame is rolled 90 degrees (phone held portrait)
            Matrix.perspectiveM(proj, 0, fovDeg, aspect, 0.05f, 100f)
            val r = quatToMat(p[3], p[4], p[5], p[6])
            // camera looks down -Z of its own frame; up = +X of the sensor frame after the 90-degree roll
            // column-major: camera x axis = (r0,r1,r2), z axis = (r8,r9,r10). Looks along -z; world-up is the sensor frame's -x (portrait)
            val fwd = floatArrayOf(-r[8], -r[9], -r[10]); val up = floatArrayOf(-r[0], -r[1], -r[2])
            Matrix.setLookAtM(view, 0, p[0], p[1], p[2], p[0] + fwd[0], p[1] + fwd[1], p[2] + fwd[2], up[0], up[1], up[2])
        } else {
            Matrix.perspectiveM(proj, 0, 50f, aspect, 0.05f, 100f)
            val ex = target[0] + dist * cos(pitch) * sin(yaw); val ey = target[1] + dist * sin(pitch); val ez = target[2] + dist * cos(pitch) * cos(yaw)
            Matrix.setLookAtM(view, 0, ex, ey, ez, target[0], target[1], target[2], 0f, 1f, 0f)
        }
        Matrix.multiplyMM(mvp, 0, proj, 0, view, 0); GLES20.glUniformMatrix4fv(uMvp, 1, false, mvp, 0)
        // grid (orbit view only)
        if (!pov) drawLines(gridLines(), 0.25f, 0.25f, 0.25f)
        // points
        val pts = points
        if (pts != null && pointCount > 0) {
            GLES20.glBindBuffer(GLES20.GL_ARRAY_BUFFER, vbo[0])
            if (uploaded != System.identityHashCode(pts)) { pts.position(0); GLES20.glBufferData(GLES20.GL_ARRAY_BUFFER, pointCount * 24, pts, GLES20.GL_DYNAMIC_DRAW); uploaded = System.identityHashCode(pts) }
            GLES20.glEnableVertexAttribArray(aPos); GLES20.glEnableVertexAttribArray(aCol)
            GLES20.glVertexAttribPointer(aPos, 3, GLES20.GL_FLOAT, false, 24, 0); GLES20.glVertexAttribPointer(aCol, 3, GLES20.GL_FLOAT, false, 24, 12)
            GLES20.glUniform1f(uUseColor, 1f); GLES20.glUniform1f(uSize, if (pov) 14f else 8f)
            GLES20.glDrawArrays(GLES20.GL_POINTS, 0, pointCount)
            GLES20.glDisableVertexAttribArray(aCol); GLES20.glBindBuffer(GLES20.GL_ARRAY_BUFFER, 0)
        }
        // trajectory and frustum (orbit view)
        if (!pov) { drawLines(path, 1f, 0.67f, 0.27f, strip = true); drawLines(frustumLines(p), 1f, 0.67f, 0.27f) }
    }
    private fun drawLines(v: FloatArray, r: Float, g: Float, b: Float, strip: Boolean = false) {
        if (v.size < 6) return
        val buf = ByteBuffer.allocateDirect(v.size * 4).order(ByteOrder.nativeOrder()).asFloatBuffer().put(v); buf.position(0)
        GLES20.glBindBuffer(GLES20.GL_ARRAY_BUFFER, 0)
        GLES20.glEnableVertexAttribArray(aPos); GLES20.glDisableVertexAttribArray(aCol)
        GLES20.glVertexAttribPointer(aPos, 3, GLES20.GL_FLOAT, false, 12, buf)
        GLES20.glUniform1f(uUseColor, 0f); GLES20.glUniform3f(uColor, r, g, b); GLES20.glLineWidth(2f)
        GLES20.glDrawArrays(if (strip) GLES20.GL_LINE_STRIP else GLES20.GL_LINES, 0, v.size / 3)
    }
    private fun gridLines(): FloatArray { val n = 12; val out = FloatArray(n * 2 * 2 * 3 + 12); var k = 0; val y = target[1] - 1.3f
        for (i in -n / 2..n / 2) { for (v in floatArrayOf(i.toFloat(), y, -n / 2f, i.toFloat(), y, n / 2f, -n / 2f, y, i.toFloat(), n / 2f, y, i.toFloat())) if (k < out.size) out[k++] = v }
        return out.copyOf(k) }
    private fun frustumLines(p: FloatArray): FloatArray {
        val r = quatToMat(p[3], p[4], p[5], p[6]); val d = 0.4f; val t = d * 0.35f; val s = d * 0.62f
        val c = floatArrayOf(0f, 0f, 0f, -s, -t, -d, s, -t, -d, s, t, -d, -s, t, -d)
        fun w(i: Int): FloatArray { val x = c[i * 3]; val y = c[i * 3 + 1]; val z = c[i * 3 + 2]
            return floatArrayOf(p[0] + r[0] * x + r[4] * y + r[8] * z, p[1] + r[1] * x + r[5] * y + r[9] * z, p[2] + r[2] * x + r[6] * y + r[10] * z) }
        val idx = intArrayOf(0, 1, 0, 2, 0, 3, 0, 4, 1, 2, 2, 3, 3, 4, 4, 1); var k = 0
        for (i in idx) { val q = w(i); frustum[k++] = q[0]; frustum[k++] = q[1]; frustum[k++] = q[2] }
        return frustum
    }
    private fun quatToMat(x: Float, y: Float, z: Float, w: Float): FloatArray {   // column-major 4x4
        val m = FloatArray(16); m[15] = 1f
        m[0] = 1 - 2 * (y * y + z * z); m[4] = 2 * (x * y - z * w); m[8] = 2 * (x * z + y * w)
        m[1] = 2 * (x * y + z * w); m[5] = 1 - 2 * (x * x + z * z); m[9] = 2 * (y * z - x * w)
        m[2] = 2 * (x * z - y * w); m[6] = 2 * (y * z + x * w); m[10] = 1 - 2 * (x * x + y * y)
        return m
    }
}
