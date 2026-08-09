package com.recon.poselogger

import android.opengl.GLES30
import android.opengl.Matrix
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer

/**
 * The square the user lines the camera up with for the next scripted shot.
 *
 * A square in the world rather than a reticle painted on the glass: it has to
 * be findable when it is *behind* the user, and something anchored in the room
 * swings past as they turn, which reads as "over there" in a way a screen-space
 * crosshair never does.
 *
 * Colour carries the state, because the two failure modes need different
 * actions from the user and a single "not yet" cannot say which:
 *
 *  * white — turn towards it,
 *  * cyan — on target, hold still (this is the moment the frame would smear;
 *    motion blur was the largest rejection category on every real capture),
 *  * green — taken.
 */
class ShotTargetRenderer {

    private var program = 0
    private var positionAttrib = 0
    private var mvpUniform = 0
    private var colorUniform = 0

    private lateinit var outline: FloatBuffer
    private lateinit var corners: FloatBuffer
    private val model = FloatArray(16)
    private val mvp = FloatArray(16)

    fun createOnGlThread() {
        program = link(
            compile(GLES30.GL_VERTEX_SHADER, VERTEX_SHADER),
            compile(GLES30.GL_FRAGMENT_SHADER, FRAGMENT_SHADER),
        )
        positionAttrib = GLES30.glGetAttribLocation(program, "a_Position")
        mvpUniform = GLES30.glGetUniformLocation(program, "u_Mvp")
        colorUniform = GLES30.glGetUniformLocation(program, "u_Color")

        // A unit square in the XY plane, as a line loop.
        outline = floats(
            floatArrayOf(
                -1f, -1f, 0f,
                +1f, -1f, 0f,
                +1f, +1f, 0f,
                -1f, +1f, 0f,
            )
        )
        // Corner ticks, drawn as separate short lines so the square reads as a
        // viewfinder rather than a picture frame hanging in the room.
        corners = floats(
            floatArrayOf(
                -1f, -0.55f, 0f, -1f, -1f, 0f, -1f, -1f, 0f, -0.55f, -1f, 0f,
                +1f, -0.55f, 0f, +1f, -1f, 0f, +1f, -1f, 0f, +0.55f, -1f, 0f,
                +1f, +0.55f, 0f, +1f, +1f, 0f, +1f, +1f, 0f, +0.55f, +1f, 0f,
                -1f, +0.55f, 0f, -1f, +1f, 0f, -1f, +1f, 0f, -0.55f, +1f, 0f,
            )
        )
    }

    /**
     * @param direction unit world vector the shot wants the camera pointed along
     * @param cameraPos where the user is, so the square sits in front of them
     * @param up world up, to keep the square level
     * @param state 0 = turn towards it, 1 = hold still, 2 = taken
     * @param pulse 0..1, wrapping
     */
    fun draw(
        viewProjection: FloatArray,
        direction: FloatArray,
        cameraPos: FloatArray,
        up: FloatArray,
        state: Int,
        pulse: Float,
    ) {
        // Placed at a fixed distance along the requested direction rather than
        // on a wall: the target is an instruction about *where to point*, and
        // hit-testing it against real geometry would make it vanish through
        // doorways and windows — exactly where the shots matter most.
        val cx = cameraPos[0] + direction[0] * DISTANCE_M
        val cy = cameraPos[1] + direction[1] * DISTANCE_M
        val cz = cameraPos[2] + direction[2] * DISTANCE_M

        // Basis with +Z along the view direction, kept upright.
        val f = normalize(floatArrayOf(direction[0], direction[1], direction[2]))
        var r = cross(up, f)
        if (length(r) < 1e-3f) r = floatArrayOf(1f, 0f, 0f) // looking straight up or down
        r = normalize(r)
        val u = cross(f, r)

        model[0] = r[0]; model[1] = r[1]; model[2] = r[2]; model[3] = 0f
        model[4] = u[0]; model[5] = u[1]; model[6] = u[2]; model[7] = 0f
        model[8] = f[0]; model[9] = f[1]; model[10] = f[2]; model[11] = 0f
        model[12] = cx; model[13] = cy; model[14] = cz; model[15] = 1f

        val size = SIZE_M * (if (state == 1) 1f + 0.06f * kotlin.math.sin(pulse * 6.28f) else 1f)
        Matrix.scaleM(model, 0, size, size, size)
        Matrix.multiplyMM(mvp, 0, viewProjection, 0, model, 0)

        GLES30.glEnable(GLES30.GL_BLEND)
        GLES30.glBlendFunc(GLES30.GL_SRC_ALPHA, GLES30.GL_ONE_MINUS_SRC_ALPHA)
        GLES30.glDepthMask(false)
        GLES30.glUseProgram(program)
        GLES30.glUniformMatrix4fv(mvpUniform, 1, false, mvp, 0)
        GLES30.glLineWidth(if (state == 0) 4f else 8f)
        when (state) {
            1 -> GLES30.glUniform4f(colorUniform, 0.30f, 0.86f, 1.00f, 0.95f)
            2 -> GLES30.glUniform4f(colorUniform, 0.24f, 0.92f, 0.45f, 0.95f)
            else -> GLES30.glUniform4f(colorUniform, 1.0f, 1.0f, 1.0f, 0.75f)
        }

        outline.position(0)
        GLES30.glVertexAttribPointer(positionAttrib, 3, GLES30.GL_FLOAT, false, 0, outline)
        GLES30.glEnableVertexAttribArray(positionAttrib)
        GLES30.glDrawArrays(GLES30.GL_LINE_LOOP, 0, 4)
        corners.position(0)
        GLES30.glVertexAttribPointer(positionAttrib, 3, GLES30.GL_FLOAT, false, 0, corners)
        GLES30.glDrawArrays(GLES30.GL_LINES, 0, 16)
        GLES30.glDisableVertexAttribArray(positionAttrib)

        GLES30.glDepthMask(true)
        GLES30.glDisable(GLES30.GL_BLEND)
    }

    private fun floats(v: FloatArray): FloatBuffer =
        ByteBuffer.allocateDirect(v.size * 4).order(ByteOrder.nativeOrder())
            .asFloatBuffer().apply { put(v); position(0) }

    private fun cross(a: FloatArray, b: FloatArray) = floatArrayOf(
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )

    private fun length(a: FloatArray) =
        kotlin.math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])

    private fun normalize(a: FloatArray): FloatArray {
        val l = length(a)
        return if (l < 1e-6f) a else floatArrayOf(a[0] / l, a[1] / l, a[2] / l)
    }

    private fun compile(type: Int, source: String): Int {
        val shader = GLES30.glCreateShader(type)
        GLES30.glShaderSource(shader, source)
        GLES30.glCompileShader(shader)
        val ok = IntArray(1)
        GLES30.glGetShaderiv(shader, GLES30.GL_COMPILE_STATUS, ok, 0)
        check(ok[0] != 0) { "shader compile failed: ${GLES30.glGetShaderInfoLog(shader)}" }
        return shader
    }

    private fun link(vertex: Int, fragment: Int): Int {
        val p = GLES30.glCreateProgram()
        GLES30.glAttachShader(p, vertex)
        GLES30.glAttachShader(p, fragment)
        GLES30.glLinkProgram(p)
        val ok = IntArray(1)
        GLES30.glGetProgramiv(p, GLES30.GL_LINK_STATUS, ok, 0)
        check(ok[0] != 0) { "program link failed: ${GLES30.glGetProgramInfoLog(p)}" }
        return p
    }

    private companion object {
        /** How far in front of the user the square hangs. */
        const val DISTANCE_M = 2.0f
        /** Half-width at that distance — about a third of the view. */
        const val SIZE_M = 0.55f

        const val VERTEX_SHADER = """#version 300 es
uniform mat4 u_Mvp;
in vec4 a_Position;
void main() { gl_Position = u_Mvp * a_Position; }
"""

        const val FRAGMENT_SHADER = """#version 300 es
precision mediump float;
uniform vec4 u_Color;
out vec4 fragColor;
void main() { fragColor = u_Color; }
"""
    }
}
