package com.recon.poselogger

import android.opengl.GLES30
import android.opengl.Matrix
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import kotlin.math.cos
import kotlin.math.sin

/**
 * The AR floor ring from plan §6.1 step 4 — where the next-best-station solver
 * wants the user to stand.
 *
 * Two concentric rings: a solid one marking the spot and a slower pulse
 * expanding past it, because a flat ring on a patterned floor at a glance is
 * easy to miss and the motion is what catches the eye. The solver hands back a
 * floor-plane position, so the geometry is built once in the XZ plane at the
 * origin and translated per draw.
 */
class StationRenderer {

    private var program = 0
    private var positionAttrib = 0
    private var mvpUniform = 0
    private var colorUniform = 0

    private lateinit var ring: FloatBuffer
    private lateinit var arrow: FloatBuffer
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

        // A unit-radius annulus as a triangle strip, scaled at draw time.
        val verts = FloatArray((SEGMENTS + 1) * 6)
        for (i in 0..SEGMENTS) {
            val a = (i.toFloat() / SEGMENTS) * (2.0 * Math.PI).toFloat()
            val c = cos(a)
            val s = sin(a)
            verts[i * 6] = c * INNER
            verts[i * 6 + 1] = 0f
            verts[i * 6 + 2] = s * INNER
            verts[i * 6 + 3] = c
            verts[i * 6 + 4] = 0f
            verts[i * 6 + 5] = s
        }
        ring = ByteBuffer.allocateDirect(verts.size * 4).order(ByteOrder.nativeOrder())
            .asFloatBuffer().apply { put(verts); position(0) }

        // A chevron lying flat on the floor, pointing down +X, drawn as a
        // triangle strip. Flat on the floor rather than billboarded upright:
        // it then reads as part of the room, and the foreshortening tells the
        // user how far away the direction is being indicated.
        val a = floatArrayOf(
            0.00f, 0f, -0.10f,
            0.00f, 0f, -0.035f,
            0.13f, 0f, 0.00f,
            0.00f, 0f, 0.035f,
            0.00f, 0f, 0.10f,
        )
        arrow = ByteBuffer.allocateDirect(a.size * 4).order(ByteOrder.nativeOrder())
            .asFloatBuffer().apply { put(a); position(0) }
    }

    /**
     * A chevron on the floor pointing from the user towards the station.
     *
     * The ring alone answers "where", but only once it is on screen. This
     * answers "which way" from wherever the user is looking, and unlike the
     * HUD's screen-edge arrow it sits in the room, so its direction is read the
     * same way as everything else in view.
     *
     * Placed a fixed distance ahead of the user rather than at the station, so
     * it stays visible when the station is metres away or behind a sofa.
     */
    fun drawArrow(
        viewProjection: FloatArray,
        userPos: FloatArray,
        stationPos: FloatArray,
        floorY: Float,
        phase: Float,
    ) {
        val dx = stationPos[0] - userPos[0]
        val dz = stationPos[2] - userPos[2]
        val len = kotlin.math.sqrt(dx * dx + dz * dz)
        if (len < RADIUS_M) return // standing on it; the ring says the rest

        GLES30.glEnable(GLES30.GL_BLEND)
        GLES30.glBlendFunc(GLES30.GL_SRC_ALPHA, GLES30.GL_ONE_MINUS_SRC_ALPHA)
        GLES30.glDepthMask(false)
        GLES30.glUseProgram(program)

        val ux = dx / len
        val uz = dz / len
        // Yaw so the chevron's +X points along the direction of travel. atan2's
        // arguments are (z, x) here because yaw is measured about +Y from +X.
        val yaw = kotlin.math.atan2(uz, ux)
        val ahead = minOf(ARROW_AHEAD_M, len - RADIUS_M * 0.5f)

        // Three chevrons marching towards the station, the nearest brightest.
        for (i in 0 until 3) {
            val slide = ((phase + i * 0.33f) % 1f)
            val d = ahead + slide * ARROW_SPACING_M
            android.opengl.Matrix.setIdentityM(model, 0)
            android.opengl.Matrix.translateM(
                model, 0, userPos[0] + ux * d, floorY + 0.01f, userPos[2] + uz * d
            )
            android.opengl.Matrix.rotateM(model, 0, -yaw * 180f / Math.PI.toFloat(), 0f, 1f, 0f)
            android.opengl.Matrix.multiplyMM(mvp, 0, viewProjection, 0, model, 0)
            GLES30.glUniformMatrix4fv(mvpUniform, 1, false, mvp, 0)
            // Fades as it travels, so the motion reads as flow rather than blink.
            GLES30.glUniform4f(colorUniform, 0.30f, 0.86f, 1.0f, 0.75f * (1f - slide))

            arrow.position(0)
            GLES30.glVertexAttribPointer(positionAttrib, 3, GLES30.GL_FLOAT, false, 0, arrow)
            GLES30.glEnableVertexAttribArray(positionAttrib)
            GLES30.glDrawArrays(GLES30.GL_TRIANGLE_STRIP, 0, 5)
            GLES30.glDisableVertexAttribArray(positionAttrib)
        }

        GLES30.glDepthMask(true)
        GLES30.glDisable(GLES30.GL_BLEND)
    }

    /**
     * @param viewProjection the camera's projection * view
     * @param position the station, on the floor plane, in world coordinates
     * @param phase 0..1, wrapping — drives both the pulse radius and its fade
     */
    fun draw(viewProjection: FloatArray, position: FloatArray, phase: Float) {
        GLES30.glEnable(GLES30.GL_BLEND)
        GLES30.glBlendFunc(GLES30.GL_SRC_ALPHA, GLES30.GL_ONE_MINUS_SRC_ALPHA)
        GLES30.glDepthMask(false)
        GLES30.glUseProgram(program)

        drawRing(viewProjection, position, RADIUS_M, 0.85f)
        // The pulse expands and fades, so it reads as a beacon rather than a
        // second target the user might walk to instead.
        val pulse = RADIUS_M * (1f + phase * 1.6f)
        drawRing(viewProjection, position, pulse, 0.5f * (1f - phase))

        GLES30.glDepthMask(true)
        GLES30.glDisable(GLES30.GL_BLEND)
    }

    private fun drawRing(viewProjection: FloatArray, position: FloatArray, radius: Float, alpha: Float) {
        if (alpha <= 0.01f) return
        Matrix.setIdentityM(model, 0)
        Matrix.translateM(model, 0, position[0], position[1], position[2])
        Matrix.scaleM(model, 0, radius, 1f, radius)
        Matrix.multiplyMM(mvp, 0, viewProjection, 0, model, 0)

        GLES30.glUniformMatrix4fv(mvpUniform, 1, false, mvp, 0)
        GLES30.glUniform4f(colorUniform, 0.30f, 0.86f, 1.0f, alpha)

        ring.position(0)
        GLES30.glVertexAttribPointer(positionAttrib, 3, GLES30.GL_FLOAT, false, 0, ring)
        GLES30.glEnableVertexAttribArray(positionAttrib)
        GLES30.glDrawArrays(GLES30.GL_TRIANGLE_STRIP, 0, (SEGMENTS + 1) * 2)
        GLES30.glDisableVertexAttribArray(positionAttrib)
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
        const val SEGMENTS = 48
        /** Ring radius: a standing footprint, not a target to aim a foot at. */
        const val RADIUS_M = 0.40f
        /** Inner edge as a fraction of the outer, giving the band its width. */
        const val INNER = 0.82f

        /** How far ahead of the user the first chevron sits. */
        const val ARROW_AHEAD_M = 1.8f

        /** How far the chevrons travel before recycling. */
        const val ARROW_SPACING_M = 0.6f

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
