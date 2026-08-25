package com.recon.poselogger

import android.opengl.GLES30
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer

/**
 * Draws the tapped footprint: a dot at every corner, a line along every wall.
 *
 * Two states the user has to be able to tell apart at a glance while walking:
 * the loop is still open (keep tapping), or the first corner is now a target
 * they can close on. The open outline is drawn in the same teal as the rest of
 * the guidance; the first corner pulses white once three corners exist, which
 * is the only animation in the app and is there because "you can finish now"
 * is the one thing the user cannot work out from the geometry alone.
 *
 * Depth testing is off. These lines mark real corners that are frequently
 * behind furniture, and an outline that disappears behind a sofa reads as a
 * bug; drawn on top, it reads as an annotation, which is what it is.
 */
class OutlineRenderer {

    private var program = 0
    private var positionAttrib = 0
    private var mvpUniform = 0
    private var colourUniform = 0
    private var pointSizeUniform = 0
    private var drawPointsUniform = 0

    private var buffer: FloatBuffer = allocate(64)

    fun createOnGlThread() {
        program = link(
            compile(GLES30.GL_VERTEX_SHADER, VERTEX_SHADER),
            compile(GLES30.GL_FRAGMENT_SHADER, FRAGMENT_SHADER),
        )
        positionAttrib = GLES30.glGetAttribLocation(program, "a_Position")
        mvpUniform = GLES30.glGetUniformLocation(program, "u_Mvp")
        colourUniform = GLES30.glGetUniformLocation(program, "u_Colour")
        pointSizeUniform = GLES30.glGetUniformLocation(program, "u_PointSize")
        drawPointsUniform = GLES30.glGetUniformLocation(program, "u_DrawPoints")
    }

    /**
     * @param pulse 0..1, drives the "you can close here" highlight on corner 0.
     */
    fun draw(
        outline: FloorOutline,
        viewProjection: FloatArray,
        pulse: Float,
        cameraRight: FloatArray,
        cameraUp: FloatArray,
    ) {
        if (outline.isEmpty()) return

        GLES30.glEnable(GLES30.GL_BLEND)
        GLES30.glBlendFunc(GLES30.GL_SRC_ALPHA, GLES30.GL_ONE_MINUS_SRC_ALPHA)
        GLES30.glDepthMask(false)
        GLES30.glDisable(GLES30.GL_DEPTH_TEST)
        GLES30.glUseProgram(program)
        GLES30.glUniformMatrix4fv(mvpUniform, 1, false, viewProjection, 0)
        GLES30.glLineWidth(6f)

        val segments = outline.segments()
        if (segments.isNotEmpty()) {
            val data = ensure(segments.size * 6)
            for ((a, b) in segments) {
                data.put(a.x).put(a.y).put(a.z)
                data.put(b.x).put(b.y).put(b.z)
            }
            data.position(0)
            // Green once the room is a closed measurement, teal while in progress.
            if (outline.isClosed) uniformColour(0.30f, 0.90f, 0.45f, 0.95f)
            else uniformColour(0.18f, 0.83f, 0.75f, 0.85f)
            drawArrays(GLES30.GL_LINES, data, segments.size * 2)
        }

        // Corners already placed.
        val corners = outline.cornersList()
        val pts = ensure(corners.size * 3)
        for (c in corners) pts.put(c.x).put(c.y).put(c.z)
        pts.position(0)
        GLES30.glUniform1f(pointSizeUniform, 22f)
        uniformColour(1f, 1f, 1f, 0.9f)
        drawArrays(GLES30.GL_POINTS, pts, corners.size)

        // A single selected corner gets a tethered handle. Drawing handles for
        // every corner made the room unreadable; a normal tap selects one, then
        // the clear amber handle is where the user starts the drag.
        outline.selectedCorner()?.let { selected ->
            val handle = dragHandle(selected, cameraRight, cameraUp)
            val tether = ensure(6)
            tether.put(selected.x).put(selected.y).put(selected.z)
            tether.put(handle.x).put(handle.y).put(handle.z)
            tether.position(0)
            GLES30.glLineWidth(3f)
            uniformColour(1f, 0.67f, 0.15f, 0.85f)
            drawArrays(GLES30.GL_LINES, tether, 2)

            val handlePoint = ensure(3)
            handlePoint.put(handle.x).put(handle.y).put(handle.z)
            handlePoint.position(0)
            GLES30.glUniform1f(pointSizeUniform, 44f + 8f * pulse)
            uniformColour(1f, 0.67f, 0.15f, 0.95f)
            drawArrays(GLES30.GL_POINTS, handlePoint, 1)
        }

        // The held corner, drawn over the plain dots so it is unambiguous which
        // one the next tap will move.
        outline.selectedCorner()?.let { sel ->
            val one = ensure(3)
            one.put(sel.x).put(sel.y).put(sel.z)
            one.position(0)
            GLES30.glUniform1f(pointSizeUniform, 40f + 10f * pulse)
            uniformColour(1f, 0.35f, 0.35f, 0.9f)
            drawArrays(GLES30.GL_POINTS, one, 1)
        }

        // The closing target, drawn last so it sits over its own corner dot.
        if (outline.canClose()) {
            outline.first()?.let { f ->
                val one = ensure(3)
                one.put(f.x).put(f.y).put(f.z)
                one.position(0)
                GLES30.glUniform1f(pointSizeUniform, 34f + 12f * pulse)
                uniformColour(1f, 0.85f, 0.25f, 0.55f + 0.45f * pulse)
                drawArrays(GLES30.GL_POINTS, one, 1)
            }
        }

        GLES30.glEnable(GLES30.GL_DEPTH_TEST)
        GLES30.glDepthMask(true)
        GLES30.glDisable(GLES30.GL_BLEND)
    }

    private fun uniformColour(r: Float, g: Float, b: Float, a: Float) {
        GLES30.glUniform4f(colourUniform, r, g, b, a)
    }

    private fun drawArrays(mode: Int, data: FloatBuffer, count: Int) {
        if (count == 0) return
        GLES30.glUniform1i(drawPointsUniform, if (mode == GLES30.GL_POINTS) 1 else 0)
        GLES30.glVertexAttribPointer(positionAttrib, 3, GLES30.GL_FLOAT, false, 12, data)
        GLES30.glEnableVertexAttribArray(positionAttrib)
        GLES30.glDrawArrays(mode, 0, count)
        GLES30.glDisableVertexAttribArray(positionAttrib)
    }

    /** Grow-only scratch buffer: this runs at 60 Hz and must not allocate. */
    private fun ensure(floats: Int): FloatBuffer {
        if (buffer.capacity() < floats) buffer = allocate(floats * 2)
        buffer.clear()
        return buffer
    }

    private fun allocate(floats: Int): FloatBuffer =
        ByteBuffer.allocateDirect(floats * 4).order(ByteOrder.nativeOrder()).asFloatBuffer()

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

    companion object {
        /** World-space handle position, with an intentionally visible tether. */
        const val HANDLE_RIGHT_M = 0.18f
        const val HANDLE_UP_M = 0.14f

        const val VERTEX_SHADER = """#version 300 es
uniform mat4 u_Mvp;
uniform float u_PointSize;
in vec4 a_Position;
void main() {
  gl_Position = u_Mvp * vec4(a_Position.xyz, 1.0);
  gl_PointSize = u_PointSize;
}
"""

        const val FRAGMENT_SHADER = """#version 300 es
precision mediump float;
uniform vec4 u_Colour;
uniform int u_DrawPoints;
out vec4 fragColor;
void main() {
  if (u_DrawPoints == 1) {
    vec2 fromCentre = gl_PointCoord - vec2(0.5);
    if (dot(fromCentre, fromCentre) > 0.25) discard;
  }
  fragColor = u_Colour;
}
"""
        /** Kept public so hit testing uses exactly the handle the renderer drew. */
        fun dragHandle(
            corner: FloorOutline.Corner,
            cameraRight: FloatArray,
            cameraUp: FloatArray,
        ): FloorOutline.Corner = FloorOutline.Corner(
            corner.x + cameraRight[0] * HANDLE_RIGHT_M + cameraUp[0] * HANDLE_UP_M,
            corner.y + cameraRight[1] * HANDLE_RIGHT_M + cameraUp[1] * HANDLE_UP_M,
            corner.z + cameraRight[2] * HANDLE_RIGHT_M + cameraUp[2] * HANDLE_UP_M,
        )
    }
}
