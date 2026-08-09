package com.recon.poselogger

import android.opengl.GLES30
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer

/**
 * Draws the core's coverage grid over the real surfaces: amber where a voxel has
 * been seen but has not cleared the quality bar, green where it has.
 *
 * This is the answer to "what have I actually mapped?", and it is deliberately
 * the *core's* view of that rather than a client-side redraw of the depth map —
 * a point the user has looked at from one bad angle is not mapped, and only the
 * grid's `quality / q_min` knows the difference.
 *
 * **Occlusion is not handled.** There is no depth buffer for the real world
 * here, so coverage on a far wall shows through a near one. Fixing it properly
 * means depth-testing against ARCore's depth texture; left undone because
 * seeing that the wall behind you is still amber is useful rather than
 * misleading, and a wrong occlusion test looks like missing coverage.
 */
class CoverageRenderer {

    private var program = 0
    private var positionAttrib = 0
    private var fillAttrib = 0
    private var mvpUniform = 0
    private var scaleUniform = 0

    private var buffer: FloatBuffer = allocate(0)
    private var count = 0

    /** Metres per voxel, so a drawn dot is about the size of the thing it stands for. */
    var voxelM: Float = 0.05f

    fun createOnGlThread() {
        program = link(
            compile(GLES30.GL_VERTEX_SHADER, VERTEX_SHADER),
            compile(GLES30.GL_FRAGMENT_SHADER, FRAGMENT_SHADER),
        )
        positionAttrib = GLES30.glGetAttribLocation(program, "a_Position")
        fillAttrib = GLES30.glGetAttribLocation(program, "a_Fill")
        mvpUniform = GLES30.glGetUniformLocation(program, "u_Mvp")
        scaleUniform = GLES30.glGetUniformLocation(program, "u_PointScale")
    }

    /**
     * Replace the drawn set. `points` is flat xyz, `fill` one entry per point,
     * exactly as `FfiCoverageCloud` delivers them.
     */
    fun update(points: List<Float>, fill: List<Float>) {
        val n = minOf(fill.size, points.size / 3)
        if (n * STRIDE_FLOATS > buffer.capacity()) {
            // Grow generously: the grid only ever gets bigger during a capture,
            // and reallocating a direct buffer every second would churn.
            buffer = allocate(maxOf(n * 2, 4096))
        }
        buffer.clear()
        for (i in 0 until n) {
            buffer.put(points[i * 3])
            buffer.put(points[i * 3 + 1])
            buffer.put(points[i * 3 + 2])
            buffer.put(fill[i])
        }
        buffer.position(0)
        count = n
    }

    fun clear() {
        count = 0
    }

    /**
     * `viewportHeight` and the projection's vertical term are what turn a voxel
     * size in metres into a pixel size on screen, so dots track the geometry
     * instead of being a fixed blob that swamps the room from across it.
     */
    fun draw(mvp: FloatArray, projection: FloatArray, viewportHeight: Int) {
        if (count == 0) return
        val pixelsPerMetreAtUnitDepth = projection[5] * viewportHeight * 0.5f

        GLES30.glEnable(GLES30.GL_BLEND)
        GLES30.glBlendFunc(GLES30.GL_SRC_ALPHA, GLES30.GL_ONE_MINUS_SRC_ALPHA)
        GLES30.glDepthMask(false)
        GLES30.glUseProgram(program)
        GLES30.glUniformMatrix4fv(mvpUniform, 1, false, mvp, 0)
        GLES30.glUniform1f(scaleUniform, pixelsPerMetreAtUnitDepth * voxelM)

        buffer.position(0)
        GLES30.glVertexAttribPointer(
            positionAttrib, 3, GLES30.GL_FLOAT, false, STRIDE_FLOATS * 4, buffer
        )
        buffer.position(3)
        GLES30.glVertexAttribPointer(
            fillAttrib, 1, GLES30.GL_FLOAT, false, STRIDE_FLOATS * 4, buffer
        )
        GLES30.glEnableVertexAttribArray(positionAttrib)
        GLES30.glEnableVertexAttribArray(fillAttrib)

        GLES30.glDrawArrays(GLES30.GL_POINTS, 0, count)

        GLES30.glDisableVertexAttribArray(positionAttrib)
        GLES30.glDisableVertexAttribArray(fillAttrib)
        GLES30.glDepthMask(true)
        GLES30.glDisable(GLES30.GL_BLEND)
    }

    private fun allocate(points: Int): FloatBuffer =
        ByteBuffer.allocateDirect(maxOf(points, 1) * STRIDE_FLOATS * 4)
            .order(ByteOrder.nativeOrder()).asFloatBuffer()

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
        /** x, y, z, fill. */
        const val STRIDE_FLOATS = 4

        const val VERTEX_SHADER = """#version 300 es
uniform mat4 u_Mvp;
uniform float u_PointScale;
in vec4 a_Position;
in float a_Fill;
out float v_Fill;
void main() {
  gl_Position = u_Mvp * a_Position;
  v_Fill = a_Fill;
  // w is view-space depth, so this is the voxel's on-screen size, clamped so a
  // dot underfoot does not fill the screen and a distant one stays visible.
  // Capped hard at the near end. A 5 cm voxel a metre from the lens is
  // geometrically ~40 px, and a screenful of 40 px blobs hides the room the
  // user is being asked to look at — the overlay stops being an overlay.
  gl_PointSize = clamp(u_PointScale / max(gl_Position.w, 0.1), 2.5, 14.0);
}
"""

        const val FRAGMENT_SHADER = """#version 300 es
precision mediump float;
in float v_Fill;
out vec4 fragColor;
void main() {
  vec2 d = gl_PointCoord - vec2(0.5);
  float r2 = dot(d, d);
  if (r2 > 0.25) discard;               // round dots rather than squares
  vec3 weak = vec3(1.00, 0.62, 0.14);   // seen, but not from enough angles
  vec3 done = vec3(0.24, 0.92, 0.45);   // cleared the quality bar
  vec3 c = mix(weak, done, smoothstep(0.5, 1.0, v_Fill));
  float edge = 1.0 - smoothstep(0.14, 0.25, r2);
  // Covered voxels fade back and unfinished ones stay forward: the eye should
  // be pulled to what still needs work, not to what is already done.
  float weight = mix(0.42, 0.22, smoothstep(0.5, 1.0, v_Fill));
  fragColor = vec4(c, weight * edge);
}
"""
    }
}
