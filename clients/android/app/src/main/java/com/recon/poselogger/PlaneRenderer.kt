package com.recon.poselogger

import android.opengl.GLES30
import com.google.ar.core.Plane
import com.google.ar.core.TrackingState
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer

/**
 * ARCore's detected planes, drawn as a grid over the real floor and walls.
 *
 * The client already asked for plane detection — it is where the floor height
 * comes from (plan §6.1's station solver needs it) — and then threw the rest
 * away. Drawing them earns its place twice over:
 *
 *  * it shows the user *what the tracker has locked onto*, which is the single
 *    best predictor of whether tracking is about to fail. A blank wall with no
 *    plane on it is the `TEXTURELESS` nag before it fires;
 *  * a floor grid that visibly slides across the real floor is a relocalisation
 *    happening in front of you, rather than a number in a bundle discovered
 *    hours later.
 *
 * A grid rather than a flat wash: the eye reads the *regularity* of squares as
 * surface orientation, so a wall the tracker has skewed looks wrong immediately.
 */
class PlaneRenderer {

    private var program = 0
    private var positionAttrib = 0
    private var mvpUniform = 0
    private var modelUniform = 0
    private var colorUniform = 0

    private var vertices: FloatBuffer = allocate(64)
    private val model = FloatArray(16)
    private val mvp = FloatArray(16)

    fun createOnGlThread() {
        program = link(
            compile(GLES30.GL_VERTEX_SHADER, VERTEX_SHADER),
            compile(GLES30.GL_FRAGMENT_SHADER, FRAGMENT_SHADER),
        )
        positionAttrib = GLES30.glGetAttribLocation(program, "a_Position")
        mvpUniform = GLES30.glGetUniformLocation(program, "u_Mvp")
        modelUniform = GLES30.glGetUniformLocation(program, "u_Local")
        colorUniform = GLES30.glGetUniformLocation(program, "u_Color")
    }

    fun draw(planes: Collection<Plane>, viewProjection: FloatArray) {
        GLES30.glEnable(GLES30.GL_BLEND)
        GLES30.glBlendFunc(GLES30.GL_SRC_ALPHA, GLES30.GL_ONE_MINUS_SRC_ALPHA)
        GLES30.glDepthMask(false)
        GLES30.glUseProgram(program)

        for (plane in planes) {
            if (plane.trackingState != TrackingState.TRACKING) continue
            // ARCore merges planes as it learns; a subsumed one is a stale
            // duplicate and drawing it double-shades the same surface.
            if (plane.subsumedBy != null) continue
            drawOne(plane, viewProjection)
        }

        GLES30.glDepthMask(true)
        GLES30.glDisable(GLES30.GL_BLEND)
    }

    private fun drawOne(plane: Plane, viewProjection: FloatArray) {
        val polygon = plane.polygon  // (x, z) pairs in the plane's local frame
        val n = polygon.limit() / 2
        if (n < 3) return

        if (n * 3 > vertices.capacity()) vertices = allocate(n * 2)
        vertices.clear()
        polygon.rewind()
        for (i in 0 until n) {
            vertices.put(polygon.get(i * 2))
            vertices.put(0f)
            vertices.put(polygon.get(i * 2 + 1))
        }
        vertices.position(0)

        plane.centerPose.toMatrix(model, 0)
        android.opengl.Matrix.multiplyMM(mvp, 0, viewProjection, 0, model, 0)
        GLES30.glUniformMatrix4fv(mvpUniform, 1, false, mvp, 0)
        GLES30.glUniformMatrix4fv(modelUniform, 1, false, model, 0)

        // Floor, ceiling and walls get different colours because they are
        // different problems: the plan's coverage targets differ per surface,
        // and "which of these am I missing" is the question being answered.
        when (plane.type) {
            Plane.Type.HORIZONTAL_UPWARD_FACING -> GLES30.glUniform4f(colorUniform, 0.25f, 0.85f, 0.75f, 0.10f)
            Plane.Type.HORIZONTAL_DOWNWARD_FACING -> GLES30.glUniform4f(colorUniform, 0.45f, 0.60f, 1.00f, 0.10f)
            else -> GLES30.glUniform4f(colorUniform, 0.70f, 0.50f, 1.00f, 0.10f)
        }

        GLES30.glVertexAttribPointer(positionAttrib, 3, GLES30.GL_FLOAT, false, 0, vertices)
        GLES30.glEnableVertexAttribArray(positionAttrib)
        // The polygon is convex and centre-ordered, so a fan fills it exactly.
        GLES30.glDrawArrays(GLES30.GL_TRIANGLE_FAN, 0, n)
        GLES30.glDisableVertexAttribArray(positionAttrib)
    }

    private fun allocate(points: Int): FloatBuffer =
        ByteBuffer.allocateDirect(maxOf(points, 1) * 3 * 4)
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
        const val VERTEX_SHADER = """#version 300 es
uniform mat4 u_Mvp;
uniform mat4 u_Local;
in vec4 a_Position;
out vec2 v_Grid;
void main() {
  gl_Position = u_Mvp * a_Position;
  // Grid in the plane's own frame, so squares stay square on a tilted wall
  // instead of shearing with the viewing angle.
  v_Grid = a_Position.xz;
}
"""

        const val FRAGMENT_SHADER = """#version 300 es
precision mediump float;
uniform vec4 u_Color;
in vec2 v_Grid;
out vec4 fragColor;
void main() {
  // 25 cm squares: fine enough to read the surface, coarse enough not to
  // alias into a grey wash across a room.
  vec2 g = abs(fract(v_Grid / 0.25) - 0.5) / fwidth(v_Grid / 0.25);
  float line = 1.0 - min(min(g.x, g.y), 1.0);
  // Mostly the grid lines themselves: ARCore finds dozens of planes in a real
  // room, and any per-plane fill worth seeing on one becomes an opaque wash
  // when thirty of them overlap.
  fragColor = vec4(u_Color.rgb, u_Color.a * (0.12 + 0.88 * line));
}
"""
    }
}
