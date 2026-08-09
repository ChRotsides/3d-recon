package com.recon.poselogger

import android.opengl.GLES30
import com.google.ar.core.PointCloud
import java.nio.FloatBuffer

/**
 * ARCore's feature points — the things it is actually tracking on.
 *
 * The client already reads this cloud every frame and then only *counts* it,
 * to feed the core's `TEXTURELESS` signal. Drawing it turns that count into
 * something a user can act on: points cluster on texture and vanish on blank
 * walls, so a wall with nothing on it visibly explains why the app is about to
 * complain, and why tracking is about to drift.
 *
 * That matters more here than it looks. The first real capture with this app
 * lost tracking four times and produced a bundle that could not be fused; the
 * warning sign — a view with almost nothing to track in it — was on screen the
 * whole time and invisible.
 *
 * Confidence, not just position: ARCore's fourth component says how much it
 * trusts each point, and a screenful of low-confidence points is a different
 * situation from a screenful of solid ones.
 */
class FeaturePointRenderer {

    private var program = 0
    private var positionAttrib = 0
    private var mvpUniform = 0

    fun createOnGlThread() {
        program = link(
            compile(GLES30.GL_VERTEX_SHADER, VERTEX_SHADER),
            compile(GLES30.GL_FRAGMENT_SHADER, FRAGMENT_SHADER),
        )
        positionAttrib = GLES30.glGetAttribLocation(program, "a_PositionConfidence")
        mvpUniform = GLES30.glGetUniformLocation(program, "u_Mvp")
    }

    /**
     * ARCore hands back a buffer of (x, y, z, confidence) in world space, which
     * is exactly the layout the shader wants, so it is bound directly rather
     * than copied into one of ours.
     */
    fun draw(cloud: PointCloud, viewProjection: FloatArray) {
        val points: FloatBuffer = cloud.points
        val n = points.remaining() / 4
        if (n == 0) return

        GLES30.glEnable(GLES30.GL_BLEND)
        GLES30.glBlendFunc(GLES30.GL_SRC_ALPHA, GLES30.GL_ONE_MINUS_SRC_ALPHA)
        GLES30.glDepthMask(false)
        GLES30.glUseProgram(program)
        GLES30.glUniformMatrix4fv(mvpUniform, 1, false, viewProjection, 0)

        points.position(0)
        GLES30.glVertexAttribPointer(positionAttrib, 4, GLES30.GL_FLOAT, false, 16, points)
        GLES30.glEnableVertexAttribArray(positionAttrib)
        GLES30.glDrawArrays(GLES30.GL_POINTS, 0, n)
        GLES30.glDisableVertexAttribArray(positionAttrib)

        GLES30.glDepthMask(true)
        GLES30.glDisable(GLES30.GL_BLEND)
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
        const val VERTEX_SHADER = """#version 300 es
uniform mat4 u_Mvp;
in vec4 a_PositionConfidence;
out float v_Confidence;
void main() {
  gl_Position = u_Mvp * vec4(a_PositionConfidence.xyz, 1.0);
  v_Confidence = a_PositionConfidence.w;
  gl_PointSize = clamp(9.0 / max(gl_Position.w, 0.1), 2.0, 9.0);
}
"""

        const val FRAGMENT_SHADER = """#version 300 es
precision mediump float;
in float v_Confidence;
out vec4 fragColor;
void main() {
  vec2 d = gl_PointCoord - vec2(0.5);
  if (dot(d, d) > 0.25) discard;
  // Pale where ARCore is unsure, white where it is certain. Deliberately low
  // contrast: these are context, not the thing the user is meant to look at.
  vec3 c = mix(vec3(0.95, 0.75, 0.35), vec3(1.0), clamp(v_Confidence, 0.0, 1.0));
  fragColor = vec4(c, 0.35 + 0.4 * clamp(v_Confidence, 0.0, 1.0));
}
"""
    }
}
