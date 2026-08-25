package com.recon.poselogger

import android.opengl.GLES30
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer

/** GPU point-splat renderer for the depth-warped previous keyframe. */
class AlignmentGhostRenderer {
    private var program = 0
    private var positionEdgeAttrib = 0
    private var colourStateAttrib = 0
    private var mvpUniform = 0
    private var scaleUniform = 0
    private var buffer: FloatBuffer = allocate(1)
    private var ghost: AlignmentGhost? = null
    private var states = FloatArray(0)
    private var count = 0

    fun createOnGlThread() {
        program = link(
            compile(GLES30.GL_VERTEX_SHADER, VERTEX_SHADER),
            compile(GLES30.GL_FRAGMENT_SHADER, FRAGMENT_SHADER),
        )
        positionEdgeAttrib = GLES30.glGetAttribLocation(program, "a_PositionEdge")
        colourStateAttrib = GLES30.glGetAttribLocation(program, "a_ColourState")
        mvpUniform = GLES30.glGetUniformLocation(program, "u_Mvp")
        scaleUniform = GLES30.glGetUniformLocation(program, "u_PointScale")
    }

    fun update(next: AlignmentGhost?) {
        ghost = next
        count = next?.positions?.size?.div(3) ?: 0
        states = FloatArray(count) { -1f }
        rebuildBuffer()
    }

    fun update(status: AlignmentStatus?) {
        if (status == null || status.pointStates.size != count) return
        states = status.pointStates
        rebuildBuffer()
    }

    private fun rebuildBuffer() {
        val current = ghost ?: run { count = 0; return }
        if (buffer.capacity() < count * STRIDE_FLOATS) buffer = allocate(count * 2)
        buffer.clear()
        for (index in 0 until count) {
            val position = index * 3
            buffer.put(current.positions[position])
            buffer.put(current.positions[position + 1])
            buffer.put(current.positions[position + 2])
            buffer.put(current.edges[index])
            buffer.put((current.colours[position].toInt() and 0xff) / 255f)
            buffer.put((current.colours[position + 1].toInt() and 0xff) / 255f)
            buffer.put((current.colours[position + 2].toInt() and 0xff) / 255f)
            buffer.put(states[index])
        }
        buffer.position(0)
    }

    fun draw(mvp: FloatArray, projection: FloatArray, viewportHeight: Int) {
        if (count == 0) return
        val pixelsPerMetreAtUnitDepth = projection[5] * viewportHeight * 0.5f
        GLES30.glEnable(GLES30.GL_BLEND)
        GLES30.glBlendFunc(GLES30.GL_SRC_ALPHA, GLES30.GL_ONE_MINUS_SRC_ALPHA)
        GLES30.glDepthMask(false)
        GLES30.glUseProgram(program)
        GLES30.glUniformMatrix4fv(mvpUniform, 1, false, mvp, 0)
        GLES30.glUniform1f(scaleUniform, pixelsPerMetreAtUnitDepth * POINT_SIZE_M)

        buffer.position(0)
        GLES30.glVertexAttribPointer(
            positionEdgeAttrib, 4, GLES30.GL_FLOAT, false, STRIDE_FLOATS * 4, buffer
        )
        buffer.position(4)
        GLES30.glVertexAttribPointer(
            colourStateAttrib, 4, GLES30.GL_FLOAT, false, STRIDE_FLOATS * 4, buffer
        )
        GLES30.glEnableVertexAttribArray(positionEdgeAttrib)
        GLES30.glEnableVertexAttribArray(colourStateAttrib)
        GLES30.glDrawArrays(GLES30.GL_POINTS, 0, count)
        GLES30.glDisableVertexAttribArray(positionEdgeAttrib)
        GLES30.glDisableVertexAttribArray(colourStateAttrib)
        GLES30.glDepthMask(true)
        GLES30.glDisable(GLES30.GL_BLEND)
    }

    private fun allocate(points: Int): FloatBuffer = ByteBuffer
        .allocateDirect(maxOf(points, 1) * STRIDE_FLOATS * 4)
        .order(ByteOrder.nativeOrder())
        .asFloatBuffer()

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
        val linked = GLES30.glCreateProgram()
        GLES30.glAttachShader(linked, vertex)
        GLES30.glAttachShader(linked, fragment)
        GLES30.glLinkProgram(linked)
        val ok = IntArray(1)
        GLES30.glGetProgramiv(linked, GLES30.GL_LINK_STATUS, ok, 0)
        check(ok[0] != 0) { "program link failed: ${GLES30.glGetProgramInfoLog(linked)}" }
        return linked
    }

    private companion object {
        /** xyz, edge weight, rgb, depth-agreement state. */
        const val STRIDE_FLOATS = 8
        const val POINT_SIZE_M = 0.035f

        const val VERTEX_SHADER = """#version 300 es
uniform mat4 u_Mvp;
uniform float u_PointScale;
in vec4 a_PositionEdge;
in vec4 a_ColourState;
out vec3 v_Colour;
out float v_Edge;
out float v_State;
void main() {
  gl_Position = u_Mvp * vec4(a_PositionEdge.xyz, 1.0);
  gl_PointSize = clamp(u_PointScale / max(gl_Position.w, 0.1), 2.0, 16.0);
  v_Colour = a_ColourState.rgb;
  v_Edge = a_PositionEdge.w;
  v_State = a_ColourState.a;
}
"""

        const val FRAGMENT_SHADER = """#version 300 es
precision mediump float;
in vec3 v_Colour;
in float v_Edge;
in float v_State;
out vec4 fragColor;
void main() {
  vec2 d = gl_PointCoord - vec2(0.5);
  float r2 = dot(d, d);
  if (r2 > 0.25) discard;
  vec3 signal = v_State > 0.5 ? vec3(0.18, 1.0, 0.38)
               : (v_State > -0.5 ? vec3(1.0, 0.18, 0.12) : vec3(0.15, 0.82, 1.0));
  vec3 colour = mix(v_Colour, signal, v_State > -0.5 ? 0.72 : 0.38);
  float softEdge = 1.0 - smoothstep(0.10, 0.25, r2);
  float alpha = mix(0.07, 0.42, v_Edge) * softEdge;
  fragColor = vec4(colour, alpha);
}
"""
    }
}
