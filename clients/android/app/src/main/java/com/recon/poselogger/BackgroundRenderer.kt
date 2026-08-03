package com.recon.poselogger

import android.opengl.GLES11Ext
import android.opengl.GLES30
import com.google.ar.core.Coordinates2d
import com.google.ar.core.Frame
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer

/**
 * Draws the ARCore camera image as a full-screen quad.
 *
 * The camera texture is an external OES texture, which is why the fragment
 * shader needs the `GL_OES_EGL_image_external_essl3` extension rather than a
 * plain sampler2D. ARCore also hands back *display*-space texture coordinates
 * that already account for device rotation and aspect — transforming them
 * ourselves is the usual source of a preview that is rotated or stretched.
 */
class BackgroundRenderer {

    var textureId: Int = -1
        private set

    private var program = 0
    private var positionAttrib = 0
    private var texCoordAttrib = 0
    private var textureUniform = 0

    private lateinit var quadCoords: FloatBuffer
    private lateinit var quadTexCoords: FloatBuffer

    private companion object {
        const val VERTEX_SHADER = """#version 300 es
in vec4 a_Position;
in vec2 a_TexCoord;
out vec2 v_TexCoord;
void main() {
  gl_Position = a_Position;
  v_TexCoord = a_TexCoord;
}
"""

        const val FRAGMENT_SHADER = """#version 300 es
#extension GL_OES_EGL_image_external_essl3 : require
precision mediump float;
in vec2 v_TexCoord;
uniform samplerExternalOES u_Texture;
out vec4 fragColor;
void main() { fragColor = texture(u_Texture, v_TexCoord); }
"""

        /** Full-screen triangle strip in clip space. */
        val QUAD = floatArrayOf(
            -1f, -1f,
            +1f, -1f,
            -1f, +1f,
            +1f, +1f,
        )
    }

    fun createOnGlThread() {
        val ids = IntArray(1)
        GLES30.glGenTextures(1, ids, 0)
        textureId = ids[0]
        GLES30.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, textureId)
        GLES30.glTexParameteri(
            GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES30.GL_TEXTURE_WRAP_S, GLES30.GL_CLAMP_TO_EDGE
        )
        GLES30.glTexParameteri(
            GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES30.GL_TEXTURE_WRAP_T, GLES30.GL_CLAMP_TO_EDGE
        )
        GLES30.glTexParameteri(
            GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES30.GL_TEXTURE_MIN_FILTER, GLES30.GL_LINEAR
        )
        GLES30.glTexParameteri(
            GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES30.GL_TEXTURE_MAG_FILTER, GLES30.GL_LINEAR
        )

        quadCoords = floatBuffer(QUAD)
        quadTexCoords = floatBuffer(FloatArray(QUAD.size))

        program = link(compile(GLES30.GL_VERTEX_SHADER, VERTEX_SHADER),
                       compile(GLES30.GL_FRAGMENT_SHADER, FRAGMENT_SHADER))
        positionAttrib = GLES30.glGetAttribLocation(program, "a_Position")
        texCoordAttrib = GLES30.glGetAttribLocation(program, "a_TexCoord")
        textureUniform = GLES30.glGetUniformLocation(program, "u_Texture")
    }

    fun draw(frame: Frame) {
        // Recompute only when the display geometry changed, as ARCore advises.
        if (frame.hasDisplayGeometryChanged()) {
            frame.transformCoordinates2d(
                Coordinates2d.OPENGL_NORMALIZED_DEVICE_COORDINATES, quadCoords,
                Coordinates2d.TEXTURE_NORMALIZED, quadTexCoords,
            )
        }
        if (frame.timestamp == 0L) return  // no image yet; drawing would flicker

        GLES30.glDisable(GLES30.GL_DEPTH_TEST)
        GLES30.glDepthMask(false)
        GLES30.glUseProgram(program)
        GLES30.glActiveTexture(GLES30.GL_TEXTURE0)
        GLES30.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, textureId)
        GLES30.glUniform1i(textureUniform, 0)

        quadCoords.position(0)
        GLES30.glVertexAttribPointer(positionAttrib, 2, GLES30.GL_FLOAT, false, 0, quadCoords)
        quadTexCoords.position(0)
        GLES30.glVertexAttribPointer(texCoordAttrib, 2, GLES30.GL_FLOAT, false, 0, quadTexCoords)
        GLES30.glEnableVertexAttribArray(positionAttrib)
        GLES30.glEnableVertexAttribArray(texCoordAttrib)

        GLES30.glDrawArrays(GLES30.GL_TRIANGLE_STRIP, 0, 4)

        GLES30.glDisableVertexAttribArray(positionAttrib)
        GLES30.glDisableVertexAttribArray(texCoordAttrib)
        GLES30.glDepthMask(true)
        GLES30.glEnable(GLES30.GL_DEPTH_TEST)
    }

    private fun floatBuffer(values: FloatArray): FloatBuffer =
        ByteBuffer.allocateDirect(values.size * 4).order(ByteOrder.nativeOrder())
            .asFloatBuffer().apply { put(values); position(0) }

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
}
