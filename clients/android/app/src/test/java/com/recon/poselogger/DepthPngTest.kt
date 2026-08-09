package com.recon.poselogger

import java.nio.file.Files
import java.io.ByteArrayInputStream
import java.io.ByteArrayOutputStream
import java.util.zip.InflaterInputStream
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Test

class DepthPngTest {
    @Test
    fun `writes metric depth as a real 16 bit grayscale png`() {
        val samples = DepthSampler.Samples(
            points = listOf(0f, 0f, -1f),
            normals = listOf(0f, 0f, 1f),
            medianDepthM = 1.5f,
            depthMm = shortArrayOf(1000, 2000),
            width = 2,
            height = 1,
        )
        val path = Files.createTempFile("depth", ".png").toFile()

        DepthPng.write(path, samples, outputWidth = 1, outputHeight = 2,
            clockwiseRotationDegrees = 90)

        val png = path.readBytes()
        assertEquals(1, intAt(png, 16))
        assertEquals(2, intAt(png, 20))
        assertEquals(16, png[24].toInt() and 0xff)
        assertEquals(0, png[25].toInt() and 0xff)

        var offset = 8
        val idat = ByteArrayOutputStream()
        while (offset < png.size) {
            val length = intAt(png, offset)
            val type = String(png, offset + 4, 4, Charsets.US_ASCII)
            if (type == "IDAT") idat.write(png, offset + 8, length)
            offset += 12 + length
        }
        val raw = InflaterInputStream(ByteArrayInputStream(idat.toByteArray())).readBytes()
        assertArrayEquals(
            byteArrayOf(0, 0x03, 0xe8.toByte(), 0, 0x07, 0xd0.toByte()),
            raw,
        )
    }

    private fun intAt(bytes: ByteArray, offset: Int): Int =
        ((bytes[offset].toInt() and 0xff) shl 24) or
            ((bytes[offset + 1].toInt() and 0xff) shl 16) or
            ((bytes[offset + 2].toInt() and 0xff) shl 8) or
            (bytes[offset + 3].toInt() and 0xff)
}
