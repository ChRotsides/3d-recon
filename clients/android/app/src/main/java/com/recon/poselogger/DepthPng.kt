package com.recon.poselogger

import java.io.ByteArrayOutputStream
import java.io.File
import java.util.zip.CRC32
import java.util.zip.DeflaterOutputStream

/** Minimal 16-bit grayscale PNG writer for ARCore's millimetre depth maps. */
object DepthPng {
    fun write(
        file: File,
        samples: DepthSampler.Samples,
        outputWidth: Int = samples.width,
        outputHeight: Int = samples.height,
        clockwiseRotationDegrees: Int = 0,
    ) {
        require(outputWidth > 0 && outputHeight > 0)
        val rotation = ((clockwiseRotationDegrees % 360) + 360) % 360
        require(rotation % 90 == 0)
        file.parentFile?.mkdirs()

        val raw = ByteArrayOutputStream(outputHeight * (1 + outputWidth * 2))
        for (y in 0 until outputHeight) {
            raw.write(0) // PNG filter: none
            for (x in 0 until outputWidth) {
                val (sx, sy) = sourcePixel(
                    x, y, outputWidth, outputHeight, samples.width, samples.height, rotation
                )
                val mm = samples.depthMm[sy * samples.width + sx].toInt() and 0xffff
                raw.write(mm ushr 8)
                raw.write(mm and 0xff)
            }
        }

        val compressed = ByteArrayOutputStream()
        DeflaterOutputStream(compressed).use { it.write(raw.toByteArray()) }
        file.outputStream().buffered().use { out ->
            out.write(PNG_SIGNATURE)
            val ihdr = ByteArrayOutputStream(13).apply {
                writeInt(outputWidth)
                writeInt(outputHeight)
                write(16) // bit depth
                write(0) // grayscale
                write(0); write(0); write(0)
            }.toByteArray()
            writeChunk(out, "IHDR", ihdr)
            writeChunk(out, "IDAT", compressed.toByteArray())
            writeChunk(out, "IEND", byteArrayOf())
        }
    }

    private fun sourcePixel(
        x: Int,
        y: Int,
        outW: Int,
        outH: Int,
        srcW: Int,
        srcH: Int,
        rotation: Int,
    ): Pair<Int, Int> {
        val rotatedW = if (rotation % 180 == 0) srcW else srcH
        val rotatedH = if (rotation % 180 == 0) srcH else srcW
        val rx = (x.toLong() * rotatedW / outW).toInt().coerceAtMost(rotatedW - 1)
        val ry = (y.toLong() * rotatedH / outH).toInt().coerceAtMost(rotatedH - 1)
        return when (rotation) {
            0 -> rx to ry
            90 -> ry to (srcH - 1 - rx)
            180 -> (srcW - 1 - rx) to (srcH - 1 - ry)
            270 -> (srcW - 1 - ry) to rx
            else -> error("unreachable")
        }
    }

    private fun ByteArrayOutputStream.writeInt(value: Int) {
        write(value ushr 24); write(value ushr 16); write(value ushr 8); write(value)
    }

    private fun writeChunk(out: java.io.OutputStream, type: String, data: ByteArray) {
        val typeBytes = type.toByteArray(Charsets.US_ASCII)
        val length = data.size
        out.write(byteArrayOf(
            (length ushr 24).toByte(), (length ushr 16).toByte(),
            (length ushr 8).toByte(), length.toByte(),
        ))
        out.write(typeBytes)
        out.write(data)
        val crc = CRC32().apply { update(typeBytes); update(data) }.value.toInt()
        out.write(byteArrayOf(
            (crc ushr 24).toByte(), (crc ushr 16).toByte(),
            (crc ushr 8).toByte(), crc.toByte(),
        ))
    }

    private val PNG_SIGNATURE = byteArrayOf(
        0x89.toByte(), 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a,
    )
}
