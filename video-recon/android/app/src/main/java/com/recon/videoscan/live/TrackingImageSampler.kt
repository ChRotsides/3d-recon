package com.recon.videoscan.live

import java.nio.ByteBuffer
import kotlin.math.ceil
import kotlin.math.floor
import kotlin.math.roundToInt

/** Copies native-size luma by rows; other sizes use pixel-area averaging, not decimation.
 * Precompute footprints once per capture geometry, outside the camera callback's hot loop.
 */
class TrackingImageSampler(val crop: ImageCrop, private val width: Int, private val height: Int) {
    private data class Span(val first: Int, val weights: DoubleArray)
    private fun spans(source: Int, target: Int) = Array(target) { p ->
        val start = p.toDouble() * source / target
        val end = (p + 1.0) * source / target
        val first = floor(start).toInt()
        val last = (ceil(end).toInt() - 1).coerceAtMost(source - 1)
        Span(first, DoubleArray(last - first + 1) { i ->
            (minOf(end, first + i + 1.0) - maxOf(start, first + i.toDouble())) / (end - start)
        })
    }
    init { require(width > 0 && height > 0 && crop.width > 0 && crop.height > 0 && crop.left >= 0 && crop.top >= 0) }
    private val xs = spans(crop.width, width)
    private val ys = spans(crop.height, height)

    fun sample(plane: ByteBuffer, rowStride: Int, pixelStride: Int): ByteArray {
        require(pixelStride > 0 && rowStride >= (crop.left + crop.width - 1) * pixelStride + 1)
        val source = plane.duplicate()
        val origin = source.position()
        require(origin.toLong() + (crop.top + crop.height - 1L) * rowStride +
            (crop.left + crop.width - 1L) * pixelStride < source.limit())
        val out = ByteArray(width * height)
        if (crop.width == width && crop.height == height && pixelStride == 1) {
            for (y in 0 until height) {
                source.position(origin + (crop.top + y) * rowStride + crop.left)
                source.get(out, y * width, width)
            }
            return out
        }
        val row = ByteArray((crop.width - 1) * pixelStride + 1)
        val sums = DoubleArray(width)
        for (y in 0 until height) {
            sums.fill(0.0)
            val sy = ys[y]
            for (j in sy.weights.indices) {
                source.position(origin + (crop.top + sy.first + j) * rowStride + crop.left * pixelStride)
                source.get(row)
                for (x in 0 until width) {
                    val sx = xs[x]; var value = 0.0
                    for (i in sx.weights.indices) value += (row[(sx.first + i) * pixelStride].toInt() and 255) * sx.weights[i]
                    sums[x] += value * sy.weights[j]
                }
            }
            for (x in 0 until width) out[y * width + x] = sums[x].roundToInt().coerceIn(0, 255).toByte()
        }
        return out
    }
}
