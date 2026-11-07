package com.recon.videoscan.live

import kotlin.math.atan
import kotlin.math.floor
import kotlin.math.roundToInt
import kotlin.math.sqrt

data class CaptureSize(val width: Int, val height: Int) {
    fun sameAspect(other: CaptureSize) = width.toLong() * other.height == height.toLong() * other.width
}

/** Prefer native-width capture, cropping excess vertical field of view without resampling.
 * Fall back to an advertised same-aspect stream only when native-width capture is absent.
 */
fun selectCaptureSize(supported: List<CaptureSize>, target: CaptureSize): CaptureSize? {
    supported.filter { it.width == target.width && it.height >= target.height &&
        (it.height - target.height) % 2 == 0 }.minByOrNull { it.height }?.let { return it }
    val matching = supported.filter { it.width > 0 && it.height > 0 && it.sameAspect(target) }
    return matching.filter { it.width >= target.width }.minByOrNull { it.width }
        ?: matching.maxByOrNull { it.width }
}

data class ImageCrop(val left: Int, val top: Int, val width: Int, val height: Int)

/** A crop changes the principal point, not focal length. The configured camera already
 * describes the central target-aspect field of view, so both image paths must use this crop.
 */
fun trackingCrop(source: CaptureSize, target: CaptureSize): ImageCrop? = when {
    source.width == target.width && source.height >= target.height && (source.height - target.height) % 2 == 0 ->
        ImageCrop(0, (source.height - target.height) / 2, target.width, target.height)
    source.width > 0 && source.height > 0 && source.sameAspect(target) -> ImageCrop(0, 0, source.width, source.height)
    else -> null
}

fun pixelCenterIndex(pixel: Int, sourceSize: Int, targetSize: Int): Int =
    ((pixel + 0.5) * sourceSize / targetSize).toInt().coerceIn(0, sourceSize - 1)

/** Pixel centres are integers. Resizing uses the half-pixel convention in both axes. */
data class PinholeCamera(val width: Int, val height: Int, val fx: Double, val fy: Double, val cx: Double, val cy: Double) {
    init {
        require(width > 0 && height > 0 && fx.isFinite() && fy.isFinite() && fx > 0 && fy > 0)
        require(cx.isFinite() && cy.isFinite())
    }
    fun resized(w: Int, h: Int): PinholeCamera {
        val sx = w.toDouble() / width; val sy = h.toDouble() / height
        return PinholeCamera(w, h, fx * sx, fy * sy, (cx + 0.5) * sx - 0.5, (cy + 0.5) * sy - 0.5)
    }
    fun intrinsics() = floatArrayOf(fx.toFloat(), fy.toFloat(), cx.toFloat(), cy.toFloat())
}

/** Immutable copy of calibration returned in the same JNI call as a frame's pose.
 * RADTAN coefficients are k1,k2,p1,p2; EQUIDISTANT coefficients are k1,k2,k3,k4.
 */
data class CameraCalibration(val camera: PinholeCamera, val model: Model,
    val d0: Double, val d1: Double, val d2: Double, val d3: Double) {
    enum class Model { RADTAN, EQUIDISTANT }
    init { require(listOf(d0, d1, d2, d3).all { it.isFinite() }) }

    /** Inverse image mapping: a ray in the target pinhole image -> distorted source pixel.
     * Camera axes are unchanged, so the associated pose needs no extra rotation.
     */
    fun rectify(rgba: ByteArray, sw: Int, sh: Int, width: Int, height: Int): RectifiedFrame {
        require(sw > 0 && sh > 0 && rgba.size == sw * sh * 4)
        val source = camera.resized(sw, sh); val target = camera.resized(width, height)
        val out = ByteArray(width * height * 4)
        for (v in 0 until height) for (u in 0 until width) {
            val x = (u - target.cx) / target.fx; val y = (v - target.cy) / target.fy
            val r2 = x * x + y * y
            val xd: Double; val yd: Double
            if (model == Model.RADTAN) {
                val radial = 1 + d0 * r2 + d1 * r2 * r2
                xd = x * radial + 2 * d2 * x * y + d3 * (r2 + 2 * x * x)
                yd = y * radial + d2 * (r2 + 2 * y * y) + 2 * d3 * x * y
            } else {
                val r = sqrt(r2); val theta = atan(r); val t2 = theta * theta
                val scale = if (r < 1e-12) 1.0 else theta * (1 + t2 * (d0 + t2 * (d1 + t2 * (d2 + t2 * d3)))) / r
                xd = x * scale; yd = y * scale
            }
            sampleRgba(rgba, sw, sh, source.fx * xd + source.cx, source.fy * yd + source.cy, out, 4 * (v * width + u))
        }
        return RectifiedFrame(target, out)
    }

    companion object {
        /** JNI layout: fx,fy,cx,cy,d0,d1,d2,d3,width,height,model (0=radtan,1=equi). */
        fun fromNative(values: DoubleArray): CameraCalibration {
            require(values.size == 11 && values.all { it.isFinite() })
            require(values[8] == values[8].toInt().toDouble() && values[9] == values[9].toInt().toDouble())
            val model = when (values[10]) { 0.0 -> Model.RADTAN; 1.0 -> Model.EQUIDISTANT; else -> error("Unsupported VIO camera model") }
            return CameraCalibration(PinholeCamera(values[8].toInt(), values[9].toInt(), values[0], values[1], values[2], values[3]),
                model, values[4], values[5], values[6], values[7])
        }
    }
}

/** Alpha is 255 only where the full interpolation footprint has camera evidence. */
class RectifiedFrame(val camera: PinholeCamera, val rgba: ByteArray) {
    init { require(rgba.size == camera.width * camera.height * 4) }
    fun resized(w: Int, h: Int): RectifiedFrame {
        if (w == camera.width && h == camera.height) return this
        val target = camera.resized(w, h); val out = ByteArray(w * h * 4)
        for (v in 0 until h) for (u in 0 until w) {
            val x = (u + 0.5) * camera.width / w - 0.5
            val y = (v + 0.5) * camera.height / h - 0.5
            sampleRgba(rgba, camera.width, camera.height, x, y, out, 4 * (v * w + u))
        }
        return RectifiedFrame(target, out)
    }
    fun maskPrediction(prediction: FloatArray) {
        require(prediction.size == camera.width * camera.height)
        for (i in prediction.indices) if (rgba[4 * i + 3] != (-1).toByte()) prediction[i] = Float.NaN
    }
}

private fun sampleRgba(src: ByteArray, w: Int, h: Int, x: Double, y: Double, dst: ByteArray, offset: Int) {
    if (!x.isFinite() || !y.isFinite() || x < -0.5 || y < -0.5 || x > w - 0.5 || y > h - 0.5) return
    val xc = x.coerceIn(0.0, w - 1.0); val yc = y.coerceIn(0.0, h - 1.0)
    val x0 = floor(xc).toInt(); val y0 = floor(yc).toInt()
    val x1 = (x0 + 1).coerceAtMost(w - 1); val y1 = (y0 + 1).coerceAtMost(h - 1)
    val a = xc - x0; val b = yc - y0
    val p00 = 4 * (y0 * w + x0); val p10 = 4 * (y0 * w + x1)
    val p01 = 4 * (y1 * w + x0); val p11 = 4 * (y1 * w + x1)
    if (src[p00 + 3] != (-1).toByte() || (a > 0 && src[p10 + 3] != (-1).toByte()) ||
        (b > 0 && src[p01 + 3] != (-1).toByte()) || (a > 0 && b > 0 && src[p11 + 3] != (-1).toByte())) return
    for (c in 0..2) {
        val top = (src[p00 + c].toInt() and 255) * (1 - a) + (src[p10 + c].toInt() and 255) * a
        val bottom = (src[p01 + c].toInt() and 255) * (1 - a) + (src[p11 + c].toInt() and 255) * a
        dst[offset + c] = (top * (1 - b) + bottom * b).roundToInt().coerceIn(0, 255).toByte()
    }
    dst[offset + 3] = -1
}
