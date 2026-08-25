package com.recon.poselogger

import android.graphics.Bitmap
import android.graphics.Color
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.roundToInt
import kotlin.math.sqrt

/**
 * A previous keyframe reprojected as metric world points, not as a flat photo.
 *
 * A flat cross-fade can only line up after a rotation around one camera centre;
 * translating the phone makes near furniture move against distant walls. These
 * points carry the previous frame's depth, so the renderer naturally produces
 * the correct parallax from the current camera pose.
 */
data class AlignmentGhost(
    /** Flat world-space xyz triples. */
    val positions: FloatArray,
    /** Flat RGB triples, one byte per channel. */
    val colours: ByteArray,
    /** 1 for an image/depth edge, 0 for a faint interior sample. */
    val edges: FloatArray,
    val capturePosition: FloatArray,
)

data class AlignmentStatus(
    val overlap: Float,
    val depthAgreement: Float,
    val baselineM: Float,
    /** -1 = no current depth comparison, 0 = mismatch, 1 = agreement. */
    val pointStates: FloatArray,
    val ready: Boolean,
)

object AlignmentGuide {
    const val MIN_BASELINE_M = 0.10f
    const val TARGET_BASELINE_M = 0.20f
    const val MIN_OVERLAP = 0.30f
    const val MIN_DEPTH_AGREEMENT = 0.55f

    /** Build a decimated coloured surface from the keyframe retained by the shutter. */
    fun build(
        bitmap: Bitmap,
        samples: DepthSampler.Samples,
        intrinsics: IntrinsicsWire,
        pose: PoseWire,
        clockwiseRotationDegrees: Int,
        step: Int = 3,
    ): AlignmentGhost? {
        require(step > 0)
        val rotation = ((clockwiseRotationDegrees % 360) + 360) % 360
        val width = if (rotation % 180 == 0) samples.width else samples.height
        val height = if (rotation % 180 == 0) samples.height else samples.width
        val sx = width.toFloat() / intrinsics.width
        val sy = height.toFloat() / intrinsics.height
        val fx = intrinsics.fx.toFloat() * sx
        val fy = intrinsics.fy.toFloat() * sy
        val cx = intrinsics.cx.toFloat() * sx
        val cy = intrinsics.cy.toFloat() * sy

        val positions = ArrayList<Float>((width / step) * (height / step) * 3)
        val colours = ArrayList<Byte>((width / step) * (height / step) * 3)
        val edges = ArrayList<Float>((width / step) * (height / step))
        val r = pose.R
        val t = pose.t

        fun depthAt(x: Int, y: Int): Float {
            val (rawX, rawY) = sourcePixel(
                x.coerceIn(0, width - 1), y.coerceIn(0, height - 1),
                samples.width, samples.height, rotation,
            )
            return (samples.depthMm[rawY * samples.width + rawX].toInt() and 0xffff) * 0.001f
        }

        for (v in step until height - step step step) {
            for (u in step until width - step step step) {
                val depth = depthAt(u, v)
                if (depth !in 0.25f..6.0f) continue
                val cameraX = (u - cx) / fx * depth
                val cameraY = -(v - cy) / fy * depth
                val cameraZ = -depth
                positions += (r[0][0] * cameraX + r[0][1] * cameraY + r[0][2] * cameraZ + t[0]).toFloat()
                positions += (r[1][0] * cameraX + r[1][1] * cameraY + r[1][2] * cameraZ + t[1]).toFloat()
                positions += (r[2][0] * cameraX + r[2][1] * cameraY + r[2][2] * cameraZ + t[2]).toFloat()

                val imageX = (u.toLong() * bitmap.width / width).toInt().coerceAtMost(bitmap.width - 1)
                val imageY = (v.toLong() * bitmap.height / height).toInt().coerceAtMost(bitmap.height - 1)
                val colour = bitmap.getPixel(imageX, imageY)
                colours += Color.red(colour).toByte()
                colours += Color.green(colour).toByte()
                colours += Color.blue(colour).toByte()

                val neighbourDepth = max(
                    abs(depthAt(u + step, v) - depth),
                    abs(depthAt(u, v + step) - depth),
                )
                val nextX = (imageX + max(1, bitmap.width * step / width))
                    .coerceAtMost(bitmap.width - 1)
                val nextY = (imageY + max(1, bitmap.height * step / height))
                    .coerceAtMost(bitmap.height - 1)
                val colourEdge = max(
                    colourDistance(colour, bitmap.getPixel(nextX, imageY)),
                    colourDistance(colour, bitmap.getPixel(imageX, nextY)),
                )
                edges += if (neighbourDepth > max(0.06f, depth * 0.04f) || colourEdge > 42) {
                    1f
                } else {
                    0f
                }
            }
        }
        if (edges.isEmpty()) return null
        return AlignmentGhost(
            positions.toFloatArray(),
            colours.toByteArray(),
            edges.toFloatArray(),
            floatArrayOf(t[0].toFloat(), t[1].toFloat(), t[2].toFloat()),
        )
    }

    /**
     * Compare the depth-warped previous frame with the current ARCore depth.
     *
     * [worldFromCamera] is column-major, matching ARCore's `Pose.toMatrix`.
     * Intrinsics must already be scaled to [depthWidth] x [depthHeight].
     */
    fun evaluate(
        ghost: AlignmentGhost,
        worldFromCamera: FloatArray,
        depthMm: ShortArray,
        depthWidth: Int,
        depthHeight: Int,
        fx: Float,
        fy: Float,
        cx: Float,
        cy: Float,
    ): AlignmentStatus {
        require(worldFromCamera.size >= 16)
        require(depthMm.size >= depthWidth * depthHeight)
        val count = ghost.positions.size / 3
        val states = FloatArray(count) { -1f }
        var visible = 0
        var comparable = 0
        var matching = 0
        val tx = worldFromCamera[12]
        val ty = worldFromCamera[13]
        val tz = worldFromCamera[14]

        for (index in 0 until count) {
            val base = index * 3
            val dx = ghost.positions[base] - tx
            val dy = ghost.positions[base + 1] - ty
            val dz = ghost.positions[base + 2] - tz
            // camera <- world = R^T (world - t), with column-major R.
            val cameraX = worldFromCamera[0] * dx + worldFromCamera[1] * dy + worldFromCamera[2] * dz
            val cameraY = worldFromCamera[4] * dx + worldFromCamera[5] * dy + worldFromCamera[6] * dz
            val cameraZ = worldFromCamera[8] * dx + worldFromCamera[9] * dy + worldFromCamera[10] * dz
            val predictedDepth = -cameraZ
            if (predictedDepth <= 0.15f) continue
            val u = (fx * cameraX / predictedDepth + cx).roundToInt()
            val v = (cy - fy * cameraY / predictedDepth).roundToInt()
            if (u !in 0 until depthWidth || v !in 0 until depthHeight) continue
            visible++
            val measuredMm = depthMm[v * depthWidth + u].toInt() and 0xffff
            if (measuredMm == 0) continue
            val measured = measuredMm * 0.001f
            if (measured !in 0.25f..6.0f) continue
            comparable++
            val tolerance = max(0.12f, predictedDepth * 0.08f)
            if (abs(measured - predictedDepth) <= tolerance) {
                states[index] = 1f
                matching++
            } else {
                states[index] = 0f
            }
        }
        val overlap = if (count == 0) 0f else visible.toFloat() / count
        val agreement = if (comparable == 0) 0f else matching.toFloat() / comparable
        val bx = tx - ghost.capturePosition[0]
        val by = ty - ghost.capturePosition[1]
        val bz = tz - ghost.capturePosition[2]
        val baseline = sqrt(bx * bx + by * by + bz * bz)
        return AlignmentStatus(
            overlap = overlap,
            depthAgreement = agreement,
            baselineM = baseline,
            pointStates = states,
            ready = baseline >= MIN_BASELINE_M &&
                overlap >= MIN_OVERLAP && agreement >= MIN_DEPTH_AGREEMENT,
        )
    }

    internal fun sourcePixel(
        x: Int,
        y: Int,
        sourceWidth: Int,
        sourceHeight: Int,
        clockwiseRotationDegrees: Int,
    ): Pair<Int, Int> = when (clockwiseRotationDegrees) {
        0 -> x to y
        90 -> y to (sourceHeight - 1 - x)
        180 -> (sourceWidth - 1 - x) to (sourceHeight - 1 - y)
        270 -> (sourceWidth - 1 - y) to x
        else -> error("rotation must be 0, 90, 180 or 270 degrees")
    }

    private fun colourDistance(first: Int, second: Int): Int = max(
        abs(Color.red(first) - Color.red(second)),
        max(
            abs(Color.green(first) - Color.green(second)),
            abs(Color.blue(first) - Color.blue(second)),
        ),
    )
}
