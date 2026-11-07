package com.recon.videoscan.live

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.io.File
import java.nio.FloatBuffer
import kotlin.math.exp
import kotlin.math.floor

internal data class XFeatGeometry(
    val sourceWidth: Int,
    val sourceHeight: Int,
    val resizedWidth: Int,
    val resizedHeight: Int,
    val padLeft: Int,
    val padTop: Int,
) {
    fun contains(x: Int, y: Int) = x in padLeft until padLeft + resizedWidth && y in padTop until padTop + resizedHeight
    fun sourceX(x: Float) = (x - padLeft + 0.5f) * sourceWidth / resizedWidth - 0.5f
    fun sourceY(y: Float) = (y - padTop + 0.5f) * sourceHeight / resizedHeight - 0.5f

    companion object {
        fun fit(sourceWidth: Int, sourceHeight: Int): XFeatGeometry {
            require(sourceWidth > 0 && sourceHeight > 0)
            val scale = minOf(XFeatRunner.INPUT_WIDTH.toDouble() / sourceWidth, XFeatRunner.INPUT_HEIGHT.toDouble() / sourceHeight)
            val width = floor(sourceWidth * scale).toInt().coerceIn(1, XFeatRunner.INPUT_WIDTH)
            val height = floor(sourceHeight * scale).toInt().coerceIn(1, XFeatRunner.INPUT_HEIGHT)
            return XFeatGeometry(sourceWidth, sourceHeight, width, height,
                (XFeatRunner.INPUT_WIDTH - width) / 2, (XFeatRunner.INPUT_HEIGHT - height) / 2)
        }
    }
}

internal object XFeatPostprocessor {
    private data class Ranked(val x: Int, val y: Int, val score: Float)

    fun candidates(
        logits: Array<Array<Array<FloatArray>>>,
        reliability: Array<Array<Array<FloatArray>>>,
        geometry: XFeatGeometry,
        threshold: Float = 0.05f,
        limit: Int = 1000,
    ): FloatArray {
        require(logits.size == 1 && logits[0].size == 65)
        require(reliability.size == 1 && reliability[0].size == 1)
        val coarseHeight = logits[0][0].size
        val coarseWidth = logits[0][0][0].size
        require(coarseHeight * 8 == XFeatRunner.INPUT_HEIGHT && coarseWidth * 8 == XFeatRunner.INPUT_WIDTH)
        val heatmap = FloatArray(XFeatRunner.INPUT_WIDTH * XFeatRunner.INPUT_HEIGHT)
        for (cy in 0 until coarseHeight) for (cx in 0 until coarseWidth) {
            var maximum = Float.NEGATIVE_INFINITY
            for (channel in 0 until 65) maximum = maxOf(maximum, logits[0][channel][cy][cx])
            var denominator = 0.0
            for (channel in 0 until 65) denominator += exp((logits[0][channel][cy][cx] - maximum).toDouble())
            for (channel in 0 until 64) {
                val x = cx * 8 + channel % 8
                val y = cy * 8 + channel / 8
                heatmap[y * XFeatRunner.INPUT_WIDTH + x] =
                    (exp((logits[0][channel][cy][cx] - maximum).toDouble()) / denominator).toFloat()
            }
        }

        val ranked = ArrayList<Ranked>()
        for (y in 0 until XFeatRunner.INPUT_HEIGHT) for (x in 0 until XFeatRunner.INPUT_WIDTH) {
            if (!geometry.contains(x, y)) continue
            val probability = heatmap[y * XFeatRunner.INPUT_WIDTH + x]
            if (probability <= threshold) continue
            var localMaximum = Float.NEGATIVE_INFINITY
            for (yy in maxOf(0, y - 2)..minOf(XFeatRunner.INPUT_HEIGHT - 1, y + 2))
                for (xx in maxOf(0, x - 2)..minOf(XFeatRunner.INPUT_WIDTH - 1, x + 2))
                    localMaximum = maxOf(localMaximum, heatmap[yy * XFeatRunner.INPUT_WIDTH + xx])
            if (probability != localMaximum) continue
            val score = probability * bilinearReliability(reliability[0][0], x, y)
            if (score > 0f && score.isFinite()) ranked += Ranked(x, y, score)
        }
        ranked.sortWith(compareByDescending<Ranked> { it.score }.thenBy { it.y }.thenBy { it.x })
        val count = minOf(limit, ranked.size)
        return FloatArray(count * 3).also { output ->
            for (i in 0 until count) {
                val point = ranked[i]
                output[3 * i] = geometry.sourceX(point.x.toFloat())
                output[3 * i + 1] = geometry.sourceY(point.y.toFloat())
                output[3 * i + 2] = point.score
            }
        }
    }

    /** Matches XFeat's InterpolateSparse2d bilinear grid_sample coordinates. */
    private fun bilinearReliability(map: Array<FloatArray>, x: Int, y: Int): Float {
        val height = map.size
        val width = map[0].size
        val sx = x.toFloat() * width / (XFeatRunner.INPUT_WIDTH - 1) - 0.5f
        val sy = y.toFloat() * height / (XFeatRunner.INPUT_HEIGHT - 1) - 0.5f
        val x0 = floor(sx).toInt(); val y0 = floor(sy).toInt()
        val tx = sx - x0; val ty = sy - y0
        fun at(xx: Int, yy: Int) = if (xx in 0 until width && yy in 0 until height) map[yy][xx] else 0f
        return at(x0, y0) * (1 - tx) * (1 - ty) + at(x0 + 1, y0) * tx * (1 - ty) +
            at(x0, y0 + 1) * (1 - tx) * ty + at(x0 + 1, y0 + 1) * tx * ty
    }
    fun mapRanked(scores: FloatArray, indices: LongArray, geometry: XFeatGeometry): FloatArray {
        require(scores.size == indices.size)
        val output = ArrayList<Float>(scores.size * 3)
        for (i in scores.indices) {
            if (scores[i] <= 0f || !scores[i].isFinite()) continue
            val x = (indices[i] % XFeatRunner.INPUT_WIDTH).toInt()
            val y = (indices[i] / XFeatRunner.INPUT_WIDTH).toInt()
            if (!geometry.contains(x, y)) continue
            output += geometry.sourceX(x.toFloat())
            output += geometry.sourceY(y.toFloat())
            output += scores[i]
        }
        return output.toFloatArray()
    }

}

class XFeatRunner(modelFile: File) : AutoCloseable {
    data class Detection(
        val candidates: FloatArray,
        val preprocessMs: Double,
        val inferenceMs: Double,
        val postprocessMs: Double,
        val descriptorMap: FloatArray = FloatArray(0),
    )

    private val environment = OrtEnvironment.getEnvironment()
    private val session: OrtSession
    private val inputName: String

    init {
        require(modelFile.isFile) { "missing XFeat model: $modelFile" }
        val options = OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(2)
            setInterOpNumThreads(1)
            addConfigEntry("session.intra_op.allow_spinning", "0")
            addConfigEntry("session.inter_op.allow_spinning", "0")
            setExecutionMode(OrtSession.SessionOptions.ExecutionMode.SEQUENTIAL)
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        }
        session = environment.createSession(modelFile.absolutePath, options)
        inputName = session.inputNames.single()
        require(session.outputNames.containsAll(listOf("descriptors", "keypoint_logits", "reliability", "top_scores", "top_indices")))
        runModel(FloatArray(INPUT_WIDTH * INPUT_HEIGHT)) // bounded warm-up before camera frames use the session
    }

    @Synchronized fun detect(gray: ByteArray, width: Int, height: Int, descriptors: Boolean = false): Detection {
        require(gray.size == width * height)
        val geometry = XFeatGeometry.fit(width, height)
        val preprocessStart = System.nanoTime()
        val input = resizeAndPad(gray, geometry)
        val inferenceStart = System.nanoTime()
        val result = runModel(input, descriptors)
        val postprocessStart = System.nanoTime()
        val candidates = XFeatPostprocessor.mapRanked(result.scores, result.indices, geometry)
        val finish = System.nanoTime()
        return Detection(candidates,
            (inferenceStart - preprocessStart) / 1e6,
            (postprocessStart - inferenceStart) / 1e6,
            (finish - postprocessStart) / 1e6, result.descriptors)
    }

    private data class ModelOutput(val scores: FloatArray, val indices: LongArray, val descriptors: FloatArray)
    @Suppress("UNCHECKED_CAST")
    private fun runModel(input: FloatArray, descriptors: Boolean = false): ModelOutput {
        OnnxTensor.createTensor(environment, FloatBuffer.wrap(input), longArrayOf(1, 1, INPUT_HEIGHT.toLong(), INPUT_WIDTH.toLong())).use { tensor ->
            val outputs = if (descriptors) setOf("top_scores", "top_indices", "descriptors") else setOf("top_scores", "top_indices")
            session.run(mapOf(inputName to tensor), outputs).use { result ->
                val scores = (result.get("top_scores").orElseThrow().value as Array<FloatArray>)[0]
                val indices = (result.get("top_indices").orElseThrow().value as Array<LongArray>)[0]
                val field = if (descriptors) FloatArray(64 * 24 * 40).also {
                    (result.get("descriptors").orElseThrow() as OnnxTensor).floatBuffer.get(it)
                } else FloatArray(0)
                return ModelOutput(scores, indices, field)
            }
        }
    }

    private fun resizeAndPad(gray: ByteArray, geometry: XFeatGeometry): FloatArray {
        val output = FloatArray(INPUT_WIDTH * INPUT_HEIGHT)
        for (y in 0 until INPUT_HEIGHT) {
            val contentY = (y - geometry.padTop).coerceIn(0, geometry.resizedHeight - 1)
            val sy = (contentY + 0.5f) * geometry.sourceHeight / geometry.resizedHeight - 0.5f
            val y0 = floor(sy).toInt().coerceIn(0, geometry.sourceHeight - 1); val y1 = minOf(y0 + 1, geometry.sourceHeight - 1); val ty = sy - floor(sy)
            for (x in 0 until INPUT_WIDTH) {
                val contentX = (x - geometry.padLeft).coerceIn(0, geometry.resizedWidth - 1)
                val sx = (contentX + 0.5f) * geometry.sourceWidth / geometry.resizedWidth - 0.5f
                val x0 = floor(sx).toInt().coerceIn(0, geometry.sourceWidth - 1); val x1 = minOf(x0 + 1, geometry.sourceWidth - 1); val tx = sx - floor(sx)
                fun value(xx: Int, yy: Int) = (gray[yy * geometry.sourceWidth + xx].toInt() and 0xff).toFloat()
                output[y * INPUT_WIDTH + x] = ((value(x0, y0) * (1 - tx) + value(x1, y0) * tx) * (1 - ty) +
                    (value(x0, y1) * (1 - tx) + value(x1, y1) * tx) * ty) / 255f
            }
        }
        return output
    }

    @Synchronized override fun close() = session.close()

    companion object {
        const val INPUT_WIDTH = 320
        const val INPUT_HEIGHT = 192
        const val MODEL_FILE = "xfeat_320x192_e92685f_ranked.onnx"
    }
}
