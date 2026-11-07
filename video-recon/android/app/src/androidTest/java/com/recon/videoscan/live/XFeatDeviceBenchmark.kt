package com.recon.videoscan.live

import android.content.Context
import java.io.File

object XFeatDeviceBenchmark {
    fun run(context: Context): String {
        val file = File(context.cacheDir, XFeatRunner.MODEL_FILE)
        context.assets.open("models/${XFeatRunner.MODEL_FILE}").use { input ->
            file.outputStream().use { input.copyTo(it) }
        }
        val gray = ByteArray(960 * 540) { i -> ((i * 37 + i / 960 * 17) and 0xff).toByte() }
        XFeatRunner(file).use { runner ->
            repeat(2) { runner.detect(gray, 960, 540) }
            val samples = ArrayList<XFeatRunner.Detection>()
            repeat(10) { samples += runner.detect(gray, 960, 540) }
            fun percentile(value: (XFeatRunner.Detection) -> Double, p: Double): Double {
                val sorted = samples.map(value).sorted()
                return sorted[(p * (sorted.size - 1)).toInt()]
            }
            return "PASS: XFeat 320x192, 10 runs, 2 ORT threads; " +
                "preprocess median/p95=${"%.1f".format(percentile({ it.preprocessMs }, .5))}/${"%.1f".format(percentile({ it.preprocessMs }, .95))} ms, " +
                "inference=${"%.1f".format(percentile({ it.inferenceMs }, .5))}/${"%.1f".format(percentile({ it.inferenceMs }, .95))} ms, " +
                "postprocess=${"%.1f".format(percentile({ it.postprocessMs }, .5))}/${"%.1f".format(percentile({ it.postprocessMs }, .95))} ms, " +
                "candidates=${samples.last().candidates.size / 3}"
        }
    }
}
