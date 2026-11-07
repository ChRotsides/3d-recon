package com.recon.videoscan.live

import android.content.Context
import java.io.File

/** Exercises the packaged weights with the same Android ONNX runtime as Live Scan. */
object DepthDeviceSmoke {
    fun run(context: Context): String {
        val dir = File(context.cacheDir, "depth-smoke-${System.nanoTime()}")
        check(dir.mkdirs())
        try {
            context.assets.open("models/${DepthRunner.DEFAULT_MODEL}").use { input ->
                File(dir, DepthRunner.DEFAULT_MODEL).outputStream().use { input.copyTo(it) }
            }
            File(dir, "metric3d_vits_182x322.onnx").writeText("invalid model")
            val camera = CameraCalibration(PinholeCamera(322, 182, 250.0, 250.0, 160.5, 90.5),
                CameraCalibration.Model.RADTAN, 0.0, 0.0, 0.0, 0.0)
            val pixels = ByteArray(322 * 182 * 4) { i ->
                if (i % 4 == 3) 255.toByte() else ((i * 31 + i / 128) % 256).toByte()
            }
            val times = ArrayList<Long>()
            DepthRunner(dir).use { runner ->
                fun checkPrediction() {
                    val p = runner.run(pixels, 322, 182, camera)
                    check(p.frame.camera.width == 322 && p.frame.camera.height == 182)
                    check(p.depth.size == 322 * 182 && p.depth.all { it.isFinite() })
                    check(p.depth.maxOrNull()!! > p.depth.minOrNull()!!)
                    times.add(p.ms)
                }
                repeat(3) { checkPrediction() }
                check(runCatching { runner.select(1) }.isFailure) { "corrupt model accepted" }
                check(runner.spec?.file == DepthRunner.DEFAULT_MODEL) { "failed switch lost working model" }
                checkPrediction()
            }
            return "PASS: packaged depth model produced finite 322x182 predictions; failed model switch retained working session; inference ms=$times"
        } finally { dir.deleteRecursively() }
    }
}
