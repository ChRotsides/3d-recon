package com.recon.videoscan.live

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.util.Log
import java.io.File
import java.nio.FloatBuffer

/**
 * On-device monocular depth with onnxruntime. Two model families are supported:
 *  - DA3-Small export (video-recon/vio/export_da3.py): input "image" [1,1,3,H,W], output depth (relative depth)
 *  - Depth Anything V2 Small (onnx-community): input "pixel_values" [1,3,H,W], output predicted_depth (relative inverse depth)
 * Candidates are tried in order (with NNAPI first for the V2 graphs) and the first that runs is kept.
 */
class DepthRunner(modelsDir: File, val w: Int = 168, val h: Int = 98) {
    private val env = OrtEnvironment.getEnvironment()
    private lateinit var session: OrtSession
    var backend = "cpu"; var name = ""; var inverse = false; private var inputName = "image"; private var fiveD = true
    private val input = FloatArray(3 * h * w)
    private val mean = floatArrayOf(0.485f, 0.456f, 0.406f); private val std = floatArrayOf(0.229f, 0.224f, 0.225f)

    init {
        val candidates = listOf(
            Triple("dav2_small_model_fp16.onnx", true, true), Triple("dav2_small_model_quantized.onnx", true, false), Triple("dav2_small_model.onnx", true, false),
            Triple("dav2_small_model_quantized.onnx", false, false), Triple("dav2_small_model.onnx", false, false), Triple("da3_small.onnx", false, false))
        var ok = false; var bestMs = Long.MAX_VALUE
        for ((file, nnapi, _) in candidates) {
            val f = File(modelsDir, file); if (!f.exists()) continue
            try {
                val opts = OrtSession.SessionOptions(); opts.setIntraOpNumThreads(4)
                if (nnapi) opts.addNnapi()
                val s = env.createSession(f.absolutePath, opts)
                val inp = s.inputInfo.keys.first(); val five = inp == "image"
                val probe = OnnxTensor.createTensor(env, FloatBuffer.wrap(input), if (five) longArrayOf(1, 1, 3, h.toLong(), w.toLong()) else longArrayOf(1, 3, h.toLong(), w.toLong()))
                val t0 = System.nanoTime(); s.run(mapOf(inp to probe)).close(); s.run(mapOf(inp to probe)).close(); val ms = (System.nanoTime() - t0) / 2_000_000; probe.close()
                Log.i("livescan", "depth model $file nnapi=$nnapi $ms ms/frame")
                if (ms < bestMs) { if (ok) session.close(); session = s; bestMs = ms; backend = if (nnapi) "nnapi" else "cpu"; name = file.removeSuffix(".onnx"); ok = true
                    inputName = inp; fiveD = inp == "image"; inverse = file.startsWith("dav2") } else s.close()
            } catch (e: Throwable) { Log.w("livescan", "depth model $file nnapi=$nnapi failed: ${e.message?.take(140)}") }
        }
        if (!ok) throw IllegalStateException("no depth model could be loaded from $modelsDir")
        Log.i("livescan", "depth model chosen: $name backend=$backend $bestMs ms/frame")
    }

    /** rgba: W*H*4 bytes at the model resolution. Returns (prediction as float[h*w], ms). Prediction is depth, or inverse depth when `inverse`. */
    fun run(rgba: ByteArray): Pair<FloatArray, Long> {
        val t0 = System.nanoTime(); val plane = h * w
        for (i in 0 until plane) { val o = i * 4
            input[i] = ((rgba[o].toInt() and 0xff) / 255f - mean[0]) / std[0]
            input[plane + i] = ((rgba[o + 1].toInt() and 0xff) / 255f - mean[1]) / std[1]
            input[2 * plane + i] = ((rgba[o + 2].toInt() and 0xff) / 255f - mean[2]) / std[2] }
        val tensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(input), if (fiveD) longArrayOf(1, 1, 3, h.toLong(), w.toLong()) else longArrayOf(1, 3, h.toLong(), w.toLong()))
        val out = session.run(mapOf(inputName to tensor))
        val flat = FloatArray(plane)
        when (val v = out[0].value) {
            is Array<*> -> { // [1,1,H,W] or [1,H,W]
                val rows: Array<FloatArray> = if (fiveD) (v as Array<Array<Array<FloatArray>>>)[0][0] else (v as Array<Array<FloatArray>>)[0]
                for (y in 0 until h) System.arraycopy(rows[y], 0, flat, y * w, w) }
            else -> throw IllegalStateException("unexpected output ${v?.javaClass}")
        }
        out.close(); tensor.close()
        return flat to (System.nanoTime() - t0) / 1_000_000
    }
}
