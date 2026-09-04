package com.recon.videoscan.live

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.util.Log
import java.io.File
import java.nio.FloatBuffer

/**
 * On-device monocular depth with onnxruntime. Model families (all files under the phone's models/ dir):
 *  - da3_small*.onnx       DA3-Small export, input "image" [1,1,3,H,W]              -> relative depth
 *  - dav2_small*.onnx      Depth Anything V2 Small, input "pixel_values" [1,3,H,W]  -> relative inverse depth
 *  - metric3d*.onnx        Metric3D v2 ViT-S, input "image" [1,3,H,W]              -> metric depth after * fx/1000
 *  - unidepth*.onnx        UniDepth v2 ViT-S, inputs "image" [1,3,H,W] + "K" [1,3,3] -> metric depth
 * The model's input size is read from the session; frames are resampled to it.
 */
class DepthRunner(private val modelsDir: File) {
    enum class Kind { RELATIVE_DEPTH, RELATIVE_INVERSE, METRIC_CANONICAL, METRIC_K }
    class Spec(val file: String, val kind: Kind, val nnapi: Boolean)
    private val env = OrtEnvironment.getEnvironment()
    val available: List<Spec> = listOf(
        Spec("dav2_small_model_quantized.onnx", Kind.RELATIVE_INVERSE, false), Spec("dav2_small_model.onnx", Kind.RELATIVE_INVERSE, false),
        Spec("dav2_small_model_fp16.onnx", Kind.RELATIVE_INVERSE, true), Spec("da3_small.onnx", Kind.RELATIVE_DEPTH, false),
        Spec("metric3d_vits_182x322.onnx", Kind.METRIC_CANONICAL, false), Spec("unidepth_vits_182x322.onnx", Kind.METRIC_K, false),
    ).filter { File(modelsDir, it.file).exists() }
    private var session: OrtSession? = null
    var spec: Spec? = null; var w = 0; var h = 0; private var iw = 0; private var ih = 0; var backend = "cpu"; var inputName = ""; private var fiveD = false; private var kName: String? = null
    val name get() = spec?.file?.removeSuffix(".onnx") ?: "none"
    val kind get() = spec?.kind ?: Kind.RELATIVE_INVERSE
    private var input = FloatArray(0)
    private val mean = floatArrayOf(0.485f, 0.456f, 0.406f); private val std = floatArrayOf(0.229f, 0.224f, 0.225f)

    init { if (available.isNotEmpty()) select(0) }

    /** Load model i of `available`; returns the probe time in ms. */
    @Synchronized fun select(i: Int): Long {
        val sp = available[i % available.size]
        val opts = OrtSession.SessionOptions(); opts.setIntraOpNumThreads(4)
        if (sp.nnapi) try { opts.addNnapi() } catch (_: Throwable) {}
        val s = env.createSession(File(modelsDir, sp.file).absolutePath, opts)
        val info = s.inputInfo; inputName = info.keys.first { it != "K" && it != "intrinsics" }; kName = info.keys.firstOrNull { it == "K" || it == "intrinsics" }
        val shape = (info[inputName]!!.info as ai.onnxruntime.TensorInfo).shape
        fiveD = shape.size == 5; val hh = shape[shape.size - 2].toInt(); val ww = shape[shape.size - 1].toInt()
        // dynamic axes (V2 exports) keep the previous size or default to 168x98
        ih = if (hh > 0) hh else 98; iw = if (ww > 0) ww else 168   // dynamic-shape exports (V2) run at the small size
        h = ih; w = iw; input = FloatArray(3 * ih * iw)
        session?.close(); session = s; spec = sp; backend = if (sp.nnapi) "nnapi" else "cpu"
        val t0 = System.nanoTime(); runRaw(FloatArray(3 * ih * iw), floatArrayOf(400f, 400f, iw / 2f, ih / 2f)); runRaw(FloatArray(3 * ih * iw), floatArrayOf(400f, 400f, iw / 2f, ih / 2f))
        val ms = (System.nanoTime() - t0) / 2_000_000; Log.i("livescan", "depth model ${sp.file} ${w}x${h} $ms ms/frame"); return ms
    }

    private fun runRaw(chw: FloatArray, k: FloatArray): FloatArray {
        val s = session!!; val feeds = HashMap<String, OnnxTensor>()
        feeds[inputName] = OnnxTensor.createTensor(env, FloatBuffer.wrap(chw), if (fiveD) longArrayOf(1, 1, 3, ih.toLong(), iw.toLong()) else longArrayOf(1, 3, ih.toLong(), iw.toLong()))
        kName?.let { feeds[it] = OnnxTensor.createTensor(env, FloatBuffer.wrap(floatArrayOf(k[0], 0f, k[2], 0f, k[1], k[3], 0f, 0f, 1f)), longArrayOf(1, 3, 3)) }
        val out = s.run(feeds)
        val v = out[0].value
        val rows: Array<FloatArray> = if (fiveD) (v as Array<Array<Array<FloatArray>>>)[0][0] else (v as Array<Array<FloatArray>>)[0]
        h = rows.size; w = rows[0].size; val flat = FloatArray(h * w)   // the output grid is what the app back-projects with
        for (y in 0 until h) System.arraycopy(rows[y], 0, flat, y * w, w)
        out.close(); feeds.values.forEach { it.close() }
        return flat
    }

    /** rgba: sw*sh*4 source frame (resampled to the model size). k: fx,fy,cx,cy at the source size. Returns (prediction, ms). */
    @Synchronized fun run(rgba: ByteArray, sw: Int, sh: Int, k: FloatArray): Pair<FloatArray, Long> {
        val t0 = System.nanoTime(); val plane = ih * iw
        for (y in 0 until ih) { val sy = y * sh / ih; for (x in 0 until iw) { val sx = x * sw / iw; val o = (sy * sw + sx) * 4; val i = y * iw + x
            input[i] = ((rgba[o].toInt() and 0xff) / 255f - mean[0]) / std[0]
            input[plane + i] = ((rgba[o + 1].toInt() and 0xff) / 255f - mean[1]) / std[1]
            input[2 * plane + i] = ((rgba[o + 2].toInt() and 0xff) / 255f - mean[2]) / std[2] } }
        val km = floatArrayOf(k[0] * iw / sw, k[1] * ih / sh, k[2] * iw / sw, k[3] * ih / sh)
        val pred = runRaw(input, km)
        if (kind == Kind.METRIC_CANONICAL) { val sc = km[0] / 1000f; for (i in pred.indices) pred[i] *= sc }   // Metric3D canonical camera fx = 1000
        return pred to (System.nanoTime() - t0) / 1_000_000
    }
}
