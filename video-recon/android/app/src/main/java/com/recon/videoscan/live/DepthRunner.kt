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
class DepthRunner(private val modelsDir: File, preferredFile: String? = null,
    includeExperimental: Boolean = false, private val cacheDir: File? = null,
    private val nativeLibDir: String? = null) : AutoCloseable {
    companion object {
        const val DEFAULT_MODEL = "dav2_small_model_quantized.onnx"
        const val CPU_UNIDEPTH = "unidepth_vits_182x322_quantized.onnx"
    }
    data class Prediction(val depth: FloatArray, val frame: RectifiedFrame, val ms: Long)
    enum class Kind { RELATIVE_DEPTH, RELATIVE_INVERSE, METRIC_CANONICAL, METRIC_K }
    class Spec(val file: String, val kind: Kind, val nnapi: Boolean, val qnn: Boolean = false)
    private val env = OrtEnvironment.getEnvironment()
    val available: List<Spec> = (listOf(
        Spec("dav2_small_model_quantized.onnx", Kind.RELATIVE_INVERSE, false), Spec("dav2_small_model.onnx", Kind.RELATIVE_INVERSE, false),
        Spec("dav2_small_model_fp16.onnx", Kind.RELATIVE_INVERSE, true), Spec("da3_small.onnx", Kind.RELATIVE_DEPTH, false),
        Spec("metric3d_vits_182x322.onnx", Kind.METRIC_CANONICAL, false), Spec("unidepth_vits_182x322.onnx", Kind.METRIC_K, false),
        Spec("unidepth_vits_182x322_quantized.onnx", Kind.METRIC_K, false),
    ) + (if (cacheDir != null && nativeLibDir != null && File(modelsDir,QnnDepthSession.CAMERA_MODEL).isFile &&
            OrtEnvironment.getAvailableProviders().any { it.name == "QNN" })
        listOf(Spec(QnnDepthSession.MAIN_MODEL,Kind.METRIC_K,false,true)) else emptyList()) +
        (if (includeExperimental) listOf(Spec("unidepth_int280x490_exactK_q.onnx", Kind.METRIC_K, false)) else emptyList()))
        .filter { File(modelsDir, it.file).exists() }
    private var session: OrtSession? = null
    private var accelerated: QnnDepthSession? = null
    var spec: Spec? = null; var w = 0; var h = 0; private var iw = 0; private var ih = 0; var backend = "cpu"; var inputName = ""; private var fiveD = false; private var kName: String? = null
    val name get() = spec?.file?.removeSuffix(".onnx") ?: "none"
    val kind get() = spec?.kind ?: Kind.RELATIVE_INVERSE
    private var input = FloatArray(0)
    private val mean = floatArrayOf(0.485f, 0.456f, 0.406f); private val std = floatArrayOf(0.229f, 0.224f, 0.225f)

    val label get() = when (spec?.file) {
        QnnDepthSession.MAIN_MODEL -> "UniDepth accelerated"
        "dav2_small_model_quantized.onnx" -> "Depth Anything fast"
        "dav2_small_model.onnx" -> "Depth Anything full"
        "dav2_small_model_fp16.onnx" -> "Depth Anything FP16"
        "unidepth_vits_182x322_quantized.onnx" -> "UniDepth fast"
        "unidepth_int280x490_exactK_q.onnx" -> "UniDepth live"
        "unidepth_vits_182x322.onnx" -> "UniDepth full"
        "metric3d_vits_182x322.onnx" -> "Metric3D"
        "da3_small.onnx" -> "Depth Anything 3"
        else -> "loading"
    }
    init {
        if (available.isNotEmpty()) {
            val preferred = available.indexOfFirst { it.file == preferredFile }
            val fallback = if (preferredFile == QnnDepthSession.MAIN_MODEL)
                available.indexOfFirst { it.file == CPU_UNIDEPTH }.coerceAtLeast(0) else 0
            try { select(if (preferred >= 0) preferred else fallback) }
            catch (e: Exception) {
                if (preferred == fallback || preferred < 0) throw e
                Log.w("livescan", "Saved model unavailable; loading default", e)
                select(fallback)
            }
        }
    }

    /** Load model i of `available`; returns the probe time in ms. */
    @Synchronized fun select(i: Int): Long {
        require(available.isNotEmpty()) { "No on-device depth model installed" }
        val sp = available[Math.floorMod(i, available.size)]
        val oldSession = session; val oldSpec = spec; val oldInput = input
        val oldAccelerated = accelerated
        val oldDims = intArrayOf(iw, ih, w, h)
        val oldName = inputName; val oldK = kName; val oldFiveD = fiveD; val oldBackend = backend
        var s: OrtSession? = null; var nextAccelerated: QnnDepthSession? = null
        try {
            val shape = if (sp.qnn) {
                nextAccelerated = QnnDepthSession(modelsDir,checkNotNull(cacheDir),checkNotNull(nativeLibDir))
                inputName = "image"; kName = "K"
                longArrayOf(1,3,QnnDepthSession.HEIGHT.toLong(),QnnDepthSession.WIDTH.toLong())
            } else {
                val opts = OrtSession.SessionOptions()
                s = try {
                    opts.setIntraOpNumThreads(4)
                    opts.addConfigEntry("session.intra_op.allow_spinning", "0")
                    opts.addConfigEntry("session.inter_op.allow_spinning", "0")
                    if (sp.nnapi) try { opts.addNnapi() } catch (_: Throwable) {}
                    env.createSession(File(modelsDir, sp.file).absolutePath, opts)
                } finally { opts.close() }
                val info = checkNotNull(s).inputInfo
                inputName = info.keys.first { it != "K" && it != "intrinsics" }
                kName = info.keys.firstOrNull { it == "K" || it == "intrinsics" }
                (info[inputName]!!.info as ai.onnxruntime.TensorInfo).shape
            }
            require(shape.size == 4 || shape.size == 5) { "Unexpected depth input shape" }
            fiveD = shape.size == 5
            val hh = shape[shape.size - 2].toInt(); val ww = shape.last().toInt()
            ih = if (hh > 0) hh else 182; iw = if (ww > 0) ww else 322
            require(ih in 1..2048 && iw in 1..2048) { "Depth model input too large" }
            h = ih; w = iw; input = FloatArray(3 * ih * iw)
            session = s; accelerated = nextAccelerated; spec = sp
            backend = if (sp.qnn) "npu" else if (sp.nnapi) "nnapi" else "cpu"
            val t0 = System.nanoTime()
            repeat(2) { runRaw(input, floatArrayOf(400f, 400f, iw / 2f, ih / 2f)) }
            val ms = (System.nanoTime() - t0) / 2_000_000
            runCatching { oldSession?.close() }.onFailure { Log.w("livescan","Closing previous depth session",it) }
            runCatching { oldAccelerated?.close() }.onFailure { Log.w("livescan","Closing previous accelerated depth",it) }
            Log.i("livescan", "depth model ${sp.file} ${w}x${h} $ms ms/frame")
            return ms
        } catch (e: Exception) {
            runCatching { s?.close() }; runCatching { nextAccelerated?.close() }
            session = oldSession; accelerated = oldAccelerated; spec = oldSpec; input = oldInput
            iw = oldDims[0]; ih = oldDims[1]; w = oldDims[2]; h = oldDims[3]
            inputName = oldName; kName = oldK; fiveD = oldFiveD; backend = oldBackend
            throw e
        }
    }

    private fun runRaw(chw: FloatArray, k: FloatArray): FloatArray {
        accelerated?.let { return it.run(chw,k) }
        val s = session!!; val feeds = HashMap<String, OnnxTensor>()
        try {
        feeds[inputName] = OnnxTensor.createTensor(env, FloatBuffer.wrap(chw), if (fiveD) longArrayOf(1, 1, 3, ih.toLong(), iw.toLong()) else longArrayOf(1, 3, ih.toLong(), iw.toLong()))
        kName?.let { feeds[it] = OnnxTensor.createTensor(env, FloatBuffer.wrap(floatArrayOf(k[0], 0f, k[2], 0f, k[1], k[3], 0f, 0f, 1f)), longArrayOf(1, 3, 3)) }
            return s.run(feeds).use { out ->
                val tensor = out[0] as? OnnxTensor ?: error("Depth output must be a tensor")
                val layout = DepthTensorLayout.from(tensor.info.shape)
                h = layout.height; w = layout.width
                val flat = FloatArray(h * w)
                val buffer = tensor.floatBuffer ?: error("Depth output must contain floating point values")
                require(buffer.remaining() == flat.size) { "Depth output shape does not match its data" }
                buffer.get(flat)
                flat
            }
        } finally { feeds.values.forEach { it.close() } }
    }

    @Synchronized override fun close() {
        try { session?.close() } finally {
            session = null
            try { accelerated?.close() } finally { accelerated = null }
        }
    }

    /** Undistort and resize into one pinhole grid using this frame's VIO calibration. */
    @Synchronized fun run(rgba: ByteArray, sw: Int, sh: Int, calibration: CameraCalibration): Prediction {
        val t0 = System.nanoTime(); val plane = ih * iw
        val rectified = calibration.rectify(rgba, sw, sh, iw, ih)
        for (i in 0 until plane) { val o = 4 * i
            input[i] = ((rectified.rgba[o].toInt() and 0xff) / 255f - mean[0]) / std[0]
            input[plane + i] = ((rectified.rgba[o + 1].toInt() and 0xff) / 255f - mean[1]) / std[1]
            input[2 * plane + i] = ((rectified.rgba[o + 2].toInt() and 0xff) / 255f - mean[2]) / std[2] }
        val km = rectified.camera.intrinsics()
        val pred = try { runRaw(input, km) } catch (e: Exception) {
            if (accelerated == null) throw e
            val fallback=available.indexOfFirst { it.file == CPU_UNIDEPTH }
            if (fallback < 0) throw e
            Log.e("livescan","Accelerated depth failed; switching to CPU model",e)
            select(fallback)
            return run(rgba,sw,sh,calibration)
        }
        if (kind == Kind.METRIC_CANONICAL) { val sc = km[0] / 1000f; for (i in pred.indices) pred[i] *= sc }   // Metric3D canonical camera fx = 1000
        val outputFrame = rectified.resized(w, h)
        outputFrame.maskPrediction(pred)
        return Prediction(pred, outputFrame, (System.nanoTime() - t0) / 1_000_000)
    }
}
