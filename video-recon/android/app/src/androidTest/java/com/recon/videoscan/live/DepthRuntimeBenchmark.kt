package com.recon.videoscan.live

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import ai.onnxruntime.providers.NNAPIFlags
import android.content.Context
import android.os.SystemClock
import java.io.DataOutputStream
import java.io.File
import java.nio.FloatBuffer
import java.util.EnumSet

/** Compare runtimes on the same real capture, with no camera or scan mutations. */
object DepthRuntimeBenchmark {
    fun run(context: Context, modelFile: String, configuration: String, profile: Boolean): String {
        val root = context.getExternalFilesDir(null)!!
        val model = File(root, "models/$modelFile").canonicalFile
        check(model.parentFile == File(root,"models").canonicalFile && model.isFile)
        val frame = DepthEvidenceReplay.read(File(root,"depth-evidence/map-1791434991165-1/depth-24.bin.gz"))
        val directory = File(root,"runtime-benchmark").apply { mkdirs() }
        val env = OrtEnvironment.getEnvironment()
        val options = OrtSession.SessionOptions()
        val session = try {
            options.setInterOpNumThreads(1)
            options.setExecutionMode(OrtSession.SessionOptions.ExecutionMode.SEQUENTIAL)
            options.addConfigEntry("session.intra_op.allow_spinning", if (configuration == "cpu4spin") "1" else "0")
            options.addConfigEntry("session.inter_op.allow_spinning", "0")
            when (configuration) {
                "cpu1" -> options.setIntraOpNumThreads(1)
                "cpu2" -> options.setIntraOpNumThreads(2)
                "cpu4", "cpu4spin" -> options.setIntraOpNumThreads(4)
                "xnn2", "xnn4" -> {
                    options.setIntraOpNumThreads(1)
                    options.addXnnpack(mapOf("intra_op_num_threads" to configuration.last().toString()))
                }
                "nnapi", "nnapi16" -> {
                    options.setIntraOpNumThreads(2)
                    val flags = EnumSet.of(NNAPIFlags.CPU_DISABLED)
                    if (configuration == "nnapi16") flags.add(NNAPIFlags.USE_FP16)
                    options.addNnapi(flags)
                }
                else -> error("Unknown configuration: $configuration")
            }
            if (profile) options.enableProfiling(File(directory,"profile-$configuration").absolutePath)
            env.createSession(model.absolutePath,options)
        } finally { options.close() }
        return session.use { s ->
            val name = s.inputNames.first { it != "K" && it != "intrinsics" }
            val shape = (s.inputInfo[name]!!.info as TensorInfo).shape
            val height=shape[shape.size-2].toInt(); val width=shape.last().toInt()
            check(width > 0 && height > 0)
            val calibration=CameraCalibration(PinholeCamera(frame.w,frame.h,frame.k[0].toDouble(),frame.k[1].toDouble(),frame.k[2].toDouble(),frame.k[3].toDouble()),
                CameraCalibration.Model.RADTAN,0.0,0.0,0.0,0.0)
            val rectified=calibration.rectify(frame.rgba,frame.w,frame.h,width,height)
            val k=rectified.camera.intrinsics(); val plane=width*height
            val mean=floatArrayOf(.485f,.456f,.406f); val std=floatArrayOf(.229f,.224f,.225f)
            val input=FloatArray(3*plane) { i -> ((rectified.rgba[(i%plane)*4+i/plane].toInt() and 255)/255f-mean[i/plane])/std[i/plane] }
            val tensors=HashMap<String,OnnxTensor>()
            try {
                tensors[name]=OnnxTensor.createTensor(env,FloatBuffer.wrap(input),shape)
                s.inputNames.firstOrNull { it == "K" || it == "intrinsics" }?.let {
                    tensors[it]=OnnxTensor.createTensor(env,FloatBuffer.wrap(floatArrayOf(k[0],0f,k[2],0f,k[1],k[3],0f,0f,1f)),longArrayOf(1,3,3))
                }
                val times=ArrayList<Long>(); var output=FloatArray(0)
                repeat(10) { index ->
                    val start=SystemClock.elapsedRealtimeNanos()
                    s.run(tensors).use { result ->
                        val tensor=result[0] as OnnxTensor
                        val buffer=tensor.floatBuffer
                        output=FloatArray(buffer.remaining()).also { buffer.get(it) }
                    }
                    if (index >= 2) times.add((SystemClock.elapsedRealtimeNanos()-start)/1_000_000)
                    check(output.count { it.isFinite() && it > 0 } > output.size*.95)
                }
                DataOutputStream(File(directory,"$configuration-${modelFile.removeSuffix(".onnx")}.bin").outputStream().buffered()).use { out ->
                    out.writeInt(output.size); output.forEach { out.writeFloat(it) }
                }
                val trace=if (profile) s.endProfiling() else "off"
                "PASS: model=$modelFile config=$configuration grid=${width}x$height inferenceMs=$times profile=$trace"
            } finally { tensors.values.forEach { it.close() } }
        }
    }
}
