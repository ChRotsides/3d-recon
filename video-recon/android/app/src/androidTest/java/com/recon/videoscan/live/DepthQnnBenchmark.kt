package com.recon.videoscan.live

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.os.SystemClock
import java.io.DataOutputStream
import java.io.File
import java.nio.FloatBuffer

/** NPU experiment isolated to instrumentation. Camera parameters change each run. */
object DepthQnnBenchmark {
    fun run(context: Context): String {
        val root=context.getExternalFilesDir(null)!!
        val models=File(root,"models")
        android.system.Os.setenv("ADSP_LIBRARY_PATH",context.applicationInfo.nativeLibraryDir+
            ";/vendor/lib/rfsa/adsp;/vendor/dsp/cdsp;/system/lib/rfsa/adsp;/dsp",true)
        val directory=File(root,"runtime-benchmark").apply { mkdirs() }
        val env=OrtEnvironment.getEnvironment()
        val options=OrtSession.SessionOptions()
        val camera=try {
            options.setIntraOpNumThreads(1)
            options.addConfigEntry("session.intra_op.allow_spinning","0")
            env.createSession(File(models,"qnn_camera.onnx").absolutePath,options)
        } finally { options.close() }
        return camera.use { c ->
            val mainOptions=OrtSession.SessionOptions()
            val start=SystemClock.elapsedRealtime()
            val main=try {
                mainOptions.setIntraOpNumThreads(2)
                mainOptions.addConfigEntry("session.intra_op.allow_spinning","0")
                mainOptions.addConfigEntry("session.disable_cpu_ep_fallback","1")
                mainOptions.enableProfiling(File(directory,"qnn").absolutePath)
                val settings=mapOf("backend_path" to "libQnnHtp.so",
                    "htp_arch" to "73",
                    "enable_htp_fp16_precision" to "1", "htp_performance_mode" to "burst",
                    "htp_graph_finalization_optimization_mode" to "1")
                // Reflection keeps this test compilable with the standard runtime.
                OrtSession.SessionOptions::class.java.getMethod("addQnn",Map::class.java).invoke(mainOptions,settings)
                env.createSession(File(models,"qnn_main_static_gelu.onnx").absolutePath,mainOptions)
            } finally { mainOptions.close() }
            val loadMs=SystemClock.elapsedRealtime()-start
            main.use { m ->
                val samples=ArrayList<String>()
                repeat(5) { i ->
                    val frame=DepthEvidenceReplay.read(File(root,"depth-evidence/map-1791434991165-1/depth-${20+i}.bin.gz"))
                    val plane=frame.w*frame.h
                    val mean=floatArrayOf(.485f,.456f,.406f); val std=floatArrayOf(.229f,.224f,.225f)
                    val x=FloatArray(3*plane) { j -> ((frame.rgba[(j%plane)*4+j/plane].toInt() and 255)/255f-mean[j/plane])/std[j/plane] }
                    val k=frame.k
                    OnnxTensor.createTensor(env,FloatBuffer.wrap(x),longArrayOf(1,3,frame.h.toLong(),frame.w.toLong())).use { image ->
                        OnnxTensor.createTensor(env,FloatBuffer.wrap(floatArrayOf(k[0],0f,k[2],0f,k[1],k[3],0f,0f,1f)),longArrayOf(1,3,3)).use { intrinsics ->
                            val t0=SystemClock.elapsedRealtimeNanos()
                            c.run(mapOf("K" to intrinsics)).use { cameraOutput ->
                                val t1=SystemClock.elapsedRealtimeNanos()
                                val feeds=HashMap<String,OnnxTensor>()
                                feeds["image"]=image
                                for (name in c.outputNames) feeds[name]=cameraOutput.get(name).get() as OnnxTensor
                                check(feeds.keys == m.inputNames)
                                m.run(feeds).use { result ->
                                    val t2=SystemClock.elapsedRealtimeNanos()
                                    val buffer=(result[0] as OnnxTensor).floatBuffer
                                    val depth=FloatArray(buffer.remaining()).also { buffer.get(it) }
                                    check(depth.size == plane && depth.all { it.isFinite() && it > 0 })
                                    DataOutputStream(File(directory,"qnn-depth-${20+i}.bin").outputStream().buffered()).use { out ->
                                        out.writeInt(depth.size); depth.forEach { out.writeFloat(it) }
                                    }
                                    samples.add("${20+i}:camera=${(t1-t0)/1e6} main=${(t2-t1)/1e6} total=${(t2-t0)/1e6}ms")
                                }
                            }
                        }
                    }
                }
                "PASS: QNN experiment load=${loadMs}ms samples=$samples profile=${m.endProfiling()}; verify provider placement and output equivalence before use"
            }
        }
    }
}
