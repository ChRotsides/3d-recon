package com.recon.videoscan.live

import android.content.Context
import android.os.SystemClock
import java.io.File

/** Uses real sessions, with no camera, preferences or saved-map mutations. */
object QnnRecoverySmoke {
    fun run(context: Context): String {
        val models=File(context.getExternalFilesDir(null),"models")
        val frame=DepthEvidenceReplay.read(File(context.getExternalFilesDir(null),
            "depth-evidence/map-1791434991165-1/depth-24.bin.gz"))
        val calibration=CameraCalibration(PinholeCamera(frame.w,frame.h,frame.k[0].toDouble(),frame.k[1].toDouble(),frame.k[2].toDouble(),frame.k[3].toDouble()),
            CameraCalibration.Model.RADTAN,0.0,0.0,0.0,0.0)
        val times=ArrayList<Long>()
        repeat(2) {
            val start=SystemClock.elapsedRealtime()
            QnnDepthSession(models,context.noBackupFilesDir,context.applicationInfo.nativeLibraryDir).use { session ->
                check(session.loadedFromCache && session.compileMillis == 0L) { "QNN graph recompiled instead of loading cache" }
                val elapsed=SystemClock.elapsedRealtime()-start
                check(elapsed < 10000) { "cached startup too slow: ${elapsed}ms" }
                times.add(elapsed)
            }
        }
        DepthRunner(models,QnnDepthSession.MAIN_MODEL,cacheDir=context.noBackupFilesDir,nativeLibDir=context.applicationInfo.nativeLibraryDir).use { runner ->
            check(runner.backend == "npu")
            val session=DepthRunner::class.java.getDeclaredField("accelerated").apply { isAccessible=true }.get(runner) as QnnDepthSession
            session.close() // A lost runtime must be handled by the real fallback path.
            val prediction=runner.run(frame.rgba,frame.w,frame.h,calibration)
            check(runner.backend == "cpu" && runner.spec?.file == DepthRunner.CPU_UNIDEPTH)
            check(prediction.depth.count { it.isFinite() && it > 0 } > prediction.depth.size*.95)
        }
        return "PASS: cached starts ${times}ms, neither recompiled; invalidated NPU session fell back to the actual CPU model and produced finite depth"
    }
}
