package com.recon.videoscan.live

import android.content.Context
import android.os.SystemClock
import java.io.DataInputStream
import java.io.DataOutputStream
import java.io.File
import java.util.zip.GZIPInputStream
import kotlin.math.roundToInt

/** Replays a frozen, explicitly supplied capture through real calibration and JNI fusion.
 * Only in the test APK. Uses isolated native maps and never writes scan segments. */
object DepthEvidenceReplay {
    internal data class Frame(val t: Double, val generation: Long, val revision: Long,
        val fused: Boolean, val w: Int, val h: Int, val pose: FloatArray, val k: FloatArray,
        val landmarks: FloatArray, val prediction: FloatArray, val depth: ShortArray,
        val confidence: ByteArray, val rgba: ByteArray)

    internal fun read(file: File): Frame = DataInputStream(GZIPInputStream(file.inputStream().buffered()).buffered(64 * 1024)).use { input ->
        check(input.readInt() == 0x44505448 && input.readInt() == 1)
        fun length(): Int = input.readInt().also { check(it in 0..16_000_000) }
        fun floats() = FloatArray(length()) { input.readFloat() }
        fun bytes() = ByteArray(length()).also { input.readFully(it) }
        val t = input.readDouble(); val generation = input.readLong(); val revision = input.readLong()
        check(input.readUTF().endsWith(":RELATIVE_INVERSE")); input.readUTF()
        val fused = input.readBoolean(); val w = input.readInt(); val h = input.readInt()
        input.readInt(); input.readInt()
        val pose = floats(); val k = floats(); repeat(length()) { input.readDouble() }
        val landmarks = floats(); val prediction = floats()
        val depth = ShortArray(length()) { input.readShort() }
        val confidence = bytes(); val rgba = bytes(); bytes()
        check(input.read() == -1)
        Frame(t,generation,revision,fused,w,h,pose,k,landmarks,prediction,depth,confidence,rgba)
    }
    private fun rotation(q: FloatArray): FloatArray {
        val x=q[3]; val y=q[4]; val z=q[5]; val w=q[6]
        return floatArrayOf(1-2*(y*y+z*z),-2*(x*y-z*w),-2*(x*z+y*w),
            2*(x*y+z*w),-(1-2*(x*x+z*z)),-2*(y*z-x*w),
            2*(x*z-y*w),-2*(y*z+x*w),-(1-2*(x*x+y*y)))
    }
    // Compare only the already visible map, before this frame can contribute
    // any depth. These are later-frame VIO checks, not ground-truth accuracy.
    private fun compareSurface(surface: FloatArray, f: Frame, r: FloatArray, errors: MutableList<Float>,
        audit: MutableList<String>? = null, frameIndex: Int = 0) {
        val points = FloatArray(surface.size / 2)
        for (i in surface.indices step 6) {
            val x=surface[i]-f.pose[0]; val y=surface[i+1]-f.pose[1]; val z=surface[i+2]-f.pose[2]
            val j=i/2
            points[j]=r[0]*x+r[3]*y+r[6]*z
            points[j+1]=r[1]*x+r[4]*y+r[7]*z
            points[j+2]=r[2]*x+r[5]*y+r[8]*z
        }
        for (i in f.landmarks.indices step 3) {
            val x=f.landmarks[i]-f.pose[0]; val y=f.landmarks[i+1]-f.pose[1]; val z=f.landmarks[i+2]-f.pose[2]
            val depth=r[2]*x+r[5]*y+r[8]*z
            if (depth !in .2f..6f) continue
            val cx=r[0]*x+r[3]*y+r[6]*z; val cy=r[1]*x+r[4]*y+r[7]*z
            val norm=kotlin.math.sqrt(cx*cx+cy*cy+depth*depth)
            val ux=cx/norm; val uy=cy/norm; val uz=depth/norm
            var nearest=Float.POSITIVE_INFINITY
            for (j in points.indices step 3) {
                val px=points[j]; val py=points[j+1]; val pz=points[j+2]
                if (pz <= .2f) continue
                val along=px*ux+py*uy+pz*uz
                val dx=px-along*ux; val dy=py-along*uy; val dz=pz-along*uz
                // A metric ray-distance limit avoids the larger, depth-dependent
                // footprint and foreground bias of a 3x3 image-pixel search.
                if (along > 0 && dx*dx+dy*dy+dz*dz <= .015f*.015f) nearest=minOf(nearest,pz)
            }
            if (nearest.isFinite()) {
                val error=kotlin.math.abs(nearest-depth)/depth
                errors.add(error); audit?.add("$frameIndex,${i/3},$error")
            }
        }
    }

    fun benchmark(context: Context, modelFile: String): String {
        val root = context.getExternalFilesDir(null)!!
        val f = read(File(root,"depth-evidence/map-1791434991165-1/depth-24.bin.gz"))
        val calibration = CameraCalibration(PinholeCamera(f.w,f.h,f.k[0].toDouble(),f.k[1].toDouble(),f.k[2].toDouble(),f.k[3].toDouble()),
            CameraCalibration.Model.RADTAN,0.0,0.0,0.0,0.0)
        val times = ArrayList<Long>()
        DepthRunner(File(root,"models"),modelFile,cacheDir=context.noBackupFilesDir,nativeLibDir=context.applicationInfo.nativeLibraryDir).use { runner ->
            check(runner.spec?.file == modelFile)
            repeat(8) {
                val prediction=runner.run(f.rgba,f.w,f.h,calibration)
                check(prediction.depth.count { it.isFinite() && it > 0 } > prediction.depth.size * .95)
                times.add(prediction.ms)
            }
        }
        check(times.sorted()[times.size/2] < 1200) { "too slow for live use: $modelFile inferenceMs=$times" }
        return "PASS: foreground $modelFile inferenceMs=$times"
    }
    fun run(context: Context, relativeDirectory: String, modelFile: String? = null): String {
        val root = context.getExternalFilesDir(null)!!.canonicalFile
        val directory = File(root, relativeDirectory).canonicalFile
        check(directory.path.startsWith(root.path + File.separator) && directory.isDirectory)
        val files = directory.listFiles()!!.filter { it.name.matches(Regex("depth-[0-9]+\\.bin\\.gz")) }
            .sortedBy { it.name.removePrefix("depth-").substringBefore('.').toInt() }
        check(files.size >= 30) { "need the frozen multi-frame phone capture" }
        val config = File(root,"vio/estimator_config.yaml").absolutePath
        var baseline = 0L; var updated = 0L
        var runner: DepthRunner? = null
        val inferenceTimes = ArrayList<Long>()
        try {
            if (modelFile != null) {
                runner = DepthRunner(File(root,"models"),modelFile,includeExperimental=true,
                    cacheDir=context.noBackupFilesDir,nativeLibDir=context.applicationInfo.nativeLibraryDir)
                check(runner.spec?.file == modelFile) { "model not installed: $modelFile" }
            }
            baseline = Native.init(config,DoubleArray(0)); updated = Native.init(config,DoubleArray(0))
            check(baseline != 0L && updated != 0L)
            val quality = DepthQuality(); var epoch: Pair<Long,Long>? = null
            var calibrated = 0; var supportedFrames = 0; var supportedPixels = 0L
            var growthFrames = 0; var priorPoints = 0; var maxQualityMs = 0L
            val geometryErrors = ArrayList<Float>(); val baselineErrors = ArrayList<Float>()
            val geometryAudit = ArrayList<String>()
            for ((frameIndex,file) in files.withIndex()) {
                var f = read(file); val key = f.generation to f.revision
                if (key != epoch) { quality.reset(); epoch = key }
                val r = rotation(f.pose); val c = f.pose.copyOfRange(0,3)
                compareSurface(Native.tsdfSurface(updated,100000),f,r,geometryErrors,geometryAudit,frameIndex)
                compareSurface(Native.tsdfSurface(baseline,100000),f,r,baselineErrors)
                if (f.fused && f.depth.isNotEmpty()) {
                    val stats = Native.tsdfIntegrate(baseline,f.depth,f.w,f.h,f.k,r,c,f.rgba,f.w,f.h,2,f.confidence)
                    check(stats[5] == 0f)
                }
                runner?.let { model ->
                    val calibration = CameraCalibration(PinholeCamera(f.w,f.h,f.k[0].toDouble(),f.k[1].toDouble(),f.k[2].toDouble(),f.k[3].toDouble()),
                        CameraCalibration.Model.RADTAN,0.0,0.0,0.0,0.0)
                    val prediction = model.run(f.rgba,f.w,f.h,calibration)
                    check(prediction.frame.camera.width == f.w && prediction.frame.camera.height == f.h)
                    f = f.copy(prediction=prediction.depth)
                    inferenceTimes.add(prediction.ms)
                    if (model.backend == "npu") {
                        val audit=File(root,"runtime-benchmark").apply { mkdirs() }
                        DataOutputStream(File(audit,"qnn-replay-$frameIndex.bin").outputStream().buffered()).use { out ->
                            out.writeInt(prediction.depth.size); prediction.depth.forEach { out.writeFloat(it) }
                        }
                    }
                }
                val started = SystemClock.elapsedRealtime()
                val ds = ArrayList<Float>(); val zs = ArrayList<Float>(); val cells = ArrayList<Int>()
                for (i in f.landmarks.indices step 3) {
                    val px=f.landmarks[i]-c[0]; val py=f.landmarks[i+1]-c[1]; val pz=f.landmarks[i+2]-c[2]
                    val z=r[2]*px+r[5]*py+r[8]*pz
                    if (z <= .2f || z >= 8f) continue
                    val u=f.k[0]*(r[0]*px+r[3]*py+r[6]*pz)/z+f.k[2]
                    val v=f.k[1]*(r[1]*px+r[4]*py+r[7]*pz)/z+f.k[3]
                    if (u < 0 || u > f.w-1 || v < 0 || v > f.h-1) continue
                    val d=f.prediction[v.roundToInt()*f.w+u.roundToInt()]
                    if (!d.isFinite() || d <= 1e-4f) continue
                    ds.add(d); zs.add(z); cells.add((u*4/f.w).toInt().coerceIn(0,3)+4*(v*3/f.h).toInt().coerceIn(0,2))
                }
                val fit = quality.fit(ds,zs,cells,runner == null || runner.kind == DepthRunner.Kind.RELATIVE_INVERSE) ?: continue
                val mm = ShortArray(f.prediction.size) { i ->
                    val z=fit.depth(f.prediction[i])
                    if (z.isFinite() && z in .2f..6f) (z*1000).roundToInt().toShort() else 0
                }
                val confidence = quality.confidence(mm,f.w,f.h,f.k,r,c,f.t)
                val supported = confidence.count { it == (-1).toByte() }
                if (supported > 0) supportedFrames++
                supportedPixels += supported
                maxQualityMs = maxOf(maxQualityMs,SystemClock.elapsedRealtime()-started)
                val stats = Native.tsdfIntegrate(updated,mm,f.w,f.h,f.k,r,c,f.rgba,f.w,f.h,2,confidence)
                check(stats[5] == 0f) { "fusion rejected captured geometry" }
                quality.remember(mm,f.w,f.h,f.k,r,c,f.t,confidence); calibrated++
                val points = Native.tsdfSurface(updated,100000).size / 6
                if (points > priorPoints && priorPoints > 0) growthFrames++
                priorPoints = points
            }
            val oldPoints=Native.tsdfSurface(baseline,100000).size/6
            val newPoints=Native.tsdfSurface(updated,100000).size/6
            fun summary(errors: List<Float>): String {
                if (errors.isEmpty()) return "no overlapping visible surface"
                val sorted=errors.sorted()
                return "n=${sorted.size} median=${sorted[sorted.size/2]} p90=${sorted[(sorted.size-1)*9/10]}"
            }
            val geometry = "later-frame anchor relative error: baseline ${summary(baselineErrors)}; updated ${summary(geometryErrors)}; " +
                "model=${modelFile ?: "captured"}; inferenceMs=$inferenceTimes"
            // Allows paired comparisons on exactly the same frame/landmark
            // samples. Comparing aggregate errors alone confounds map coverage.
            val auditDirectory=File(root,"runtime-benchmark").apply { mkdirs() }
            File(auditDirectory,"geometry-${modelFile ?: "captured"}.csv")
                .writeText("frame,anchor,relative_error\n"+geometryAudit.joinToString("\n"))
            check(calibrated >= 20 && supportedFrames >= 8 && growthFrames >= 5 && newPoints > oldPoints) {
                "insufficient progress: calibrated=$calibrated supported=$supportedFrames growth=$growthFrames points=$oldPoints->$newPoints; $geometry"
            }
            val sorted=geometryErrors.sorted()
            check(sorted.size >= 20 && sorted[sorted.size/2] <= .05f && sorted[(sorted.size-1)*9/10] <= .12f) {
                "geometry check failed: $geometry; points=$oldPoints->$newPoints; calibrated=$calibrated supported=$supportedFrames growth=$growthFrames"
            }
            return "PASS: ${files.size} captured phone frames; calibrated=$calibrated supported=$supportedFrames " +
                "supportedPixels=$supportedPixels; cloud grew on $growthFrames subsequent frames; " +
                "visible points $oldPoints -> $newPoints; maximum calibration/confirmation ${maxQualityMs}ms; $geometry. " +
                "Saved-frame replay, not a new physical scan."
        } finally {
            runner?.close()
            if (baseline != 0L) Native.destroy(baseline)
            if (updated != 0L) Native.destroy(updated)
        }
    }
}
