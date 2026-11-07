package com.recon.videoscan.live

import android.app.Activity
import android.app.Instrumentation
import android.content.Intent
import android.os.Bundle
import android.os.SystemClock
import java.io.File
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicLong
import java.util.concurrent.locks.ReentrantReadWriteLock
import kotlin.concurrent.write

/** Synthetic integration check of the actual Activity/JNI recovery, not a physical drift test.
 * Runs only from the separately installed test APK; never included in the application APK.
 */
class RecoveryInstrumentation : Instrumentation() {
    private var launchArguments: Bundle? = null
    override fun onCreate(arguments: Bundle?) { super.onCreate(arguments); launchArguments = arguments; start() }
    private fun field(a: LiveActivity, name: String): Any? =
        LiveActivity::class.java.getDeclaredField(name).apply { isAccessible = true }.get(a)
    private fun set(a: LiveActivity, name: String, value: Any) =
        LiveActivity::class.java.getDeclaredField(name).apply { isAccessible = true }.set(a, value)
    private fun waitFor(message: String, ready: () -> Boolean) {
        val end = SystemClock.elapsedRealtime() + 15000
        while (!ready()) {
            check(SystemClock.elapsedRealtime() < end) { message }
            SystemClock.sleep(20)
        }
    }
    override fun onStart() {
        if (launchArguments?.getString("qnn_recovery_smoke") == "1") {
            var host: Activity? = null
            val result = try {
                host=startActivitySync(Intent(targetContext,DepthBenchmarkActivity::class.java).addFlags(Intent.FLAG_ACTIVITY_NEW_TASK))
                QnnRecoverySmoke.run(targetContext)
            } catch (e: Throwable) { "FAIL: ${e.stackTraceToString()}" }
            finally { host?.let { runOnMainSync { it.finish() } } }
            finish(if (result.startsWith("PASS")) Activity.RESULT_OK else Activity.RESULT_CANCELED,
                Bundle().apply { putString("stream",result) })
            return
        }
        if (launchArguments?.getString("depth_qnn_benchmark") == "1") {
            var host: Activity? = null
            val result = try {
                host=startActivitySync(Intent(targetContext,DepthBenchmarkActivity::class.java).addFlags(Intent.FLAG_ACTIVITY_NEW_TASK))
                DepthQnnBenchmark.run(targetContext)
            } catch (e: Throwable) { "FAIL: ${e.stackTraceToString()}" }
            finally { host?.let { runOnMainSync { it.finish() } } }
            finish(if (result.startsWith("PASS")) Activity.RESULT_OK else Activity.RESULT_CANCELED,
                Bundle().apply { putString("stream",result) })
            return
        }
        launchArguments?.getString("depth_runtime_benchmark")?.let { modelFile ->
            var host: Activity? = null
            val result = try {
                host = startActivitySync(Intent(targetContext,DepthBenchmarkActivity::class.java).addFlags(Intent.FLAG_ACTIVITY_NEW_TASK))
                check((targetContext.getSystemService(android.content.Context.POWER_SERVICE) as android.os.PowerManager).isInteractive) { "phone screen must be awake" }
                DepthRuntimeBenchmark.run(targetContext,modelFile,launchArguments?.getString("runtime_configuration") ?: "cpu4",
                    launchArguments?.getString("runtime_profile") == "1")
            } catch (e: Throwable) { "FAIL: ${e.stackTraceToString()}" }
            finally { host?.let { runOnMainSync { it.finish() } } }
            finish(if (result.startsWith("PASS")) Activity.RESULT_OK else Activity.RESULT_CANCELED,
                Bundle().apply { putString("stream",result) })
            return
        }
        launchArguments?.getString("depth_benchmark")?.let { modelFile ->
            var host: Activity? = null
            val result = try {
                host = startActivitySync(Intent(targetContext,DepthBenchmarkActivity::class.java).addFlags(Intent.FLAG_ACTIVITY_NEW_TASK))
                check((targetContext.getSystemService(android.content.Context.POWER_SERVICE) as android.os.PowerManager).isInteractive) { "phone screen must be awake" }
                DepthEvidenceReplay.benchmark(targetContext,modelFile)
            } catch (e: Throwable) { "FAIL: ${e.stackTraceToString()}" }
            finally { host?.let { runOnMainSync { it.finish() } } }
            finish(if (result.startsWith("PASS")) Activity.RESULT_OK else Activity.RESULT_CANCELED,
                Bundle().apply { putString("stream",result) })
            return
        }
        launchArguments?.getString("depth_replay")?.let { directory ->
            var host: Activity? = null
            val result = try {
                host = startActivitySync(Intent(targetContext,DepthBenchmarkActivity::class.java).addFlags(Intent.FLAG_ACTIVITY_NEW_TASK))
                DepthEvidenceReplay.run(targetContext, directory, launchArguments?.getString("depth_replay_model"))
            }
                catch (e: Throwable) { "FAIL: ${e.stackTraceToString()}" }
            finally { host?.let { runOnMainSync { it.finish() } } }
            finish(if (result.startsWith("PASS")) Activity.RESULT_OK else Activity.RESULT_CANCELED,
                Bundle().apply { putString("stream", result) })
            return
        }
        if (launchArguments?.getString("depth_smoke") == "1") {
            val result = try { DepthDeviceSmoke.run(targetContext) } catch (e: Throwable) { "FAIL: ${e.stackTraceToString()}" }
            finish(if (result.startsWith("PASS")) Activity.RESULT_OK else Activity.RESULT_CANCELED,
                Bundle().apply { putString("stream", result) })
            return
        }
        if (launchArguments?.getString("landmark_reference") == "1") {
            var handle = 0L
            val result = try {
                handle = Native.init(File(targetContext.getExternalFilesDir(null), "vio/estimator_config.yaml").absolutePath, DoubleArray(0))
                check(handle != 0L)
                val landmarks = FloatArray(64*3)
                repeat(64) { i ->
                    landmarks[3*i]=(i%8-3.5f)*.35f
                    landmarks[3*i+1]=-(i/8-3.5f)*.25f
                    landmarks[3*i+2]=-(3f+.15f*(i%5))
                }
                val random=java.util.Random(42)
                val descriptors=FloatArray(64*24*40) { random.nextFloat()-.5f }
                val map=Native.mapFrame(handle,1.0,true,true,floatArrayOf(0f,0f,0f,0f,0f,0f,1f),
                    landmarks,64,FloatArray(0),descriptors)
                check(map[0] == 1.0 && map[1] == 1.0) { "tracked landmarks did not create reference without detector overlap: ${map.contentToString()}" }
                val pose=floatArrayOf(0f,0f,0f,0f,0f,0f,1f)
                val dip=Native.mapFrame(handle,1.1,false,true,pose,FloatArray(0),0,null,null)
                check(dip[0] == 1.0) { "brief low-quality frame invalidated map alignment" }
                val recovered=Native.mapFrame(handle,1.2,true,true,pose,FloatArray(0),0,null,null)
                check(recovered[0] == 1.0 && recovered[1] == 1.0)
                val jump=Native.mapFrame(handle,1.3,false,true,pose,FloatArray(0),0,null,null,true)
                check(jump[0] == 0.0) { "actual frame discontinuity must require relocalization" }
                "PASS: landmark reference created; brief tracking dip preserves alignment; actual discontinuity holds map"
            } catch (e: Throwable) { "FAIL: ${e.stackTraceToString()}" }
            finally { if(handle != 0L) Native.destroy(handle) }
            finish(if(result.startsWith("PASS")) Activity.RESULT_OK else Activity.RESULT_CANCELED,
                Bundle().apply { putString("stream",result) })
            return
        }
        if (launchArguments?.getString("empty_map_recovery") == "1") {
            val handles = ArrayList<Long>()
            val result = try {
                val config = File(targetContext.getExternalFilesDir(null), "vio/estimator_config.yaml")
                fun create(): Long = Native.init(config.absolutePath, DoubleArray(0)).also { check(it != 0L); handles.add(it) }
                val old = create(); val next = create()
                Native.retainMap(old, next)
                val pose = floatArrayOf(1f,2f,3f,0f,0f,0f,1f)
                val status = Native.mapFrame(next,1.0,true,true,pose,FloatArray(0),0,null,null)
                check(status[0] == 1.0 && status[1] == 0.0) { "empty map cannot acquire its first reference" }
                check(pose[0] == 1f && pose[1] == 2f && pose[2] == 3f) { "new pose not published" }
                // An unconfirmed first frame allocates blocks but has no visible
                // surface/reference. Recovery must not wait forever for PnP.
                val provisional=create(); val restarted=create()
                val stats=Native.tsdfIntegrate(provisional,ShortArray(64*48) { 1500 },64,48,
                    floatArrayOf(60f,60f,32f,24f),floatArrayOf(1f,0f,0f,0f,1f,0f,0f,0f,1f),
                    floatArrayOf(0f,0f,0f),null,0,0,1,ByteArray(64*48) { 64 })
                check(stats[3] > 0 && Native.tsdfSurface(provisional,100000).isEmpty())
                Native.retainMap(provisional,restarted)
                check(Native.mapFrame(restarted,1.1,true,true,pose,FloatArray(0),0,null,null)[0] == 1.0) {
                    "provisional-only map prevented recovery without a reference"
                }
                repeat(4) { Native.tsdfIntegrate(next,ShortArray(64*48) { 1500 },64,48,
                    floatArrayOf(60f,60f,32f,24f),floatArrayOf(1f,0f,0f,0f,1f,0f,0f,0f,1f),
                    floatArrayOf(0f,0f,0f),null,0,0,1) }
                val count = Native.tsdfSurface(next,300000).size; check(count > 0)
                val recovered = create(); Native.retainMap(next,recovered)
                check(Native.mapFrame(recovered,2.0,true,true,pose,FloatArray(0),0,null,null)[0] == 0.0)
                check(Native.tsdfSurface(recovered,300000).size == count)
                "PASS: empty and provisional-only restarts permit new poses/reference creation; confirmed geometry stays held and preserved"
            } catch (e: Throwable) { "FAIL: ${e.stackTraceToString()}" }
            finally { handles.forEach { Native.destroy(it) } }
            finish(if (result.startsWith("PASS")) Activity.RESULT_OK else Activity.RESULT_CANCELED,
                Bundle().apply { putString("stream",result) })
            return
        }
        if (launchArguments?.getString("overlay_smoke") == "1") {
            val result = try {
                lateinit var overlay: FeatureOverlayView
                val points = FeatureOverlayView::class.java.getDeclaredField("points").apply { isAccessible = true }
                repeat(20) {
                    runOnMainSync {
                        overlay = FeatureOverlayView(targetContext)
                        overlay.layout(0, 0, 200, 200)
                        overlay.submit(ByteArray(16), 2, 2, 2, 2, FloatArray(300) { 1f }, 1.0, 90)
                        // Hold the UI thread while recovery requests a clear. It must
                        // not mutate the array that this draw is about to traverse.
                        Thread { overlay.clear() }.apply { start(); join() }
                        check((points.get(overlay) as FloatArray).size == 300)
                        val bitmap = android.graphics.Bitmap.createBitmap(200, 200, android.graphics.Bitmap.Config.ARGB_8888)
                        overlay.draw(android.graphics.Canvas(bitmap)); bitmap.recycle()
                    }
                    runOnMainSync { check((points.get(overlay) as FloatArray).isEmpty()) }
                }
                "PASS: 20 background recovery clears serialized with overlay drawing"
            } catch (e: Throwable) { "FAIL: ${e.stackTraceToString()}" }
            finish(if (result.startsWith("PASS")) Activity.RESULT_OK else Activity.RESULT_CANCELED,
                Bundle().apply { putString("stream", result) })
            return
        }
        if (launchArguments?.getString("xfeat_benchmark") == "1") {
            val result = try { XFeatDeviceBenchmark.run(targetContext) } catch (e: Throwable) { "FAIL: ${e.stackTraceToString()}" }
            finish(if (result.startsWith("PASS")) Activity.RESULT_OK else Activity.RESULT_CANCELED, Bundle().apply { putString("stream", result) })
            return
        }
        var activity: LiveActivity? = null
        try {
            val a = startActivitySync(Intent(targetContext, LiveActivity::class.java)
                .addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)) as LiveActivity
            activity = a
            val gate = field(a, "sessionGate") as ReentrantReadWriteLock
            waitFor("native initialization timed out") { gate.write { field(a, "handle") != 0L } }
            // Stop actual camera and IMU producers; the controlled scene below is synthetic.
            runOnMainSync { callActivityOnPause(a) }
            val pauseSaved = java.util.concurrent.CountDownLatch(1)
            (field(a, "vioHandler") as android.os.Handler).post { pauseSaved.countDown() }
            check(pauseSaved.await(15, java.util.concurrent.TimeUnit.SECONDS)) { "pause save did not finish" }
            val generation = field(a, "generation") as AtomicLong
            val pending = field(a, "recoveryPending") as AtomicBoolean
            val beforeGeneration = generation.get()
            val root = File(a.getExternalFilesDir(null), "segments")
            val beforeFiles = root.listFiles()?.map { it.name }?.toSet() ?: emptySet()
            val health = field(a, "trackingHealth") as TrackingHealth
            val now = SystemClock.elapsedRealtimeNanos() * 1e-9
            val pose = floatArrayOf(0f, 0f, 0f, 0f, 0f, 0f, 1f)
            var surfaceCount = 0
            lateinit var stalePermit: TrackingHealth.Permit
            gate.write {
                val h = field(a, "handle") as Long
                repeat(6) {
                    Native.tsdfIntegrate(h, ShortArray(64 * 48) { 1500 }, 64, 48,
                        floatArrayOf(60f, 60f, 32f, 24f),
                        floatArrayOf(1f, 0f, 0f, 0f, 1f, 0f, 0f, 0f, 1f),
                        floatArrayOf(0f, 0f, 0f), null, 0, 0, 1)
                }
                surfaceCount = Native.tsdfSurface(h, 1000000).size / 6
                check(surfaceCount > 0) { "test scene produced no surface" }
                health.reset()
                repeat(3) { i ->
                    val t = now + i * .01
                    health.update(t, pose, VioDiagnostics(t, t, 40, 10, 8, 0, 0, false, .1, .05))
                }
                stalePermit = checkNotNull(health.permit(now + .02))
                val warm = field(a, "warmStart") as BiasWarmStart
                val prior = doubleArrayOf(.0001, .0002, .0003, 1e-6, 1e-6, 1e-6)
                warm.observe(now - 1.1, true, true, .001) { prior }
                warm.observe(now, true, true, .001) { prior }
                health.update(now + .03, null, null)
                check(health.update(now + 3.04, null, null).state == TrackingHealth.State.RESTARTING)
                set(a, "active", true)
                LiveActivity::class.java.getDeclaredMethod("recoverIfNeeded", TrackingHealth.Status::class.java)
                    .apply { isAccessible = true }.invoke(a, health.current(now + 3.04))
            }
            waitFor("automatic recovery timed out") {
                gate.write { !pending.get() && (field(a, "nativeGeneration") as Long) > beforeGeneration }
            }
            gate.write {
                check(field(a, "timingError") == null) { "recovery reported an error" }
                check(Native.tsdfSurface(field(a, "handle") as Long, 1000000).size / 6 == surfaceCount) { "recovery lost cloud coverage" }
                check(field(a, "mapAligned") == false) { "fusion must wait for a saved-view match" }
                check(!health.commit(stalePermit, now + .03) { error("stale fusion committed") })
                check(health.current(SystemClock.elapsedRealtimeNanos() * 1e-9).state == TrackingHealth.State.INITIALIZING)
            }
            val saved = root.listFiles()!!.filter { it.name !in beforeFiles && it.extension == "ply" }
            check(saved.isEmpty()) { "automatic recovery should retain the map, not create a separate segment" }
            val result = "PASS: actual Activity recovery retained $surfaceCount points, " +
                "paused map fusion pending relocalization, rejected old fusion permit, and returned to INITIALIZING. Synthetic test only."
            finish(Activity.RESULT_OK, Bundle().apply { putString("stream", result) })
        } catch (e: Throwable) {
            finish(Activity.RESULT_CANCELED, Bundle().apply { putString("stream", "FAIL: ${e.stackTraceToString()}") })
        } finally {
            activity?.let { a -> runOnMainSync { a.finish() } }
        }
    }
}
