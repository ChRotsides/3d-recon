package com.recon.videoscan.live

import java.io.File
import java.util.concurrent.ArrayBlockingQueue
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicLong

/** Bounded asynchronous CSV capture. No images/audio, no network upload, no deletion.
 * Stops after 120 seconds or 12 MiB per session. A full queue drops records, never
 * blocks sensor/VIO producers; the dropped count is written to the session summary.
 */
class DiagnosticRecorder(private val directory: File, private val start: Double,
    private val duration: Double = 120.0, private val byteLimit: Long = 12L * 1024 * 1024) {
    private val queue = ArrayBlockingQueue<Pair<Int, String>>(4096)
    private val dropped = AtomicLong()
    private var lastFeatureSample = Double.NEGATIVE_INFINITY
    @Volatile private var accepting = true
    @Volatile var state = "recording"
        private set
    private val thread = Thread({
        try {
            directory.mkdirs()
            File(directory, "imu.csv").bufferedWriter().use { imu ->
                File(directory, "frames.csv").bufferedWriter().use { frames ->
                    File(directory, "events.csv").bufferedWriter().use { events ->
                    File(directory, "features.csv").bufferedWriter().use { features ->
                    File(directory, "geometry.csv").bufferedWriter().use { geometry ->
                    features.appendLine("t,id,xn,yn")
                    geometry.appendLine("t,offset,bgx,bgy,bgz,r00,r01,r02,r10,r11,r12,r20,r21,r22")
                    events.appendLine("t,phase")
                    imu.appendLine("t_sensor_boot_seconds,type,x,y,z")
                    frames.appendLine("t,health,reason,pose_valid,px,py,pz,qx,qy,qz,qw,tracks,msckf_attempted,msckf_accepted,slam_attempted,slam_accepted,zupt,speed,position_sigma,offset,fx,fy,cx,cy,d0,d1,d2,d3,width,height,model,v_before_x,v_before_y,v_before_z,v_after_x,v_after_y,v_after_z,a_world_before_x,a_world_before_y,a_world_before_z,a_world_after_x,a_world_after_y,a_world_after_z,velocity_cov_before_trace,velocity_cov_after_trace,stationary_evidence,velocity_update,long_motion_px,short_motion_px,gyro_rms,accel_std,witness_count,witness_coverage,v_legacy_x,v_legacy_y,v_legacy_z,trend_px_s,trend_p75_px_s,gyro_mean,velocity_evidence,startup_accepted,startup_gyro_older,startup_gyro_newer,startup_accel_std,msckf_short,msckf_triangulation,msckf_refinement,msckf_chi2,startup_gyro_rms,startup_gyro_mean_change,stationary_rejection_bits,klt_input,klt_forward,klt_roundtrip,klt_geometric,tracks_age2,tracks_age5,track_median_observations,tri_condition,tri_behind,tri_near,tri_far,tri_nonfinite,clones,observations_removed,observations_seeded,tri_success,tri_min_condition,tri_max_abs_depth,klt_homography,klt_photometric")
                    var bytes = 0L
                    while (accepting || queue.isNotEmpty()) {
                        val row = queue.poll(200, TimeUnit.MILLISECONDS) ?: continue
                        if (bytes + row.second.length + 1 > byteLimit) { state = "size limit"; accepting = false; queue.clear(); break }
                        (when (row.first) { 0 -> imu; 1 -> frames; 2 -> events; 3 -> features; else -> geometry }).appendLine(row.second)
                        bytes += row.second.length + 1
                    }
                    if (state == "recording") state = "saved"
                    File(directory, "summary.txt").writeText("status=$state\ndropped_records=${dropped.get()}\nbytes=$bytes\n")
                    }
                    }
                    }
                }
            }
            if (state == "recording") state = "saved"
        } catch (e: Exception) { state = "error: ${e.message}"; accepting = false }
    }, "scan-diagnostics").apply { isDaemon = true; start() }

    @Synchronized private fun offer(t: Double, kind: Int, row: String) {
        if (!accepting) return
        if (t - start > duration) { state = "time limit"; accepting = false; return }
        if (!queue.offer(kind to row)) dropped.incrementAndGet()
    }
    fun imu(t: Double, type: Int, x: Float, y: Float, z: Float) = offer(t, 0, "$t,$type,$x,$y,$z")
    fun marker(t: Double, phase: String) {
        require(phase in listOf("STILL", "TILT", "HOLD", "TRANSLATE", "STOP", "DONE"))
        offer(t, 2, "$t,$phase")
    }
    @Synchronized fun featureSamples(t: Double, samples: DoubleArray) {
        if (!accepting || t-lastFeatureSample < .2 || samples.size < 14 || samples.any { !it.isFinite() }) return
        val count = samples[0].toInt()
        if (count !in 0..192 || samples[0] != count.toDouble() || samples.size != 14+3*count) return
        lastFeatureSample=t
        offer(t,4,"$t,${samples.sliceArray(1..13).joinToString(",")}")
        // The transformer consumes up to 192 live tracks; keep diagnostic CSV volume at 64.
        val recorded = minOf(count, 64)
        if (recorded>0) offer(t,3,(0 until recorded).joinToString("\n") { i ->
            "$t,${samples[14+3*i].toLong()},${samples[15+3*i]},${samples[16+3*i]}"
        })
    }
    fun frame(t: Double, health: TrackingHealth.Status, pose: FloatArray?, d: VioDiagnostics?, offset: Double, calibration: DoubleArray, extra: DoubleArray = DoubleArray(60) { Double.NaN }) {
        val p = pose?.joinToString(",") ?: "NaN,NaN,NaN,NaN,NaN,NaN,NaN"
        offer(t, 1, "$t,${health.state},${health.reason.replace(',', ';')},${pose != null},$p,${d?.tracks ?: 0},${d?.msckfAttempted ?: 0},${d?.msckfAccepted ?: 0},${d?.slamAttempted ?: 0},${d?.slamAccepted ?: 0},${d?.stationary ?: false},${d?.speed ?: Double.NaN},${d?.positionSigma ?: Double.NaN},$offset,${calibration.joinToString(",")},${extra.joinToString(",")}")
    }
    @Synchronized fun close() { accepting = false }
    fun awaitClosed(timeoutMs: Long = 2000) { thread.join(timeoutMs) }
}
