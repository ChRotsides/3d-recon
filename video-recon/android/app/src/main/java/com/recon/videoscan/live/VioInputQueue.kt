package com.recon.videoscan.live

import java.util.ArrayDeque

/** All times are seconds. Accelerometer timestamps form the common IMU grid; gyro is
 * interpolated, never extrapolated. Producers and the single VIO consumer may differ.
 */
class VioInputQueue<T>(private val maxSamples: Int = 2048, private val maxFrames: Int = 3) {
    data class Sample(val t: Double, val x: Double, val y: Double, val z: Double)
    data class Imu(val acceleration: Sample, val gyro: Sample)
    private data class Frame<T>(val t: Double, val value: T)
    private val gyros = ArrayDeque<Sample>()
    private val accelerations = ArrayDeque<Sample>()
    private val frames = ArrayDeque<Frame<T>>()
    private var minTime = Double.NEGATIVE_INFINITY
    private var lastGyro = Double.NEGATIVE_INFINITY
    private var lastAcceleration = Double.NEGATIVE_INFINITY
    private var lastFrame = Double.NEGATIVE_INFINITY
    private var discontinuity: String? = null
    @Volatile var droppedFrames = 0L
        private set

    init { require(maxSamples >= 2 && maxFrames >= 1) }

    @Synchronized fun reset(startTime: Double) {
        gyros.clear(); accelerations.clear(); frames.clear()
        minTime = startTime; lastGyro = startTime; lastAcceleration = startTime; lastFrame = startTime
        droppedFrames = 0
        discontinuity = null
    }

    @Synchronized fun addGyro(sample: Sample) {
        if (!valid(sample) || sample.t <= lastGyro) return
        if (lastGyro > minTime && sample.t - lastGyro > MAX_IMU_GAP_SECONDS)
            discontinuity = "gyroscope sample gap"
        lastGyro = sample.t; gyros.addLast(sample)
        while (gyros.size > maxSamples) { gyros.removeFirst(); discontinuity = "gyroscope queue overflow" }
    }

    @Synchronized fun addAcceleration(sample: Sample) {
        if (!valid(sample) || sample.t <= lastAcceleration) return
        if (lastAcceleration > minTime && sample.t - lastAcceleration > MAX_IMU_GAP_SECONDS)
            discontinuity = "accelerometer sample gap"
        lastAcceleration = sample.t; accelerations.addLast(sample)
        while (accelerations.size > maxSamples) { accelerations.removeFirst(); discontinuity = "accelerometer queue overflow" }
    }

    private fun valid(s: Sample) = s.t.isFinite() && s.t > minTime &&
        s.x.isFinite() && s.y.isFinite() && s.z.isFinite()

    @Synchronized fun drainImu(): List<Imu> {
        val result = ArrayList<Imu>()
        while (accelerations.isNotEmpty() && gyros.size >= 2) {
            val a = accelerations.first
            if (a.t < gyros.first.t) { accelerations.removeFirst(); continue }
            if (a.t > gyros.last.t) break
            while (gyros.size > 2 && gyros.elementAt(1).t <= a.t) gyros.removeFirst()
            val left = gyros.first; val right = gyros.elementAt(1)
            // Never fabricate motion through an unobserved interval.
            if (right.t - left.t > MAX_IMU_GAP_SECONDS) { accelerations.removeFirst(); continue }
            val alpha = (a.t - left.t) / (right.t - left.t)
            val g = Sample(a.t, left.x + alpha * (right.x - left.x),
                left.y + alpha * (right.y - left.y), left.z + alpha * (right.z - left.z))
            result.add(Imu(a, g)); accelerations.removeFirst()
        }
        return result
    }

    @Synchronized fun addFrame(t: Double, value: T) {
        if (!t.isFinite() || t <= minTime || t <= lastFrame) return
        lastFrame = t; frames.addLast(Frame(t, value))
        while (frames.size > maxFrames) { frames.removeFirst(); droppedFrames++ }
    }

    /** latestImu is the last sample delivered to OpenVINS. Recheck its current offset for
     * every frame. A small lead supplies a right endpoint at sample boundaries too.
     */
    @Synchronized fun pollFrame(latestImu: Double, cameraImuOffset: Double): T? {
        val frame = frames.peekFirst() ?: return null
        if (!latestImu.isFinite() || !cameraImuOffset.isFinite() ||
            latestImu < frame.t + cameraImuOffset + IMU_LEAD_SECONDS) return null
        return frames.removeFirst().value
    }

    @Synchronized fun takeDiscontinuity(): String? = discontinuity.also { discontinuity = null }

    companion object {
        const val IMU_LEAD_SECONDS = 0.002
        const val MAX_IMU_GAP_SECONDS = 0.05
    }
}
