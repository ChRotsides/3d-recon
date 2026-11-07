package com.recon.videoscan.live

import kotlin.math.abs
import kotlin.math.sqrt

data class VioDiagnostics(val timestamp: Double, val stateTime: Double, val tracks: Int,
    val msckfAttempted: Int, val msckfAccepted: Int, val slamAttempted: Int, val slamAccepted: Int,
    val stationary: Boolean, val speed: Double, val positionSigma: Double) {
    val attempted get() = msckfAttempted + slamAttempted
    val accepted get() = msckfAccepted + slamAccepted
    val notAccepted get() = attempted - accepted
    companion object {
        fun fromNative(v: DoubleArray): VioDiagnostics? {
            if (v.size !in listOf(10, 35, 50, 70) || v.take(8).any { !it.isFinite() }) return null
            if ((2..6).any { v[it] < 0 || v[it] != v[it].toInt().toDouble() } ||
                v[4] > v[3] || v[6] > v[5] || v[7] !in listOf(0.0, 1.0)) return null
            return VioDiagnostics(v[0], v[1], v[2].toInt(), v[3].toInt(), v[4].toInt(),
                v[5].toInt(), v[6].toInt(), v[7] == 1.0, v[8], v[9])
        }
    }
}

/** Conservative fusion guard, not a detector of every possible VIO failure.
 * Exported 3D landmark count is deliberately not used as a tracking-quality proxy.
 */
class TrackingHealth {
    enum class State { INITIALIZING, ACQUIRING, TRACKING, STATIONARY, RECOVERING, RESTARTING }
    data class Status(val state: State, val reason: String, val epoch: Long) {
        val poseReliable get() = state == State.TRACKING || state == State.STATIONARY
        val fusionAllowed get() = state == State.TRACKING || state == State.STATIONARY
    }
    data class Permit(val epoch: Long, val timestamp: Double)
    private var status = Status(State.INITIALIZING, "waiting for initialization", 0)
    private var lastFrame = Double.NaN
    private var lastPose: FloatArray? = null
    private var badSince = Double.NaN
    private var goodFrames = 0
    private var everReliable = false
    private val recoverySeconds = 3.0
    private var lastVisualConstraint = Double.NaN
    private var hasEstimate = false

    @Synchronized fun reset() {
        status = Status(State.INITIALIZING, "waiting for initialization", status.epoch + 1)
        lastFrame = Double.NaN; lastPose = null
        badSince = Double.NaN; goodFrames = 0; everReliable = false
        lastVisualConstraint = Double.NaN; hasEstimate = false
    }
    private fun set(state: State, reason: String): Status {
        val nextFusionAllowed = state == State.TRACKING || state == State.STATIONARY
        val epoch = status.epoch + if (status.fusionAllowed && !nextFusionAllowed) 1 else 0
        status = Status(state, reason, epoch)
        return status
    }
    private fun fail(t: Double, reason: String): Status {
        goodFrames = 0
        if (!badSince.isFinite()) badSince = t
        return set(if ((everReliable || hasEstimate) && t - badSince >= recoverySeconds) State.RESTARTING else State.RECOVERING, reason)
    }
    /** An unobserved IMU interval or a changed coordinate frame cannot be fused. */
    @Synchronized fun inputDiscontinuity(reason: String): Status {
        goodFrames = 0
        return set(State.RESTARTING, reason)
    }

    @Synchronized fun update(t: Double, pose: FloatArray?, d: VioDiagnostics?): Status {
        if (status.state == State.RESTARTING) return status
        if (!t.isFinite() || (lastFrame.isFinite() && t <= lastFrame))
            return fail(lastFrame, "non-monotonic camera time")
        val dt = t - lastFrame
        val previous = lastPose
        lastFrame = t
        if (pose == null) {
            return if (!hasEstimate) set(State.INITIALIZING, "waiting for a current pose")
                else fail(t, "no current pose")
        }
        hasEstimate = true
        if (pose.size != 7 || pose.any { !it.isFinite() } || d == null ||
            !d.timestamp.isFinite() || !d.stateTime.isFinite() ||
            abs(d.timestamp - t) > 1e-6 || abs(d.stateTime - t) > 1e-6 ||
            !d.speed.isFinite() || d.speed < 0 || !d.positionSigma.isFinite() || d.positionSigma < 0 ||
            (3..6).sumOf { pose[it].toDouble() * pose[it] } !in 0.9..1.1)
            return fail(t, "invalid or mismatched estimate")
        if (everReliable && dt > 0.75) return fail(t, "camera update gap")
        if (d.positionSigma > 0.75) return fail(t, "position uncertainty too high")
        if (d.speed > 3.0) return fail(t, "implausible speed for handheld scan")
        // The first visual updates can correct an unconstrained startup position.
        // There is no trusted map frame to preserve until acquisition succeeds.
        if (everReliable && previous != null && dt > 0 && dt < 0.75) {
            val step = sqrt((0..2).sumOf { (pose[it] - previous[it]).toDouble().let { x -> x * x } })
            if (step > 0.25 + 3.0 * dt) return inputDiscontinuity("pose discontinuity; matching saved map again")
        }
        lastPose = pose.copyOf()
        // Image tracks alone are not a position constraint. A rejected EKF update
        // can leave hundreds of tracks while the pose propagates on IMU alone.
        val stationary = d.stationary && d.speed <= 0.15
        if (d.accepted > 0 || stationary) lastVisualConstraint = t
        if (d.tracks < 12) return fail(t, "too few image tracks")
        if (!lastVisualConstraint.isFinite() || t - lastVisualConstraint > 1.0)
            return fail(t, "waiting for accepted visual or stationary constraints")
        if (everReliable && badSince.isFinite() && t - badSince >= recoverySeconds)
            return set(State.RESTARTING, "tracking gap; matching saved map again")
        goodFrames++
        if (goodFrames < 3) return set(State.ACQUIRING, "confirming tracking")
        badSince = Double.NaN; everReliable = true
        return set(if (stationary) State.STATIONARY else State.TRACKING,
            if (stationary) "stationary constraint active" else "visual tracking active")
    }
    @Synchronized fun current(now: Double): Status {
        if (status.state != State.RESTARTING && lastFrame.isFinite() && now - lastFrame > 0.75) {
            if (!badSince.isFinite()) badSince = lastFrame + 0.75
            fail(now, "camera/IMU updates stalled")
        }
        return status
    }
    @Synchronized fun permit(t: Double): Permit? =
        if (status.fusionAllowed && t == lastFrame) Permit(status.epoch, t) else null

    /** A completed depth image uses its captured pose. Bounded inference latency
     * can be accommodated only while tracking remains continuously reliable in
     * the same epoch. Camera freshness, loss, discontinuity, and reset checks
     * are unchanged; queued work still uses the short default budget. */
    @Synchronized fun commit(permit: Permit, now: Double, maxAgeSeconds: Double = 1.5, action: () -> Unit): Boolean {
        current(now)
        if (!maxAgeSeconds.isFinite() || maxAgeSeconds <= 0 || maxAgeSeconds > 3.0 ||
            !now.isFinite() || !permit.timestamp.isFinite() || !status.fusionAllowed || permit.epoch != status.epoch || now - permit.timestamp > maxAgeSeconds || now < permit.timestamp) return false
        action()
        return true
    }
}
