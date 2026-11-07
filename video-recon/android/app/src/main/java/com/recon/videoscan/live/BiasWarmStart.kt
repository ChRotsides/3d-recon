package com.recon.videoscan.live

/** Short-lived prior from successful stationary corrections, never a lost state. */
class BiasWarmStart {
    private var since = Double.NaN
    private var savedAt = Double.NEGATIVE_INFINITY
    private var saved: DoubleArray? = null
    fun observe(t: Double, stationary: Boolean, corrected: Boolean, speed: Double, read: () -> DoubleArray) {
        if (!stationary || !speed.isFinite() || speed > .02) { since = Double.NaN; return }
        if (!since.isFinite()) since = t
        if (!corrected || t - since < 1.0 || t - savedAt < 1.0) return
        val value = read()
        if (value.size != 6 || value.any { !it.isFinite() } ||
            value.take(3).any { kotlin.math.abs(it) > .005 } || value.drop(3).any { it < 0 || it > 1e-4 }) return
        saved = value.copyOf(); savedAt = t
    }
    fun prior(now: Double): DoubleArray {
        val value = saved ?: return doubleArrayOf()
        val age = now - savedAt
        if (!age.isFinite() || age < 0 || age > 120) return doubleArrayOf()
        return value.copyOf(7).also {
            for (k in 3..5) it[k] = maxOf(1e-6, it[k]) + age * 1e-8
            it[6] = savedAt + 120 // expiration is checked again when initialization actually accepts
        }
    }
    fun resetEvidence() { since = Double.NaN }
}
