package com.recon.videoscan.live

/** Five-second UX deadline is not permission to fuse an unobservable estimate. */
class StartupProgress {
    private var began = Double.NaN
    private var ready = false
    @Synchronized fun start(now: Double, recovering: Boolean) {
        // Repeated failures before first acquisition must not reset the visible clock.
        if (!recovering || ready || !began.isFinite()) began = now
        ready = false
    }
    @Synchronized fun observe(reliable: Boolean, now: Double = Double.NaN): Double? {
        if (!reliable || ready) return null
        ready = true
        return if (began.isFinite() && now.isFinite()) (now-began).coerceAtLeast(0.0) else null
    }
    @Synchronized fun message(now: Double): String {
        if (ready || !began.isFinite()) return ""
        val elapsed = (now - began).coerceAtLeast(0.0)
        return if (elapsed < 5.0)
            "Starting handheld (${elapsed.toInt()}/5 s). Gently move sideways and turn toward visible detail.\n"
        else "Not ready after 5 s: need clearer features or more motion. Fusion stays paused; retrying in background.\n"
    }
}
