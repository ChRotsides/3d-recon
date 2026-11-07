package com.recon.videoscan.live

/** One in-flight frame and at most one pending frame. Slow optional inference
 * replaces pending work with the newest image instead of growing a Handler queue.
 * offer() returns true exactly when the caller must schedule the single consumer.
 */
class FrameMailbox<T> {
    private var pending: T? = null
    private var scheduled = false
    @Volatile var dropped = 0L
        private set

    @Synchronized fun offer(frame: T): Boolean {
        if (pending != null) dropped++
        pending = frame
        if (scheduled) return false
        scheduled = true
        return true
    }

    @Synchronized fun poll(): T? {
        val frame = pending
        pending = null
        if (frame == null) scheduled = false
        return frame
    }

    @Synchronized fun clear() { pending = null }
}
