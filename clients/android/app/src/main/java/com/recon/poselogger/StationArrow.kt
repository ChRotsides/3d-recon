package com.recon.poselogger

import kotlin.math.abs
import kotlin.math.atan2

/**
 * Which way to point the on-screen arrow at a next-best station that is not
 * visible.
 *
 * Split out of the Activity because it is the one piece of the overlay with a
 * trap in it, and the trap is silent: **behind the camera the perspective divide
 * mirrors the point**, so the obvious `clip.xy / clip.w` sends the user in
 * exactly the wrong direction. An arrow that is merely absent is a missing
 * feature; an arrow confidently pointing away from where they should walk is
 * worse than nothing, and on a device it looks like a plausible suggestion.
 * [StationArrowTest] pins it without a phone.
 */
object StationArrow {

    /**
     * @param clipX,clipY,clipW the station in clip space, i.e. projection *
     *        view * position
     * @return the bearing from the screen centre in radians, 0 pointing right
     *         and increasing downward (screen convention, not clip), or null
     *         when the station is already on screen and needs no arrow
     */
    fun bearing(clipX: Float, clipY: Float, clipW: Float): Float? {
        if (abs(clipW) < 1e-6f) return null // on the eye plane; no meaningful direction
        val behind = clipW <= 0f
        var ndcX = clipX / clipW
        var ndcY = clipY / clipW
        if (behind) {
            ndcX = -ndcX
            ndcY = -ndcY
        }
        // Visible: the ring itself is the guidance, and an arrow to something in
        // plain sight is clutter.
        if (!behind && abs(ndcX) <= 1f && abs(ndcY) <= 1f) return null
        // Clip y grows upward, screen y downward.
        return atan2(-ndcY, ndcX)
    }
}
