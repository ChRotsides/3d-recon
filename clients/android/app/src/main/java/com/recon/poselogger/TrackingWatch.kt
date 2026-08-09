package com.recon.poselogger

import uniffi.recon_core.FfiTracking

/**
 * Deciding when tracking has been lost, and when the world moved under us.
 *
 * Both matter more than they look. ARCore relocalises by **moving the world
 * origin**, so poses either side of a relocalisation are not in a common frame.
 * Fusing across one lays the same room down twice, offset — a defect no depth
 * model can repair, because nothing about it is a depth error. The first real
 * capture with this app relocalised four times (jumps of 5.6, 5.3, 3.7 and
 * 2.5 m against a median inter-keyframe motion of 0.12 m) and produced exactly
 * that, while the UI showed "tracking" throughout: the client was passing
 * `NORMAL` to the core unconditionally and dropping every non-tracking frame
 * before the core could see it, so `TRACKING_LOST` was unreachable.
 *
 * Pure functions, so [TrackingWatchTest] pins them on a plain JVM rather than
 * only when someone walks a real room and loses tracking on purpose.
 */
object TrackingWatch {

    /**
     * ARCore's state in the core's vocabulary.
     *
     * Booleans rather than ARCore enums so this is testable without a device.
     * `badState` is ARCore's `BAD_STATE`/`CAMERA_UNAVAILABLE`, which is the
     * session recovering from an interruption — that is a relocalisation, and
     * the world may move. The other failure reasons (dim light, fast motion,
     * blank walls) degrade tracking without necessarily rebasing the world.
     */
    fun classify(tracking: Boolean, badState: Boolean): FfiTracking = when {
        tracking -> FfiTracking.NORMAL
        badState -> FfiTracking.RELOCALIZING
        else -> FfiTracking.LIMITED
    }

    /**
     * Whether two consecutive keyframes are too far apart to be one walk.
     *
     * A relocalisation is not always announced — ARCore can rebase between two
     * frames that both report tracking — so the pose is checked as well as the
     * state. Speed rather than distance alone: keyframes can be seconds apart,
     * and 2 m over four seconds is someone walking, while 2 m in a fifth of a
     * second is the world moving.
     */
    fun isTeleport(distanceM: Float, dtSeconds: Double): Boolean {
        if (distanceM < MIN_BREAK_M) return false
        if (dtSeconds <= 0.0) return true // no time passed; it cannot be motion
        return distanceM / dtSeconds > MAX_WALK_SPEED_M_S
    }

    /** Below this, a jump is ordinary capture movement whatever the speed. */
    const val MIN_BREAK_M = 1.5f

    /** Nobody carrying a phone through a room covers ground faster than this. */
    const val MAX_WALK_SPEED_M_S = 3.0
}
