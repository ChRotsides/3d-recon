package com.recon.poselogger

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import uniffi.recon_core.FfiTracking

/**
 * Tracking-loss detection, pinned on a plain JVM.
 *
 * These exist because of a real capture: 150 keyframes across four ARCore
 * relocalisations, fused into a room laid down five times over. The client
 * reported `NORMAL` for every frame and dropped the rest before the core saw
 * them, so the one nag whose whole purpose is this — `TRACKING_LOST`, priority
 * 0, "nothing else can be fixed while it persists" — could never fire.
 *
 * The failure is silent by construction, so it has to be tested somewhere that
 * does not require losing tracking on a real phone on purpose.
 */
class TrackingWatchTest {

    @Test
    fun a_tracking_frame_is_normal() {
        assertEquals(FfiTracking.NORMAL, TrackingWatch.classify(tracking = true, badState = false))
        // Whatever ARCore says about a failure reason, tracking wins.
        assertEquals(FfiTracking.NORMAL, TrackingWatch.classify(tracking = true, badState = true))
    }

    @Test
    fun a_bad_state_is_relocalizing_not_merely_limited() {
        // This is the case that moves the world origin, so it must be
        // distinguishable from dim light or a blank wall.
        assertEquals(
            FfiTracking.RELOCALIZING,
            TrackingWatch.classify(tracking = false, badState = true),
        )
    }

    @Test
    fun other_failures_are_limited() {
        assertEquals(
            FfiTracking.LIMITED,
            TrackingWatch.classify(tracking = false, badState = false),
        )
    }

    @Test
    fun ordinary_capture_movement_is_not_a_teleport() {
        // The real capture's median inter-keyframe motion was 0.12 m and its
        // p95 was 0.45 m; none of that may register as a break.
        assertFalse(TrackingWatch.isTeleport(0.12f, 0.5))
        assertFalse(TrackingWatch.isTeleport(0.45f, 0.8))
        // Crossing a room over several seconds is a walk, not a rebase.
        assertFalse(TrackingWatch.isTeleport(2.5f, 4.0))
    }

    @Test
    fun the_captures_real_jumps_are_teleports() {
        // The four breaks actually observed, at plausible keyframe spacing.
        for (d in listOf(5.56f, 5.26f, 3.72f, 2.52f)) {
            assertTrue("$d m should read as a break", TrackingWatch.isTeleport(d, 0.5))
        }
    }

    @Test
    fun a_jump_with_no_elapsed_time_is_always_a_teleport() {
        // Two keyframes at the same timestamp cannot be separated by motion, so
        // the speed test would divide by zero rather than decide.
        assertTrue(TrackingWatch.isTeleport(2.0f, 0.0))
        // ...but a small offset at the same instant is still not a break, since
        // below the distance floor nothing counts.
        assertFalse(TrackingWatch.isTeleport(0.3f, 0.0))
    }

    @Test
    fun slow_drift_over_a_long_gap_is_not_a_break() {
        // A user standing still for 20 s while ARCore wanders 2 m is bad
        // tracking, but it is not the discontinuity this detects, and calling
        // it one would split a capture that could still be fused.
        assertFalse(TrackingWatch.isTeleport(2.0f, 20.0))
    }
}
