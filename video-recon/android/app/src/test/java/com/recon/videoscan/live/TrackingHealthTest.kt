package com.recon.videoscan.live

import org.junit.Assert.*
import org.junit.Test

/** Tracking quality must reflect constraints, not just a finite EKF prediction. */
class TrackingHealthTest {
    private fun pose(x: Float = 0f) = floatArrayOf(x, 0f, 0f, 0f, 0f, 0f, 1f)
    private fun d(t: Double, tracks: Int = 100, attempted: Int = 10, accepted: Int = 8,
        stationary: Boolean = false, speed: Double = 0.0, sigma: Double = .05) =
        VioDiagnostics(t, t, tracks, attempted, accepted, 0, 0, stationary, speed, sigma)
    private fun healthy(h: TrackingHealth, start: Double = 1.0) {
        for (i in 0..2) { val t = start + i * .05; h.update(t, pose(), d(t)) }
        assertEquals(TrackingHealth.State.TRACKING, h.current(start + .1).state)
    }

    @Test fun `first pose is acquiring until three consecutive frames`() {
        val h = TrackingHealth()
        assertEquals(TrackingHealth.State.INITIALIZING, h.update(1.0, null, d(1.0)).state)
        assertEquals(TrackingHealth.State.ACQUIRING, h.update(1.1, pose(), d(1.1)).state)
        assertNull(h.permit(1.1))
    }
    @Test fun `healthy tracking requires consecutive supported frames`() {
        val h = TrackingHealth()
        assertFalse(h.update(1.0, pose(), d(1.0)).fusionAllowed)
        assertFalse(h.update(1.05, pose(), d(1.05)).fusionAllowed)
        assertTrue(h.update(1.1, pose(), d(1.1)).fusionAllowed)
        var wrote = false
        assertTrue(h.commit(h.permit(1.1)!!, 1.15) { wrote = true })
        assertTrue(wrote)
    }
    @Test fun `first visual correction before acquisition does not restart the estimator`() {
        val h = TrackingHealth()
        assertFalse(h.update(1.0, pose(), d(1.0, accepted = 0)).fusionAllowed)
        assertEquals(TrackingHealth.State.ACQUIRING, h.update(1.05, pose(.65f), d(1.05)).state)
        assertNull(h.permit(1.05))
        assertFalse(h.update(1.1, pose(.66f), d(1.1)).fusionAllowed)
        assertTrue(h.update(1.15, pose(.67f), d(1.15)).fusionAllowed)
        assertEquals(TrackingHealth.State.RESTARTING, h.update(1.2, pose(2f), d(1.2)).state)
    }
    @Test fun `stationary frames keep fusion running`() {
        val h = TrackingHealth()
        for (i in 0..2) { val t = 1.0 + .05 * i; h.update(t, pose(), d(t, attempted = 0, accepted = 0, stationary = true, speed = 0.02)) }
        val s = h.current(1.1)
        assertEquals(TrackingHealth.State.STATIONARY, s.state)
        assertTrue(s.poseReliable); assertTrue(s.fusionAllowed)
        assertNotNull(h.permit(1.1))
    }
    @Test fun `IMU-only drift loses fusion permission even with many image tracks`() {
        val h = TrackingHealth(); healthy(h)
        val old = h.permit(1.1)!!
        assertTrue(h.update(1.2, pose(), d(1.2, accepted = 0)).fusionAllowed)
        assertTrue(h.update(1.7, pose(), d(1.7, attempted = 0, accepted = 0)).fusionAllowed)
        assertFalse(h.update(2.2, pose(), d(2.2, attempted = 40, accepted = 0)).fusionAllowed)
        assertFalse(h.commit(old, 2.2) { fail("unconstrained depth fused") })
        val few = TrackingHealth(); healthy(few)
        assertFalse(few.update(1.2, pose(), d(1.2, tracks = 3)).fusionAllowed)
    }
    @Test fun `sustained loss latches a restart until estimator reset`() {
        val h = TrackingHealth(); healthy(h)
        for (i in 0..120) {
            val t = 1.2 + i * .05
            assertFalse(h.update(t, pose(), d(t, sigma = 5.0)).fusionAllowed)
            assertNull(h.permit(t))
        }
        assertEquals(TrackingHealth.State.RESTARTING, h.update(7.5, pose(), d(7.5)).state)
        h.reset(); healthy(h, 7.6)
    }
    @Test fun `old depth job stays invalid after a short loss and recovery`() {
        val h = TrackingHealth(); healthy(h)
        val permit = h.permit(1.1)!!
        h.update(1.2, pose(), d(1.2, sigma = 5.0))
        healthy(h, 1.3)
        assertFalse(h.commit(permit, 1.45) { fail("stale job fused") })
        assertTrue(h.commit(h.permit(1.3 + 2 * .05)!!, 1.45) {})
    }
    @Test fun `camera stall invalidates in-flight fusion even without another frame`() {
        val h = TrackingHealth(); healthy(h)
        val permit = h.permit(1.1)!!
        assertFalse(h.commit(permit, 2.0) { fail("stalled camera fused") })
        assertEquals(TrackingHealth.State.RECOVERING, h.current(2.0).state)
        assertEquals(TrackingHealth.State.RESTARTING, h.current(5.1).state)
        assertEquals(TrackingHealth.State.RESTARTING, h.update(6.0, pose(), d(6.0)).state)
        h.reset(); healthy(h, 6.05)
    }
    @Test fun `reset invalidates queued depth jobs`() {
        val h = TrackingHealth(); healthy(h)
        val permit = h.permit(1.1)!!
        h.reset(); healthy(h, 1.2)
        assertFalse(h.commit(permit, 1.4) {})
    }
    @Test fun `uncertainty implausible speed nonfinite poses and discontinuities are rejected`() {
        for (test in 0..4) {
            val h = TrackingHealth(); healthy(h)
            val p = when (test) { 2 -> pose(Float.NaN); 3 -> pose(2f); else -> pose() }
            val diag = when (test) { 0 -> d(1.2, sigma = 4.0); 1 -> d(1.2, speed = 5.0); 4 -> d(1.2).copy(stateTime = 1.1); else -> d(1.2) }
            assertFalse(h.update(1.2, p, diag).poseReliable)
        }
    }
    @Test fun `metre scale uncertainty cannot contaminate a room map`() {
        val h = TrackingHealth(); healthy(h)
        assertFalse(h.update(1.2, pose(), d(1.2, sigma = 1.0)).fusionAllowed)
    }
    @Test fun `native counts cannot silently claim more acceptances than attempts`() {
        val v = doubleArrayOf(1.0, 1.0, 100.0, 10.0, 8.0, 5.0, 3.0, 0.0, .1, .05)
        assertEquals(4, VioDiagnostics.fromNative(v)!!.notAccepted)
        v[4] = 11.0
        assertNull(VioDiagnostics.fromNative(v))
    }
    @Test fun `one second interruption recovers without restarting or reusing stale depth`() {
        val h = TrackingHealth(); healthy(h)
        val old = h.permit(1.1)!!
        for (i in 0..24) { val t = 1.2 + i * .05; assertFalse(h.update(t, pose(), d(t, sigma = 5.0)).fusionAllowed) }
        healthy(h, 2.5)
        assertFalse(h.commit(old, 2.6) { fail("old segment depth reused") })
    }

    @Test fun `a pose jump cannot silently resume fusion in the old map frame`() {
        val h = TrackingHealth(); healthy(h)
        assertEquals(TrackingHealth.State.RESTARTING, h.update(1.2, pose(2f), d(1.2)).state)
        for (i in 1..10) {
            val t = 1.2 + i * .05
            assertFalse(h.update(t, pose(2f), d(t)).poseReliable)
        }
    }

    @Test fun `stationary flag with drifting velocity is not a position constraint`() {
        val h = TrackingHealth()
        for (i in 0..80) {
            val t = 1.0 + i * .05
            assertFalse(h.update(t, pose(), d(t, accepted = 0, stationary = true, speed = .5)).fusionAllowed)
        }
        assertEquals(TrackingHealth.State.RESTARTING, h.current(5.0).state)
    }

    @Test fun `nonfinite diagnostic time and invalid quaternion are rejected`() {
        val h = TrackingHealth(); healthy(h)
        assertFalse(h.update(1.2, pose(), d(1.2).copy(stateTime = Double.NaN)).fusionAllowed)
        h.reset(); healthy(h)
        assertFalse(h.update(1.2, FloatArray(7), d(1.2)).fusionAllowed)
    }

    @Test fun `sensor discontinuity immediately invalidates an in-flight depth job`() {
        val h = TrackingHealth(); healthy(h)
        val permit = h.permit(1.1)!!
        assertEquals(TrackingHealth.State.RESTARTING, h.inputDiscontinuity("IMU gap").state)
        assertFalse(h.commit(permit, 1.15) { fail("sensor gap fused") })
    }

    @Test fun `slow depth can commit its captured pose only through continuously healthy tracking`() {
        val h = TrackingHealth(); healthy(h); val permit = h.permit(1.1)!!
        for (i in 0..21) { val t=1.2+i*.1; h.update(t,pose(),d(t)) }
        assertFalse(h.commit(permit,3.3) { fail("default queue budget bypassed") })
        assertTrue(h.commit(permit,3.3,2.75) {})
        assertFalse(h.commit(permit,3.3,10.0) { fail("unbounded latency accepted") })
        for (i in 1..11) { val t=3.3+i*.1; h.update(t,pose(),d(t)) }
        assertFalse(h.commit(permit,4.4,3.0) { fail("old result fused") })
    }
    @Test fun `inference budget never permits a tracking gap or a recovered epoch`() {
        val h = TrackingHealth(); healthy(h); val permit = h.permit(1.1)!!
        assertFalse(h.commit(permit,2.0,3.0) { fail("camera stall ignored") })
        h.update(2.05,pose(),d(2.05)) // first returning image still exposes the camera gap
        healthy(h,2.1)
        assertFalse(h.commit(permit,2.3,3.0) { fail("recovered epoch reused an old pose") })
        h.reset(); healthy(h,2.4)
        assertFalse(h.commit(permit,2.6,3.0) { fail("reset epoch reused an old pose") })
    }
}
