package com.recon.videoscan.live

import org.junit.Assert.*
import org.junit.Test

class VioInputQueueTest {
    private fun gyro(t: Double) = VioInputQueue.Sample(t, 2 * t, -3 * t, 4 * t)
    private fun acc(t: Double) = VioInputQueue.Sample(t, 1.0, 2.0, 9.81)

    @Test fun `waits for future gyro and interpolates all axes without changing acceleration`() {
        val q = VioInputQueue<String>()
        q.addGyro(gyro(1.0)); q.addAcceleration(acc(1.005))
        assertTrue(q.drainImu().isEmpty())
        q.addGyro(gyro(1.010))
        val imu = q.drainImu().single()
        assertEquals(acc(1.005), imu.acceleration)
        assertEquals(1.005, imu.gyro.t, 1e-12)
        assertEquals(2.010, imu.gyro.x, 1e-12)
        assertEquals(-3.015, imu.gyro.y, 1e-12)
        assertEquals(4.020, imu.gyro.z, 1e-12)
        assertTrue(q.drainImu().isEmpty())
    }

    @Test fun `batched independent callbacks give the same interpolated stream`() {
        val batched = VioInputQueue<String>()
        val interleaved = VioInputQueue<String>()
        val actual = ArrayList<VioInputQueue.Imu>()
        for (i in 0..100) batched.addGyro(gyro(1.0 + i * 0.005))
        for (i in 0 until 100) batched.addAcceleration(acc(1.0025 + i * 0.005))
        for (i in 0..100) {
            interleaved.addGyro(gyro(1.0 + i * 0.005))
            if (i < 100) interleaved.addAcceleration(acc(1.0025 + i * 0.005))
            actual.addAll(interleaved.drainImu())
        }
        assertEquals(100, actual.size)
        assertEquals(batched.drainImu(), actual)
        actual.forEach { assertEquals(2 * it.acceleration.t, it.gyro.x, 1e-12) }
    }

    @Test fun `does not extrapolate before first or after last gyro`() {
        val q = VioInputQueue<String>()
        q.addAcceleration(acc(0.99)); q.addAcceleration(acc(1.01))
        q.addGyro(gyro(1.0)); q.addGyro(gyro(1.005))
        assertTrue(q.drainImu().isEmpty())
        q.addGyro(gyro(1.02))
        assertEquals(1.01, q.drainImu().single().gyro.t, 1e-12)
    }

    @Test fun `duplicate out of order and nonfinite samples cannot corrupt interpolation`() {
        val q = VioInputQueue<String>()
        q.addGyro(gyro(1.0)); q.addGyro(gyro(1.0)); q.addGyro(gyro(0.9))
        q.addGyro(gyro(Double.NaN))
        q.addGyro(VioInputQueue.Sample(1.2, Double.NaN, 0.0, 0.0))
        q.addGyro(gyro(1.02))
        q.addAcceleration(acc(1.01)); q.addAcceleration(acc(1.01)); q.addAcceleration(acc(1.005))
        q.addAcceleration(acc(Double.POSITIVE_INFINITY))
        assertEquals(2.02, q.drainImu().single().gyro.x, 1e-12)
    }

    @Test fun `camera waits for delivered imu including positive offset and lead`() {
        val q = VioInputQueue<String>()
        q.addFrame(10.0, "frame")
        assertNull(q.pollFrame(Double.NEGATIVE_INFINITY, 0.015))
        assertNull(q.pollFrame(10.010, 0.015))
        assertNull(q.pollFrame(10.015, 0.015))
        assertEquals("frame", q.pollFrame(10.018, 0.015))
        assertNull(q.pollFrame(11.0, 0.015))
    }

    @Test fun `negative offset is used with the correct sign`() {
        val q = VioInputQueue<String>()
        q.addFrame(10.0, "frame")
        assertNull(q.pollFrame(9.981, -0.020))
        assertEquals("frame", q.pollFrame(9.983, -0.020))
    }

    @Test fun `offset changes are reevaluated between camera updates`() {
        val q = VioInputQueue<String>()
        q.addFrame(10.0, "first"); q.addFrame(10.03, "second")
        assertEquals("first", q.pollFrame(10.04, 0.0))
        assertNull(q.pollFrame(10.04, 0.020))
        assertEquals("second", q.pollFrame(10.06, 0.020))
    }

    @Test fun `received gyro alone does not release camera before paired imu is delivered`() {
        val q = VioInputQueue<String>()
        q.addFrame(1.01, "image")
        q.addGyro(gyro(1.0)); q.addGyro(gyro(1.02))
        q.addAcceleration(acc(1.005))
        val firstDelivered = q.drainImu().last().acceleration.t
        assertNull(q.pollFrame(firstDelivered, 0.0))
        q.addAcceleration(acc(1.015))
        val nextDelivered = q.drainImu().last().acceleration.t
        assertEquals("image", q.pollFrame(nextDelivered, 0.0))
    }

    @Test fun `frame backlog stays bounded and preserves increasing timestamps`() {
        val q = VioInputQueue<Int>(maxFrames = 3)
        for (i in 1..100) q.addFrame(i.toDouble(), i)
        q.addFrame(99.0, -1); q.addFrame(100.0, -1)
        assertEquals(97L, q.droppedFrames)
        assertEquals(98, q.pollFrame(101.0, 0.0))
        assertEquals(99, q.pollFrame(101.0, 0.0))
        assertEquals(100, q.pollFrame(101.0, 0.0))
        assertNull(q.pollFrame(101.0, 0.0))
    }

    @Test fun `sensor backlog stays bounded without extrapolating discarded history`() {
        val q = VioInputQueue<String>(maxSamples = 3)
        for (i in 1..100) { q.addGyro(gyro(i * .005)); q.addAcceleration(acc(i * .005)) }
        assertEquals(listOf(98 * .005, 99 * .005, 100 * .005), q.drainImu().map { it.acceleration.t })
    }

    @Test fun `reset discards queued work and rejects delayed samples from old session`() {
        val q = VioInputQueue<String>()
        q.addGyro(gyro(1.0)); q.addAcceleration(acc(1.05)); q.addFrame(1.05, "old")
        q.reset(2.0)
        q.addGyro(gyro(1.1)); q.addAcceleration(acc(1.1)); q.addFrame(1.1, "late old")
        assertTrue(q.drainImu().isEmpty()); assertNull(q.pollFrame(3.0, 0.0))
        q.addGyro(gyro(2.1)); q.addGyro(gyro(2.12)); q.addAcceleration(acc(2.11))
        q.addFrame(2.105, "new")
        assertEquals("new", q.pollFrame(q.drainImu().single().acceleration.t, 0.0))
    }

    @Test fun `a missing gyro interval is never synthesized and requests recovery`() {
        val q = VioInputQueue<String>()
        q.addGyro(gyro(1.0)); q.addGyro(gyro(1.2)); q.addAcceleration(acc(1.1))
        assertTrue(q.drainImu().isEmpty())
        assertEquals("gyroscope sample gap", q.takeDiscontinuity())
        assertNull(q.takeDiscontinuity())
        q.reset(2.0)
        q.addGyro(gyro(2.1)); q.addGyro(gyro(2.11)); q.addAcceleration(acc(2.105))
        assertEquals(1, q.drainImu().size)
        assertNull(q.takeDiscontinuity())
    }

    @Test fun `accelerometer outage and queue overflow are reported`() {
        val q = VioInputQueue<String>(maxSamples = 3)
        q.addAcceleration(acc(1.0)); q.addAcceleration(acc(1.2))
        assertEquals("accelerometer sample gap", q.takeDiscontinuity())
        for (i in 0..4) q.addGyro(gyro(1.0 + i * .005))
        assertEquals("gyroscope queue overflow", q.takeDiscontinuity())
    }

    @Test fun `nonfinite timing never releases a frame`() {
        val q = VioInputQueue<String>()
        q.addFrame(Double.NaN, "invalid"); q.addFrame(1.0, "valid")
        assertNull(q.pollFrame(Double.NaN, 0.0))
        assertNull(q.pollFrame(2.0, Double.NaN))
        assertEquals("valid", q.pollFrame(2.0, 0.0))
    }
}
