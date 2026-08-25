package com.recon.poselogger

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class AlignmentGuideTest {
    private fun identityPose(): FloatArray = floatArrayOf(
        1f, 0f, 0f, 0f,
        0f, 1f, 0f, 0f,
        0f, 0f, 1f, 0f,
        0f, 0f, 0f, 1f,
    )

    private fun ghost(captureX: Float = 0.15f) = AlignmentGhost(
        positions = floatArrayOf(
            -0.4f, 0f, -2f,
            0f, 0f, -2f,
            0.4f, 0f, -2f,
        ),
        colours = ByteArray(9),
        edges = floatArrayOf(1f, 1f, 1f),
        capturePosition = floatArrayOf(captureX, 0f, 0f),
    )

    @Test
    fun `matching depth with overlap and baseline is ready`() {
        val result = AlignmentGuide.evaluate(
            ghost(), identityPose(), ShortArray(20 * 20) { 2000 },
            depthWidth = 20, depthHeight = 20,
            fx = 10f, fy = 10f, cx = 10f, cy = 10f,
        )

        assertEquals(1f, result.overlap, 1e-5f)
        assertEquals(1f, result.depthAgreement, 1e-5f)
        assertEquals(0.15f, result.baselineM, 1e-5f)
        assertTrue(result.ready)
        assertTrue(result.pointStates.all { it == 1f })
    }

    @Test
    fun `wrong live depth paints mismatches and refuses capture`() {
        val result = AlignmentGuide.evaluate(
            ghost(), identityPose(), ShortArray(20 * 20) { 1000 },
            depthWidth = 20, depthHeight = 20,
            fx = 10f, fy = 10f, cx = 10f, cy = 10f,
        )

        assertEquals(0f, result.depthAgreement, 1e-5f)
        assertFalse(result.ready)
        assertTrue(result.pointStates.all { it == 0f })
    }

    @Test
    fun `matching image without translation still asks for baseline`() {
        val result = AlignmentGuide.evaluate(
            ghost(captureX = 0f), identityPose(), ShortArray(20 * 20) { 2000 },
            depthWidth = 20, depthHeight = 20,
            fx = 10f, fy = 10f, cx = 10f, cy = 10f,
        )

        assertEquals(0f, result.baselineM, 1e-5f)
        assertFalse(result.ready)
    }

    @Test
    fun `clockwise depth rotation maps output back to sensor pixels`() {
        assertEquals(1 to 2, AlignmentGuide.sourcePixel(1, 1, 3, 4, 90))
        assertEquals(1 to 2, AlignmentGuide.sourcePixel(1, 1, 3, 4, 180))
        assertEquals(1 to 1, AlignmentGuide.sourcePixel(1, 1, 3, 4, 270))
    }
}
