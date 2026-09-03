package com.recon.poselogger.rgbscan

import org.junit.Assert.assertEquals
import org.junit.Test
import kotlin.math.cos
import kotlin.math.sin

class SensorDirectionMathTest {
    private val identity = floatArrayOf(1f, 0f, 0f, 0f, 1f, 0f, 0f, 0f, 1f)

    @Test fun unchangedSensorKeepsRgbDirection() {
        val result = SensorDirectionMath.estimate(identity, identity, 35.0, -12.0)
        assertEquals(35.0, result.first, 0.001)
        assertEquals(-12.0, result.second, 0.001)
    }

    @Test fun relativeYawAndPitchUpdateTheHudDirection() {
        val yaw = rotationY(Math.toRadians(30.0))
        val turned = SensorDirectionMath.estimate(identity, yaw, 0.0, 0.0)
        assertEquals(-30.0, turned.first, 0.001)

        val pitch = rotationX(Math.toRadians(20.0))
        val tilted = SensorDirectionMath.estimate(identity, pitch, 0.0, 0.0)
        assertEquals(20.0, tilted.second, 0.001)
    }

    private fun rotationX(angle: Double) = floatArrayOf(
        1f, 0f, 0f,
        0f, cos(angle).toFloat(), (-sin(angle)).toFloat(),
        0f, sin(angle).toFloat(), cos(angle).toFloat(),
    )

    private fun rotationY(angle: Double) = floatArrayOf(
        cos(angle).toFloat(), 0f, sin(angle).toFloat(),
        0f, 1f, 0f,
        (-sin(angle)).toFloat(), 0f, cos(angle).toFloat(),
    )
}
