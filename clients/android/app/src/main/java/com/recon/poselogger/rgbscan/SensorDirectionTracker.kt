package com.recon.poselogger.rgbscan

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import kotlin.math.asin
import kotlin.math.cos
import kotlin.math.sin

/**
 * Interpolates an RGB-localized direction with relative IMU motion for the HUD only.
 * No sensor sample leaves the phone or enters reconstruction geometry.
 */
internal class SensorDirectionTracker(
    context: Context,
    private val onDirection: (yawDegrees: Double, pitchDegrees: Double) -> Unit,
) : SensorEventListener {
    private val manager = context.getSystemService(SensorManager::class.java)
    private val sensor = manager.getDefaultSensor(Sensor.TYPE_GAME_ROTATION_VECTOR)
        ?: manager.getDefaultSensor(Sensor.TYPE_ROTATION_VECTOR)
    private val lock = Any()
    private var current: FloatArray? = null
    private var anchorSensor: FloatArray? = null
    private var anchorYaw = 0.0
    private var anchorPitch = 0.0
    private var lastDispatchNs = 0L
    private var smoothYaw: Double? = null
    private var smoothPitch: Double? = null

    fun start() {
        sensor?.let { manager.registerListener(this, it, SensorManager.SENSOR_DELAY_GAME) }
    }

    fun stop() = manager.unregisterListener(this)

    fun captureOrientation(): FloatArray? = synchronized(lock) { current?.copyOf() }

    fun anchor(sensorAtCapture: FloatArray?, yawDegrees: Double, pitchDegrees: Double) {
        if (sensorAtCapture == null || !yawDegrees.isFinite() || !pitchDegrees.isFinite()) return
        synchronized(lock) {
            anchorSensor = sensorAtCapture.copyOf()
            anchorYaw = yawDegrees
            anchorPitch = pitchDegrees
            smoothYaw = yawDegrees
            smoothPitch = pitchDegrees
        }
    }

    override fun onSensorChanged(event: SensorEvent) {
        val matrix = FloatArray(9)
        SensorManager.getRotationMatrixFromVector(matrix, event.values)
        val estimate = synchronized(lock) {
            current = matrix
            val anchor = anchorSensor ?: return@synchronized null
            if (event.timestamp - lastDispatchNs < DISPATCH_INTERVAL_NS) return@synchronized null
            lastDispatchNs = event.timestamp
            val raw = SensorDirectionMath.estimate(anchor, matrix, anchorYaw, anchorPitch)
            val previousYaw = smoothYaw ?: raw.first
            val previousPitch = smoothPitch ?: raw.second
            val yaw = previousYaw + Smoothing * SphericalGuidanceView.wrapDegrees(raw.first - previousYaw)
            val pitch = previousPitch + Smoothing * (raw.second - previousPitch)
            smoothYaw = SphericalGuidanceView.wrapDegrees(yaw)
            smoothPitch = pitch.coerceIn(-89.0, 89.0)
            smoothYaw!! to smoothPitch!!
        } ?: return
        onDirection(estimate.first, estimate.second)
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) = Unit

    companion object {
        private const val DISPATCH_INTERVAL_NS = 33_000_000L
        private const val Smoothing = 0.28
    }
}

internal object SensorDirectionMath {
    fun estimate(
        anchorDeviceToWorld: FloatArray,
        currentDeviceToWorld: FloatArray,
        anchorYawDegrees: Double,
        anchorPitchDegrees: Double,
    ): Pair<Double, Double> {
        require(anchorDeviceToWorld.size == 9 && currentDeviceToWorld.size == 9)
        val relative = multiply(transpose(anchorDeviceToWorld), currentDeviceToWorld)
        val roomFromAnchorDevice = roomFromDevice(anchorYawDegrees, anchorPitchDegrees)
        val roomFromCurrentDevice = multiply(roomFromAnchorDevice, relative)
        // The back camera looks through the negative device-Z axis.
        val forwardX = -roomFromCurrentDevice[2].toDouble()
        val forwardY = -roomFromCurrentDevice[5].toDouble()
        val forwardZ = -roomFromCurrentDevice[8].toDouble()
        val yaw = Math.toDegrees(kotlin.math.atan2(forwardX, forwardZ))
        val pitch = Math.toDegrees(asin(forwardY.coerceIn(-1.0, 1.0)))
        return SphericalGuidanceView.wrapDegrees(yaw) to pitch
    }

    private fun roomFromDevice(yawDegrees: Double, pitchDegrees: Double): FloatArray {
        val yaw = Math.toRadians(yawDegrees)
        val pitch = Math.toRadians(pitchDegrees)
        val forwardX = cos(pitch) * sin(yaw)
        val forwardY = sin(pitch)
        val forwardZ = cos(pitch) * cos(yaw)
        val rightLength = kotlin.math.hypot(forwardX, forwardZ).coerceAtLeast(1e-9)
        val rightX = forwardZ / rightLength
        val rightZ = -forwardX / rightLength
        val upX = -forwardY * rightZ
        val upY = forwardZ * rightX - forwardX * rightZ
        val upZ = forwardY * rightX
        return floatArrayOf(
            rightX.toFloat(), upX.toFloat(), (-forwardX).toFloat(),
            0f, upY.toFloat(), (-forwardY).toFloat(),
            rightZ.toFloat(), upZ.toFloat(), (-forwardZ).toFloat(),
        )
    }

    private fun transpose(matrix: FloatArray) = floatArrayOf(
        matrix[0], matrix[3], matrix[6],
        matrix[1], matrix[4], matrix[7],
        matrix[2], matrix[5], matrix[8],
    )

    private fun multiply(first: FloatArray, second: FloatArray): FloatArray {
        val result = FloatArray(9)
        for (row in 0 until 3) for (column in 0 until 3) {
            result[row * 3 + column] = (0 until 3).sumOf {
                (first[row * 3 + it] * second[it * 3 + column]).toDouble()
            }.toFloat()
        }
        return result
    }
}
