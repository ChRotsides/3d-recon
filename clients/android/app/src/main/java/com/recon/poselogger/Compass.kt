package com.recon.poselogger

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.view.Surface

/**
 * Device heading, degrees clockwise from magnetic north.
 *
 * Exists for one job: telling a user which way to turn after tracking is lost.
 * That is the one moment when ARCore's pose cannot be used — there isn't a
 * trustworthy one — and it is also the moment when a relocalisation may have
 * moved the world origin out from under any bearing remembered in world
 * coordinates. A compass is measured against the earth, so it survives both.
 *
 * It is a poor *absolute* reference indoors: steel frames, wiring and
 * appliances bend the field by tens of degrees. It is a serviceable *relative*
 * one, because two readings taken minutes apart in the same room share most of
 * that distortion, and the difference cancels it. Nothing here is fed to the
 * reconstruction; the core uses it only to phrase an instruction.
 *
 * [Sensor.TYPE_ROTATION_VECTOR] rather than raw magnetometer readings: it is
 * already fused with the gyroscope and accelerometer, so it is gravity-aligned
 * and far steadier than a bare compass, which on a hand-held phone swings
 * enough to be unreadable.
 */
class Compass(context: Context) : SensorEventListener {

    private val sensors = context.getSystemService(SensorManager::class.java)
    private val rotationVector: Sensor? =
        sensors?.getDefaultSensor(Sensor.TYPE_ROTATION_VECTOR)

    private val matrix = FloatArray(9)
    private val remapped = FloatArray(9)
    private val orientation = FloatArray(3)

    /** Latest heading in degrees [0, 360), or null before the first reading. */
    @Volatile
    var headingDeg: Float? = null
        private set

    /** False when the device has no rotation-vector sensor at all. */
    val available: Boolean get() = rotationVector != null

    /**
     * Display rotation, so a phone held in landscape does not report a heading
     * 90 degrees out. Set from the Activity, which is what knows.
     */
    @Volatile
    var displayRotation: Int = Surface.ROTATION_0

    fun start() {
        val sensor = rotationVector ?: return
        // Game-rate is far more than a human turn needs, and the reading is
        // only ever consumed once per frame; UI rate keeps the radio quiet.
        sensors?.registerListener(this, sensor, SensorManager.SENSOR_DELAY_UI)
    }

    fun stop() {
        sensors?.unregisterListener(this)
    }

    override fun onSensorChanged(event: SensorEvent) {
        if (event.sensor.type != Sensor.TYPE_ROTATION_VECTOR) return
        SensorManager.getRotationMatrixFromVector(matrix, event.values)

        // The camera looks along the device's -Z, so the axis that means
        // "where am I pointing" is the screen's up axis once the matrix is
        // remapped for the current display orientation.
        val (axisX, axisY) = when (displayRotation) {
            Surface.ROTATION_90 -> SensorManager.AXIS_Y to SensorManager.AXIS_MINUS_X
            Surface.ROTATION_180 -> SensorManager.AXIS_MINUS_X to SensorManager.AXIS_MINUS_Y
            Surface.ROTATION_270 -> SensorManager.AXIS_MINUS_Y to SensorManager.AXIS_X
            else -> SensorManager.AXIS_X to SensorManager.AXIS_Y
        }
        SensorManager.remapCoordinateSystem(matrix, axisX, axisY, remapped)
        SensorManager.getOrientation(remapped, orientation)

        // getOrientation returns azimuth about the screen normal; the camera
        // points opposite the screen, so add half a turn to get the direction
        // the *camera* faces, which is what the user is aiming.
        val azimuth = Math.toDegrees(orientation[0].toDouble()).toFloat() + 180f
        headingDeg = ((azimuth % 360f) + 360f) % 360f
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) = Unit
}
