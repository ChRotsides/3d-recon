package com.recon.videoscan

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.os.Handler
import android.os.HandlerThread
import java.io.BufferedWriter
import java.io.File
import java.io.FileWriter
import java.util.concurrent.atomic.AtomicLong

/**
 * Streams every inertial and magnetic sample to `imu.csv` on its own thread.
 *
 * Timestamps are `SensorEvent.timestamp`, which on any device that reports its
 * camera timestamps as REALTIME is the same monotonic clock the camera stamps
 * frames with — that is the whole basis for matching frames to orientation.
 */
class ImuLogger(context: Context, private val file: File) : SensorEventListener {
    private val manager = context.getSystemService(Context.SENSOR_SERVICE) as SensorManager
    private val thread = HandlerThread("imu-log").apply { start() }
    private val handler = Handler(thread.looper)
    private var writer: BufferedWriter? = null
    private val line = StringBuilder(96)
    private val counts = HashMap<String, AtomicLong>()
    private val names = HashMap<Int, String>()

    val available: List<String>
        get() = names.values.sorted()

    fun start() {
        file.parentFile?.mkdirs()
        writer = BufferedWriter(FileWriter(file), 1 shl 16).apply {
            write(ImuLog.HEADER)
            newLine()
        }
        for ((type, name) in SENSORS) {
            val sensor = manager.getDefaultSensor(type) ?: continue
            names[type] = name
            counts[name] = AtomicLong()
            manager.registerListener(this, sensor, SensorManager.SENSOR_DELAY_FASTEST, handler)
        }
    }

    fun stop() {
        manager.unregisterListener(this)
        // Flush on the logging thread so no sample lands after close().
        handler.post {
            writer?.flush()
            writer?.close()
            writer = null
        }
        thread.quitSafely()
        thread.join(2000)
    }

    fun sampleCounts(): Map<String, Long> = counts.mapValues { it.value.get() }

    override fun onSensorChanged(event: SensorEvent) {
        val name = names[event.sensor.type] ?: return
        val w = writer ?: return
        ImuLog.line(event.timestamp, name, event.values, line)
        w.append(line)
        w.newLine()
        counts[name]?.incrementAndGet()
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) = Unit

    companion object {
        val SENSORS = listOf(
            Sensor.TYPE_ACCELEROMETER to "acc",
            Sensor.TYPE_GYROSCOPE to "gyr",
            Sensor.TYPE_MAGNETIC_FIELD to "mag",
            Sensor.TYPE_ROTATION_VECTOR to "rotv",
            Sensor.TYPE_GAME_ROTATION_VECTOR to "gamerotv",
            Sensor.TYPE_GRAVITY to "grav",
            Sensor.TYPE_LINEAR_ACCELERATION to "linacc",
        )
    }
}
