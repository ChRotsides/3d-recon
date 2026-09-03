package com.recon.videoscan

import org.json.JSONArray
import org.json.JSONObject

/**
 * Everything the offline pipeline needs to know about how a recording was made.
 *
 * Plain values in, JSON text out, so the shape of `meta.json` is pinned by a JVM
 * unit test rather than discovered on the phone.
 */
object RecordingMeta {
    data class Size(val width: Int, val height: Int)

    data class Camera(
        val id: String,
        val sensorOrientationDeg: Int,
        val timestampSource: String,
        val activeArray: Size,
        val preCorrectionActiveArray: Size?,
        /** LENS_INTRINSIC_CALIBRATION `[fx, fy, cx, cy, s]` in pre-correction pixels, or null. */
        val intrinsicCalibration: FloatArray?,
        /** LENS_DISTORTION as Android reports it: a *correction*, not OpenCV's model. */
        val lensDistortionAndroid: FloatArray?,
        val focalLengthMm: Float,
        val physicalSizeMm: Pair<Float, Float>,
        val hardwareLevel: String,
        val focusMode: String,
        val focusDiopters: Float?,
        val videoStabilization: String,
        val opticalStabilization: String,
    )

    data class Video(val width: Int, val height: Int, val fps: Int, val bitRate: Int)

    fun build(
        device: Map<String, String>,
        camera: Camera,
        video: Video,
        startElapsedRealtimeNs: Long,
        startWallClockMs: Long,
        framesLogged: Long? = null,
        durationNs: Long? = null,
    ): String {
        val root = JSONObject()
        root.put("format", "videoscan-recording-v1")
        root.put("device", JSONObject(device))
        root.put(
            "camera",
            JSONObject().apply {
                put("id", camera.id)
                put("sensor_orientation_deg", camera.sensorOrientationDeg)
                put("timestamp_source", camera.timestampSource)
                put("active_array", size(camera.activeArray))
                put("pre_correction_active_array", camera.preCorrectionActiveArray?.let { size(it) } ?: JSONObject.NULL)
                put("intrinsic_calibration", camera.intrinsicCalibration?.let { floats(it) } ?: JSONObject.NULL)
                put("lens_distortion_android", camera.lensDistortionAndroid?.let { floats(it) } ?: JSONObject.NULL)
                put("focal_length_mm", camera.focalLengthMm.toDouble())
                put(
                    "physical_size_mm",
                    JSONObject().put("width", camera.physicalSizeMm.first.toDouble())
                        .put("height", camera.physicalSizeMm.second.toDouble()),
                )
                put("hardware_level", camera.hardwareLevel)
                put("focus_mode", camera.focusMode)
                put("focus_diopters", camera.focusDiopters?.toDouble() ?: JSONObject.NULL)
                put("video_stabilization", camera.videoStabilization)
                put("optical_stabilization", camera.opticalStabilization)
            },
        )
        root.put(
            "video",
            JSONObject().put("width", video.width).put("height", video.height)
                .put("fps", video.fps).put("bit_rate", video.bitRate).put("file", "video.mp4"),
        )
        root.put("start_elapsed_realtime_ns", startElapsedRealtimeNs)
        root.put("start_wall_clock_ms", startWallClockMs)
        root.put("frames_logged", framesLogged ?: JSONObject.NULL)
        root.put("duration_ns", durationNs ?: JSONObject.NULL)
        root.put("imu_file", "imu.csv")
        root.put("frames_file", "frames.csv")
        return root.toString(2)
    }

    private fun size(s: Size) = JSONObject().put("width", s.width).put("height", s.height)

    private fun floats(values: FloatArray): JSONArray {
        val arr = JSONArray()
        for (v in values) arr.put(v.toDouble())
        return arr
    }
}

/** One row of `frames.csv`: one camera frame the sensor actually exposed. */
object FrameLog {
    const val HEADER = "frame_number,sensor_timestamp_ns,exposure_ns,focus_diopters,iso"

    fun line(frameNumber: Long, sensorTimestampNs: Long, exposureNs: Long?, focusDiopters: Float?, iso: Int?): String =
        "$frameNumber,$sensorTimestampNs,${exposureNs ?: ""},${focusDiopters ?: ""},${iso ?: ""}"
}

/** One row of `imu.csv`. `w` is only present for the rotation-vector sensors. */
object ImuLog {
    const val HEADER = "timestamp_ns,sensor,x,y,z,w"

    fun line(timestampNs: Long, sensor: String, values: FloatArray, into: StringBuilder): StringBuilder {
        into.setLength(0)
        into.append(timestampNs).append(',').append(sensor)
        for (i in 0 until 3) into.append(',').append(if (i < values.size) values[i] else 0f)
        into.append(',')
        if (values.size > 3) into.append(values[3])
        return into
    }
}
