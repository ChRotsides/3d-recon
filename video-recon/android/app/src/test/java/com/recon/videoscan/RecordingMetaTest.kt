package com.recon.videoscan

import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class RecordingMetaTest {
    private val camera = RecordingMeta.Camera(
        id = "0", sensorOrientationDeg = 90, timestampSource = "REALTIME",
        activeArray = RecordingMeta.Size(4000, 3000), preCorrectionActiveArray = RecordingMeta.Size(4032, 3024),
        intrinsicCalibration = floatArrayOf(3000f, 3000f, 2000f, 1500f, 0f),
        lensDistortionAndroid = floatArrayOf(0.01f, 0f, 0f, 0f, 0f),
        focalLengthMm = 6.3f, physicalSizeMm = Pair(9.8f, 7.3f), hardwareLevel = "LEVEL_3",
        focusMode = "fixed", focusDiopters = 0.4f, videoStabilization = "off", opticalStabilization = "off",
    )

    @Test
    fun `meta json uses the wire names the python loader reads`() {
        val text = RecordingMeta.build(
            mapOf("model" to "SM-S918B"), camera, RecordingMeta.Video(1920, 1080, 30, 24_000_000),
            startElapsedRealtimeNs = 123L, startWallClockMs = 456L, framesLogged = 900L, durationNs = 30_000_000_000L,
        )
        val json = JSONObject(text)
        val cam = json.getJSONObject("camera")
        assertEquals(90, cam.getInt("sensor_orientation_deg"))
        assertEquals("REALTIME", cam.getString("timestamp_source"))
        assertEquals(4000, cam.getJSONObject("active_array").getInt("width"))
        assertEquals(4032, cam.getJSONObject("pre_correction_active_array").getInt("width"))
        assertEquals(3000.0, cam.getJSONArray("intrinsic_calibration").getDouble(0), 1e-9)
        assertTrue(cam.has("lens_distortion_android"))
        assertTrue(!cam.has("distortion_opencv"))
        assertEquals(1920, json.getJSONObject("video").getInt("width"))
        assertEquals(900L, json.getLong("frames_logged"))
        assertEquals("SM-S918B", json.getJSONObject("device").getString("model"))
    }

    @Test
    fun `meta json before stop has null frame count`() {
        val text = RecordingMeta.build(emptyMap(), camera, RecordingMeta.Video(1280, 720, 30, 1), 1L, 2L)
        assertTrue(JSONObject(text).isNull("frames_logged"))
    }

    @Test
    fun `csv rows are plain and stable`() {
        assertEquals("frame_number,sensor_timestamp_ns,exposure_ns,focus_diopters,iso", FrameLog.HEADER)
        assertEquals("7,123456789,10000000,0.4,400", FrameLog.line(7, 123456789L, 10_000_000L, 0.4f, 400))
        assertEquals("7,123456789,,,", FrameLog.line(7, 123456789L, null, null, null))
        val sb = StringBuilder()
        assertEquals("5,rotv,0.1,0.2,0.3,0.9", ImuLog.line(5, "rotv", floatArrayOf(0.1f, 0.2f, 0.3f, 0.9f), sb).toString())
        assertEquals("5,acc,1.0,2.0,3.0,", ImuLog.line(5, "acc", floatArrayOf(1f, 2f, 3f), sb).toString())
    }
}
