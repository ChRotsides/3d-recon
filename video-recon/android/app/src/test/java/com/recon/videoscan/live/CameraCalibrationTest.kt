package com.recon.videoscan.live

import org.junit.Assert.*
import org.junit.Test

class CameraCalibrationTest {
    @Test fun `unsupported 960x540 prefers native width capture without tracking resampling`() {
        val sizes = listOf(CaptureSize(960, 720), CaptureSize(1920, 1080), CaptureSize(640, 360), CaptureSize(1280, 720))
        assertEquals(CaptureSize(960, 720), selectCaptureSize(sizes, CaptureSize(960, 540)))
        assertEquals(CaptureSize(1280, 720), selectCaptureSize(sizes.filter { it.width != 960 }, CaptureSize(960, 540)))
        assertEquals(CaptureSize(960, 540), selectCaptureSize(sizes + CaptureSize(960, 540), CaptureSize(960, 540)))
        assertNull(selectCaptureSize(listOf(CaptureSize(800, 600)), CaptureSize(960, 540)))
    }

    @Test fun `native width crop matches calibrated central field of view`() {
        val crop = trackingCrop(CaptureSize(960, 720), CaptureSize(960, 540))!!
        assertEquals(ImageCrop(0, 90, 960, 540), crop)
        // Factory principal point in full 4:3 capture moves up by the crop's 90 rows.
        val fullCy = 358.5
        assertEquals(268.5, fullCy - crop.top, 0.0)
        val v = 639.3 * .2 + fullCy
        assertEquals((v - fullCy) / 639.3, (v - crop.top - 268.5) / 639.3, 1e-12)
        assertEquals(ImageCrop(0, 0, 1280, 720), trackingCrop(CaptureSize(1280, 720), CaptureSize(960, 540)))
        assertNull(trackingCrop(CaptureSize(800, 600), CaptureSize(960, 540)))
    }

    @Test fun `camera downsampling samples the centres of the same aspect image`() {
        assertEquals(0, pixelCenterIndex(0, 1280, 960))
        assertEquals(1279, pixelCenterIndex(959, 1280, 960))
        assertEquals(719, pixelCenterIndex(539, 720, 540))
        assertEquals(640, pixelCenterIndex(480, 1280, 960))
        assertEquals(360, pixelCenterIndex(270, 720, 540))
    }

    private val camera = PinholeCamera(9, 7, 4.0, 5.0, 4.0, 3.0)
    private fun calibration(model: CameraCalibration.Model = CameraCalibration.Model.RADTAN,
        d0: Double = 0.0, d1: Double = 0.0, d2: Double = 0.0, d3: Double = 0.0) =
        CameraCalibration(camera, model, d0, d1, d2, d3)
    // A linear colour ramp makes the sampled source coordinate independently observable.
    private fun image(w: Int = 9, h: Int = 7) = ByteArray(w * h * 4).also { pixels ->
        for (y in 0 until h) for (x in 0 until w) {
            val i = 4 * (y * w + x)
            pixels[i] = (20 * x).toByte(); pixels[i + 1] = (30 * y).toByte()
            pixels[i + 2] = 77; pixels[i + 3] = -1
        }
    }

    @Test fun `zero radtan distortion preserves all pixels including borders`() {
        val source = image()
        val result = calibration().rectify(source, 9, 7, 9, 7)
        assertArrayEquals(source, result.rgba)
        assertEquals(camera, result.camera)
    }

    @Test fun `radtan remapping agrees with independent OpenCV projection`() {
        val result = calibration(d0 = 0.1, d1 = 0.01, d2 = 0.02, d3 = -0.03).rectify(image(), 9, 7, 9, 7)
        val i = 4 * (5 * 9 + 7)
        // cv2.projectPoints([.75,.4,1], K, [.1,.01,.02,-.03]) = [7.05871019,5.16919013].
        assertEquals(141, result.rgba[i].toInt() and 255)
        assertEquals(155, result.rgba[i + 1].toInt() and 255)
        assertEquals(255, result.rgba[i + 3].toInt() and 255)
    }

    @Test fun `equidistant model agrees with independent OpenCV projection`() {
        val result = calibration(CameraCalibration.Model.EQUIDISTANT, 0.1, 0.01, 0.02, -0.03).rectify(image(), 9, 7, 9, 7)
        val i = 4 * (5 * 9 + 7)
        // cv2.fisheye.projectPoints gives [6.61753338,4.74502225].
        assertEquals(132, result.rgba[i].toInt() and 255)
        assertEquals(142, result.rgba[i + 1].toInt() and 255)
        assertEquals(77, result.rgba[4 * (3 * 9 + 4) + 2].toInt() and 255) // optical axis stays finite
    }

    @Test fun `anisotropic resize scales each focal axis and pixel centre independently`() {
        val raw = PinholeCamera(960, 540, 639.2, 639.3, 481.0, 268.5)
        val resized = raw.resized(320, 180)
        assertEquals(639.2 / 3, resized.fx, 1e-10)
        assertEquals(639.3 / 3, resized.fy, 1e-10)
        assertEquals(160.0, resized.cx, 1e-10)
        assertEquals(89.16666666666667, resized.cy, 1e-10)
        val stretched = raw.resized(480, 180)
        assertEquals(319.6, stretched.fx, 1e-10)
        assertEquals(213.1, stretched.fy, 1e-10)
    }

    @Test fun `different model input and output dimensions preserve camera rays`() {
        val raw = PinholeCamera(960, 540, 639.2, 639.3, 481.0, 268.5)
        val throughModel = raw.resized(322, 182).resized(320, 180)
        val direct = raw.resized(320, 180)
        assertArrayEquals(direct.intrinsics(), throughModel.intrinsics(), 1e-5f)
        // Ray (.2,-.1,1) projects to the corresponding full-frame and output-grid pixels.
        val uRaw = raw.fx * 0.2 + raw.cx; val vRaw = raw.fy * -0.1 + raw.cy
        assertEquals((uRaw + 0.5) / 3 - 0.5, throughModel.fx * 0.2 + throughModel.cx, 1e-10)
        assertEquals((vRaw + 0.5) / 3 - 0.5, throughModel.fy * -0.1 + throughModel.cy, 1e-10)
    }

    @Test fun `resampling colours follows the same half pixel convention as calibration`() {
        val resized = calibration().rectify(image(), 9, 7, 3, 7)
        // Target column 1 samples source column 4, column 0 samples column 1.
        assertEquals(80, resized.rgba[4].toInt() and 255)
        assertEquals(20, resized.rgba[0].toInt() and 255)
        assertEquals(1.0, resized.camera.cx, 1e-12)
    }

    @Test fun `unsupported borders cannot become inferred depth surfaces`() {
        val result = calibration(d0 = 10.0).rectify(image(), 9, 7, 9, 7)
        val predictions = FloatArray(9 * 7) { 2f }
        result.maskPrediction(predictions)
        assertTrue(predictions[0].isNaN())
        assertEquals(2f, predictions[3 * 9 + 4], 0f)
        val smaller = result.resized(3, 3)
        val smallerPredictions = FloatArray(9) { 2f }
        smaller.maskPrediction(smallerPredictions)
        assertTrue(smallerPredictions[0].isNaN())
        assertEquals(2f, smallerPredictions[4], 0f)
    }

    @Test fun `native calibration snapshot owns values even if JNI array is reused`() {
        val values = doubleArrayOf(639.2, 639.3, 481.0, 268.5, .1, .01, .02, -.03, 960.0, 540.0, 0.0)
        val snapshot = CameraCalibration.fromNative(values)
        values.fill(0.0)
        assertEquals(639.2, snapshot.camera.fx, 0.0)
        assertEquals(-.03, snapshot.d3, 0.0)
        assertEquals(CameraCalibration.Model.RADTAN, snapshot.model)
    }

    @Test fun `invalid native calibration is rejected instead of falling back to factory values`() {
        val good = doubleArrayOf(639.2, 639.3, 481.0, 268.5, 0.0, 0.0, 0.0, 0.0, 960.0, 540.0, 0.0)
        for ((index, value) in listOf(0 to 0.0, 1 to -1.0, 4 to Double.NaN, 8 to 0.0, 9 to 540.5, 10 to 2.0)) {
            val invalid = good.copyOf(); invalid[index] = value
            assertThrows(RuntimeException::class.java) { CameraCalibration.fromNative(invalid) }
        }
    }
}
