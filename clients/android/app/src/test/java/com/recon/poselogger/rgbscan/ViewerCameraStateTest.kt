package com.recon.poselogger.rgbscan

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class ViewerCameraStateTest {
    @Test fun pinchDirectionsAndLimitsAreStable() {
        val camera = ViewerCameraState()
        camera.zoomBy(0.5f)
        assertEquals(2f, camera.distance, 0.0001f)
        camera.zoomBy(1000f)
        assertEquals(ViewerCameraState.MAX_DISTANCE, camera.distance, 0.0001f)
        camera.zoomBy(0.00001f)
        assertEquals(ViewerCameraState.MIN_DISTANCE, camera.distance, 0.0001f)
    }

    @Test fun twoFingerDragPansInEveryScreenDirection() {
        val camera = ViewerCameraState()
        camera.panByPixels(100f, 100f, 1000)
        assertTrue(camera.panX > 0f)
        assertTrue(camera.panY < 0f)
        camera.panByPixels(-200f, -200f, 1000)
        assertTrue(camera.panX < 0f)
        assertTrue(camera.panY > 0f)
    }

    @Test fun oneFingerOrbitIsViewportScaledAndPitchClamped() {
        val camera = ViewerCameraState()
        camera.orbitByViewport(500f, 500f, 1000, 1000)
        assertEquals(90f, camera.yaw, 0.0001f)
        assertEquals(75f, camera.pitch, 0.0001f)
        camera.orbitByViewport(0f, 1000f, 1000, 1000)
        assertEquals(85f, camera.pitch, 0.0001f)
    }

    @Test fun doubleTapResetRestoresTheInitialView() {
        val camera = ViewerCameraState()
        camera.orbit(100f, 50f)
        camera.panByPixels(100f, 100f, 1000)
        camera.zoomBy(0.5f)
        camera.reset()
        assertEquals(0f, camera.yaw, 0.0001f)
        assertEquals(0f, camera.pitch, 0.0001f)
        assertEquals(0f, camera.panX, 0.0001f)
        assertEquals(0f, camera.panY, 0.0001f)
        assertEquals(ViewerCameraState.DEFAULT_DISTANCE, camera.distance, 0.0001f)
    }

    @Test fun resetReturnsToTheCloudSpecificFitDistance() {
        val camera = ViewerCameraState(7.5f)
        camera.zoomBy(0.5f)
        camera.reset()
        assertEquals(7.5f, camera.distance, 0.0001f)
    }
}
