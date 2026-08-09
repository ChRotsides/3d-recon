package com.recon.poselogger

import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.sin
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * The arrow to an off-screen next-best station, pinned on a plain JVM.
 *
 * Worth testing without a device precisely because the failure is not a crash:
 * a mirrored bearing renders a confident arrow pointing away from where the user
 * should walk, which on a phone looks like a real suggestion and sends them the
 * wrong way. The same reason [FrameConversionTest] guards the coordinate frame.
 *
 * Clip space here is what `projection * view * position` produces: x right,
 * y up, w the view-space depth, positive in front of the camera.
 */
class StationArrowTest {

    /** Bearings are screen convention: 0 right, +PI/2 down, -PI/2 up. */
    private fun assertPointsRoughly(expected: Float, actual: Float?, what: String) {
        assertNotNull("expected an arrow $what", actual)
        // Compare as directions so the +/-PI wrap does not make an equal
        // direction look like a maximal error.
        val dot = cos(expected) * cos(actual!!) + sin(expected) * sin(actual)
        assertTrue("$what: expected ~$expected, got $actual", dot > 0.9f)
    }

    @Test
    fun a_station_on_screen_gets_no_arrow() {
        // Dead centre, two metres ahead.
        assertNull(StationArrow.bearing(0f, 0f, 2f))
        // Off-centre but still inside the frustum.
        assertNull(StationArrow.bearing(1f, -0.5f, 2f))
    }

    @Test
    fun a_station_past_the_right_edge_points_right() {
        // ndc x = 3, well outside [-1, 1].
        assertPointsRoughly(0f, StationArrow.bearing(6f, 0f, 2f), "to the right")
    }

    @Test
    fun a_station_above_the_top_edge_points_up() {
        // Clip y is up, screen y is down, so up is -PI/2 on screen. Getting this
        // flip wrong sends the user's gaze to the floor when the ceiling is what
        // is missing — the single most common gap in a room capture.
        assertPointsRoughly(-(PI / 2).toFloat(), StationArrow.bearing(0f, 6f, 2f), "upwards")
    }

    @Test
    fun a_station_behind_the_camera_and_to_the_right_still_points_right() {
        // The trap. Behind the camera w is negative, so the perspective divide
        // flips the sign and the naive bearing points left — away from a station
        // that is physically over the user's right shoulder.
        assertPointsRoughly(0f, StationArrow.bearing(1f, 0f, -2f), "behind and right")
    }

    @Test
    fun a_station_directly_behind_still_gets_an_arrow() {
        // Nothing is on screen behind the camera, whatever the ndc says.
        assertNotNull(StationArrow.bearing(0.1f, 0f, -2f))
    }

    @Test
    fun a_station_on_the_eye_plane_gets_no_arrow() {
        // w == 0 is the degenerate case: the direction is undefined and the
        // divide would be an infinity, so it must not produce a bearing at all.
        assertNull(StationArrow.bearing(1f, 1f, 0f))
    }

    @Test
    fun the_bearing_is_a_direction_not_a_distance() {
        // Twice as far off-screen in the same direction is the same arrow.
        val near = StationArrow.bearing(6f, 6f, 2f)
        val far = StationArrow.bearing(60f, 60f, 2f)
        assertNotNull(near)
        assertNotNull(far)
        assertEquals(near!!, far!!, 1e-4f)
    }
}
