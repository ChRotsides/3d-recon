package com.recon.poselogger

import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.sin
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Test

/**
 * Guards the flipped/mirrored-scene failure mode from plan §7.
 *
 * These tests exercise the matrix arithmetic directly rather than through
 * ARCore's `Pose`, so they run on a plain JVM with no device and no emulator —
 * which means they run in CI on every commit, which is the only way a
 * coordinate-frame regression gets caught before it ships a mirrored house.
 */
class FrameConversionTest {

    /** Column-major 4x4, the layout ARCore's `Pose.toMatrix` produces. */
    private fun columnMajor(r: Array<DoubleArray>, t: DoubleArray): FloatArray {
        val m = FloatArray(16)
        for (col in 0 until 3) {
            for (row in 0 until 3) {
                m[col * 4 + row] = r[row][col].toFloat()
            }
        }
        m[12] = t[0].toFloat(); m[13] = t[1].toFloat(); m[14] = t[2].toFloat()
        m[15] = 1f
        return m
    }

    private fun convert(m: FloatArray): PoseWire {
        val r = List(3) { row -> List(3) { col -> m[col * 4 + row].toDouble() } }
        val t = listOf(m[12].toDouble(), m[13].toDouble(), m[14].toDouble())
        val det = det3(r)
        if (det < 0.0) throw FrameConversionException("mirrored, det=$det")
        if (abs(det - 1.0) > 1e-3) throw FrameConversionException("not a rotation, det=$det")
        return PoseWire(r, t)
    }

    private fun det3(r: List<List<Double>>) =
        r[0][0] * (r[1][1] * r[2][2] - r[1][2] * r[2][1]) -
            r[0][1] * (r[1][0] * r[2][2] - r[1][2] * r[2][0]) +
            r[0][2] * (r[1][0] * r[2][1] - r[1][1] * r[2][0])

    @Test
    fun `column major input is not transposed`() {
        // A 90 deg yaw is asymmetric, so a transpose is visible in the result.
        val yaw = Math.PI / 2
        val r = arrayOf(
            doubleArrayOf(cos(yaw), 0.0, sin(yaw)),
            doubleArrayOf(0.0, 1.0, 0.0),
            doubleArrayOf(-sin(yaw), 0.0, cos(yaw)),
        )
        val wire = convert(columnMajor(r, doubleArrayOf(1.0, 2.0, 3.0)))

        assertEquals(1.0, wire.R[0][2], 1e-6)   // sin(90) in row 0, col 2
        assertEquals(-1.0, wire.R[2][0], 1e-6)  // -sin(90) in row 2, col 0
        assertEquals(listOf(1.0, 2.0, 3.0), wire.t)
    }

    @Test
    fun `identity round trips`() {
        val r = arrayOf(
            doubleArrayOf(1.0, 0.0, 0.0),
            doubleArrayOf(0.0, 1.0, 0.0),
            doubleArrayOf(0.0, 0.0, 1.0),
        )
        val wire = convert(columnMajor(r, doubleArrayOf(0.0, 0.0, 0.0)))
        assertEquals(listOf(listOf(1.0, 0.0, 0.0), listOf(0.0, 1.0, 0.0), listOf(0.0, 0.0, 1.0)),
            wire.R)
    }

    @Test
    fun `a mirrored pose is refused rather than uploaded`() {
        // This is what a botched left-handed conversion produces.
        val mirrored = arrayOf(
            doubleArrayOf(-1.0, 0.0, 0.0),
            doubleArrayOf(0.0, 1.0, 0.0),
            doubleArrayOf(0.0, 0.0, 1.0),
        )
        try {
            convert(columnMajor(mirrored, doubleArrayOf(0.0, 0.0, 0.0)))
            fail("a mirrored pose must not be accepted")
        } catch (e: FrameConversionException) {
            assertTrue(e.message!!.contains("mirrored"))
        }
    }

    @Test
    fun `a scaled pose is refused`() {
        val scaled = arrayOf(
            doubleArrayOf(2.0, 0.0, 0.0),
            doubleArrayOf(0.0, 2.0, 0.0),
            doubleArrayOf(0.0, 0.0, 2.0),
        )
        try {
            convert(columnMajor(scaled, doubleArrayOf(0.0, 0.0, 0.0)))
            fail("a pose carrying scale must not be accepted")
        } catch (e: FrameConversionException) {
            assertTrue(e.message!!.contains("not a rotation"))
        }
    }

    @Test
    fun `camera looks down minus z`() {
        // Canonical frame (plan §5): forward is the negated third column of R.
        val r = arrayOf(
            doubleArrayOf(1.0, 0.0, 0.0),
            doubleArrayOf(0.0, 1.0, 0.0),
            doubleArrayOf(0.0, 0.0, 1.0),
        )
        val wire = convert(columnMajor(r, doubleArrayOf(0.0, 0.0, 0.0)))
        val forward = listOf(-wire.R[0][2], -wire.R[1][2], -wire.R[2][2])
        // Element-wise with a delta, not list equality: negating a zero yields
        // -0.0, and Double.equals treats -0.0 and 0.0 as different values.
        val expected = listOf(0.0, 0.0, -1.0)
        for (i in expected.indices) {
            assertEquals("component $i of $forward", expected[i], forward[i], 1e-12)
        }
    }

    @Test
    fun `rotating pixels rotates intrinsics through the same reprojection`() {
        val source = IntrinsicsWire(
            fx = 800.0, fy = 820.0, cx = 301.25, cy = 211.75, width = 640, height = 480,
        )
        val point = doubleArrayOf(0.17, -0.09, 1.4)
        val u = source.fx * point[0] / point[2] + source.cx
        val v = source.fy * point[1] / point[2] + source.cy

        for (rotation in listOf(0, 90, 180, 270)) {
            val savedWidth = if (rotation % 180 == 0) source.width else source.height
            val savedHeight = if (rotation % 180 == 0) source.height else source.width
            val rotated = FrameConversion.rotateAndScaleIntrinsics(
                source, rotation, savedWidth, savedHeight
            )
            val expected = when (rotation) {
                0 -> doubleArrayOf(u, v)
                90 -> doubleArrayOf(source.height - 1.0 - v, u)
                180 -> doubleArrayOf(source.width - 1.0 - u, source.height - 1.0 - v)
                else -> doubleArrayOf(v, source.width - 1.0 - u)
            }
            val projected = when (rotation) {
                0 -> doubleArrayOf(
                    rotated.fx * point[0] / point[2] + rotated.cx,
                    rotated.fy * point[1] / point[2] + rotated.cy,
                )
                90 -> doubleArrayOf(
                    rotated.fx * (-point[1]) / point[2] + rotated.cx,
                    rotated.fy * point[0] / point[2] + rotated.cy,
                )
                180 -> doubleArrayOf(
                    rotated.fx * (-point[0]) / point[2] + rotated.cx,
                    rotated.fy * (-point[1]) / point[2] + rotated.cy,
                )
                else -> doubleArrayOf(
                    rotated.fx * point[1] / point[2] + rotated.cx,
                    rotated.fy * (-point[0]) / point[2] + rotated.cy,
                )
            }
            assertEquals("u at $rotation degrees", expected[0], projected[0], 1e-9)
            assertEquals("v at $rotation degrees", expected[1], projected[1], 1e-9)
        }
    }

    @Test
    fun `intrinsics rotation round trips for every device orientation`() {
        val source = IntrinsicsWire(
            fx = 790.0, fy = 810.0, cx = 300.5, cy = 220.25, width = 640, height = 480,
        )
        for (rotation in listOf(0, 90, 180, 270)) {
            val w = if (rotation % 180 == 0) source.width else source.height
            val h = if (rotation % 180 == 0) source.height else source.width
            val rotated = FrameConversion.rotateAndScaleIntrinsics(source, rotation, w, h)
            val restored = FrameConversion.rotateAndScaleIntrinsics(
                rotated, (360 - rotation) % 360, source.width, source.height
            )
            assertEquals(source.fx, restored.fx, 1e-9)
            assertEquals(source.fy, restored.fy, 1e-9)
            assertEquals(source.cx, restored.cx, 1e-9)
            assertEquals(source.cy, restored.cy, 1e-9)
            assertEquals(source.width, restored.width)
            assertEquals(source.height, restored.height)
        }
    }

    @Test
    fun `pixel rotation and camera basis preserve every world ray`() {
        val identity = PoseWire(
            R = listOf(
                listOf(1.0, 0.0, 0.0),
                listOf(0.0, 1.0, 0.0),
                listOf(0.0, 0.0, 1.0),
            ),
            t = listOf(1.0, 2.0, 3.0),
        )
        val oldRay = doubleArrayOf(0.17, -0.09, -1.4)
        for (rotation in listOf(0, 90, 180, 270)) {
            val newRay = when (rotation) {
                0 -> oldRay
                90 -> doubleArrayOf(oldRay[1], -oldRay[0], oldRay[2])
                180 -> doubleArrayOf(-oldRay[0], -oldRay[1], oldRay[2])
                else -> doubleArrayOf(-oldRay[1], oldRay[0], oldRay[2])
            }
            val pose = FrameConversion.rotateCameraAxes(identity, rotation)
            val world = DoubleArray(3) { row ->
                pose.t[row] + (0 until 3).sumOf { col -> pose.R[row][col] * newRay[col] }
            }
            for (axis in 0 until 3) {
                assertEquals(
                    "world ray axis $axis at $rotation degrees",
                    identity.t[axis] + oldRay[axis],
                    world[axis],
                    1e-9,
                )
            }
        }
    }

    @Test
    fun `serialised json uses the wire field names the harness expects`() {
        val bundle = SessionBundleWire(
            sessionId = "s", roomId = "r",
            device = DeviceInfo("Pixel 9", "Android 15", "arcore", "ARCore"),
            gravity = FrameConversion.GRAVITY_DOWN,
        )
        val json = Json { encodeDefaults = true }.encodeToString(bundle)
        // Pydantic's SessionBundle forbids unknown fields, so a camelCase slip
        // here fails the upload rather than being silently dropped.
        for (key in listOf(
            "\"session_id\"", "\"room_id\"", "\"ar_platform\"",
            "\"sparse_points\"", "\"reference_measurements\"", "\"schema_version\"",
            "\"up_axis\"", "\"camera_forward\"",
        )) {
            assertTrue("missing $key in $json", json.contains(key))
        }
        assertTrue("camelCase leaked into the wire format", !json.contains("sessionId"))
    }

    @Test
    fun `a tapped footprint serialises under the key the harness expects`() {
        val bundle = SessionBundleWire(
            sessionId = "s", roomId = "r",
            device = DeviceInfo("Pixel 9", "Android 15", "arcore", "ARCore"),
            gravity = FrameConversion.GRAVITY_DOWN,
            floorOutline = FloorOutlineWire(
                corners = listOf(
                    listOf(0.0, 0.0, 0.0),
                    listOf(4.0, 0.0, 0.0),
                    listOf(4.0, 0.0, 3.0),
                    listOf(0.0, 0.0, 3.0),
                ),
                cornerIds = listOf(1, 2, 3, 4),
                closed = true,
                referencePointAdjustments = listOf(
                    ReferencePointAdjustmentWire(
                        cornerId = 2,
                        fromPoint = listOf(4.0, 0.0, 0.0),
                        toPoint = listOf(4.25, 0.0, -0.1),
                        delta = listOf(0.25, 0.0, -0.1),
                        startedAtNs = 10,
                        endedAtNs = 20,
                    )
                ),
            ),
        )
        val json = Json { encodeDefaults = true }.encodeToString(bundle)
        // `floor_outline`, not `floorOutline`: SessionBundle forbids unknown
        // fields, so the wrong case fails the whole upload rather than dropping
        // just the footprint.
        assertTrue("missing floor_outline in $json", json.contains("\"floor_outline\""))
        assertTrue(json.contains("\"corners\""))
        assertTrue(json.contains("\"corner_ids\""))
        assertTrue(json.contains("\"closed\""))
        assertTrue(json.contains("\"reference_point_adjustments\""))
        assertTrue(json.contains("\"corner_id\""))
        assertTrue(json.contains("\"from\""))
        assertTrue(json.contains("\"started_at_ns\""))
        assertTrue("camelCase leaked", !json.contains("floorOutline"))
    }

    @Test
    fun `a bundle with no footprint omits it rather than sending an empty one`() {
        val bundle = SessionBundleWire(
            sessionId = "s", roomId = "r",
            device = DeviceInfo("Pixel 9", "Android 15", "arcore", "ARCore"),
            gravity = FrameConversion.GRAVITY_DOWN,
        )
        val json = Json { encodeDefaults = true }.encodeToString(bundle)
        // An empty corner list would read downstream as "the user traced
        // nothing", which is true, but so does null — and null is the shape
        // every bundle captured before this feature already has.
        assertTrue("expected a null floor_outline, got $json",
            json.contains("\"floor_outline\": null") || json.contains("\"floor_outline\":null"))
    }
}
