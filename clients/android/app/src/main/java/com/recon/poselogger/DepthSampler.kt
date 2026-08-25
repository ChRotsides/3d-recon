package com.recon.poselogger

import android.media.Image
import com.google.ar.core.Camera
import com.google.ar.core.Frame
import com.google.ar.core.exceptions.NotYetAvailableException
import kotlin.math.abs
import kotlin.math.sqrt

/**
 * ARCore's depth map turned into the oriented world-space surface samples the
 * shared core's coverage grid consumes (plan §6.1).
 *
 * Plan §3.6 has this geometry coming from an on-device DA3-Small pass, which
 * does not exist yet — and until something feeds the grid, `observe` is called
 * with empty points, every coverage figure is 0 and the next-best-station solver
 * has nothing to reason about. That is a capture UI that cannot tell the user
 * what it has mapped.
 *
 * ARCore's Depth API closes that gap today. It is **not** a replacement for the
 * planned depth pass: this is guidance-only geometry, never written to the
 * bundle and never reconstructed from. It only has to be good enough to answer
 * "have I seen that wall well enough yet", and for that a coarse 160x120 depth
 * map at keyframe rate is plenty.
 *
 * Not every device supports it, so [supportedBy] is checked before the session
 * is configured and the app degrades to nags and keyframe counting without it.
 */
class DepthSampler(
    /** Sample every Nth pixel in each axis. 6 over a 160x120 map is ~500 points. */
    private val stride: Int = 6,
) {

    class Samples(
        /** Flat xyz triples, world frame — the layout `FfiCaptureSession.observe` wants. */
        val points: List<Float>,
        val normals: List<Float>,
        /** Median of the sampled depths, which drives the core's TOO_CLOSE nag. */
        val medianDepthM: Float,
        /** Full millimetre depth image, tightly packed row-major, for bundle evidence. */
        val depthMm: ShortArray,
        val width: Int,
        val height: Int,
    )

    sealed interface Probe {
        data class Ready(val samples: Samples) : Probe
        data object Waiting : Probe
        data class Failed(val reason: String) : Probe
    }

    /**
     * Sample the frame, or null if depth is not ready.
     *
     * ARCore takes a few frames to produce its first depth map and throws
     * [NotYetAvailableException] until then, which is expected rather than an
     * error worth surfacing.
     */
    fun probe(frame: Frame, camera: Camera): Probe {
        val image: Image = try {
            frame.acquireDepthImage16Bits()
        } catch (e: NotYetAvailableException) {
            return Probe.Waiting
        } catch (e: Exception) {
            return Probe.Failed("${e.javaClass.simpleName}: ${e.message ?: "depth acquisition failed"}")
        }
        return image.use { depth ->
            val w = depth.width
            val h = depth.height
            val plane = depth.planes[0]
            // `order(nativeOrder())` is load-bearing. A plane's ByteBuffer
            // defaults to big-endian while DEPTH16 is little-endian, so
            // `asShortBuffer()` on its own byte-swaps every sample: 3000 mm
            // (0x0BB8) reads back as 0xB80B = 47 m and is discarded as out of
            // range. The failure is near-silent — the occasional value that is
            // still plausible when swapped gets through, so the coverage grid
            // fills with a handful of voxels instead of none, which reads as
            // "the room is not covered yet" rather than "the depth pass is
            // broken". Measured on a real capture: 4 voxels after 81 keyframes.
            val buf = plane.buffer.order(java.nio.ByteOrder.nativeOrder()).asShortBuffer()
            val rowStrideShorts = plane.rowStride / 2
            val depthMm = ShortArray(w * h)
            for (row in 0 until h) {
                for (col in 0 until w) {
                    depthMm[row * w + col] = buf.get(row * rowStrideShorts + col)
                }
            }

            // ARCore reports intrinsics against the CPU image, which is a much
            // larger rendering of the same frustum. Same aspect, so a uniform
            // scale to the depth map's size is the whole correction — skipping
            // it back-projects everything at ~4x the true angle off-axis.
            val intr = camera.imageIntrinsics
            val dims = intr.imageDimensions
            val s = w.toFloat() / dims[0].toFloat()
            val fx = intr.focalLength[0] * s
            val fy = intr.focalLength[1] * s
            val cx = intr.principalPoint[0] * s
            val cy = intr.principalPoint[1] * s

            // world <- camera, applied by hand: ARCore's transformPoint would
            // allocate a FloatArray per sample, and there are thousands.
            val m = FloatArray(16)
            camera.pose.toMatrix(m, 0)

            val n = (w / stride) * (h / stride)
            val points = ArrayList<Float>(n * 3)
            val normals = ArrayList<Float>(n * 3)
            val depths = ArrayList<Float>(n)

            fun depthAt(u: Int, v: Int): Float {
                val mm = buf.get(v * rowStrideShorts + u).toInt() and 0xFFFF
                return mm * 0.001f
            }

            var v = 0
            while (v + stride < h) {
                var u = 0
                while (u + stride < w) {
                    val d = depthAt(u, v)
                    if (d in MIN_DEPTH_M..MAX_DEPTH_M) {
                        val dr = depthAt(u + stride, v)
                        val dd = depthAt(u, v + stride)
                        // A normal straddling a depth discontinuity is a fiction
                        // — it would point along the silhouette rather than at
                        // the surface, and the coverage grid weights by exactly
                        // that direction.
                        if (dr in MIN_DEPTH_M..MAX_DEPTH_M && dd in MIN_DEPTH_M..MAX_DEPTH_M &&
                            abs(dr - d) < CONTINUITY * d && abs(dd - d) < CONTINUITY * d
                        ) {
                            // Camera frame: +X right, +Y up, -Z forward, so the
                            // image's downward v axis flips sign.
                            val px = (u - cx) / fx * d
                            val py = -(v - cy) / fy * d
                            val pz = -d
                            val rx = (u + stride - cx) / fx * dr
                            val ry = -(v - cy) / fy * dr
                            val rz = -dr
                            val bx = (u - cx) / fx * dd
                            val by = -(v + stride - cy) / fy * dd
                            val bz = -dd

                            val ax = rx - px; val ay = ry - py; val az = rz - pz
                            val ex = bx - px; val ey = by - py; val ez = bz - pz
                            var nx = ey * az - ez * ay
                            var ny = ez * ax - ex * az
                            var nz = ex * ay - ey * ax
                            val len = sqrt(nx * nx + ny * ny + nz * nz)
                            if (len > 1e-6f) {
                                nx /= len; ny /= len; nz /= len
                                // Face the observer. The grid discards
                                // back-facing samples outright, so a normal
                                // pointing into the wall contributes nothing at
                                // all — and the sign out of a cross product
                                // depends on the winding, not on the surface.
                                if (nx * -px + ny * -py + nz * -pz < 0f) {
                                    nx = -nx; ny = -ny; nz = -nz
                                }
                                points.add(m[0] * px + m[4] * py + m[8] * pz + m[12])
                                points.add(m[1] * px + m[5] * py + m[9] * pz + m[13])
                                points.add(m[2] * px + m[6] * py + m[10] * pz + m[14])
                                // A direction, so translation must not apply.
                                normals.add(m[0] * nx + m[4] * ny + m[8] * nz)
                                normals.add(m[1] * nx + m[5] * ny + m[9] * nz)
                                normals.add(m[2] * nx + m[6] * ny + m[10] * nz)
                                depths.add(d)
                            }
                        }
                    }
                    u += stride
                }
                v += stride
            }
            if (points.isEmpty()) return@use Probe.Waiting
            depths.sort()
            Probe.Ready(
                Samples(points, normals, depths[depths.size / 2], depthMm, w, h)
            )
        }
    }

    fun sample(frame: Frame, camera: Camera): Samples? =
        (probe(frame, camera) as? Probe.Ready)?.samples

    companion object {
        /**
         * Whether this device can do depth at all. Checked before the session is
         * configured, because asking for an unsupported mode throws.
         */
        fun supportedBy(session: com.google.ar.core.Session): Boolean =
            session.isDepthModeSupported(com.google.ar.core.Config.DepthMode.AUTOMATIC)

        /** Below this ARCore's depth is mostly noise; above it, useless for a room. */
        private const val MIN_DEPTH_M = 0.25f
        private const val MAX_DEPTH_M = 6.0f

        /** Fractional depth step tolerated between neighbours before a normal is discarded. */
        private const val CONTINUITY = 0.06f
    }
}
