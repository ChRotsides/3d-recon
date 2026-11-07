package com.recon.videoscan.live

import org.junit.Assert.*
import org.junit.Test

class DepthQualityTest {
    @Test fun `feature fit tolerates gross outliers but rejects concentrated anchors`() {
        val q = DepthQuality(); val ds = (1..20).map { it / 10f + .5f }
        val zs = ds.mapIndexed { i, d -> if (i < 4) 7f else d * 2f }
        val cells = ds.indices.map { it % 12 }
        val fit = q.fit(ds, zs, cells, false)!!
        assertEquals(2f, fit.a, .001f); assertEquals(16, fit.inliers)
        assertNull(q.fit(ds, zs, List(20) { 0 }, false))
        assertNull(q.fit(ds.take(3), zs.take(3), cells.take(3), false))
        assertNull(q.fit(ds.drop(4).take(5), zs.drop(4).take(5), cells.drop(4).take(5), false))
    }
    @Test fun `inverse fit recovers offset without reusing a previous scale`() {
        val q = DepthQuality(); val ds = (1..20).map { it / 20f + .1f }
        val zs = ds.mapIndexed { i, d -> if (i < 3) 7f else 1f / (.4f * d + .1f) }
        val f = q.fit(ds, zs, ds.indices.map { it % 12 }, true)!!
        assertEquals(.4f, f.a, .001f); assertEquals(.1f, f.b, .001f)
        assertNull(q.fit(List(20) { 1f }, List(20) { 2f }, ds.indices.map { it % 12 }, true))
    }
    @Test fun `metric predictions cannot contradict distributed landmark depths`() {
        val q = DepthQuality(); val zs = (1..20).map { 1f + it / 10f }; val cells = zs.indices.map { it % 12 }
        assertTrue(q.metricConsistent(zs.map { it * 1.02f }, zs, cells))
        assertFalse(q.metricConsistent(zs.map { it * 1.5f }, zs, cells))
        assertFalse(q.metricConsistent(zs.mapIndexed { i, z -> if (i < 9) z else z * 2f }, zs, cells))
        assertTrue(q.metricConsistent(emptyList(), emptyList(), emptyList()))
    }

    @Test fun `bad or mismatched anchor arrays cannot manufacture a fit`() {
        val q = DepthQuality(); val ds = List(12) { 1f }; val zs = List(12) { 2f }; val cells = (0..11).toList()
        assertNull(q.fit(ds, zs.drop(1), cells, false))
        assertNull(q.fit(ds, zs, cells.drop(1), false))
        assertNull(q.fit(List(12) { Float.NaN }, zs, cells, false))
        assertNull(q.fit(ds, List(12) { -1f }, cells, false))
        assertNull(q.fit(ds, zs, List(12) { 20 }, false))
    }

    @Test fun `new coverage is provisional until translated view agrees`() {
        val q = DepthQuality(); val w = 40; val h = 30
        val depth = ShortArray(w * h) { 2000 }; val k = floatArrayOf(40f, 40f, 20f, 15f)
        val r = floatArrayOf(1f,0f,0f,0f,1f,0f,0f,0f,1f); val c = floatArrayOf(0f,0f,0f)
        assertTrue(q.confidence(depth,w,h,k,r,c,1.0).all { it.toInt() == 64 })
        q.remember(depth,w,h,k,r,c,1.0,ByteArray(w*h) { (-1).toByte() })
        assertFalse(q.confidence(depth,w,h,k,r,c,1.2).any { it == (-1).toByte() })
        val moved = floatArrayOf(.1f,0f,0f)
        val weights = q.confidence(depth,w,h,k,r,moved,1.4)
        assertTrue(weights.count { it == (-1).toByte() } > 1000)
        assertTrue(weights.any { it.toInt() == 64 })
        assertTrue(q.confidence(ShortArray(w*h) { 3000 },w,h,k,r,moved,1.4).count { it.toInt() == 0 } > 1000)
        q.reset()
        assertTrue(q.confidence(depth,w,h,k,r,moved,1.6).all { it.toInt() == 64 })
    }

    @Test fun `phone calibration keeps usable distributed hypotheses and rejects bad frames`() {
        val rows = javaClass.getResourceAsStream("/depth-anchors-s23.csv")!!.bufferedReader().useLines { lines ->
            lines.filter { !it.startsWith("#") }.map { it.split(",") }.toList()
        }.groupBy { it[0].toInt() }
        val q = DepthQuality()
        val calibrated = rows.mapNotNull { (frame, anchors) ->
            val ds = anchors.map { it[1].toFloat() }; val zs = anchors.map { it[2].toFloat() }
            val cells = anchors.map { it[3].toInt() }
            val fit = q.fit(ds, zs, cells, true) ?: return@mapNotNull null
            // Every accepted result must still meet the geometric support rules.
            val agreeing = ds.indices.filter { kotlin.math.abs(fit.depth(ds[it]) - zs[it]) <= maxOf(.12f, .1f * zs[it]) }
            assertTrue(agreeing.size >= 8 && agreeing.size >= ds.size * .65)
            assertTrue(agreeing.map { cells[it] }.toSet().size >= 4)
            frame
        }.toSet()
        // These real frames previously selected unusable hypotheses or failed the
        // overly tight model-calibration tolerance despite distributed support.
        assertTrue(calibrated.containsAll(listOf(2, 5, 6, 10, 11, 12, 24, 25, 28, 29, 32, 33)))
        assertTrue(calibrated.size >= 20)
        assertFalse(calibrated.contains(15)) // inverted/noisy depth relation
        assertFalse(calibrated.contains(27)) // no sufficient consensus
        assertFalse(calibrated.contains(40))
    }

    @Test fun `unobservable wall calibration has an actionable reason`() {
        val q = DepthQuality(); val ds = (1..20).map { it / 10f + 1f }
        val zs = ds.map { 1f / (.005f * it + .5f) }
        val result = q.calibrate(ds, zs, ds.indices.map { it % 12 }, true)
        assertNull(result.fit)
        assertTrue(result.reason.contains("varied distances"))
    }

    @Test fun `view history bridges a calibration gap but expires and resets`() {
        val q = DepthQuality(); val w = 40; val h = 30
        val depth = ShortArray(w*h) { 2000 }; val k = floatArrayOf(40f,40f,20f,15f)
        val r = floatArrayOf(1f,0f,0f,0f,1f,0f,0f,0f,1f)
        val origin = floatArrayOf(0f,0f,0f); val moved = floatArrayOf(.1f,0f,0f)
        q.remember(depth,w,h,k,r,origin,1.0)
        assertTrue(q.confidence(depth,w,h,k,r,moved,5.0).count { it == (-1).toByte() } > 1000)
        // The latest accepted view has no independent baseline from the current
        // view. The older translated view remains useful for confirmation.
        q.remember(depth,w,h,k,r,moved,5.0)
        assertTrue(q.confidence(depth,w,h,k,r,moved,5.5).count { it == (-1).toByte() } > 1000)
        assertFalse(q.confidence(depth,w,h,k,r,moved,10.0).any { it == (-1).toByte() })
        q.reset()
        assertFalse(q.confidence(depth,w,h,k,r,origin,10.1).any { it == (-1).toByte() })
    }

    @Test fun `recent foreground conflict cannot be overruled by older matching background`() {
        val q = DepthQuality(); val w = 40; val h = 30
        val depth = ShortArray(w*h) { 2000 }; val k = floatArrayOf(40f,40f,20f,15f)
        val r = floatArrayOf(1f,0f,0f,0f,1f,0f,0f,0f,1f)
        q.remember(depth,w,h,k,r,floatArrayOf(0f,0f,0f),1.0)
        q.remember(ShortArray(w*h) { 1000 },w,h,k,r,floatArrayOf(.1f,0f,0f),2.0,ByteArray(w*h) { (-1).toByte() })
        val weights = q.confidence(depth,w,h,k,r,floatArrayOf(.2f,0f,0f),3.0)
        assertEquals(0, weights[15*w+20].toInt())
    }

    @Test fun `inverse fit excludes unsupported disparity instead of inventing distant surfaces`() {
        val ds = (1..24).map { .5f + it / 20f }; val zs = ds.map { 1f / (.4f * it + .1f) }
        val fit = DepthQuality().fit(ds,zs,ds.indices.map { it % 12 },true)!!
        assertTrue(fit.depth(.1f).isNaN())
        assertTrue(fit.depth(3f).isNaN())
        assertEquals(2f,fit.depth(1f),.001f)
    }

    @Test fun `provisional foreground cannot veto older agreement and rejected pixels cannot confirm`() {
        val q = DepthQuality(); val w = 40; val h = 30
        val depth = ShortArray(w*h) { 2000 }; val k = floatArrayOf(40f,40f,20f,15f)
        val identity = floatArrayOf(1f,0f,0f,0f,1f,0f,0f,0f,1f)
        q.remember(depth,w,h,k,identity,floatArrayOf(0f,0f,0f),1.0)
        q.remember(ShortArray(w*h) { 1000 },w,h,k,identity,floatArrayOf(.1f,0f,0f),2.0)
        assertEquals(255,q.confidence(depth,w,h,k,identity,floatArrayOf(.2f,0f,0f),3.0)[15*w+20].toInt() and 255)
        q.reset()
        q.remember(depth,w,h,k,identity,floatArrayOf(0f,0f,0f),1.0,ByteArray(w*h))
        assertFalse(q.confidence(depth,w,h,k,identity,floatArrayOf(.1f,0f,0f),2.0).any { it == (-1).toByte() })
    }
}
