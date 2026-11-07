package com.recon.videoscan.live

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class XFeatRunnerTest {
    @Test fun `sixteen by nine frame is resized without changing rays`() {
        val geometry = XFeatGeometry.fit(960, 540)
        assertEquals(320, geometry.resizedWidth)
        assertEquals(180, geometry.resizedHeight)
        assertEquals(0, geometry.padLeft)
        assertEquals(6, geometry.padTop)
        assertEquals(481f, geometry.sourceX(160f), 1e-5f)
        assertEquals(271f, geometry.sourceY(96f), 1e-5f)
    }

    @Test fun `official logits and reliability score map back to source pixels`() {
        val logits = Array(1) { Array(65) { Array(24) { FloatArray(40) } } }
        val reliability = Array(1) { Array(1) { Array(24) { FloatArray(40) { 1f } } } }
        // Channel zero is the top-left pixel of its 8x8 cell, matching XFeat get_kpts_heatmap.
        logits[0][0][12][20] = 10f
        val candidates = XFeatPostprocessor.candidates(logits, reliability, XFeatGeometry.fit(960, 540))
        assertTrue(candidates.size >= 3)
        assertEquals(481f, candidates[0], 1e-4f)
        assertEquals(271f, candidates[1], 1e-4f)
        assertTrue(candidates[2] > 0.9f)
    }

    @Test fun `letterbox padding never becomes a camera feature`() {
        val logits = Array(1) { Array(65) { Array(24) { FloatArray(40) } } }
        val reliability = Array(1) { Array(1) { Array(24) { FloatArray(40) { 1f } } } }
        logits[0][0][0][20] = 10f
        assertTrue(XFeatPostprocessor.candidates(logits, reliability, XFeatGeometry.fit(960, 540)).isEmpty())
    }

    @Test fun `ranked graph outputs discard padding and map pixel centers`() {
        val indices = longArrayOf(96L * XFeatRunner.INPUT_WIDTH + 160L, 160L)
        val candidates = XFeatPostprocessor.mapRanked(floatArrayOf(0.8f, 0.9f), indices, XFeatGeometry.fit(960, 540))
        assertEquals(3, candidates.size)
        assertEquals(481f, candidates[0], 1e-5f)
        assertEquals(271f, candidates[1], 1e-5f)
        assertEquals(0.8f, candidates[2], 1e-6f)
    }

}
