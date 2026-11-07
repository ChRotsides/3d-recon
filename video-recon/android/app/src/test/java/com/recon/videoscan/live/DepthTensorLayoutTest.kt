package com.recon.videoscan.live

import org.junit.Assert.*
import org.junit.Test

class DepthTensorLayoutTest {
    @Test fun `depth layout is independent of the input tensor rank`() {
        for (shape in listOf(longArrayOf(30,40), longArrayOf(1,30,40),
            longArrayOf(1,1,30,40), longArrayOf(1,1,1,30,40))) {
            assertEquals(DepthTensorLayout(40,30), DepthTensorLayout.from(shape))
        }
    }
    @Test fun `multi image and malformed model outputs fail clearly`() {
        for (shape in listOf(longArrayOf(2,30,40), longArrayOf(1,3,30,40),
            longArrayOf(0,40), longArrayOf(-1,40), longArrayOf(40), longArrayOf(100000,100000))) {
            assertThrows(IllegalArgumentException::class.java) { DepthTensorLayout.from(shape) }
        }
    }
}
