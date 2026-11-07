package com.recon.videoscan.live

import java.nio.ByteBuffer
import org.junit.Assert.*
import org.junit.Test

class TrackingImageSamplerTest {
    @Test fun `native crop copies exact rows without changing plane position`() {
        val pixels = ByteBuffer.wrap(ByteArray(8 * 6) { (it / 8 * 10 + it % 8).toByte() })
        val sampler = TrackingImageSampler(ImageCrop(0, 2, 4, 2), 4, 2)
        assertArrayEquals(byteArrayOf(20, 21, 22, 23, 30, 31, 32, 33), sampler.sample(pixels, 8, 1))
        assertEquals(0, pixels.position())
    }

    @Test fun `area filtering suppresses alternating features instead of dropping pixels`() {
        val pixels = ByteBuffer.wrap(byteArrayOf(0, -1, 0, -1, 0, -1, 0, -1))
        val sampler = TrackingImageSampler(ImageCrop(0, 0, 4, 2), 3, 1)
        assertArrayEquals(byteArrayOf(64, 128.toByte(), 191.toByte()), sampler.sample(pixels, 4, 1))
    }

    @Test fun `two dimensional area average preserves constant and mean intensity`() {
        val sampler = TrackingImageSampler(ImageCrop(0, 0, 2, 2), 1, 1)
        assertArrayEquals(byteArrayOf(60), sampler.sample(ByteBuffer.wrap(byteArrayOf(0, 40, 80, 120)), 2, 1))
        val constant = TrackingImageSampler(ImageCrop(0, 0, 8, 6), 6, 4)
        assertArrayEquals(ByteArray(24) { 77 }, constant.sample(ByteBuffer.wrap(ByteArray(48) { 77 }), 8, 1))
    }

    @Test fun `pixel stride buffer offset and unpadded final row are respected`() {
        val pixels = ByteBuffer.wrap(byteArrayOf(99, 10, 99, 20, 99, 99, 99, 30, 99, 40))
        pixels.position(1)
        val sampler = TrackingImageSampler(ImageCrop(0, 0, 2, 2), 2, 2)
        assertArrayEquals(byteArrayOf(10, 20, 30, 40), sampler.sample(pixels, 6, 2))
        assertEquals(1, pixels.position())
    }

    @Test fun `truncated plane and invalid row stride are rejected`() {
        val sampler = TrackingImageSampler(ImageCrop(0, 0, 4, 2), 4, 2)
        assertThrows(IllegalArgumentException::class.java) { sampler.sample(ByteBuffer.allocate(7), 4, 1) }
        assertThrows(IllegalArgumentException::class.java) { sampler.sample(ByteBuffer.allocate(8), 3, 1) }
    }
}
