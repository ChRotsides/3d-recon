package com.recon.poselogger.rgbscan

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Test
import java.nio.file.Files

class LocalFrameStoreTest {
    @Test fun `one persist makes one stable jpeg`() {
        val root = Files.createTempDirectory("rgb-frame").toFile()
        val jpeg = byteArrayOf(0xff.toByte(), 0xd8.toByte(), 1, 2, 0xff.toByte(), 0xd9.toByte())
        val frame = LocalFrameStore(root).persist(jpeg, 123)
        assertEquals(1, root.listFiles()!!.size)
        assertEquals(123, frame.timestampNs)
        assertArrayEquals(jpeg, frame.jpeg.readBytes())
    }
}
