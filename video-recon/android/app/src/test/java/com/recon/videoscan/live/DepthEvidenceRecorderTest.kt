package com.recon.videoscan.live

import java.io.DataInputStream
import java.io.File
import java.nio.file.Files
import java.util.zip.GZIPInputStream
import org.junit.Assert.*
import org.junit.Test

class DepthEvidenceRecorderTest {
    private fun frame() = DepthEvidenceRecorder.Frame(12.5, 3, 4, "model", "integrated", true,
        2, 1, 2, 1, floatArrayOf(1f,2f,3f,0f,0f,0f,1f), floatArrayOf(2f,2f,1f,0f),
        DoubleArray(11) { it.toDouble() }, floatArrayOf(1f,2f,3f), floatArrayOf(.5f,1f),
        shortArrayOf(2000,6000), byteArrayOf(64,-1), ByteArray(8) { it.toByte() }, byteArrayOf(7,8))

    @Test fun `closed recorder drains complete evidence with matching timestamp and geometry`() {
        val dir = Files.createTempDirectory("depth-evidence-test").toFile()
        try {
            val recorder = DepthEvidenceRecorder(dir); recorder.offer(frame()); recorder.close(); recorder.awaitClosed()
            assertEquals("saved", recorder.state)
            DataInputStream(GZIPInputStream(File(dir,"depth-0.bin.gz").inputStream())).use { input ->
                assertEquals(0x44505448,input.readInt()); assertEquals(1,input.readInt())
                assertEquals(12.5,input.readDouble(),0.0); assertEquals(3L,input.readLong()); assertEquals(4L,input.readLong())
                assertEquals("model",input.readUTF()); assertEquals("integrated",input.readUTF()); assertTrue(input.readBoolean())
                for (dimension in intArrayOf(2,1,2,1)) assertEquals(dimension,input.readInt())
                fun floats(expected: FloatArray) { assertEquals(expected.size,input.readInt()); expected.forEach { assertEquals(it,input.readFloat(),0f) } }
                floats(frame().pose); floats(frame().k)
                assertEquals(11,input.readInt()); repeat(11) { assertEquals(it.toDouble(),input.readDouble(),0.0) }
                floats(frame().landmarks); floats(frame().prediction)
                assertEquals(2,input.readInt()); assertEquals(2000,input.readUnsignedShort()); assertEquals(6000,input.readUnsignedShort())
                for (bytes in listOf(frame().confidence,frame().rgba,frame().gray)) {
                    assertEquals(bytes.size,input.readInt()); val actual=ByteArray(bytes.size); input.readFully(actual); assertArrayEquals(bytes,actual)
                }
                assertEquals(-1,input.read())
            }
            assertTrue(File(dir,"summary.txt").readText().contains("dropped=0"))
        } finally { dir.deleteRecursively() }
    }
    @Test fun `quota stops recording without deleting existing evidence`() {
        val dir = Files.createTempDirectory("depth-evidence-quota").toFile()
        try {
            File(dir,"existing").writeText("retain")
            val recorder=DepthEvidenceRecorder(dir,1);recorder.offer(frame());recorder.close();recorder.awaitClosed()
            assertEquals("quota reached",recorder.state)
            assertEquals("retain",File(dir,"existing").readText());assertFalse(File(dir,"depth-0.bin.gz").exists())
        } finally { dir.deleteRecursively() }
    }
}
