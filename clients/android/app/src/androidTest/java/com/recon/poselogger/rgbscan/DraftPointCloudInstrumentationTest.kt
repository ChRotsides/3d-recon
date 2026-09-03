package com.recon.poselogger.rgbscan

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Test
import org.junit.runner.RunWith
import java.io.ByteArrayOutputStream
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.security.MessageDigest
import java.util.zip.GZIPOutputStream

@RunWith(AndroidJUnit4::class)
class DraftPointCloudInstrumentationTest {
    @Test fun evidenceAssetDecodesAsOnePureVersion() {
        val raw = ByteBuffer.allocate(56).order(ByteOrder.LITTLE_ENDIAN).apply {
            record(floatArrayOf(1f, 2f, 3f), byteArrayOf(4, 5, 6), 0.8f, 0.5f, 9)
            record(floatArrayOf(-1f, 0f, 2f), byteArrayOf(7, 8, 9), 0.2f, 1f, 9)
        }.array()
        val compressed = ByteArrayOutputStream().also { sink ->
            GZIPOutputStream(sink).use { it.write(raw) }
        }.toByteArray()
        val root = InstrumentationRegistry.getInstrumentation().targetContext.cacheDir
        val binary = File(root, "phase3-draft.bin.gz").apply { writeBytes(compressed) }
        val metadata = File(root, "phase3-draft.json").apply {
            writeText(JSONObject()
                .put("schema", PointCloudDecoder.DRAFT_SCHEMA)
                .put("version", 9).put("source_panorama_version", 9).put("count", 2)
                .put("record_bytes", 28).put("compression", "gzip")
                .put("asset_bytes", compressed.size).put("decoded_bytes", raw.size)
                .put("asset_sha256", sha256(compressed)).put("decoded_sha256", sha256(raw))
                .put("min", JSONArray(listOf(-1, 0, 2)))
                .put("max", JSONArray(listOf(1, 2, 3))).toString())
        }
        val asset = PointCloudDecoder.decode(metadata, binary)
        assertEquals(9, asset.version)
        assertEquals(9, asset.sourcePanoramaVersion)
        assertEquals(2, asset.count)
        assertArrayEquals(floatArrayOf(0.8f, 0.2f), asset.confidence!!, 0.0001f)
        assertArrayEquals(floatArrayOf(0.5f, 1f), asset.panoramaSupport!!, 0.0001f)
    }

    private fun ByteBuffer.record(
        xyz: FloatArray,
        rgb: ByteArray,
        confidence: Float,
        support: Float,
        version: Int,
    ) {
        xyz.forEach { putFloat(it) }
        put(rgb); put(0); putFloat(confidence); putFloat(support); putInt(version)
    }

    private fun sha256(bytes: ByteArray) = MessageDigest.getInstance("SHA-256")
        .digest(bytes).joinToString("") { "%02x".format(it) }
}
