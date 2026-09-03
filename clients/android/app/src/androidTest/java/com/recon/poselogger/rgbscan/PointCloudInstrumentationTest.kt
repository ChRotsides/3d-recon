package com.recon.poselogger.rgbscan

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.json.JSONObject
import org.json.JSONArray
import org.junit.Assert.assertEquals
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.security.MessageDigest

@RunWith(AndroidJUnit4::class)
class PointCloudInstrumentationTest {
    @Test fun acceptedAssetReachesRenderableState() {
        val root = InstrumentationRegistry.getInstrumentation().targetContext.cacheDir
        val binary = File(root, "phase1-cloud.bin")
        val payload = ByteBuffer.allocate(15).order(ByteOrder.LITTLE_ENDIAN)
            .putFloat(0f).putFloat(0f).putFloat(0f).put(byteArrayOf(1, 2, 3)).array()
        binary.writeBytes(payload)
        val hash = MessageDigest.getInstance("SHA-256").digest(payload).joinToString("") { "%02x".format(it) }
        val metadata = File(root, "phase1-cloud.json")
        metadata.writeText(JSONObject().put("schema", PointCloudDecoder.SCHEMA).put("count", 1)
            .put("asset_sha256", hash).put("min", JSONArray(listOf(0, 0, 0)))
            .put("max", JSONArray(listOf(0, 0, 0))).toString())
        assertEquals(1, PointCloudDecoder.decode(metadata, binary).count)
    }
}
