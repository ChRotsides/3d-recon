package com.recon.poselogger.rgbscan

import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class GuidanceStateStoreTest {
    @Test fun `delayed top level response is rejected`() {
        val store = GuidanceStateStore()
        assertEquals(3, store.accept(payload(version = 3, coverageVersion = 3))!!.version)
        assertNull(store.accept(payload(version = 2, coverageVersion = 4)))
        assertEquals(3, store.current().coverage!!.version)
    }

    @Test fun `new response cannot roll an individual asset backwards`() {
        val store = GuidanceStateStore()
        store.accept(payload(version = 3, coverageVersion = 3, localizationVersion = 3))
        val result = store.accept(payload(version = 4, coverageVersion = 2, localizationVersion = 4))!!
        assertEquals(3, result.coverage!!.version)
        assertEquals(4, result.localization!!.version)
    }

    @Test fun `explicit unlocalized state suppresses target arrow`() {
        val store = GuidanceStateStore()
        val unlocalized = store.accept(payload(version = 1, localization = "unlocalized"))!!
        assertFalse(unlocalized.targetVisible)
        val localized = store.accept(payload(version = 2, localization = "localized"))!!
        assertTrue(localized.targetVisible)
    }

    @Test fun `null target from newer response clears previous target`() {
        val store = GuidanceStateStore()
        assertTrue(store.accept(payload(version = 1))!!.targetVisible)
        val withoutTarget = payload(version = 2).put("target", JSONObject.NULL)
        val result = store.accept(withoutTarget)!!
        assertNull(result.target)
        assertFalse(result.targetVisible)
    }

    @Test fun `localized direction is retained for relative guidance`() {
        val snapshot = GuidanceStateStore().accept(payload(version = 1))!!
        assertEquals(0.0, snapshot.localization!!.yawDegrees!!, 0.0)
        assertEquals(0.0, snapshot.localization!!.pitchDegrees!!, 0.0)
        assertEquals(-150.0, SphericalGuidanceView.wrapDegrees(210.0), 0.0)
    }

    private fun payload(
        version: Long,
        coverageVersion: Long = version,
        localizationVersion: Long = version,
        localization: String = "localized",
    ): JSONObject {
        val channels = JSONObject()
        GuidanceStateStore.CHANNELS.forEach { channels.put(it, 0.5) }
        val cell = JSONObject()
            .put("index", 0).put("row", 0).put("column", 0).put("state", "red")
            .put("channels", channels).put("reason_mask", JSONArray(listOf("missing_coverage")))
        val coverage = JSONObject().put("version", coverageVersion).put("width", 1).put("height", 1)
            .put("cells", JSONArray(listOf(cell)))
        val localizationObject = JSONObject().put("version", localizationVersion)
            .put("state", localization).put("confidence", if (localization == "localized") 0.9 else 0.0)
            .put("graph_version", version)
            .put("yaw_deg", 0.0).put("pitch_deg", 0.0)
        val target = JSONObject().put("version", version).put("target_id", "target-1")
            .put("yaw_deg", 30.0).put("pitch_deg", -5.0).put("priority", 0.8)
            .put("reasons", JSONArray(listOf("missing_coverage")))
        return JSONObject().put("schema", GuidanceStateStore.SCHEMA).put("version", version)
            .put("graph_version", version).put("coverage", coverage)
            .put("localization", localizationObject).put("target", target)
    }
}
