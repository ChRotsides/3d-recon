package com.recon.poselogger.rgbscan

import android.graphics.Bitmap
import android.graphics.Canvas
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class GuidanceViewInstrumentationTest {
    @Test fun unlocalizedCoverageRendersWithoutDirectionalTarget() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        instrumentation.runOnMainSync {
            val channels = JSONObject()
            GuidanceStateStore.CHANNELS.forEach { channels.put(it, 0.5) }
            val cell = JSONObject().put("index", 0).put("row", 0).put("column", 0)
                .put("state", "amber").put("channels", channels)
                .put("reason_mask", JSONArray(listOf("weak_overlap")))
            val payload = JSONObject().put("schema", GuidanceStateStore.SCHEMA)
                .put("version", 2).put("graph_version", 2)
                .put("coverage", JSONObject().put("version", 2).put("width", 1)
                    .put("height", 1).put("cells", JSONArray(listOf(cell))))
                .put("localization", JSONObject().put("version", 2)
                    .put("state", "unlocalized").put("confidence", 0.0).put("graph_version", 2))
                .put("target", JSONObject().put("version", 2).put("target_id", "hidden")
                    .put("yaw_deg", 30).put("pitch_deg", 0).put("priority", 1)
                    .put("reasons", JSONArray(listOf("missing_coverage"))))
            val snapshot = GuidanceStateStore().accept(payload)!!
            val view = SphericalGuidanceView(instrumentation.targetContext)
            view.show(snapshot)
            view.measure(
                android.view.View.MeasureSpec.makeMeasureSpec(600, android.view.View.MeasureSpec.EXACTLY),
                android.view.View.MeasureSpec.makeMeasureSpec(240, android.view.View.MeasureSpec.EXACTLY),
            )
            view.layout(0, 0, 600, 240)
            view.draw(Canvas(Bitmap.createBitmap(600, 240, Bitmap.Config.ARGB_8888)))
            assertTrue(view.contentDescription.contains("Cannot locate this view"))
        }
    }

    @Test fun localizedHudShowsRelativeActionInsteadOfCoverageGrid() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        instrumentation.runOnMainSync {
            val channels = JSONObject()
            GuidanceStateStore.CHANNELS.forEach { channels.put(it, 0.0) }
            val cell = JSONObject().put("index", 0).put("row", 0).put("column", 0)
                .put("state", "red").put("channels", channels)
                .put("reason_mask", JSONArray(listOf("missing")))
            val payload = JSONObject().put("schema", GuidanceStateStore.SCHEMA)
                .put("version", 3).put("graph_version", 2)
                .put("coverage", JSONObject().put("version", 2).put("width", 1)
                    .put("height", 1).put("cells", JSONArray(listOf(cell))))
                .put("localization", JSONObject().put("version", 3)
                    .put("state", "localized").put("confidence", 0.9).put("graph_version", 2)
                    .put("yaw_deg", 0.0).put("pitch_deg", 0.0))
                .put("target", JSONObject().put("version", 3).put("target_id", "right")
                    .put("yaw_deg", 30.0).put("pitch_deg", 0.0).put("priority", 1)
                    .put("reasons", JSONArray(listOf("missing"))))
            val view = SphericalGuidanceView(instrumentation.targetContext)
            view.show(GuidanceStateStore().accept(payload)!!)
            view.measure(
                android.view.View.MeasureSpec.makeMeasureSpec(600, android.view.View.MeasureSpec.EXACTLY),
                android.view.View.MeasureSpec.makeMeasureSpec(240, android.view.View.MeasureSpec.EXACTLY),
            )
            view.layout(0, 0, 600, 240)
            view.draw(Canvas(Bitmap.createBitmap(600, 240, Bitmap.Config.ARGB_8888)))
            assertTrue(view.contentDescription.contains("Turn right 30°"))
        }
    }
}
