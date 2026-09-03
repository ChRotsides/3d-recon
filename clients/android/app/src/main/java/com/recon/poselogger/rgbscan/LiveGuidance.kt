package com.recon.poselogger.rgbscan

import org.json.JSONObject

internal enum class CoverageState { RED, AMBER, GREEN }

internal enum class LocalizationState { LOCALIZED, UNLOCALIZED, NOT_REQUESTED }

internal data class CoverageCell(
    val index: Int,
    val row: Int,
    val column: Int,
    val state: CoverageState,
    val channels: Map<String, Double>,
    val reasonMask: List<String>,
)

internal data class GuidanceCoverage(
    val version: Long,
    val width: Int,
    val height: Int,
    val cells: List<CoverageCell>,
)

internal data class GuidanceLocalization(
    val version: Long,
    val state: LocalizationState,
    val confidence: Double,
    val graphVersion: Long,
    val yawDegrees: Double? = null,
    val pitchDegrees: Double? = null,
    val reasons: List<String> = emptyList(),
)

internal data class GuidanceTarget(
    val version: Long,
    val id: String,
    val yawDegrees: Double,
    val pitchDegrees: Double,
    val priority: Double,
    val reasons: List<String>,
)

internal data class GuidanceSnapshot(
    val version: Long = -1,
    val graphVersion: Long = -1,
    val coverage: GuidanceCoverage? = null,
    val localization: GuidanceLocalization? = null,
    val target: GuidanceTarget? = null,
) {
    val targetVisible: Boolean
        get() = localization?.state == LocalizationState.LOCALIZED &&
            localization.yawDegrees != null && localization.pitchDegrees != null && target != null
}

/**
 * Applies independently versioned Phase-2 assets without allowing a delayed response to roll
 * any visible asset backwards. The service response is immutable after parsing.
 */
internal class GuidanceStateStore {
    private var value = GuidanceSnapshot()
    private var targetVersion = -1L

    @Synchronized
    fun current(): GuidanceSnapshot = value

    @Synchronized
    fun accept(json: JSONObject): GuidanceSnapshot? {
        require(json.getString("schema") == SCHEMA) { "unsupported guidance schema" }
        val responseVersion = json.getLong("version")
        if (responseVersion <= value.version) return null

        val candidateCoverage = json.optJSONObject("coverage")?.let(::parseCoverage)
        val candidateLocalization = json.optJSONObject("localization")?.let(::parseLocalization)
        val candidateTarget = if (json.isNull("target")) null else json.optJSONObject("target")?.let(::parseTarget)

        val coverage = when {
            candidateCoverage == null -> value.coverage
            value.coverage == null || candidateCoverage.version > value.coverage!!.version -> candidateCoverage
            else -> value.coverage
        }
        val localization = when {
            candidateLocalization == null -> value.localization
            value.localization == null || candidateLocalization.version > value.localization!!.version -> candidateLocalization
            else -> value.localization
        }
        val nextTargetVersion = candidateTarget?.version ?: responseVersion
        val target = if (nextTargetVersion > targetVersion) {
            targetVersion = nextTargetVersion
            candidateTarget
        } else {
            value.target
        }

        value = GuidanceSnapshot(
            version = responseVersion,
            graphVersion = maxOf(value.graphVersion, json.optLong("graph_version", -1)),
            coverage = coverage,
            localization = localization,
            target = target,
        )
        return value
    }

    private fun parseCoverage(json: JSONObject): GuidanceCoverage {
        val width = json.getInt("width")
        val height = json.getInt("height")
        require(width > 0 && height > 0) { "invalid coverage dimensions" }
        val source = json.getJSONArray("cells")
        val cells = ArrayList<CoverageCell>(source.length())
        for (offset in 0 until source.length()) {
            val cell = source.getJSONObject(offset)
            val channelsJson = cell.getJSONObject("channels")
            val channels = CHANNELS.associateWith { name -> channelsJson.getDouble(name) }
            val reasonsJson = cell.getJSONArray("reason_mask")
            val reasons = (0 until reasonsJson.length()).map(reasonsJson::getString)
            val state = when (cell.getString("state")) {
                "red" -> CoverageState.RED
                "amber" -> CoverageState.AMBER
                "green" -> CoverageState.GREEN
                else -> error("unknown coverage state")
            }
            val index = cell.getInt("index")
            val row = cell.getInt("row")
            val column = cell.getInt("column")
            require(index in 0 until width * height && row in 0 until height && column in 0 until width) {
                "coverage cell outside grid"
            }
            cells += CoverageCell(index, row, column, state, channels, reasons)
        }
        return GuidanceCoverage(json.getLong("version"), width, height, cells)
    }

    private fun parseLocalization(json: JSONObject): GuidanceLocalization {
        val state = when (json.getString("state")) {
            "localized" -> LocalizationState.LOCALIZED
            "unlocalized" -> LocalizationState.UNLOCALIZED
            "not_requested" -> LocalizationState.NOT_REQUESTED
            else -> error("unknown localization state")
        }
        val confidence = json.getDouble("confidence")
        require(confidence in 0.0..1.0) { "localization confidence outside [0, 1]" }
        val reasonsJson = json.optJSONArray("reason_mask")
        val reasons = if (reasonsJson == null) emptyList() else
            (0 until reasonsJson.length()).map(reasonsJson::getString)
        val yaw = if (json.has("yaw_deg")) json.getDouble("yaw_deg") else null
        val pitch = if (json.has("pitch_deg")) json.getDouble("pitch_deg") else null
        require(yaw == null || yaw.isFinite()) { "non-finite localization yaw" }
        require(pitch == null || pitch.isFinite()) { "non-finite localization pitch" }
        return GuidanceLocalization(
            json.getLong("version"), state, confidence, json.getLong("graph_version"),
            yaw, pitch, reasons,
        )
    }

    private fun parseTarget(json: JSONObject): GuidanceTarget {
        val reasonsJson = json.getJSONArray("reasons")
        return GuidanceTarget(
            version = json.getLong("version"),
            id = json.getString("target_id"),
            yawDegrees = json.getDouble("yaw_deg"),
            pitchDegrees = json.getDouble("pitch_deg"),
            priority = json.getDouble("priority"),
            reasons = (0 until reasonsJson.length()).map(reasonsJson::getString),
        )
    }

    companion object {
        const val SCHEMA = "live-rgb-guidance-v2"
        val CHANNELS = listOf(
            "photographed", "independent_views", "edge_strength", "feature_spread",
            "reprojection", "cycle_consistency", "sharpness", "exposure",
            "parallax_risk", "seam_support",
        )
    }
}
