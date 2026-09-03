package com.recon.poselogger.rgbscan

import org.json.JSONObject
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.security.MessageDigest
import java.util.zip.GZIPInputStream

data class PointCloudAsset(
    val positions: FloatArray,
    val colours: ByteArray,
    val count: Int,
    val version: Int = 0,
    val confidence: FloatArray? = null,
    val panoramaSupport: FloatArray? = null,
    val sourcePanoramaVersion: Int = 0,
)

object PointCloudDecoder {
    const val FINAL_SCHEMA = "live-rgb-cloud-v1"
    const val SCHEMA = FINAL_SCHEMA // Phase 1 instrumentation compatibility
    const val DRAFT_SCHEMA = "live-rgb-draft-cloud-v3"
    private const val DRAFT_RECORD_BYTES = 28
    private const val MAX_DRAFT_POINTS = 250_000
    private const val MAX_TRANSPORT_BYTES = 12 * 1024 * 1024

    fun decode(metadataFile: File, binaryFile: File): PointCloudAsset {
        val metadata = JSONObject(metadataFile.readText())
        return when (metadata.getString("schema")) {
            FINAL_SCHEMA -> decodeFinal(metadata, binaryFile)
            DRAFT_SCHEMA -> decodeDraft(metadata, binaryFile)
            else -> error("unsupported cloud schema")
        }
    }

    private fun decodeFinal(metadata: JSONObject, binaryFile: File): PointCloudAsset {
        val count = metadata.getInt("count")
        require(count in 1..800_000) { "invalid point count" }
        val payload = binaryFile.readBytes()
        require(payload.size == count * 15) { "cloud byte length mismatch" }
        require(sha256(payload) == metadata.getString("asset_sha256")) { "cloud checksum mismatch" }
        val positions = FloatArray(count * 3)
        ByteBuffer.wrap(payload, 0, count * 12).order(ByteOrder.LITTLE_ENDIAN)
            .asFloatBuffer().get(positions)
        require(positions.all { it.isFinite() }) { "non-finite cloud coordinate" }
        val low = metadata.getJSONArray("min")
        val high = metadata.getJSONArray("max")
        positions.forEachIndexed { index, value ->
            val axis = index % 3
            require(value >= low.getDouble(axis) - 1e-5 && value <= high.getDouble(axis) + 1e-5) {
                "cloud coordinate outside declared bounds"
            }
        }
        return PointCloudAsset(positions, payload.copyOfRange(count * 12, payload.size), count)
    }

    private fun decodeDraft(metadata: JSONObject, binaryFile: File): PointCloudAsset {
        val count = metadata.getInt("count")
        val version = metadata.getInt("version")
        val sourceVersion = metadata.getInt("source_panorama_version")
        require(version > 0 && sourceVersion > 0) { "invalid draft version" }
        require(count in 1..MAX_DRAFT_POINTS) { "invalid draft point count" }
        require(metadata.getInt("record_bytes") == DRAFT_RECORD_BYTES) { "invalid draft stride" }
        require(metadata.getString("compression") == "gzip") { "unsupported draft compression" }
        val compressed = binaryFile.readBytes()
        require(compressed.size <= MAX_TRANSPORT_BYTES) { "draft transport is too large" }
        require(compressed.size == metadata.getInt("asset_bytes")) { "draft byte length mismatch" }
        require(sha256(compressed) == metadata.getString("asset_sha256")) {
            "draft checksum mismatch"
        }
        val expectedDecoded = count * DRAFT_RECORD_BYTES
        require(expectedDecoded == metadata.getInt("decoded_bytes")) { "invalid decoded draft length" }
        val decoded = GZIPInputStream(compressed.inputStream()).use { input ->
            val output = java.io.ByteArrayOutputStream(expectedDecoded)
            val buffer = ByteArray(32 * 1024)
            while (true) {
                val read = input.read(buffer)
                if (read < 0) break
                require(output.size() + read <= expectedDecoded) { "draft expands beyond declared size" }
                output.write(buffer, 0, read)
            }
            output.toByteArray()
        }
        require(decoded.size == expectedDecoded) { "decoded draft byte length mismatch" }
        require(sha256(decoded) == metadata.getString("decoded_sha256")) {
            "decoded draft checksum mismatch"
        }
        val positions = FloatArray(count * 3)
        val colours = ByteArray(count * 3)
        val confidence = FloatArray(count)
        val support = FloatArray(count)
        val wire = ByteBuffer.wrap(decoded).order(ByteOrder.LITTLE_ENDIAN)
        repeat(count) { point ->
            repeat(3) { axis -> positions[point * 3 + axis] = wire.float }
            repeat(3) { channel -> colours[point * 3 + channel] = wire.get() }
            wire.get() // explicit padding byte in the cross-platform record
            confidence[point] = wire.float
            support[point] = wire.float
            require(wire.int == sourceVersion) { "mixed source versions in one draft" }
        }
        require(positions.all { it.isFinite() }) { "non-finite draft coordinate" }
        require(confidence.all { it.isFinite() }) { "non-finite draft confidence" }
        require(support.all { it.isFinite() && it in 0f..1f }) { "invalid panorama support" }
        validateBounds(metadata, positions)
        return PointCloudAsset(
            positions = positions,
            colours = colours,
            count = count,
            version = version,
            confidence = confidence,
            panoramaSupport = support,
            sourcePanoramaVersion = sourceVersion,
        )
    }

    private fun validateBounds(metadata: JSONObject, positions: FloatArray) {
        val low = metadata.getJSONArray("min")
        val high = metadata.getJSONArray("max")
        positions.forEachIndexed { index, value ->
            val axis = index % 3
            require(value >= low.getDouble(axis) - 1e-5 && value <= high.getDouble(axis) + 1e-5) {
                "cloud coordinate outside declared bounds"
            }
        }
    }

    private fun sha256(bytes: ByteArray): String = MessageDigest.getInstance("SHA-256")
        .digest(bytes).joinToString("") { "%02x".format(it) }
}
