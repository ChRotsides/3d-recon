package com.recon.poselogger.rgbscan

import org.json.JSONObject
import java.io.File
import java.net.HttpURLConnection
import java.net.URL
import java.nio.file.Files
import java.nio.file.StandardCopyOption

internal class RgbProtocol(private val server: String) {
    fun createSession(sessionId: String) = json(
        "POST", "/v1/sessions", JSONObject().put("session_id", sessionId).toString()
    )

    fun createStation(sessionId: String, stationId: String) = json(
        "POST", "/v1/stations", JSONObject()
            .put("session_id", sessionId).put("station_id", stationId).toString()
    )

    fun upload(sessionId: String, stationId: String, frameId: String, timestampNs: Long, jpeg: File): JSONObject {
        val connection = open("PUT", "/v1/frames")
        connection.setRequestProperty("Content-Type", "image/jpeg")
        connection.setRequestProperty(
            "X-RGB-Frame",
            JSONObject().put("session_id", sessionId).put("station_id", stationId)
                .put("frame_id", frameId).put("timestamp_ns", timestampNs).toString(),
        )
        connection.doOutput = true
        jpeg.inputStream().use { input -> connection.outputStream.use { input.copyTo(it) } }
        requireSuccess(connection)
        return JSONObject(connection.inputStream.bufferedReader().use { it.readText() })
    }

    fun uploadPreview(
        sessionId: String,
        stationId: String,
        previewId: String,
        timestampNs: Long,
        jpeg: File,
    ): JSONObject {
        val connection = open("PUT", "/v1/previews", 30_000)
        connection.setRequestProperty("Content-Type", "image/jpeg")
        connection.setRequestProperty(
            "X-RGB-Preview",
            JSONObject().put("session_id", sessionId).put("station_id", stationId)
                .put("preview_id", previewId).put("timestamp_ns", timestampNs).toString(),
        )
        connection.doOutput = true
        jpeg.inputStream().use { input -> connection.outputStream.use { input.copyTo(it) } }
        requireSuccess(connection)
        return JSONObject(connection.inputStream.bufferedReader().use { it.readText() })
    }

    fun guidance(sessionId: String, stationId: String): JSONObject {
        val connection = open("GET", "/v1/guidance/$sessionId/$stationId", 5_000)
        requireSuccess(connection)
        return JSONObject(connection.inputStream.bufferedReader().use { it.readText() })
    }

    fun latestDraft(sessionId: String, stationId: String): JSONObject {
        val connection = open("GET", "/v1/drafts/$sessionId/$stationId/latest", 5_000)
        requireSuccess(connection)
        return JSONObject(connection.inputStream.bufferedReader().use { it.readText() })
    }

    fun downloadDraft(
        sessionId: String,
        stationId: String,
        version: Int,
        name: String,
        target: File,
    ) {
        val connection = open("GET", "/v1/drafts/$sessionId/$stationId/$version/$name")
        requireSuccess(connection)
        target.parentFile?.mkdirs()
        val temporary = File(target.parentFile, ".${target.name}.part")
        try {
            connection.inputStream.use { input -> temporary.outputStream().use { input.copyTo(it) } }
            Files.move(
                temporary.toPath(),
                target.toPath(),
                StandardCopyOption.ATOMIC_MOVE,
                StandardCopyOption.REPLACE_EXISTING,
            )
        } finally {
            temporary.delete()
        }
    }

    fun finalize(sessionId: String, stationId: String): JSONObject = JSONObject(
        json("POST", "/v1/stations/finalize", JSONObject()
            .put("session_id", sessionId).put("station_id", stationId).toString())
    )

    fun download(sessionId: String, stationId: String, version: Int, name: String, target: File) {
        val connection = open("GET", "/v1/results/$sessionId/$stationId/$version/$name")
        requireSuccess(connection)
        target.parentFile?.mkdirs()
        connection.inputStream.use { input -> target.outputStream().use { input.copyTo(it) } }
    }

    private fun json(method: String, path: String, body: String): String {
        val connection = open(method, path)
        connection.setRequestProperty("Content-Type", "application/json")
        connection.doOutput = true
        connection.outputStream.bufferedWriter().use { it.write(body) }
        requireSuccess(connection)
        return connection.inputStream.bufferedReader().use { it.readText() }
    }

    private fun open(method: String, path: String, responseTimeoutMs: Int = 30 * 60_000): HttpURLConnection =
        (URL(server.trimEnd('/') + path).openConnection() as HttpURLConnection).apply {
            requestMethod = method
            connectTimeout = 10_000
            readTimeout = responseTimeoutMs
        }

    private fun requireSuccess(connection: HttpURLConnection) {
        if (connection.responseCode !in 200..299) {
            val message = connection.errorStream?.bufferedReader()?.use { it.readText() }
            throw IllegalStateException("HTTP ${connection.responseCode}: $message")
        }
    }
}
