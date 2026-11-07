package com.recon.videoscan.live

import java.io.DataOutputStream
import java.io.File
import java.util.concurrent.ArrayBlockingQueue
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicLong
import java.util.zip.GZIPOutputStream

/** Bounded asynchronous evidence for diagnosing model shape versus trajectory errors. */
class DepthEvidenceRecorder(private val directory: File, private val limit: Long = 256L * 1024 * 1024) {
    data class Frame(val t: Double, val generation: Long, val revision: Long, val model: String,
        val reason: String, val fused: Boolean, val w: Int, val h: Int, val sourceW: Int, val sourceH: Int,
        val pose: FloatArray, val k: FloatArray, val sourceCalibration: DoubleArray,
        val landmarks: FloatArray, val prediction: FloatArray, val depth: ShortArray,
        val confidence: ByteArray, val rgba: ByteArray, val gray: ByteArray)
    private val queue = ArrayBlockingQueue<Frame>(2)
    private val dropped = AtomicLong()
    @Volatile private var accepting = true
    @Volatile var state = "recording"
        private set
    private val thread = Thread({
        var bytes = 0L; var count = 0
        try {
            directory.mkdirs()
            while (accepting || queue.isNotEmpty()) {
                val f = queue.poll(200, TimeUnit.MILLISECONDS) ?: continue
                // Conservative uncompressed budget; no deletion or unbounded backlog.
                val estimate = 4096L + 4L * (f.prediction.size + f.landmarks.size) + 2L * f.depth.size + f.confidence.size + f.rgba.size + f.gray.size
                if (bytes + estimate > limit) { state = "quota reached"; accepting = false; queue.clear(); break }
                val file = File(directory, "depth-$count.bin.gz")
                val pending = File(directory, "depth-$count.bin.gz.tmp")
                DataOutputStream(GZIPOutputStream(pending.outputStream().buffered()).buffered(64 * 1024)).use { out ->
                    out.writeInt(0x44505448); out.writeInt(1)
                    out.writeDouble(f.t); out.writeLong(f.generation); out.writeLong(f.revision)
                    out.writeUTF(f.model); out.writeUTF(f.reason); out.writeBoolean(f.fused)
                    out.writeInt(f.w); out.writeInt(f.h); out.writeInt(f.sourceW); out.writeInt(f.sourceH)
                    fun floats(a: FloatArray) { out.writeInt(a.size); a.forEach { out.writeFloat(it) } }
                    fun bytes(a: ByteArray) { out.writeInt(a.size); out.write(a) }
                    floats(f.pose); floats(f.k)
                    out.writeInt(f.sourceCalibration.size); f.sourceCalibration.forEach { out.writeDouble(it) }
                    floats(f.landmarks); floats(f.prediction)
                    out.writeInt(f.depth.size); f.depth.forEach { out.writeShort(it.toInt()) }
                    bytes(f.confidence); bytes(f.rgba); bytes(f.gray)
                }
                check(pending.renameTo(file)) { "could not publish evidence file" }
                count++
                bytes += estimate
            }
            if (state == "recording") state = "saved"
        } catch (e: Exception) { state = "error: ${e.message}"; accepting = false; queue.clear() }
        finally { runCatching { File(directory, "summary.txt").writeText("state=$state\nframes=$count\ndropped=${dropped.get()}\nbudget_bytes=$bytes\n") } }
    }, "depth-evidence").apply { isDaemon = true; start() }
    @Synchronized fun offer(frame: Frame) { if (accepting && !queue.offer(frame)) dropped.incrementAndGet() }
    @Synchronized fun close() { accepting = false }
    fun awaitClosed(timeoutMs: Long = 2000) { thread.join(timeoutMs) }
}
