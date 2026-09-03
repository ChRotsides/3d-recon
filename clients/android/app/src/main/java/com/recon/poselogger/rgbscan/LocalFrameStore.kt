package com.recon.poselogger.rgbscan

import java.io.File
import java.util.UUID

internal data class LocalFrame(val id: String, val timestampNs: Long, val jpeg: File)

internal class LocalFrameStore(private val root: File) {
    @Synchronized
    fun persist(jpeg: ByteArray, timestampNs: Long = System.nanoTime()): LocalFrame {
        require(jpeg.size >= 4 && jpeg[0] == 0xff.toByte() && jpeg[1] == 0xd8.toByte())
        val frameId = "frame-${UUID.randomUUID()}"
        root.mkdirs()
        val temporary = File(root, ".$frameId.tmp")
        temporary.outputStream().use { output ->
            output.write(jpeg)
            output.flush()
            output.fd.sync()
        }
        val target = File(root, "$frameId.jpg")
        check(temporary.renameTo(target)) { "could not atomically persist frame" }
        return LocalFrame(frameId, timestampNs, target)
    }
}
