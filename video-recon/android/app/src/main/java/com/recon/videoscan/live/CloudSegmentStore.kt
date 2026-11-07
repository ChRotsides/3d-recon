package com.recon.videoscan.live

import java.io.File
import java.io.IOException
import java.io.DataOutputStream
import java.io.RandomAccessFile

/** Completed segments have independent coordinate frames. Never silently join them. */
class CloudSegmentStore(private val root: File, private val quotaBytes: Long = 512L * 1024 * 1024) {
    fun save(writeSurface: (File) -> Long): File? {
        if (!root.isDirectory && !root.mkdirs()) throw IOException("cannot create segment folder")
        val used = root.walkTopDown().filter { it.isFile }.sumOf { it.length() }
        if (used >= quotaBytes) throw IOException("saved segment storage quota reached")
        val pending = File.createTempFile("segment-", ".partial", root)
        try {
            val count = writeSurface(pending)
            if (count < 0) throw IOException("surface export failed")
            if (count == 0L) { pending.delete(); return null }
            if (pending.length() == 0L || used + pending.length() > quotaBytes) throw IOException("segment exceeds storage quota")
            RandomAccessFile(pending, "rw").use { it.fd.sync() }
            val finished = File(root, pending.name.removeSuffix(".partial") + ".ply")
            if (finished.exists()) throw IOException("segment name collision")
            if (!pending.renameTo(finished)) throw IOException("could not commit saved segment")
            return finished
        } catch (e: Exception) {
            // Only our incomplete temporary file; caller still owns the live map.
            pending.delete()
            throw e
        }
    }
    companion object {
        fun writeRaw(file: File, points: List<Float>): Long {
            require(points.size % 6 == 0 && points.all { it.isFinite() })
            val count = points.size / 6
            DataOutputStream(file.outputStream().buffered()).use { out ->
                out.writeBytes("ply\nformat binary_little_endian 1.0\ncomment independent_local_frame not_aligned_to_other_segments\nelement vertex $count\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
                for (i in points.indices step 6) {
                    for (k in 0..2) out.writeInt(Integer.reverseBytes(java.lang.Float.floatToRawIntBits(points[i+k])))
                    for (k in 3..5) out.writeByte((points[i+k].coerceIn(0f, 1f) * 255 + .5f).toInt())
                }
            }
            return count.toLong()
        }
    }
}
