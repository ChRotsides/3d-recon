package com.recon.videoscan.live

import java.io.File
import org.junit.Assert.*
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder

class DiagnosticRecorderTest {
    @get:Rule val temporary = TemporaryFolder()
    @Test fun `records raw IMU and per-frame diagnostics in separate parseable CSVs`() {
        val dir = temporary.newFolder(); val r = DiagnosticRecorder(dir, 1.0)
        r.imu(1.1, 4, .1f, .2f, .3f)
        r.marker(1.1, "STILL")
        val h = TrackingHealth().current(1.1)
        r.frame(1.1, h, null, null, .02, DoubleArray(11), DoubleArray(60) { it.toDouble() })
        r.close(); r.awaitClosed()
        assertEquals(2, File(dir, "imu.csv").readLines().size)
        assertEquals(listOf("t,phase", "1.1,STILL"), File(dir, "events.csv").readLines())
        val frames = File(dir, "frames.csv").readLines()
        assertEquals(2, frames.size)
        assertEquals(frames[0].split(',').size, frames[1].split(',').size)
        val fields = frames[0].split(',').zip(frames[1].split(',')).toMap()
        assertEquals(91, fields.size)
        assertEquals("29.0", fields["startup_accepted"])
        assertEquals("36.0", fields["msckf_chi2"])
        assertEquals("39.0", fields["stationary_rejection_bits"])
        assertEquals("54.0", fields["observations_seeded"])
        assertTrue(File(dir, "summary.txt").readText().contains("dropped_records=0"))
    }
    @Test fun `recording stops at duration limit without deleting earlier evidence`() {
        val dir = temporary.newFolder(); val r = DiagnosticRecorder(dir, 1.0, duration = .5)
        r.imu(1.1, 1, 0f, 0f, 9.81f)
        r.imu(2.0, 1, 0f, 0f, 9.81f)
        r.awaitClosed()
        assertEquals(2, File(dir, "imu.csv").readLines().size)
        assertEquals("time limit", r.state)
    }
    @Test fun `size limit bounds recorded data`() {
        val dir = temporary.newFolder(); val r = DiagnosticRecorder(dir, 1.0, byteLimit = 1)
        r.imu(1.1, 1, 0f, 0f, 9.81f)
        r.awaitClosed()
        assertEquals(1, File(dir, "imu.csv").readLines().size)
        assertEquals("size limit", r.state)
    }
    @Test fun `ray samples are bounded timestamped and rate limited`() {
        val dir=temporary.newFolder(); val r=DiagnosticRecorder(dir,1.0)
        val sample=DoubleArray(17).apply { this[0]=1.0; this[14]=42.0; this[15]=.1; this[16]=-.2 }
        r.featureSamples(1.1,sample); r.featureSamples(1.15,sample); r.featureSamples(1.4,sample)
        r.featureSamples(1.7,DoubleArray(14).apply { this[0]=65.0 })
        r.close(); r.awaitClosed()
        assertEquals(listOf("t,id,xn,yn","1.1,42,0.1,-0.2","1.4,42,0.1,-0.2"),File(dir,"features.csv").readLines())
        val rows=File(dir,"geometry.csv").readLines(); assertEquals(3,rows.size)
        assertTrue(rows.all { it.split(',').size==14 })
    }
}
