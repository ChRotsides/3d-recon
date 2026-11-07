package com.recon.videoscan.live

import org.junit.Assert.*
import org.junit.Test
import java.nio.file.Files
import java.io.IOException

class RecoveryStorageTest {
    @Test fun `saved clouds are separate atomic files with exact point payload`() {
        val root = Files.createTempDirectory("segments-test").toFile()
        val points = listOf(1f,2f,3f,1f,0f,.5f)
        val store = CloudSegmentStore(root)
        val a = store.save { CloudSegmentStore.writeRaw(it, points) }!!
        val b = store.save { CloudSegmentStore.writeRaw(it, points) }!!
        assertNotEquals(a, b)
        val bytes = a.readBytes(); val header = bytes.toString(Charsets.ISO_8859_1).substringBefore("end_header\n") + "end_header\n"
        assertTrue(header.contains("independent_local_frame"))
        assertEquals(header.length + 15, bytes.size)
        assertEquals(2, root.listFiles()!!.size)
        assertArrayEquals(a.readBytes(), b.readBytes())
    }
    @Test fun `failed export does not replace existing cloud and does not invoke destruction`() {
        val root = Files.createTempDirectory("segments-test").toFile()
        val store = CloudSegmentStore(root)
        val saved = store.save { it.writeText("existing cloud"); 1 }!!
        var destroyed = false
        try { store.save { it.writeText("incomplete"); throw IOException("disk failure") }; destroyed = true } catch (_: IOException) {}
        assertFalse(destroyed)
        assertEquals("existing cloud", saved.readText())
        assertEquals(1, root.listFiles()!!.size)
    }
    @Test fun `quota failure preserves existing data and empty clouds do not make segments`() {
        val root = Files.createTempDirectory("segments-test").toFile()
        val store = CloudSegmentStore(root, 20)
        assertNull(store.save { 0 })
        try { store.save { it.writeText("x".repeat(21)); 1 }; fail("quota bypass") } catch (_: IOException) {}
        assertTrue(root.listFiles()!!.isEmpty())
    }
    @Test fun `warm bias requires sustained stationary corrections and expires`() {
        val cache = BiasWarmStart(); val value = doubleArrayOf(.0002,.0003,.0004,1e-8,1e-8,1e-8)
        cache.observe(1.0,true,true,.001) { value }
        assertTrue(cache.prior(1.5).isEmpty())
        cache.observe(2.1,true,true,.001) { value }
        assertEquals(7,cache.prior(3.0).size)
        assertEquals(.0002,cache.prior(3.0)[0],1e-12)
        assertTrue(cache.prior(3.0)[3]>=1e-6)
        cache.observe(4.0,false,true,2.0) { error("must not read a lost state") }
        assertEquals(.0002,cache.prior(4.0)[0],1e-12)
        assertTrue(cache.prior(123.0).isEmpty())
    }
    @Test fun `bad warm bias and interrupted stationary evidence are rejected`() {
        val cache = BiasWarmStart()
        cache.observe(1.0,true,true,.001) { doubleArrayOf() }
        cache.observe(1.5,false,false,.001) { error("not stationary") }
        cache.observe(2.1,true,true,.001) { error("hold restarted") }
        cache.observe(3.2,true,true,.001) { doubleArrayOf(.1,0.0,0.0,0.0,0.0,0.0) }
        assertTrue(cache.prior(3.3).isEmpty())
    }
}
