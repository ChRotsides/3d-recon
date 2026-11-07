package com.recon.videoscan.live

import org.junit.Assert.*
import org.junit.Test

class StartupProgressTest {
    @Test fun `handheld guidance reaches a truthful five second deadline`() {
        val p=StartupProgress(); p.start(10.0,false)
        assertTrue(p.message(14.99).contains("Starting handheld"))
        assertTrue(p.message(15.0).contains("Not ready after 5 s"))
        assertFalse(p.message(15.0).contains("Support phone"))
    }
    @Test fun `failed automatic retries do not restart the startup countdown`() {
        val p=StartupProgress(); p.start(10.0,false); p.start(14.0,true)
        assertTrue(p.message(15.0).contains("Not ready after 5 s"))
        p.observe(true); assertEquals("",p.message(16.0))
        p.start(20.0,true); assertTrue(p.message(20.0).contains("0/5 s"))
    }
}
