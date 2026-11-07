package com.recon.videoscan.live

import org.junit.Assert.*
import org.junit.Test

class FrameMailboxTest {
    @Test fun `slow inference cannot grow a backlog and receives newest frame`() {
        val mailbox = FrameMailbox<Int>()
        assertTrue(mailbox.offer(1))
        assertEquals(1, mailbox.poll())
        for (frame in 2..1000) assertFalse(mailbox.offer(frame))
        assertEquals(998L, mailbox.dropped)
        assertEquals(1000, mailbox.poll())
        assertNull(mailbox.poll())
        assertTrue(mailbox.offer(1001))
        assertEquals(1001, mailbox.poll())
        assertNull(mailbox.poll())
    }

    @Test fun `reset clears pending frames without scheduling a second consumer`() {
        val mailbox = FrameMailbox<String>()
        assertTrue(mailbox.offer("old"))
        mailbox.clear()
        assertFalse(mailbox.offer("new"))
        assertEquals("new", mailbox.poll())
        assertNull(mailbox.poll())
        assertTrue(mailbox.offer("next"))
    }
}
