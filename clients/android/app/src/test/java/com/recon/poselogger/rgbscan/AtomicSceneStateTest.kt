package com.recon.poselogger.rgbscan

import org.junit.Assert.assertEquals
import org.junit.Test

class AtomicSceneStateTest {
    @Test fun aSceneCommitPublishesVersionAndCountTogether() {
        val state = AtomicSceneState()
        state.commit(3, 42)
        assertEquals(RenderedSceneSnapshot(3, 42), state.snapshot())
        state.commit(7, 99)
        assertEquals(RenderedSceneSnapshot(7, 99), state.snapshot())
    }
}
