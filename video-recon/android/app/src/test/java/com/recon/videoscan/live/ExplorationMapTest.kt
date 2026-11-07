package com.recon.videoscan.live

import org.junit.Assert.*
import org.junit.Test

class ExplorationMapTest {
    private val w=64; private val h=48
    private val k=floatArrayOf(60f,60f,32f,24f)
    private val r=floatArrayOf(1f,0f,0f,0f,1f,0f,0f,0f,1f)
    private val c=floatArrayOf(0f,0f,0f)
    private val depth=ShortArray(w*h) { 2000 }
    @Test fun `wall observations reveal space up to wall and require repeated support for outline`() {
        val map=ExplorationMap();val confidence=ByteArray(w*h) { -1 }
        map.observe(depth,confidence,w,h,k,r,c)
        assertTrue(map.snapshot().cells.isNotEmpty());assertTrue(map.snapshot().walls.isEmpty())
        map.observe(depth,confidence,w,h,k,r,c)
        val s=map.snapshot();assertTrue(s.walls.isNotEmpty())
        for(i in 1 until s.cells.size step 2) assertTrue("revealed behind wall",s.cells[i] <= 2.01f)
        for(i in 1 until s.walls.size step 2) assertEquals(2f,s.walls[i],.11f)
        map.clear();assertTrue(map.snapshot().cells.isEmpty());assertTrue(map.snapshot().walls.isEmpty())
    }
    @Test fun `provisional depth cannot reveal unknown room or create walls`() {
        val map=ExplorationMap()
        repeat(4) { map.observe(depth,ByteArray(w*h) { 64 },w,h,k,r,c) }
        assertTrue(map.snapshot().cells.isEmpty());assertTrue(map.snapshot().walls.isEmpty())
        map.visit(-.2f,-.3f);assertEquals(2,map.snapshot().cells.size)
    }
    @Test fun `ceiling does not become a wall outline`() {
        val map=ExplorationMap();val tilted=floatArrayOf(1f,0f,0f,0f,0f,-1f,0f,1f,0f)
        repeat(3) { map.observe(depth,ByteArray(w*h) { -1 },w,h,k,tilted,c) }
        assertTrue(map.snapshot().walls.isEmpty())
    }
}
