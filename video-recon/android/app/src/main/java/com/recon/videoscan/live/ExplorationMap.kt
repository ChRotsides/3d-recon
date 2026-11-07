package com.recon.videoscan.live

import kotlin.math.*

/** Top-down observed space; no inferred rooms or filling behind unseen walls. */
class ExplorationMap {
    data class Snapshot(val cells: FloatArray, val walls: FloatArray, val cellSize: Float)
    private var cell = .1f
    private val seen = HashSet<Long>()
    private val wallVotes = HashMap<Long, Int>()
    private fun key(x: Int, z: Int) = (x.toLong() shl 32) or (z.toLong() and 0xffffffffL)
    private fun x(k: Long) = (k shr 32).toInt()
    private fun z(k: Long) = k.toInt()
    private fun key(x: Float, z: Float) = key(floor(x / cell).toInt(), floor(z / cell).toInt())
    @Synchronized fun clear() { seen.clear(); wallVotes.clear(); cell = .1f }
    @Synchronized fun visit(px: Float, pz: Float) { if (px.isFinite() && pz.isFinite()) seen.add(key(px,pz)) }

    @Synchronized fun observe(depth: ShortArray, confidence: ByteArray, w: Int, h: Int, k: FloatArray, r: FloatArray, c: FloatArray) {
        val wallsThisView = HashSet<Long>()
        fun point(u: Int, v: Int): FloatArray? {
            val i=v*w+u; val d=(depth[i].toInt() and 65535)/1000f
            if (d !in .2f..6f || (confidence[i].toInt() and 255) != 255) return null
            return floatArrayOf((u-k[2])/k[0]*d,(v-k[3])/k[1]*d,d)
        }
        for (v in 2 until h-2 step 4) for (u in 2 until w-2 step 4) {
            val p=point(u,v) ?: continue
            val py=r[3]*p[0]+r[4]*p[1]+r[5]*p[2]
            // A horizontal slice near camera height avoids revealing space from
            // ceiling/floor rays or treating a table top as a room boundary.
            if (abs(py) > .6f) continue
            val px=c[0]+r[0]*p[0]+r[1]*p[1]+r[2]*p[2]
            val pz=c[2]+r[6]*p[0]+r[7]*p[1]+r[8]*p[2]
            reveal(c[0],c[2],px,pz)
            val a=point(u+2,v) ?: continue; val b=point(u,v+2) ?: continue
            if (abs(a[2]-p[2]) > .15f || abs(b[2]-p[2]) > .15f) continue
            val ax=a[0]-p[0];val ay=a[1]-p[1];val az=a[2]-p[2]
            val bx=b[0]-p[0];val by=b[1]-p[1];val bz=b[2]-p[2]
            val nx=ay*bz-az*by; val ny=az*bx-ax*bz; val nz=ax*by-ay*bx
            val norm=sqrt(nx*nx+ny*ny+nz*nz)
            if (norm > 1e-7f && abs(r[3]*nx+r[4]*ny+r[5]*nz)/norm < .3f)
                wallsThisView.add(key(px,pz))
        }
        for (wall in wallsThisView) wallVotes[wall]=minOf(3,(wallVotes[wall] ?: 0)+1)
        // Bound memory by reducing resolution across the entire explored area.
        if (seen.size > 60000) {
            val coarser=seen.map { key(Math.floorDiv(x(it),2),Math.floorDiv(z(it),2)) }.toHashSet()
            val wallCoarser=HashMap<Long,Int>()
            for ((id,votes) in wallVotes) { val id2=key(Math.floorDiv(x(id),2),Math.floorDiv(z(id),2));wallCoarser[id2]=maxOf(wallCoarser[id2] ?: 0,votes) }
            seen.clear();seen.addAll(coarser);wallVotes.clear();wallVotes.putAll(wallCoarser);cell*=2
        }
    }
    private fun reveal(sx: Float, sz: Float, ex: Float, ez: Float) {
        val count=maxOf(1,ceil(hypot(ex-sx,ez-sz)/(cell*.5f)).toInt()).coerceAtMost(256)
        for (i in 0..count) { val t=i.toFloat()/count;seen.add(key(sx+(ex-sx)*t,sz+(ez-sz)*t)) }
    }
    @Synchronized fun snapshot(): Snapshot {
        fun coordinates(ids: Collection<Long>): FloatArray {
            val out=FloatArray(ids.size*2);var i=0
            for (id in ids) {out[i++]=x(id)*cell;out[i++]=z(id)*cell};return out
        }
        return Snapshot(coordinates(seen),coordinates(wallVotes.filterValues { it >= 2 }.keys),cell)
    }
}
