package com.recon.videoscan.live

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.util.AttributeSet
import android.view.View

class ExplorationMapView @JvmOverloads constructor(context: Context, attrs: AttributeSet? = null) : View(context, attrs) {
    private val paint=Paint(Paint.ANTI_ALIAS_FLAG)
    @Volatile private var map=ExplorationMap.Snapshot(FloatArray(0),FloatArray(0),.1f)
    private data class Position(val x: Float,val z: Float)
    @Volatile private var position: Position?=null
    @Volatile var tracking=false
        set(value) { field=value;postInvalidate() }
    fun submit(snapshot: ExplorationMap.Snapshot) { map=snapshot;postInvalidate() }
    fun position(x: Float,z: Float) { position=Position(x,z);postInvalidate() }
    fun clear() { map=ExplorationMap.Snapshot(FloatArray(0),FloatArray(0),.1f);position=null;postInvalidate() }
    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas);canvas.drawColor(Color.BLACK)
        val s=map;val p=position;val density=resources.displayMetrics.density
        var minX=(p?.x ?: 0f)-2;var maxX=minX+4
        var minZ=(p?.z ?: 0f)-2;var maxZ=minZ+4
        for(i in s.cells.indices step 2) {minX=minOf(minX,s.cells[i]-.5f);maxX=maxOf(maxX,s.cells[i]+.5f);minZ=minOf(minZ,s.cells[i+1]-.5f);maxZ=maxOf(maxZ,s.cells[i+1]+.5f)}
        val top=26*density;val bottom=22*density
        val scale=minOf((width-16*density)/(maxX-minX),(height-top-bottom)/(maxZ-minZ))
        if(scale<=0) return
        val ox=width/2f-(minX+maxX)/2*scale;val oz=top+(height-top-bottom)/2-(minZ+maxZ)/2*scale
        paint.color=Color.LTGRAY
        for(i in s.cells.indices step 2) canvas.drawRect(ox+s.cells[i]*scale,oz+s.cells[i+1]*scale,ox+(s.cells[i]+s.cellSize)*scale+.5f,oz+(s.cells[i+1]+s.cellSize)*scale+.5f,paint)
        // White marks trace only repeatedly observed vertical surfaces.
        paint.color=Color.WHITE
        for(i in s.walls.indices step 2) canvas.drawRect(ox+s.walls[i]*scale,oz+s.walls[i+1]*scale,ox+(s.walls[i]+s.cellSize)*scale,oz+(s.walls[i+1]+s.cellSize)*scale,paint)
        if(p!=null) {
            paint.color=Color.BLACK;canvas.drawCircle(ox+p.x*scale,oz+p.z*scale,6*density,paint)
            paint.color=Color.rgb(70,255,100);canvas.drawCircle(ox+p.x*scale,oz+p.z*scale,4*density,paint)
        }
        paint.color=Color.WHITE;paint.textSize=11*density
        canvas.drawText(if(p==null) "2D MAP · waiting for pose" else if(tracking) "2D MAP · you" else "2D MAP · position held",8*density,17*density,paint)
        paint.textSize=9*density
        canvas.drawText("Grey: explored · white: walls",8*density,height-7*density,paint)
    }
}
