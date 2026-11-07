package com.recon.videoscan.live

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.os.SystemClock
import android.os.Handler
import android.os.Looper
import android.util.AttributeSet
import android.view.View

/** Display model candidates on their own image, never stale points on a newer frame. */
class FeatureOverlayView @JvmOverloads constructor(context: Context, attrs: AttributeSet? = null) : View(context, attrs) {
    private val imagePaint = Paint(Paint.FILTER_BITMAP_FLAG)
    private val dot = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = Color.rgb(80, 255, 130) }
    private val text = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = Color.WHITE; textSize = 11 * resources.displayMetrics.scaledDensity }
    private val shade = Paint().apply { color = 0xcc000000.toInt() }
    private var bitmap: Bitmap? = null
    private var points = FloatArray(0)
    private var candidateCount = 0
    private var sourceWidth = 1
    private var sourceHeight = 1
    private var rotation = 90
    private var timestamp = 0.0
    private var showFeatures = true
    private val ui = Handler(Looper.getMainLooper())

    init {
        isClickable = true
        setOnClickListener { showFeatures = !showFeatures; invalidate() }
    }

    // UI thread only; RGB and candidates share the same tracking crop and timestamp.
    fun submit(rgba: ByteArray, rw: Int, rh: Int, width: Int, height: Int,
               candidates: FloatArray, time: Double, sensorRotation: Int) {
        require(rgba.size == rw * rh * 4 && candidates.size % 3 == 0 && width > 0 && height > 0)
        val pixels = IntArray(rw * rh) { i ->
            val o = i * 4
            Color.rgb(rgba[o].toInt() and 255, rgba[o + 1].toInt() and 255, rgba[o + 2].toInt() and 255)
        }
        if (bitmap?.width != rw || bitmap?.height != rh) bitmap = Bitmap.createBitmap(rw, rh, Bitmap.Config.ARGB_8888)
        bitmap!!.setPixels(pixels, 0, rw, 0, 0, rw, rh)
        candidateCount = candidates.size / 3
        points = candidates.copyOfRange(0, minOf(candidates.size, 180 * 3))
        sourceWidth = width; sourceHeight = height; timestamp = time; rotation = sensorRotation
        invalidate()
    }

    // Recovery can originate on the VIO worker. Serialize the entire state
    // change with submit/onDraw; clearing points mid-draw invalidates its indices.
    fun clear() {
        if (Looper.myLooper() != Looper.getMainLooper()) { ui.post { clear() }; return }
        bitmap = null; points = FloatArray(0); candidateCount = 0; timestamp = 0.0; invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val frame = bitmap
        if (showFeatures && frame != null) {
            canvas.drawColor(Color.BLACK)
            val sideways = rotation % 180 != 0
            val scale = minOf(width.toFloat() / (if (sideways) frame.height else frame.width),
                height.toFloat() / (if (sideways) frame.width else frame.height))
            if (scale > 0f) {
                canvas.save()
                canvas.translate(width / 2f, height / 2f)
                canvas.rotate(rotation.toFloat())
                canvas.scale(scale, scale)
                canvas.translate(-frame.width / 2f, -frame.height / 2f)
                canvas.drawBitmap(frame, 0f, 0f, imagePaint)
                val radius = 1.3f * resources.displayMetrics.density / scale
                for (i in points.indices step 3) {
                    val x = (points[i] + 0.5f) * frame.width / sourceWidth - 0.5f
                    val y = (points[i + 1] + 0.5f) * frame.height / sourceHeight - 0.5f
                    if (x.isFinite() && y.isFinite()) canvas.drawCircle(x, y, radius, dot)
                }
                canvas.restore()
            }
        }
        val age = maxOf(0.0, SystemClock.elapsedRealtimeNanos() * 1e-9 - timestamp)
        val label = when {
            !showFeatures -> "Live camera · tap for features"
            frame == null -> "XFeat · waiting for detection"
            else -> "XFeat · ${points.size / 3}/$candidateCount shown"
        }
        val lineHeight = text.textSize * 1.5f
        canvas.drawRect(0f, 0f, width.toFloat(), lineHeight * 3 + 8f, shade)
        canvas.drawText(label, 6f, lineHeight, text)
        if (showFeatures) {
            canvas.drawText(if (frame == null) "No model frame yet" else "Sampled frame · ${"%.1f".format(age)}s ago", 6f, lineHeight * 2, text)
            canvas.drawText("Tap for live camera", 6f, lineHeight * 3, text)
        }
        if (isShown && showFeatures && frame != null) postInvalidateDelayed(250)
    }
}
