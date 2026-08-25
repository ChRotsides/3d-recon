package com.recon.poselogger

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.graphics.Typeface
import android.util.AttributeSet
import android.util.TypedValue
import android.view.View

/**
 * A deliberately small capture HUD. The depth ghost and the current 3D target
 * do the spatial work; this view presents only status and the single action the
 * user should take next.
 */
class GuidanceHud @JvmOverloads constructor(
    context: Context, attrs: AttributeSet? = null, defStyle: Int = 0,
) : View(context, attrs, defStyle) {

    data class State(
        val statusLine: String = "",
        val recording: Boolean = false,
        /**
         * How to get back to where tracking worked. Set only while tracking is
         * lost, and drawn above everything else — while it is showing, nothing
         * else on screen is being collected.
         */
        val recovery: String? = null,
        val complete: Boolean = false,
        /**
         * Relocalisations so far. Shown because each one rebases ARCore's world,
         * and a bundle spanning one cannot be fused as a single room — the user
         * is the only one who can decide to walk it again, and only if told.
         */
        val trackingBreaks: Int = 0,
        /** "3 of 12" — a countable instruction, which a percentage is not. */
        val shotTaken: Int = 0,
        val shotTotal: Int = 0,
        /** Shots given up on, shown beside the count so it cannot flatter. */
        val shotSkipped: Int = 0,
        /** The only non-critical instruction visible on the capture screen. */
        val primaryInstruction: String? = null,
        val primaryReady: Boolean = false,
    )

    var state: State = State()
        set(value) {
            field = value
            invalidate()
    }

    private val panel = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = 0xB3000000.toInt() }
    private val status = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE
        textSize = sp(14f)
        typeface = Typeface.MONOSPACE
    }
    private val banner = Paint(Paint.ANTI_ALIAS_FLAG)
    private val bannerText = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.BLACK
        textSize = sp(14f)
        typeface = Typeface.create(Typeface.DEFAULT, Typeface.BOLD)
        textAlign = Paint.Align.CENTER
    }
    private val rect = RectF()

    override fun onDraw(canvas: Canvas) {
        val s = state
        val pad = dp(14f)
        val panelHeight = dp(50f)
        rect.set(0f, 0f, width.toFloat(), panelHeight)
        canvas.drawRect(rect, panel)

        // Clipped rather than ellipsised is how this line used to fail: the
        // status ran past the right edge and the last field silently vanished,
        // so a reader could not tell a short status from a truncated one.
        canvas.drawText(ellipsise(s.statusLine, status, width - pad * 2), pad, dp(30f), status)
        if (!s.recording) return

        val progress = when {
            s.shotTotal <= 0 -> ""
            s.shotSkipped > 0 -> " · ${s.shotTaken}/${s.shotTotal} · ${s.shotSkipped} skipped"
            else -> " · ${s.shotTaken}/${s.shotTotal}"
        }
        val instruction: Pair<String, Int>? = when {
            s.recovery != null -> s.recovery to RED
            s.trackingBreaks > 0 ->
                "Tracking reset — restart this capture" to RED
            s.complete -> "Room complete — you can stop" to GREEN
            s.primaryInstruction != null ->
                "${s.primaryInstruction}$progress" to if (s.primaryReady) GREEN else AMBER
            else -> null
        }
        instruction?.let { (text, colour) ->
            drawBanner(canvas, text, colour, panelHeight + dp(10f))
        }
    }

    private fun drawBanner(canvas: Canvas, text: String, color: Int, y: Float) {
        val pad = dp(14f)
        banner.color = color
        rect.set(pad, y, width - pad, y + dp(36f))
        canvas.drawRoundRect(rect, dp(8f), dp(8f), banner)
        canvas.drawText(
            ellipsise(text, bannerText, width - pad * 4),
            width / 2f,
            y + dp(23f),
            bannerText,
        )
    }

    /**
     * Trim `text` to `maxWidth`, marking the cut with an ellipsis.
     *
     * Canvas.drawText does not clip, it just draws past the edge, which is how
     * the status line lost its last field on a 1080 px phone without anything
     * looking wrong. An explicit ellipsis at least says "there was more".
     */
    private fun ellipsise(text: String, paint: Paint, maxWidth: Float): String {
        if (maxWidth <= 0f || paint.measureText(text) <= maxWidth) return text
        val ellipsis = "…"
        val budget = maxWidth - paint.measureText(ellipsis)
        if (budget <= 0f) return ellipsis
        var end = text.length
        while (end > 0 && paint.measureText(text, 0, end) > budget) end--
        return text.substring(0, end).trimEnd() + ellipsis
    }

    private fun dp(v: Float) =
        TypedValue.applyDimension(TypedValue.COMPLEX_UNIT_DIP, v, resources.displayMetrics)

    // Not displayMetrics.scaledDensity, which is deprecated from API 34 and was
    // always a single scale factor for every text size regardless of its size.
    private fun sp(v: Float) =
        TypedValue.applyDimension(TypedValue.COMPLEX_UNIT_SP, v, resources.displayMetrics)

    private companion object {
        const val AMBER = 0xFFFF9E24.toInt()
        const val GREEN = 0xFF3DEB73.toInt()
        const val RED = 0xFFFF6B6B.toInt()
    }
}
