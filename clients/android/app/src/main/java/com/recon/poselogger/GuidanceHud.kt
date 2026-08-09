package com.recon.poselogger

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Path
import android.graphics.RectF
import android.graphics.Typeface
import android.util.AttributeSet
import android.util.TypedValue
import android.view.View
import kotlin.math.cos
import kotlin.math.min
import kotlin.math.roundToInt
import kotlin.math.sin

/**
 * The 2D half of capture guidance: coverage bars, the keyframe budget, the
 * core's nag, and an arrow to a next-best station that is behind the user.
 *
 * The 3D overlay answers "which surfaces still need work"; this answers "how
 * close am I to done" and "where do I point the phone". Both read from the same
 * `FfiGuidance`, so the numbers here and the colours out there cannot disagree.
 */
class GuidanceHud @JvmOverloads constructor(
    context: Context, attrs: AttributeSet? = null, defStyle: Int = 0,
) : View(context, attrs, defStyle) {

    data class State(
        val statusLine: String = "",
        val recording: Boolean = false,
        val floor: Float = 0f,
        val walls: Float = 0f,
        val ceiling: Float = 0f,
        val floorTarget: Float = 0.9f,
        val wallTarget: Float = 0.9f,
        val ceilingTarget: Float = 0.7f,
        val keyframes: Int = 0,
        val maxKeyframes: Int = 150,
        val mappedVoxels: Int = 0,
        val nag: String? = null,
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
        /** What the user should do about the target square right now. */
        val shotInstruction: String? = null,
        /** Set when the device cannot do depth: coverage is then unknowable, not zero. */
        val noDepth: Boolean = false,
        /**
         * What ARCore itself has to work with. Shown always, not just while
         * recording: a room with no planes and few feature points is one where
         * tracking is about to fail, and that is worth knowing *before*
         * committing to a capture rather than after losing one.
         */
        val planeCount: Int = 0,
        val featurePoints: Int = 0,
        /** Surface samples the last keyframe fed the coverage grid. */
        val depthSamples: Int = 0,
        val floorSectors: Int = 0,
        val wallSectors: Int = 0,
        val ceilingSectors: Int = 0,
        /** Screen-space bearing to an off-screen station, radians, 0 = right. */
        val stationBearingRad: Float? = null,
        val stationDistanceM: Float? = null,
    )

    var state: State = State()
        set(value) {
            field = value
            invalidate()
        }

    private val panel = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = 0xB3000000.toInt() }
    private val track = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = 0x33FFFFFF }
    private val bar = Paint(Paint.ANTI_ALIAS_FLAG)
    private val tick = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = 0xCCFFFFFF.toInt()
        strokeWidth = dp(2f)
    }
    private val label = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE
        textSize = sp(13f)
        typeface = Typeface.MONOSPACE
    }
    private val value = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE
        textSize = sp(13f)
        typeface = Typeface.MONOSPACE
        textAlign = Paint.Align.RIGHT
    }
    private val caption = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = 0x99FFFFFF.toInt()
        textSize = sp(11f)
    }
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
    private val verdictPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        textSize = sp(13f)
        typeface = Typeface.create(Typeface.DEFAULT, Typeface.BOLD)
    }
    private val arrow = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = CYAN }
    private val arrowText = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = CYAN
        textSize = sp(13f)
        typeface = Typeface.create(Typeface.DEFAULT, Typeface.BOLD)
        textAlign = Paint.Align.CENTER
    }
    private val rect = RectF()
    private val arrowPath = Path()

    override fun onDraw(canvas: Canvas) {
        val s = state
        val pad = dp(14f)
        val panelHeight = if (s.recording) dp(182f) else dp(58f)
        rect.set(0f, 0f, width.toFloat(), panelHeight)
        canvas.drawRect(rect, panel)

        // Clipped rather than ellipsised is how this line used to fail: the
        // status ran past the right edge and the last field silently vanished,
        // so a reader could not tell a short status from a truncated one.
        canvas.drawText(ellipsise(s.statusLine, status, width - pad * 2), pad, dp(30f), status)
        canvas.drawText(
            "%d planes · %d feature points".format(s.planeCount, s.featurePoints),
            pad, dp(46f), caption,
        )
        if (!s.recording) {
            drawStationArrow(canvas, s)
            return
        }

        var y = dp(74f)
        drawBar(canvas, "floor", s.floor, s.floorTarget, pad, y)
        y += dp(24f)
        drawBar(canvas, "walls", s.walls, s.wallTarget, pad, y)
        y += dp(24f)
        drawBar(canvas, "ceiling", s.ceiling, s.ceilingTarget, pad, y)
        y += dp(20f)

        if (s.noDepth) {
            canvas.drawText(
                "no depth on this device — coverage cannot be measured", pad, y, caption
            )
        } else {
            canvas.drawText(
                "%,d voxels · %d/%d frames · %,d depth pts · sectors %d/%d/%d"
                    .format(
                        s.mappedVoxels, s.keyframes, s.maxKeyframes, s.depthSamples,
                        s.floorSectors, s.wallSectors, s.ceilingSectors,
                    ),
                pad, y, caption,
            )
            // Say what the bars actually measure. The core reports the fraction
            // of *observed* voxels that clear the quality bar, not the fraction
            // of the room — mid-capture the room's extent is unknown. Without
            // this line a user reads "90%" as "nearly done" having swept one
            // corner well.
            canvas.drawText(
                "bars: how well you have seen it, not how much of the room",
                pad, y + dp(15f), caption,
            )
        }

        var bannerY = panelHeight + dp(10f)
        // First, and red: while this is showing the capture is collecting
        // nothing at all, so every other instruction on screen is asking the
        // user to do something that cannot currently work.
        s.recovery?.let {
            drawBanner(canvas, it, RED, bannerY)
            bannerY += dp(44f)
        }
        // Above everything else while a script is running: it is the one thing
        // the user is being asked to do, and a coverage bar is context, not an
        // instruction.
        s.shotInstruction?.let {
            // Skips ride with the count rather than beside it: "20/24" alone
            // reads as a room four shots from done, when four of them are
            // never coming.
            val progress = if (s.shotSkipped > 0) {
                "${s.shotTaken}/${s.shotTotal} · ${s.shotSkipped} skipped"
            } else {
                "${s.shotTaken}/${s.shotTotal}"
            }
            drawBanner(canvas, "$it  ·  $progress", CYAN, bannerY)
            bannerY += dp(44f)
        }
        // Above the nag, and red rather than amber: a nag is advice about the
        // next few seconds, whereas a break has already cost the part of the
        // room captured before it. Nothing later in the pipeline can undo one.
        if (s.trackingBreaks > 0) {
            val n = s.trackingBreaks
            drawBanner(
                canvas,
                "Tracking reset ${n}× — this capture cannot be joined up",
                RED, bannerY,
            )
            bannerY += dp(44f)
        }
        if (s.complete) {
            drawBanner(canvas, "Room complete — you can stop", GREEN, bannerY)
            bannerY += dp(44f)
        }
        s.nag?.let { drawBanner(canvas, it, AMBER, bannerY) }

        drawStationArrow(canvas, s)
    }

    private fun drawBar(canvas: Canvas, name: String, v: Float, target: Float, pad: Float, y: Float) {
        val labelWidth = dp(66f)
        val valueWidth = dp(52f)
        val left = pad + labelWidth
        val right = width - pad - valueWidth
        val h = dp(9f)

        canvas.drawText(name, pad, y + h, label)
        rect.set(left, y, right, y + h)
        canvas.drawRoundRect(rect, h / 2, h / 2, track)

        val frac = v.coerceIn(0f, 1f)
        if (frac > 0f) {
            // Amber until it clears the target, green past it — the same reading
            // the user gets from the paint on the walls.
            bar.color = if (v >= target) GREEN else AMBER
            rect.set(left, y, left + (right - left) * frac, y + h)
            canvas.drawRoundRect(rect, h / 2, h / 2, bar)
        }

        val tx = left + (right - left) * target.coerceIn(0f, 1f)
        canvas.drawLine(tx, y - dp(3f), tx, y + h + dp(3f), tick)
        canvas.drawText("${(frac * 100).roundToInt()}%", width - pad, y + h, value)
    }

    private fun drawBanner(canvas: Canvas, text: String, color: Int, y: Float) {
        val pad = dp(14f)
        banner.color = color
        rect.set(pad, y, width - pad, y + dp(36f))
        canvas.drawRoundRect(rect, dp(8f), dp(8f), banner)
        canvas.drawText(text, width / 2f, y + dp(23f), bannerText)
    }

    /**
     * When the suggested station is off-screen the ring is invisible and the
     * user has no idea which way to turn, which is the moment guidance matters
     * most — so the arrow only appears then.
     */
    private fun drawStationArrow(canvas: Canvas, s: State) {
        val bearing = s.stationBearingRad ?: return
        val cx = width / 2f
        val cy = height / 2f
        val radius = min(width, height) * 0.34f
        val x = cx + cos(bearing) * radius
        val y = cy + sin(bearing) * radius
        val size = dp(20f)

        arrowPath.reset()
        arrowPath.moveTo(x + cos(bearing) * size, y + sin(bearing) * size)
        arrowPath.lineTo(
            x + cos(bearing + 2.5f) * size * 0.8f, y + sin(bearing + 2.5f) * size * 0.8f
        )
        arrowPath.lineTo(
            x + cos(bearing - 2.5f) * size * 0.8f, y + sin(bearing - 2.5f) * size * 0.8f
        )
        arrowPath.close()
        canvas.drawPath(arrowPath, arrow)

        s.stationDistanceM?.let {
            canvas.drawText("%.1f m".format(it), x, y + size * 2f, arrowText)
        }
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
        const val CYAN = 0xFF4DDBFF.toInt()
        const val RED = 0xFFFF6B6B.toInt()
    }
}
