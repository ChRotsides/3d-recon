package com.recon.poselogger.rgbscan

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Path
import android.graphics.RectF
import android.graphics.Typeface
import android.view.View
import kotlin.math.abs
import kotlin.math.hypot
import kotlin.math.roundToInt

/** Action-first guidance HUD. RGB localization is authoritative; sensors may interpolate it. */
internal class SphericalGuidanceView(context: Context) : View(context) {
    private val density = resources.displayMetrics.density
    private val title = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE
        textAlign = Paint.Align.CENTER
        textSize = resources.displayMetrics.scaledDensity * 18f
        typeface = Typeface.DEFAULT_BOLD
    }
    private val detail = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.rgb(185, 194, 204)
        textAlign = Paint.Align.CENTER
        textSize = resources.displayMetrics.scaledDensity * 12f
    }
    private val arrow = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = ACCENT
        style = Paint.Style.STROKE
        strokeCap = Paint.Cap.ROUND
        strokeJoin = Paint.Join.ROUND
        strokeWidth = 5f * density
    }
    private val progressTrack = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = Color.rgb(47, 55, 64) }
    private val progressFill = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = ACCENT }
    private var state = GuidanceSnapshot()
    private var assistedYaw: Double? = null
    private var assistedPitch: Double? = null

    fun show(snapshot: GuidanceSnapshot) {
        state = snapshot
        assistedYaw = snapshot.localization?.yawDegrees
        assistedPitch = snapshot.localization?.pitchDegrees
        updateDescription()
        invalidate()
    }

    fun showSensorDirection(yawDegrees: Double, pitchDegrees: Double) {
        if (!yawDegrees.isFinite() || !pitchDegrees.isFinite()) return
        assistedYaw = yawDegrees
        assistedPitch = pitchDegrees.coerceIn(-89.0, 89.0)
        updateDescription()
        postInvalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        canvas.drawColor(Color.rgb(17, 22, 27))
        val coverage = state.coverage
        if (coverage == null || coverage.cells.isEmpty()) {
            drawMessage(canvas, "Take a photo to begin", "Guidance uses visible room details")
            drawProgress(canvas, 0.0)
            return
        }

        when (state.localization?.state) {
            LocalizationState.UNLOCALIZED -> drawMessage(
                canvas,
                "Can't locate this view",
                "Point at a textured object and hold steady",
            )
            LocalizationState.NOT_REQUESTED, null -> drawMessage(
                canvas,
                "Take a photo for direction",
                "Keep rotating from the same position",
            )
            LocalizationState.LOCALIZED -> drawLocalized(canvas)
        }
        drawProgress(canvas, coverageProgress(coverage))
    }

    private fun drawLocalized(canvas: Canvas) {
        val target = state.target
        val currentYaw = assistedYaw
        val currentPitch = assistedPitch
        if (target == null) {
            drawMessage(canvas, "Coverage looks good", "You can finish or add detail photos")
            return
        }
        if (currentYaw == null || currentPitch == null) {
            drawMessage(canvas, "Refreshing direction…", "Hold the phone steady")
            return
        }
        val deltaYaw = wrapDegrees(target.yawDegrees - currentYaw)
        val deltaPitch = target.pitchDegrees - currentPitch
        val reached = abs(deltaYaw) <= 10.0 && abs(deltaPitch) <= 8.0
        if (reached) {
            arrow.color = READY
            drawReticle(canvas)
            drawMessage(canvas, "Capture this view", "Next missing room direction", reserveArrow = true)
            return
        }

        arrow.color = ACCENT
        drawDirectionArrow(canvas, deltaYaw, deltaPitch)
        drawMessage(
            canvas,
            instruction(deltaYaw, deltaPitch),
            "Next photo • ${coveragePercent()}% covered",
            reserveArrow = true,
        )
    }

    private fun drawMessage(
        canvas: Canvas,
        headline: String,
        subtitle: String,
        reserveArrow: Boolean = false,
    ) {
        val centre = width / 2f
        val headlineY = if (reserveArrow) 27f * density else 35f * density
        val subtitleY = if (reserveArrow) 76f * density else 61f * density
        canvas.drawText(headline, centre, headlineY, title)
        canvas.drawText(subtitle, centre, subtitleY.coerceAtMost(height - 17f * density), detail)
    }

    private fun drawDirectionArrow(canvas: Canvas, deltaYaw: Double, deltaPitch: Double) {
        val magnitude = hypot(deltaYaw, deltaPitch).coerceAtLeast(1.0)
        val dx = (deltaYaw / magnitude * 25f * density).toFloat()
        val dy = (-deltaPitch / magnitude * 25f * density).toFloat()
        val startX = width / 2f - dx * 0.45f
        val startY = 50f * density - dy * 0.45f
        val endX = width / 2f + dx * 0.55f
        val endY = 50f * density + dy * 0.55f
        canvas.drawLine(startX, startY, endX, endY, arrow)
        val perpendicularX = -dy * 0.28f
        val perpendicularY = dx * 0.28f
        val backX = endX - dx * 0.42f
        val backY = endY - dy * 0.42f
        val head = Path().apply {
            moveTo(backX + perpendicularX, backY + perpendicularY)
            lineTo(endX, endY)
            lineTo(backX - perpendicularX, backY - perpendicularY)
        }
        canvas.drawPath(head, arrow)
    }

    private fun drawReticle(canvas: Canvas) {
        val x = width / 2f
        val y = 50f * density
        canvas.drawCircle(x, y, 13f * density, arrow)
        canvas.drawLine(x - 20f * density, y, x - 11f * density, y, arrow)
        canvas.drawLine(x + 11f * density, y, x + 20f * density, y, arrow)
        canvas.drawLine(x, y - 20f * density, x, y - 11f * density, arrow)
        canvas.drawLine(x, y + 11f * density, x, y + 20f * density, arrow)
    }

    private fun drawProgress(canvas: Canvas, progress: Double) {
        val left = 12f * density
        val right = width - left
        val bottom = height - 6f * density
        val top = bottom - 5f * density
        val radius = 3f * density
        canvas.drawRoundRect(RectF(left, top, right, bottom), radius, radius, progressTrack)
        canvas.drawRoundRect(
            RectF(left, top, left + (right - left) * progress.toFloat(), bottom),
            radius,
            radius,
            progressFill,
        )
    }

    private fun instruction(deltaYaw: Double, deltaPitch: Double): String {
        val horizontal = if (deltaYaw >= 0) "right" else "left"
        val vertical = if (deltaPitch >= 0) "up" else "down"
        val yaw = abs(deltaYaw)
        val pitch = abs(deltaPitch)
        return when {
            yaw > pitch * 1.5 -> "Turn $horizontal ${friendlyDegrees(yaw)}°"
            pitch > yaw * 1.5 -> "Tilt $vertical ${friendlyDegrees(pitch)}°"
            else -> "Aim $vertical-$horizontal ${friendlyDegrees(hypot(yaw, pitch))}°"
        }
    }

    private fun evidenceLabel(target: GuidanceTarget): String = when {
        "blur" in target.reasons -> "Needs a sharper photo"
        "exposure" in target.reasons -> "Needs better exposure"
        "single_view" in target.reasons -> "Needs another angle"
        else -> "Missing coverage"
    }

    private fun coveragePercent(): Int =
        ((state.coverage?.let(::coverageProgress) ?: 0.0) * 100).roundToInt().coerceIn(0, 100)

    private fun coverageProgress(coverage: GuidanceCoverage): Double {
        val bandHeight = minOf(4, coverage.height)
        val firstRow = (coverage.height - bandHeight) / 2
        val required = coverage.cells.filter { it.row in firstRow until (firstRow + bandHeight) }
        if (required.isEmpty()) return 0.0
        // A direction is complete after one usable photo. Amber cells are
        // optional quality repairs and must not hold the user hostage.
        return required.count { it.state != CoverageState.RED }.toDouble() / required.size
    }

    private fun updateDescription() {
        val localization = state.localization
        contentDescription = when {
            localization?.state == LocalizationState.UNLOCALIZED ->
                "Cannot locate this view. Point at a textured object and hold steady."
            state.targetVisible && assistedYaw != null && assistedPitch != null -> {
                val yaw = wrapDegrees(state.target!!.yawDegrees - assistedYaw!!)
                val pitch = state.target!!.pitchDegrees - assistedPitch!!
                instruction(yaw, pitch)
            }
            localization?.state == LocalizationState.LOCALIZED && state.target == null ->
                "Coverage looks good."
            else -> "Take a photo for directional guidance."
        }
    }

    companion object {
        private val ACCENT = Color.rgb(77, 163, 255)
        private val READY = Color.rgb(54, 207, 123)

        internal fun wrapDegrees(value: Double): Double =
            ((value + 180.0) % 360.0 + 360.0) % 360.0 - 180.0

        private fun friendlyDegrees(value: Double): Int =
            ((value / 5.0).roundToInt() * 5).coerceAtLeast(5)
    }
}
