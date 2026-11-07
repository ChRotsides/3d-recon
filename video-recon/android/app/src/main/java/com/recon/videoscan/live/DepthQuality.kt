package com.recon.videoscan.live

import kotlin.math.*

/** Robust feature anchors and independent-view agreement. No previous scale is reused. */
class DepthQuality {
    data class Fit(val a: Float, val b: Float, val inverse: Boolean, val inliers: Int,
        val minPrediction: Float = Float.NEGATIVE_INFINITY, val maxPrediction: Float = Float.POSITIVE_INFINITY,
        val minDepth: Float = 0f, val maxDepth: Float = Float.POSITIVE_INFINITY) {
        fun depth(d: Float): Float {
            if (d !in minPrediction..maxPrediction) return Float.NaN
            val z = if (inverse) 1f / (a * d + b) else a * d
            return if (z in minDepth..maxDepth) z else Float.NaN
        }
    }
    data class Calibration(val fit: Fit?, val reason: String)
    // Calibration reconciles two noisy estimators; surface confirmation below
    // stays at 6 cm / 2.5% and remains mandatory before TSDF visibility.
    private fun anchorTolerance(z: Float) = maxOf(.12f, .10f * z)

    fun fit(ds: List<Float>, zs: List<Float>, cells: List<Int>, inverse: Boolean): Fit? =
        calibrate(ds, zs, cells, inverse).fit

    fun calibrate(ds: List<Float>, zs: List<Float>, cells: List<Int>, inverse: Boolean): Calibration {
        if (!validAnchors(ds, zs, cells)) return Calibration(null, "invalid depth anchors")
        if (ds.size < 8 || cells.toSet().size < 4)
            return Calibration(null, "depth needs more coverage: ${ds.size} anchors in ${cells.toSet().size}/4 regions")
        val candidates = ArrayList<Pair<Float, Float>>()
        if (!inverse) ds.indices.forEach { candidates.add(zs[it] / ds[it] to 0f) }
        else {
            // Deterministic pair hypotheses avoid letting a bad landmark dominate least squares.
            val step = maxOf(1, ds.size / 24)
            for (i in ds.indices step step) for (j in i + 1 until ds.size step step) {
                val dx = ds[j] - ds[i]
                if (abs(dx) > 1e-5f) {
                    val a = (1f / zs[j] - 1f / zs[i]) / dx
                    if (a.isFinite() && a > 0) candidates.add(a to (1f / zs[i] - a * ds[i]))
                }
            }
        }
        fun inliers(f: Fit) = ds.indices.filter {
            val z = f.depth(ds[it])
            z.isFinite() && z > 0 && abs(z - zs[it]) <= anchorTolerance(zs[it])
        }
        fun supported(ids: List<Int>) = ids.size >= 8 && ids.size >= ds.size * .65 &&
            ids.map { cells[it] }.toSet().size >= 4
        fun observable(ids: List<Int>) = !inverse ||
            (ids.maxOf { zs[it] } / ids.minOf { zs[it] } >= 1.2f &&
                ids.maxOf { ds[it] } - ids.minOf { ds[it] } > 1e-5f)
        fun cost(f: Fit) = ds.indices.sumOf {
            val z = f.depth(ds[it])
            if (!z.isFinite() || z <= 0) 1.0 else {
                val error = abs(z - zs[it]) / anchorTolerance(zs[it])
                minOf(1.0, error.toDouble() * error)
            }
        }
        var best = emptyList<Int>(); var bestFit: Fit? = null; var bestCost = Double.POSITIVE_INFINITY
        var mostInliers = 0; var insufficientDepthRange = false
        for ((a, b) in candidates) {
            val f = Fit(a, b, inverse, 0)
            val ids = inliers(f)
            mostInliers = maxOf(mostInliers, ids.size)
            // Select among usable hypotheses. An excellent fit to one small patch
            // must not veto another hypothesis supported across the whole image.
            if (!supported(ids)) continue
            if (!observable(ids)) { insufficientDepthRange = true; continue }
            val score = cost(f)
            if (score < bestCost) { best = ids; bestFit = f; bestCost = score }
        }
        val selected = bestFit ?: return Calibration(null, if (insufficientDepthRange)
            "depth needs varied distances; include a corner or nearby object" else
            "depth calibration disagrees: $mostInliers/${ds.size} anchors agree")
        val refined = if (!inverse) {
            val ratios = best.map { zs[it] / ds[it] }.sorted()
            Fit(ratios[ratios.size / 2], 0f, false, best.size)
        } else {
            val xs = best.map { ds[it].toDouble() }; val ys = best.map { 1.0 / zs[it] }
            // Weight inverse-depth residuals by z^2 to approximate the relative
            // metric-depth error used by consensus, rather than privileging near objects.
            val weights = best.map { zs[it].toDouble() * zs[it] }
            val sum = weights.sum()
            val mx = xs.indices.sumOf { weights[it] * xs[it] } / sum
            val my = ys.indices.sumOf { weights[it] * ys[it] } / sum
            val variance = xs.indices.sumOf { weights[it] * (xs[it] - mx) * (xs[it] - mx) }
            val a = xs.indices.sumOf { weights[it] * (xs[it] - mx) * (ys[it] - my) } / variance
            Fit(a.toFloat(), (my - a * mx).toFloat(), true, best.size)
        }
        // Refinement must retain the same consensus and observability requirements.
        val refinedIds = inliers(refined)
        val result = if (refined.a.isFinite() && refined.a > 0 && supported(refinedIds) &&
            observable(refinedIds) && cost(refined) <= bestCost)
            refined.copy(inliers = refinedIds.size) else selected.copy(inliers = best.size)
        val errors = ds.indices.map { abs(result.depth(ds[it]) - zs[it]) / zs[it] }.sorted()
        if (errors[errors.size / 2] > .08f)
            return Calibration(null, "depth shape unstable against tracked geometry")
        // Do not extrapolate inverse-affine calibration into unobserved disparity
        // ranges, especially near its pole. Invalid pixels stay absent, not clamped
        // onto an invented surface at the nearest supported depth.
        val ids = inliers(result)
        val low = ids.minOf { ds[it] }; val high = ids.maxOf { ds[it] }
        val margin = .1f * (high - low)
        val bounded = if (inverse) result.copy(minPrediction = low - margin, maxPrediction = high + margin,
            minDepth = ids.minOf { zs[it] } / 1.1f, maxDepth = ids.maxOf { zs[it] } * 1.1f) else result
        return Calibration(bounded, "calibrated ${result.inliers}/${ds.size} anchors")
    }

    private fun validAnchors(ds: List<Float>, zs: List<Float>, cells: List<Int>) =
        ds.size == zs.size && ds.size == cells.size && ds.all { it.isFinite() && it > 0 } &&
            zs.all { it.isFinite() && it > 0 } && cells.all { it in 0..11 }

    /** A metric network still has scene-dependent bias. When independent landmarks
     * cover the image, reject predictions that contradict their geometry. With too
     * few anchors, translated-view confirmation remains mandatory for fusion. */
    fun metricConsistent(ds: List<Float>, zs: List<Float>, cells: List<Int>): Boolean {
        if (!validAnchors(ds, zs, cells)) return false
        if (ds.size < 8 || cells.toSet().size < 4) return true
        val agreeing = ds.indices.filter { abs(ds[it] - zs[it]) <= maxOf(.12f, .10f * zs[it]) }
        return agreeing.size >= 8 && agreeing.size >= ds.size * .65 &&
            agreeing.map { cells[it] }.toSet().size >= 4
    }

    private data class View(val depth: ShortArray, val confidence: ByteArray, val w: Int, val h: Int, val k: FloatArray, val r: FloatArray, val c: FloatArray, val t: Double)
    // A rejected calibration or a brief pause must not discard the only useful
    // translated view. Bound both time and memory; reset on every map revision.
    private val views = ArrayDeque<View>()
    fun reset() { views.clear() }
    /** 64 = provisional new coverage, 255 = cross-view supported, 0 = invalid/conflicting. */
    fun confidence(mm: ShortArray, w: Int, h: Int, k: FloatArray, r: FloatArray, c: FloatArray, t: Double): ByteArray {
        val weights = ByteArray(mm.size) { if ((mm[it].toInt() and 65535) in 200..6000) 64 else 0 }
        for (p in views.reversed()) {
            if (t <= p.t || t - p.t > 8.0) continue
            val baseline = sqrt(c.indices.sumOf { ((c[it] - p.c[it]) * (c[it] - p.c[it])).toDouble() })
            // Rotation alone provides no independent triangulation baseline.
            if (baseline < .04) continue
            val projected = FloatArray(mm.size) { Float.POSITIVE_INFINITY }
            val confirmed = BooleanArray(mm.size)
            for (v in 0 until p.h) for (u in 0 until p.w) {
                val source = v * p.w + u
                val z = (p.depth[source].toInt() and 65535) / 1000f
                if (z !in .2f..6f || p.confidence[source] == 0.toByte()) continue
                val x = (u - p.k[2]) / p.k[0] * z; val y = (v - p.k[3]) / p.k[1] * z
                val px = p.c[0] + p.r[0] * x + p.r[1] * y + p.r[2] * z - c[0]
                val py = p.c[1] + p.r[3] * x + p.r[4] * y + p.r[5] * z - c[1]
                val pz = p.c[2] + p.r[6] * x + p.r[7] * y + p.r[8] * z - c[2]
                val nz = r[2] * px + r[5] * py + r[8] * pz
                if (nz <= .2f) continue
                val nu = (k[0] * (r[0] * px + r[3] * py + r[6] * pz) / nz + k[2]).roundToInt()
                val nv = (k[1] * (r[1] * px + r[4] * py + r[7] * pz) / nz + k[3]).roundToInt()
                if (nu in 0 until w && nv in 0 until h) {
                    val i = nv * w + nu
                    if (nz < projected[i]) {
                        projected[i] = nz; confirmed[i] = p.confidence[source] == (-1).toByte()
                    }
                }
            }
            for (i in mm.indices) {
                val z = (mm[i].toInt() and 65535) / 1000f
                // Confirmed foreground blocks older evidence. A provisional
                // prediction cannot veto a match to another translated view.
                if (weights[i] != 64.toByte() || !projected[i].isFinite()) continue
                val tolerance = maxOf(.06f, .025f * z)
                weights[i] = when {
                    abs(z - projected[i]) <= tolerance -> 255.toByte()
                    z < projected[i] - tolerance -> 64 // newly occluding surface: needs another view
                    confirmed[i] -> 0 // independently confirmed foreground contradicts this surface
                    else -> 64 // an unconfirmed estimate cannot establish an occlusion
                }
            }
        }
        return weights
    }
    fun remember(mm: ShortArray, w: Int, h: Int, k: FloatArray, r: FloatArray, c: FloatArray, t: Double,
        confidence: ByteArray = ByteArray(mm.size) { 64 }) {
        require(confidence.size == mm.size)
        while (views.isNotEmpty() && t - views.first().t > 8.0) views.removeFirst()
        val p = views.lastOrNull()
        if (p != null && t <= p.t) return
        val moved = p == null || sqrt(c.indices.sumOf { ((c[it] - p.c[it]) * (c[it] - p.c[it])).toDouble() }) >= .04
        if (!moved && t - p!!.t <= 2.0) return
        // Refresh a stationary view without filling the history with duplicates.
        if (!moved) views.removeLast()
        views.addLast(View(mm.copyOf(), confidence.copyOf(), w, h, k.copyOf(), r.copyOf(), c.copyOf(), t))
        while (views.size > 4) views.removeFirst()
    }
}
