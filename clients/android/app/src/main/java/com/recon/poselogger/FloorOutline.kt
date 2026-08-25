package com.recon.poselogger

import kotlin.math.abs
import kotlin.math.hypot

/**
 * The room's footprint, tapped corner by corner while the user walks the walls,
 * and editable afterwards.
 *
 * Everything else in this app infers the room's shape — planes from ARCore,
 * barriers from fitted geometry, a floorplan from a point cloud. All of it can
 * be wrong in ways nobody notices until the floorplan is out by a metre. This
 * is the one channel where the person standing in the room states the answer
 * directly: tap corner 1, tap corner 2, a wall appears between them, and so on
 * until the last tap lands back on the first and the room closes.
 *
 * Corners are wrong often enough that placing them is only half the feature.
 * A corner behind a sofa gets tapped at the sofa; a doorway gets missed and has
 * to be added to a wall already drawn. So every corner can be selected, moved,
 * deleted, or inserted into an existing wall — including after the loop closes.
 *
 * Deliberately pure — no ARCore, no OpenGL, no Android — because the rules that
 * matter here (when does a tap close the loop, when does an edit break the
 * polygon) are exactly the rules worth testing on a plain JVM in CI, without a
 * device.
 *
 * ## Frame
 *
 * Corners are world-space metres in the capture frame, which is Y-up (see
 * [FrameConversion.GRAVITY_DOWN]). Closing distance, area and self-intersection
 * are all computed on the horizontal (x, z) projection: two corners at opposite
 * ends of a sloping floor are still different corners, and a user who taps a
 * skirting board 4 cm lower than the first tap still means "close the loop".
 */
class FloorOutline {

    data class Corner(val x: Float, val y: Float, val z: Float)

    /** What a tap or an edit did. The UI says something different for each. */
    enum class Tap {
        /** A new corner, and a wall segment back to the previous one. */
        ADDED,

        /** Landed on the first corner: the footprint is now a closed polygon. */
        CLOSED,

        /** An existing corner was picked up, or put down somewhere new. */
        SELECTED,
        MOVED,
        DESELECTED,

        /** A corner was taken out, or added into the middle of a wall. */
        DELETED,
        INSERTED,

        /** Within a few centimetres of the last corner — a double tap, not a wall. */
        TOO_CLOSE,

        /**
         * The change would make the footprint cross itself.
         *
         * Rooms are non-convex often enough (an L is a room), but a *self
         * crossing* footprint is not a room in any house — it is a mis-tap or a
         * bad drag, and the moment to say so is now, while the user is standing
         * at the corner and can retap it. Left in, it produces a bow-tie polygon
         * whose shoelace area partly cancels: the number comes out plausible and
         * small rather than obviously broken, which is the worst way for a
         * measurement to be wrong.
         */
        CROSSES,

        /** A closed room cannot drop below three corners and stay a room. */
        TOO_FEW,

        /** The loop is already closed; edit it, undo, or reset. */
        ALREADY_CLOSED,

        /** Nothing to act on. */
        NOTHING,
    }

    private val corners = mutableListOf<Corner>()
    // The position is deliberately not the corner's identity.  A corner can be
    // moved while a capture is in progress, and the bundle must still be able
    // to say which reference point moved after another corner is inserted or
    // deleted.  This id never escapes into the displayed footprint; it is only
    // used by the adjustment log.
    private val cornerIds = mutableListOf<Long>()
    private var nextCornerId = 1L

    var isClosed: Boolean = false
        private set

    /** Index of the corner being edited, or -1. */
    var selected: Int = NONE
        private set

    val size: Int get() = corners.size
    fun cornersList(): List<Corner> = corners.toList()
    /** Stable ids in the same order as [cornersList]. */
    fun cornerIdsList(): List<Long> = cornerIds.toList()
    fun first(): Corner? = corners.firstOrNull()
    fun last(): Corner? = corners.lastOrNull()
    fun isEmpty(): Boolean = corners.isEmpty()
    fun corner(i: Int): Corner? = corners.getOrNull(i)
    /** Stable identity for a corner, retained across moves and insertions. */
    fun cornerId(i: Int): Long? = cornerIds.getOrNull(i)
    fun selectedCorner(): Corner? = corners.getOrNull(selected)

    /**
     * True once a tap on the first corner would close the loop, which is what
     * the UI needs to know to start highlighting it as a target.
     */
    fun canClose(): Boolean = !isClosed && corners.size >= MIN_CORNERS

    // -- placing ------------------------------------------------------------

    fun add(x: Float, y: Float, z: Float): Tap {
        if (isClosed) return Tap.ALREADY_CLOSED
        val p = Corner(x, y, z)

        corners.lastOrNull()?.let {
            if (horizontalDistance(it, p) < MIN_SPACING_M) return Tap.TOO_CLOSE
        }

        val start = corners.firstOrNull()
        if (start != null && corners.size >= MIN_CORNERS &&
            horizontalDistance(start, p) < CLOSE_RADIUS_M
        ) {
            // Close on the *existing* first corner rather than the tap: the loop
            // should meet exactly, not within a tap's aim.
            if (selfIntersects(corners, closed = true)) return Tap.CROSSES
            isClosed = true
            return Tap.CLOSED
        }

        val candidate = corners + p
        if (selfIntersects(candidate, closed = false)) return Tap.CROSSES
        corners += p
        cornerIds += nextCornerId++
        return Tap.ADDED
    }

    /**
     * One tap on the floor, resolved against what is already there.
     *
     * This is the whole interaction, and it lives here rather than in the
     * activity so that the precedence between its cases is testable without a
     * device — the precedence is where the bugs are, not the arithmetic.
     *
     * In order:
     *
     * 1. **Closing beats selecting.** Corner picking reaches 35 cm and the
     *    closing radius 40 cm, so without this the tap that finishes the room —
     *    on the first corner, the one the UI has been pulsing at the user —
     *    would pick that corner up for editing instead.
     * 2. **Holding a corner and tapping open floor moves it.** Checked before
     *    selection so that tapping a *different* corner while holding one still
     *    re-selects, rather than dropping the held corner on top of it.
     * 3. **Corners beat everything else.** A user aiming at a corner they can
     *    see has been unambiguous, and "add another corner 4 cm away" is the
     *    one thing they never mean.
     * 4. Then the room's state decides: an open outline is still being drawn,
     *    so a tap extends it; a closed one is being edited, so a tap on a wall
     *    splits it.
     */
    fun tap(x: Float, y: Float, z: Float): Tap {
        val onCorner = cornerAt(x, z, CORNER_PICK_M)

        if (!isClosed && canClose() && onCorner == 0 && selected == NONE) {
            return add(x, y, z)
        }
        if (selected != NONE && onCorner == NONE) return moveSelected(x, y, z)
        if (onCorner != NONE) return select(onCorner)
        if (isClosed) {
            val wall = wallAt(x, z, WALL_PICK_M)
            return if (wall >= 0) insertBefore(wall, x, y, z) else Tap.ALREADY_CLOSED
        }
        return add(x, y, z)
    }

    /**
     * Long press: delete what is held, otherwise take back the last corner.
     */
    fun longPress(): Tap = when {
        selected != NONE -> deleteSelected()
        undo() -> Tap.DELETED
        else -> Tap.NOTHING
    }

    // -- editing ------------------------------------------------------------

    /** Pick a corner up, or put it down if it was already held. */
    fun select(i: Int): Tap {
        if (i !in corners.indices) return Tap.NOTHING
        return if (selected == i) {
            selected = NONE
            Tap.DESELECTED
        } else {
            selected = i
            Tap.SELECTED
        }
    }

    fun deselect() {
        selected = NONE
    }

    /**
     * Move the held corner somewhere new.
     *
     * Refused if it would make the footprint cross itself, which is exactly the
     * case a user drags into when they mean to move a corner but pick the wrong
     * one — the polygon inverts and the area silently halves.
     */
    fun moveSelected(x: Float, y: Float, z: Float): Tap {
        val i = selected
        if (i !in corners.indices) return Tap.NOTHING
        val candidate = corners.toMutableList()
        candidate[i] = Corner(x, y, z)
        if (selfIntersects(candidate, isClosed)) return Tap.CROSSES
        corners[i] = candidate[i]
        return Tap.MOVED
    }

    /**
     * Delete a corner.
     *
     * Removing a vertex replaces two walls with one, and in a non-convex room
     * that chord can cut across the far side — so this is checked like any
     * other edit rather than assumed safe.
     */
    fun delete(i: Int): Tap {
        if (i !in corners.indices) return Tap.NOTHING
        if (isClosed && corners.size <= MIN_CORNERS) return Tap.TOO_FEW
        val candidate = corners.toMutableList()
        candidate.removeAt(i)
        if (selfIntersects(candidate, isClosed)) return Tap.CROSSES
        corners.removeAt(i)
        cornerIds.removeAt(i)
        selected = when {
            selected == i -> NONE
            selected > i -> selected - 1
            else -> selected
        }
        return Tap.DELETED
    }

    fun deleteSelected(): Tap = if (selected == NONE) Tap.NOTHING else delete(selected)

    /**
     * Split the wall that *ends* at [i], putting a new corner in the middle of
     * it. The new corner becomes the selection, because a user who inserts one
     * almost always wants to move it next.
     */
    fun insertBefore(i: Int, x: Float, y: Float, z: Float): Tap {
        if (corners.size < 2) return Tap.NOTHING
        if (i !in 0..corners.size) return Tap.NOTHING
        val candidate = corners.toMutableList()
        candidate.add(i, Corner(x, y, z))
        if (selfIntersects(candidate, isClosed)) return Tap.CROSSES
        corners.add(i, Corner(x, y, z))
        cornerIds.add(i, nextCornerId++)
        selected = i
        return Tap.INSERTED
    }

    /**
     * Which wall a point lies on, or -1.
     *
     * Returns the index of the wall's *end* corner, which is what
     * [insertBefore] wants. The closing wall of a shut room reports index 0.
     */
    fun wallAt(x: Float, z: Float, tolerance: Float): Int {
        var best = -1
        var bestD = tolerance
        for ((i, w) in wallsOf(corners, isClosed).withIndex()) {
            val d = distanceToSegment(x, z, w.first, w.second)
            if (d < bestD) {
                bestD = d
                // Wall i runs corner i -> corner i+1; the closing wall wraps to 0.
                best = (i + 1) % corners.size
            }
        }
        return best
    }

    /** Nearest corner within [tolerance] horizontal metres, or -1. */
    fun cornerAt(x: Float, z: Float, tolerance: Float): Int {
        var best = -1
        var bestD = tolerance
        for ((i, c) in corners.withIndex()) {
            val d = hypot(c.x - x, c.z - z)
            if (d < bestD) {
                bestD = d
                best = i
            }
        }
        return best
    }

    /** Take back the last thing that happened — the closure, or a corner. */
    fun undo(): Boolean {
        if (isClosed) {
            isClosed = false
            return true
        }
        if (corners.isEmpty()) return false
        corners.removeAt(corners.size - 1)
        cornerIds.removeAt(cornerIds.size - 1)
        if (selected >= corners.size) selected = NONE
        return true
    }

    fun reset() {
        corners.clear()
        cornerIds.clear()
        isClosed = false
        selected = NONE
    }

    // -- geometry -----------------------------------------------------------

    /** Wall segments to draw, including the closing one once the loop is shut. */
    fun segments(): List<Pair<Corner, Corner>> = wallsOf(corners, isClosed)

    /** Total wall length, metres. Open outlines report what exists so far. */
    fun perimeterM(): Float =
        segments().fold(0f) { acc, (a, b) -> acc + horizontalDistance(a, b) }

    /**
     * Footprint area in m², or 0 while the outline is still open.
     *
     * Shoelace on the (x, z) projection. Open polygons deliberately report 0
     * rather than the area of an implied closing edge the user has not agreed
     * to — a half-drawn room has no area, and showing one invites the user to
     * stop early.
     */
    fun areaM2(): Float {
        if (!isClosed || corners.size < MIN_CORNERS) return 0f
        var sum = 0.0
        for (i in corners.indices) {
            val a = corners[i]
            val b = corners[(i + 1) % corners.size]
            sum += a.x.toDouble() * b.z - b.x.toDouble() * a.z
        }
        return abs(sum / 2.0).toFloat()
    }

    private companion object {
        const val NONE = -1

        /** A tap this near the first corner means "close the loop". */
        const val CLOSE_RADIUS_M = 0.40f

        /**
         * Two corners nearer than this are one corner tapped twice. Rooms do
         * have short walls — a door reveal is ~10 cm — so this is set below
         * anything architectural and only catches finger jitter.
         */
        const val MIN_SPACING_M = 0.08f

        /** A polygon needs three corners; two taps are a line, not a room. */
        const val MIN_CORNERS = 3

        /**
         * How near a tap must land to count as aiming at an existing corner
         * rather than placing a new one.
         *
         * Generous, because the cost is asymmetric: picking the wrong corner is
         * one more tap to undo, while accidentally adding a corner on top of an
         * existing one puts a near-zero-length wall into the footprint.
         */
        const val CORNER_PICK_M = 0.35f

        /** Tighter than corner picking, so a wall never steals a corner's tap. */
        const val WALL_PICK_M = 0.20f

        fun horizontalDistance(a: Corner, b: Corner): Float = hypot(a.x - b.x, a.z - b.z)

        fun wallsOf(cs: List<Corner>, closed: Boolean): List<Pair<Corner, Corner>> {
            if (cs.size < 2) return emptyList()
            val out = cs.zipWithNext().toMutableList()
            if (closed) out += cs.last() to cs.first()
            return out
        }

        /**
         * Does this footprint cross itself?
         *
         * Whole-polygon rather than incremental: an edit in the middle of a
         * closed room can break a pair of walls neither of which it touches, so
         * checking only the walls that changed misses exactly the cases editing
         * introduces. n is a room's corner count, so O(n²) is nothing.
         */
        fun selfIntersects(cs: List<Corner>, closed: Boolean): Boolean {
            val walls = wallsOf(cs, closed)
            for (i in walls.indices) {
                for (j in i + 2 until walls.size) {
                    // Walls sharing an endpoint touch by construction. In a
                    // closed ring the last wall also meets the first.
                    if (closed && i == 0 && j == walls.size - 1) continue
                    if (segmentsIntersect(
                            walls[i].first, walls[i].second,
                            walls[j].first, walls[j].second,
                        )
                    ) {
                        return true
                    }
                }
            }
            return false
        }

        fun distanceToSegment(x: Float, z: Float, a: Corner, b: Corner): Float {
            val vx = b.x - a.x
            val vz = b.z - a.z
            val len2 = vx * vx + vz * vz
            if (len2 < 1e-9f) return hypot(x - a.x, z - a.z)
            val t = (((x - a.x) * vx + (z - a.z) * vz) / len2).coerceIn(0f, 1f)
            return hypot(x - (a.x + t * vx), z - (a.z + t * vz))
        }

        fun orientation(px: Float, pz: Float, qx: Float, qz: Float, rx: Float, rz: Float): Int {
            val v = (qz - pz) * (rx - qx) - (qx - px) * (rz - qz)
            return when {
                v > 1e-6f -> 1
                v < -1e-6f -> -1
                else -> 0
            }
        }

        fun onSegment(px: Float, pz: Float, qx: Float, qz: Float, rx: Float, rz: Float): Boolean =
            qx <= maxOf(px, rx) && qx >= minOf(px, rx) &&
                qz <= maxOf(pz, rz) && qz >= minOf(pz, rz)

        /** Standard orientation test, on the horizontal projection. */
        fun segmentsIntersect(p1: Corner, q1: Corner, p2: Corner, q2: Corner): Boolean {
            val o1 = orientation(p1.x, p1.z, q1.x, q1.z, p2.x, p2.z)
            val o2 = orientation(p1.x, p1.z, q1.x, q1.z, q2.x, q2.z)
            val o3 = orientation(p2.x, p2.z, q2.x, q2.z, p1.x, p1.z)
            val o4 = orientation(p2.x, p2.z, q2.x, q2.z, q1.x, q1.z)
            if (o1 != o2 && o3 != o4) return true
            if (o1 == 0 && onSegment(p1.x, p1.z, p2.x, p2.z, q1.x, q1.z)) return true
            if (o2 == 0 && onSegment(p1.x, p1.z, q2.x, q2.z, q1.x, q1.z)) return true
            if (o3 == 0 && onSegment(p2.x, p2.z, p1.x, p1.z, q2.x, q2.z)) return true
            if (o4 == 0 && onSegment(p2.x, p2.z, q1.x, q1.z, q2.x, q2.z)) return true
            return false
        }
    }
}
