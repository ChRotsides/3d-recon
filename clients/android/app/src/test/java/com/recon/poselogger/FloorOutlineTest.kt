package com.recon.poselogger

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * The corner-tapping rules, exercised on a plain JVM.
 *
 * A 4 m x 3 m room, tapped anticlockwise from the origin corner, is the shape
 * every test starts from: the numbers (12 m², 14 m perimeter) are ones you can
 * check by hand, which is the point.
 */
class FloorOutlineTest {

    private fun room(o: FloorOutline) {
        o.add(0f, 0f, 0f)
        o.add(4f, 0f, 0f)
        o.add(4f, 0f, 3f)
        o.add(0f, 0f, 3f)
    }

    @Test
    fun `four corners and a closing tap make a room`() {
        val o = FloorOutline()
        room(o)
        assertEquals(4, o.size)
        assertFalse(o.isClosed)
        // Three walls so far; the fourth only exists once the loop is closed.
        assertEquals(3, o.segments().size)

        assertEquals(FloorOutline.Tap.CLOSED, o.add(0.05f, 0f, 0.05f))
        assertTrue(o.isClosed)
        assertEquals(4, o.segments().size)
        assertEquals(12f, o.areaM2(), 1e-4f)
        assertEquals(14f, o.perimeterM(), 1e-4f)
    }

    @Test
    fun `closing snaps to the first corner rather than to where the finger landed`() {
        val o = FloorOutline()
        room(o)
        o.add(0.3f, 0f, -0.2f)   // within the closing radius, but not exact
        // The sloppy tap must not become a fifth corner, or the footprint grows
        // a spurious 30 cm spur and the area is wrong by the user's aim.
        assertEquals(4, o.size)
        assertEquals(12f, o.areaM2(), 1e-4f)
    }

    @Test
    fun `an open outline has no area`() {
        val o = FloorOutline()
        room(o)
        assertEquals(0f, o.areaM2(), 0f)
        assertEquals(11f, o.perimeterM(), 1e-4f)   // 4 + 3 + 4, no closing wall
    }

    @Test
    fun `two corners cannot close a room`() {
        val o = FloorOutline()
        o.add(0f, 0f, 0f)
        o.add(2f, 0f, 0f)
        assertFalse(o.canClose())
        // Tapping back on the start is a new corner, not a closure...
        assertEquals(FloorOutline.Tap.ADDED, o.add(0.01f, 0f, 0.3f))
        assertFalse(o.isClosed)
    }

    @Test
    fun `a double tap on the same corner is not a wall`() {
        val o = FloorOutline()
        o.add(0f, 0f, 0f)
        assertEquals(FloorOutline.Tap.TOO_CLOSE, o.add(0.02f, 0f, 0.02f))
        assertEquals(1, o.size)
    }

    @Test
    fun `a crossing wall is refused`() {
        // A bow tie: the fourth corner sends the wall back across the first.
        val o = FloorOutline()
        o.add(0f, 0f, 0f)
        o.add(4f, 0f, 0f)
        o.add(4f, 0f, 3f)
        assertEquals(FloorOutline.Tap.CROSSES, o.add(2f, 0f, -1f))
        assertEquals(3, o.size)
    }

    @Test
    fun `an L shaped room is allowed`() {
        // Non-convex is normal architecture and must not be confused with
        // self-intersecting.
        val o = FloorOutline()
        for (c in listOf(
            floatArrayOf(0f, 0f), floatArrayOf(4f, 0f), floatArrayOf(4f, 2f),
            floatArrayOf(2f, 2f), floatArrayOf(2f, 4f), floatArrayOf(0f, 4f),
        )) {
            assertEquals(FloorOutline.Tap.ADDED, o.add(c[0], 0f, c[1]))
        }
        assertEquals(FloorOutline.Tap.CLOSED, o.add(0.1f, 0f, 0.1f))
        // 4x2 + 2x2 = 12
        assertEquals(12f, o.areaM2(), 1e-4f)
    }

    @Test
    fun `undo takes back the closure first then the corners`() {
        val o = FloorOutline()
        room(o)
        o.add(0f, 0f, 0f)
        assertTrue(o.isClosed)

        assertTrue(o.undo())
        assertFalse(o.isClosed)
        assertEquals(4, o.size)     // the corners survive re-opening

        assertTrue(o.undo())
        assertEquals(3, o.size)
        o.reset()
        assertTrue(o.isEmpty())
        assertFalse(o.undo())
    }

    @Test
    fun `a closed outline refuses further corners`() {
        val o = FloorOutline()
        room(o)
        o.add(0f, 0f, 0f)
        assertEquals(FloorOutline.Tap.ALREADY_CLOSED, o.add(9f, 0f, 9f))
        assertEquals(4, o.size)
    }

    @Test
    fun `winding direction does not flip the sign of the area`() {
        val cw = FloorOutline()
        cw.add(0f, 0f, 0f); cw.add(0f, 0f, 3f); cw.add(4f, 0f, 3f); cw.add(4f, 0f, 0f)
        cw.add(0f, 0f, 0f)
        assertTrue(cw.isClosed)
        assertEquals(12f, cw.areaM2(), 1e-4f)
    }

    // -- editing ------------------------------------------------------------

    private fun closedRoom(): FloorOutline {
        val o = FloorOutline()
        room(o)
        o.add(0f, 0f, 0f)
        return o
    }

    @Test
    fun `a corner can be picked up and put down somewhere else`() {
        val o = closedRoom()
        val i = o.cornerAt(4f, 0f, 0.3f)
        assertEquals(1, i)
        assertEquals(FloorOutline.Tap.SELECTED, o.select(i))

        // The room was 4 x 3; pull the corner out to 6 and it becomes 6 x 3
        // along that edge — a trapezium of area (4+6)/2 * 3 = 15.
        assertEquals(FloorOutline.Tap.MOVED, o.moveSelected(6f, 0f, 0f))
        assertEquals(15f, o.areaM2(), 1e-4f)
        assertTrue(o.isClosed)
    }

    @Test
    fun `tapping the held corner again puts it down`() {
        val o = closedRoom()
        o.select(2)
        assertEquals(2, o.selected)
        assertEquals(FloorOutline.Tap.DESELECTED, o.select(2))
        assertEquals(-1, o.selected)
    }

    @Test
    fun `a move that would invert the room is refused`() {
        val o = closedRoom()
        o.select(0)
        // Dragging corner 0 across the far wall folds the polygon into a bow
        // tie whose shoelace area partly cancels — the failure that looks like
        // a plausible number rather than an error.
        assertEquals(FloorOutline.Tap.CROSSES, o.moveSelected(4f, 0f, 5f))
        assertEquals(12f, o.areaM2(), 1e-4f)   // unchanged
    }

    @Test
    fun `a corner can be deleted and the room stays closed`() {
        val o = closedRoom()
        assertEquals(FloorOutline.Tap.DELETED, o.delete(3))
        assertEquals(3, o.size)
        assertTrue(o.isClosed)
        // The remaining triangle (0,0) (4,0) (4,3) is half the rectangle.
        assertEquals(6f, o.areaM2(), 1e-4f)
    }

    @Test
    fun `a closed room refuses to drop below three corners`() {
        val o = closedRoom()
        o.delete(3)
        assertEquals(FloorOutline.Tap.TOO_FEW, o.delete(0))
        assertEquals(3, o.size)
    }

    @Test
    fun `deleting shifts the selection rather than leaving it dangling`() {
        val o = closedRoom()
        o.select(3)
        o.delete(1)
        // The held corner moved down one index; a stale index here would edit
        // the wrong corner on the next move.
        assertEquals(2, o.selected)
        assertEquals(FloorOutline.Corner(0f, 0f, 3f), o.selectedCorner())
    }

    @Test
    fun `deleting the held corner clears the selection`() {
        val o = closedRoom()
        o.select(2)
        o.delete(2)
        assertEquals(-1, o.selected)
        assertNull(o.selectedCorner())
    }

    @Test
    fun `a corner can be inserted into an existing wall`() {
        val o = closedRoom()
        // A doorway noticed after the fact, halfway along the south wall.
        val wall = o.wallAt(2f, 0f, 0.3f)
        assertEquals(1, wall)
        assertEquals(FloorOutline.Tap.INSERTED, o.insertBefore(wall, 2f, 0f, 0f))
        assertEquals(5, o.size)
        // Collinear insertion changes nothing about the footprint...
        assertEquals(12f, o.areaM2(), 1e-4f)
        // ...and the new corner is held, because moving it is the next thing
        // anyone does after inserting one.
        assertEquals(1, o.selected)

        // Pushing it 1 m out adds a triangle spanning the whole 4 m wall:
        // 0.5 * 4 * 1 = 2 m².
        o.moveSelected(2f, 0f, -1f)
        assertEquals(14f, o.areaM2(), 1e-3f)
    }

    @Test
    fun `the closing wall of a shut room can be split too`() {
        val o = closedRoom()
        // The wall from corner 3 back to corner 0 — the one that only exists
        // because the loop closed, and the easy one to leave uneditable.
        val wall = o.wallAt(0f, 1.5f, 0.3f)
        assertEquals(0, wall)
        assertEquals(FloorOutline.Tap.INSERTED, o.insertBefore(wall, 0f, 0f, 1.5f))
        assertEquals(5, o.size)
        assertEquals(12f, o.areaM2(), 1e-4f)
    }

    @Test
    fun `picking finds nothing when the tap is far from everything`() {
        val o = closedRoom()
        assertEquals(-1, o.cornerAt(2f, 1.5f, 0.3f))   // middle of the room
        assertEquals(-1, o.wallAt(2f, 1.5f, 0.3f))
        assertEquals(FloorOutline.Tap.NOTHING, o.select(9))
    }

    @Test
    fun `an open outline is editable before it is closed`() {
        val o = FloorOutline()
        room(o)
        assertFalse(o.isClosed)
        o.select(1)
        assertEquals(FloorOutline.Tap.MOVED, o.moveSelected(5f, 0f, 0f))
        assertEquals(FloorOutline.Tap.DELETED, o.delete(0))
        assertEquals(3, o.size)
    }

    @Test
    fun `a reference corner keeps its identity when it is moved or another corner is inserted`() {
        val o = closedRoom()
        val originalId = requireNotNull(o.cornerId(1))
        o.select(1)
        assertEquals(FloorOutline.Tap.MOVED, o.moveSelected(4.2f, 0f, 0f))
        assertEquals(originalId, o.cornerId(1))

        assertEquals(FloorOutline.Tap.INSERTED, o.insertBefore(1, 2f, 0f, 0f))
        // Inserting ahead of the reference moves its index, but never changes
        // the id written into the capture's correction record.
        assertEquals(originalId, o.cornerId(2))
    }

    // -- tap resolution -----------------------------------------------------
    //
    // The precedence between "close", "select", "move", "insert" and "add" is
    // where this feature can go wrong without anything looking broken, so each
    // rule gets a test rather than being left to the order of a when-block.

    @Test
    fun `tapping the first corner closes the room rather than selecting it`() {
        // Corner picking reaches 35 cm and the closing radius 40 cm, so these
        // two rules genuinely overlap and the order between them is load
        // bearing: get it wrong and the room can never be closed by tapping the
        // target the app is pulsing at the user.
        val o = FloorOutline()
        room(o)
        assertEquals(FloorOutline.Tap.CLOSED, o.tap(0.05f, 0f, 0.05f))
        assertTrue(o.isClosed)
        assertEquals(-1, o.selected)
    }

    @Test
    fun `tapping open floor while a corner is held moves it`() {
        val o = closedRoom()
        o.select(1)
        assertEquals(FloorOutline.Tap.MOVED, o.tap(6f, 0f, 0f))
        assertEquals(15f, o.areaM2(), 1e-4f)
    }

    @Test
    fun `tapping a different corner while holding one re-selects instead of stacking them`() {
        val o = closedRoom()
        o.select(1)
        assertEquals(FloorOutline.Tap.SELECTED, o.tap(4f, 0f, 3f))
        assertEquals(2, o.selected)
        // Corner 1 must still be where it was, not dropped onto corner 2.
        assertEquals(FloorOutline.Corner(4f, 0f, 0f), o.corner(1))
    }

    @Test
    fun `tapping a wall of a closed room inserts a corner there`() {
        val o = closedRoom()
        assertEquals(FloorOutline.Tap.INSERTED, o.tap(2f, 0f, 0f))
        assertEquals(5, o.size)
        assertEquals(12f, o.areaM2(), 1e-4f)
    }

    @Test
    fun `tapping the middle of a closed room does nothing`() {
        val o = closedRoom()
        assertEquals(FloorOutline.Tap.ALREADY_CLOSED, o.tap(2f, 0f, 1.5f))
        assertEquals(4, o.size)
    }

    @Test
    fun `tapping open floor while drawing adds a corner`() {
        val o = FloorOutline()
        assertEquals(FloorOutline.Tap.ADDED, o.tap(0f, 0f, 0f))
        assertEquals(FloorOutline.Tap.ADDED, o.tap(4f, 0f, 0f))
        assertEquals(2, o.size)
    }

    @Test
    fun `long press deletes the held corner but otherwise undoes`() {
        val o = closedRoom()
        o.select(2)
        assertEquals(FloorOutline.Tap.DELETED, o.longPress())
        assertEquals(3, o.size)
        assertEquals(-1, o.selected)

        // Nothing held: falls back to undo, which re-opens a closed room.
        assertEquals(FloorOutline.Tap.DELETED, o.longPress())
        assertFalse(o.isClosed)
        assertEquals(3, o.size)
    }

    @Test
    fun `long press on an empty outline does nothing`() {
        assertEquals(FloorOutline.Tap.NOTHING, FloorOutline().longPress())
    }

    @Test
    fun `corner height differences do not stop the loop closing`() {
        // Skirting boards, thresholds and a sloping floor all mean the closing
        // tap is rarely at the same height as the first corner.
        val o = FloorOutline()
        o.add(0f, 0.00f, 0f)
        o.add(4f, 0.02f, 0f)
        o.add(4f, -0.03f, 3f)
        o.add(0f, 0.01f, 3f)
        assertEquals(FloorOutline.Tap.CLOSED, o.add(0.1f, 0.35f, 0.1f))
        assertEquals(12f, o.areaM2(), 1e-3f)
    }
}
