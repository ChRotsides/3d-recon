"""Per-object segments, and what can be done with them across frames.

The measurement that motivates this: letting each *region* of a frame move on
its own closes the frame-to-frame disagreement from 7.5 cm raw to 3.5 cm, where
one rigid motion for the whole frame only reaches 5.2 cm. Segments are the
principled version of "region" — a couch moves as a couch, not as whichever
1 m cube it happens to straddle.

The hard part is not the segmenting, it is the **correspondence**. Spatial blocks
got matching for free by being defined in world coordinates; SAM's segments are
per-frame and unlabelled, so segment 7 of frame 3 and segment 12 of frame 4 have
to be recognised as the same couch before either can be aligned to the other.
Everything here except `composite_labels` exists to serve that.

Nothing in this module imports torch. SAM runs in `scripts/capture/sam_masks.py`
and leaves a label map per keyframe behind; this operates on those.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Label value for pixels no mask claimed. SAM's automatic mode covers about 70%
#: of a frame, so this is a large minority of every image and must never be
#: mistaken for segment 0.
BACKGROUND = -1


@dataclass(frozen=True)
class SegmentMatch:
    """One segment in one frame identified with a segment in another."""

    frame_a: int
    segment_a: int
    frame_b: int
    segment_b: int
    #: Points of A whose nearest neighbour in B lands in `segment_b`, over all
    #: of A's points that found any neighbour at all.
    agreement: float
    overlap_points: int
    #: Median nearest-neighbour distance between the two, in metres.
    distance_m: float


def composite_labels(
    masks: list[np.ndarray] | np.ndarray,
    *,
    min_area: int = 200,
) -> np.ndarray:
    """Flatten overlapping SAM masks into one label per pixel.

    SAM's automatic mode returns overlapping masks at several scales — a cushion,
    the couch it sits on, and sometimes the whole wall behind both. Alignment
    needs one segment per 3-D point, so the overlaps have to be resolved.

    Painted largest first, so the smallest mask covering a pixel wins. That is
    the right way round: the specific object is what moves rigidly, while the
    large mask that swallows it is usually several objects at once, and aligning
    on the union of a couch and the wall behind it is exactly the whole-frame
    fit we already know bottoms out.
    """
    if isinstance(masks, np.ndarray):
        masks = list(masks)
    if not masks:
        raise ValueError("no masks to composite")

    shape = np.asarray(masks[0]).shape
    areas = np.array([int(np.asarray(m).sum()) for m in masks])
    keep = np.flatnonzero(areas >= min_area)
    labels = np.full(shape, BACKGROUND, dtype=np.int16)
    for out_id, source in enumerate(keep[np.argsort(-areas[keep])]):
        mask = np.asarray(masks[source], dtype=bool)
        if mask.shape != shape:
            raise ValueError(f"mask {source} has shape {mask.shape}, expected {shape}")
        labels[mask] = out_id
    return labels


def sample_labels_to_grid(labels: np.ndarray, height: int, width: int) -> np.ndarray:
    """Nearest-neighbour resample of a label map onto the depth grid.

    Nearest neighbour rather than any interpolation, for the same reason the
    cloud builder samples the image on the depth grid rather than resizing the
    depth: an interpolated label is a segment id that does not exist, and it
    lands exactly on object boundaries where it does the most harm.
    """
    if labels.shape == (height, width):
        return labels
    rows, cols = labels.shape
    ys = np.clip((np.arange(height) * rows / height).astype(np.int32), 0, rows - 1)
    xs = np.clip((np.arange(width) * cols / width).astype(np.int32), 0, cols - 1)
    return labels[ys][:, xs]


def segment_sizes(labels: np.ndarray) -> np.ndarray:
    """Pixel count per segment id, indexed by id."""
    valid = labels[labels >= BACKGROUND + 1]
    if len(valid) == 0:
        return np.zeros(0, dtype=np.int64)
    return np.bincount(valid.ravel())


@dataclass(frozen=True)
class SegmentAlignStats:
    input_points: int
    frames: int
    segments: int
    aligned_segments: int
    unmatched_segments: int
    rejected_segments: int
    carried_points: int
    rounds: int
    median_translation_m: float
    max_translation_m: float
    median_rotation_deg: float


def _procrustes(
    src: np.ndarray,
    dst: np.ndarray,
    *,
    anchor_src: np.ndarray | None = None,
    anchor_dst: np.ndarray | None = None,
    anchor_weight: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Rigid transform taking `src` onto `dst`, no scaling, optionally anchored.

    The reflection guard is not decorative: without forcing det(R) = +1, a
    near-degenerate segment — a flat patch of wall, which many of these are —
    solves to a mirror and turns the object inside out.

    The anchor exists for a worse version of the same degeneracy. **Rigid ICP on
    a planar patch is under-determined**: a piece of wall can slide anywhere
    within its own plane and spin about its normal while every point still has a
    neighbour a millimetre away. Unanchored, 365 of 556 segments ran past the
    displacement cap doing exactly that. Pairing each source point with where it
    *started*, at `anchor_weight`, leaves the well-constrained directions free
    and pulls only the free ones back to where they were.

    It also means the residual this minimises is no longer a fair score for the
    result — a sliding patch lowers it while getting further from the truth.
    """
    if anchor_weight > 0.0 and anchor_src is not None and anchor_dst is not None:
        weights = np.concatenate(
            [np.ones(len(src)), np.full(len(anchor_src), anchor_weight)]
        )
        src = np.concatenate([src, anchor_src])
        dst = np.concatenate([dst, anchor_dst])
    else:
        weights = np.ones(len(src))

    total = weights.sum()
    cs = (weights[:, None] * src).sum(axis=0) / total
    cd = (weights[:, None] * dst).sum(axis=0) / total
    u, _, vt = np.linalg.svd(((src - cs) * weights[:, None]).T @ (dst - cd))
    sign = np.sign(np.linalg.det(vt.T @ u.T))
    rotation = vt.T @ np.diag([1.0, 1.0, sign]) @ u.T
    return rotation, cd - rotation @ cs


def align_segments_to_consensus(
    points: np.ndarray,
    frame_ids: np.ndarray,
    segment_ids: np.ndarray,
    *,
    rounds: int = 2,
    iterations: int = 6,
    max_correspondence_m: float = 0.25,
    min_segment_points: int = 400,
    min_matches: int = 200,
    #: Matches from one partner frame needed before that frame may nominate a
    #: corresponding segment. Below this the vote is noise.
    min_partner_matches: int = 80,
    min_agreement: float = 0.5,
    #: Weight of the "stay where you started" pairing that keeps a planar
    #: segment from sliding within its own plane. Scaled per segment by how
    #: degenerate it is — full for a flat patch, zero for a segment whose
    #: smallest extent is at least a quarter of its largest, which constrains
    #: all six degrees of freedom on its own. A flat anchor for everything
    #: damped the couch as hard as the wall and only closed 12 cm to 6.4.
    anchor_weight: float = 0.6,
    #: A correction larger than the disagreement being corrected (7.5 cm median)
    #: is a wrong match, not a fix.
    max_translation_m: float = 0.15,
    max_rotation_deg: float = 5.0,
    sample_per_segment: int = 8_000,
    sample_per_frame: int = 45_000,
    seed: int = 0,
) -> tuple[np.ndarray, SegmentAlignStats]:
    """Move each segment onto the same object as seen by the other frames.

    Measured ceiling on bundle_e4a5da81: 7.5 cm raw, 5.2 cm for one rigid motion
    per frame, 3.5 cm per 1 m block, **2.4 cm per segment**. Per-object motion is
    the only granularity so far that gets inside the 2 cm voxel, because the
    residual error varies smoothly across a frame and a single rigid motion
    cannot follow it.

    The guard that makes this safe is `min_agreement`. A segment is moved only
    when its points, matched into another frame, land predominantly in *one*
    segment there — that is what says the couch was recognised as the same couch.
    Segments without such a partner keep their position rather than being fitted
    to whatever happened to be nearby, which is the failure that made an earlier
    plane-based attempt worse than doing nothing.

    Points no mask claimed — about 19% of a frame — are carried by the nearest
    aligned segment of their own frame, so background does not tear away from the
    objects standing on it.
    """
    p = np.asarray(points, dtype=np.float64)
    frames = np.asarray(frame_ids)
    segments = np.asarray(segment_ids)
    if p.ndim != 2 or p.shape[1] != 3:
        raise ValueError(f"expected points shaped (N, 3), got {p.shape}")
    if not (len(p) == len(frames) == len(segments)):
        raise ValueError(f"{len(p)} points, {len(frames)} frames, {len(segments)} segments")
    if rounds < 1 or iterations < 1:
        raise ValueError("rounds and iterations must be at least one")

    from scipy.spatial import cKDTree

    rng = np.random.default_rng(seed)
    work = p.copy()
    # A globally unique id per (frame, segment), so a vote can name one object.
    span = int(segments.max()) + 2 if len(segments) else 2
    key = frames.astype(np.int64) * span + np.maximum(segments, -1)
    groups = [g for g in np.unique(key[segments >= 0])]

    shifts: list[float] = []
    angles: list[float] = []
    aligned = unmatched = rejected = 0

    for _ in range(rounds):
        aligned = unmatched = rejected = 0
        shifts, angles = [], []
        # One tree per source frame, over every *other* frame's points. Querying
        # a tree that contains the source frame is useless: a point's nearest
        # neighbour is then itself at distance zero, every match is rejected as
        # same-frame, and no segment ever finds a partner. Excluding the frame
        # up front is exact, where taking the k nearest and filtering afterwards
        # only works while k happens to be large enough.
        for source_frame in np.unique(frames):
            here = [g for g in groups if g // span == source_frame]
            if not here:
                continue
            at = []
            for f in np.unique(frames):
                if f == source_frame:
                    continue
                where = np.flatnonzero(frames == f)
                if len(where) > sample_per_frame:
                    where = rng.choice(where, size=sample_per_frame, replace=False)
                at.append(where)
            if not at:
                continue
            at = np.concatenate(at)
            tree = cKDTree(work[at])
            target_key = key[at]
            target_frame = frames[at]

            for group in here:
                member = np.flatnonzero(key == group)
                if len(member) < min_segment_points:
                    continue
                sample = member
                if len(sample) > sample_per_segment:
                    sample = rng.choice(member, size=sample_per_segment, replace=False)

                # How much of the rigid fit this segment can actually
                # constrain. A flat patch leaves three degrees of freedom free
                # and needs holding; a couch constrains all six and must not be
                # held, or the correction it exists to make is damped away.
                centred = work[sample] - work[sample].mean(axis=0)
                moments = centred.T @ centred / len(sample)
                spread = np.sqrt(np.maximum(np.linalg.eigvalsh(moments), 0))
                shape = float(spread[0] / max(spread[2], 1e-9))
                held = anchor_weight * float(np.clip(1.0 - shape / 0.25, 0.0, 1.0))

                rotation = np.eye(3)
                translation = np.zeros(3)
                moving = work[sample].copy()
                matched_any = False
                for _ in range(iterations):
                    dist, idx = tree.query(moving, k=1)
                    near = dist < max_correspondence_m
                    if near.sum() < min_matches:
                        break
                    # Agreement is decided *per partner frame*, then unioned.
                    # Taking one dominant partner across all 26 other frames at
                    # once splits the vote between them: it fell under the bar
                    # for 193 segments and locked 273 more onto a wrong partner,
                    # which is what the runaway fits were. Each frame gets to
                    # nominate its own matching segment, and only frames that
                    # nominate one clearly contribute correspondences.
                    hit_frame = target_frame[idx]
                    hit_key = target_key[idx]
                    ok = np.zeros(len(moving), dtype=bool)
                    for partner in np.unique(hit_frame[near]):
                        from_partner = near & (hit_frame == partner)
                        if from_partner.sum() < min_partner_matches:
                            continue
                        votes, counts = np.unique(hit_key[from_partner], return_counts=True)
                        if counts.max() / counts.sum() < min_agreement:
                            continue
                        ok |= from_partner & (hit_key == votes[counts.argmax()])
                    if ok.sum() < min_matches:
                        break
                    matched_any = True
                    step_r, step_t = _procrustes(
                        moving[ok],
                        tree.data[idx[ok]],
                        anchor_src=moving,
                        anchor_dst=work[sample],
                        anchor_weight=held,
                    )
                    moving = moving @ step_r.T + step_t
                    rotation = step_r @ rotation
                    translation = step_r @ translation + step_t

                if not matched_any:
                    unmatched += 1
                    continue

                anchor = work[sample].mean(axis=0)
                shift = float(np.linalg.norm(translation + (rotation - np.eye(3)) @ anchor))
                cosine = np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0)
                angle = float(np.degrees(np.arccos(cosine)))
                if shift > max_translation_m or angle > max_rotation_deg:
                    rejected += 1
                    continue

                work[member] = work[member] @ rotation.T + translation
                aligned += 1
                shifts.append(shift)
                angles.append(angle)

    # Background travels with the nearest segment of its own frame.
    carried = 0
    loose = np.flatnonzero(segments < 0)
    if len(loose):
        for f in np.unique(frames[loose]):
            here = loose[frames[loose] == f]
            labelled = np.flatnonzero((frames == f) & (segments >= 0))
            if len(labelled) < min_segment_points:
                continue
            anchors = labelled
            if len(anchors) > sample_per_frame:
                anchors = rng.choice(anchors, size=sample_per_frame, replace=False)
            # Move each loose point by the displacement its nearest labelled
            # neighbour received, taken before and after in the same frame.
            tree = cKDTree(p[anchors])
            nearest = tree.query(p[here], k=1)[1]
            work[here] += work[anchors][nearest] - p[anchors][nearest]
            carried += len(here)

    stats = SegmentAlignStats(
        input_points=len(p),
        frames=int(len(np.unique(frames))),
        segments=len(groups),
        aligned_segments=aligned,
        unmatched_segments=unmatched,
        rejected_segments=rejected,
        carried_points=carried,
        rounds=rounds,
        median_translation_m=float(np.median(shifts)) if shifts else 0.0,
        max_translation_m=float(np.max(shifts)) if shifts else 0.0,
        median_rotation_deg=float(np.median(angles)) if angles else 0.0,
    )
    return work.astype(np.float32), stats
