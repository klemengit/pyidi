"""Turning a score image and a mask into the points to track.

A threshold on its own is not a selection. On the sparse grid the old filter
scored, "keep everything above 0.2 of the maximum" was a reasonable answer
because the grid had already spaced the candidates out. On a dense score image
it returns a solid blob of adjacent pixels around every strong corner -- a
thousand subsets stacked on top of each other, all tracking the same feature.

So the selection has one other control: a **separation**, the distance no two
selected points may come closer than. Thinning the thresholded pixels any
other way does not work, and the numbers are worth recording, because "just keep
every n-th of them" is the obvious thing to reach for. On a 1024x1024 frame with
357k pixels above the threshold, thinned to twenty thousand points:

  =========================  ==========  ================
  rule                       median gap  pairs under 3 px
  =========================  ==========  ================
  every n-th, best first     2.0 px      78%
  every n-th, in scan order  1.0 px      92%
  separation                 >= n px     0%
  =========================  ==========  ================

Keeping every n-th of a list ordered by score fails because consecutive ranks
are neighbours on the same feature; keeping every n-th in scan order fails
because the stride aliases against the row length and lands in columns. Either
way the great majority of subsets end up on top of another one.

Spacing is enforced by a greedy walk from best to worst, accepting a candidate
only if nothing already accepted is within the separation of it -- what
``goodFeaturesToTrack`` does. The walk uses a boolean occupancy array rather
than pairwise distances: accepting a point stamps a disc into it, and a
candidate is rejected by a single lookup.

That walk is exact but it is linear in the *candidates*, and a loose threshold
leaves hundreds of thousands of them -- 40 ms to 300 ms, which no slider can
drag. So the candidates are reduced first, to the best pixel in each cell of a
grid half the separation across. It is an approximation, and what it costs is
yield: at a separation of 11 it finds 1708 points where the exact walk finds
2193, in 9 ms instead of 39. What it does not cost is the guarantee -- the walk
still runs, so the separation still holds exactly -- and a point count is what
the separation control is for adjusting anyway.

The occupancy array also gives the merge with hand-picked points for free --
stamp those into it before picking starts and no automatic point can ever crowd
one.
"""

import inspect

import numpy as np

from ..selection_geometry import _as_size_pair

#: Cap on the number of points a single selection returns. Finite by default:
#: a dense score image with a separation of 1 can otherwise produce tens of
#: thousands of subsets from one careless slider drag.
DEFAULT_MAX_POINTS = 20000

#: Default threshold: keep anything at least this good a fraction of the best
#: feature in the region. See :data:`ROBUST_MAXIMUM_PERCENTILE` for why "best"
#: is not the literal maximum.
DEFAULT_THRESHOLD = 0.01

#: How many candidates the suppression walk tests for occupancy at a time.
#: Large enough that the vectorised read dominates the Python loop, small enough
#: that a `max_points` cap still stops early rather than doing a whole pass.
SUPPRESS_CHUNK = 4096

#: Percentile of the eligible scores taken as "the best feature here".
#:
#: Not the maximum, which one dust mote or specular highlight can push an order
#: of magnitude above everything real, dragging every useful quality setting
#: into the bottom of the slider. The 99.9th percentile is the same number on a
#: well-behaved frame and survives a handful of outliers on a bad one.
ROBUST_MAXIMUM_PERCENTILE = 99.9

#: Default separation, in pixels: the distance no two selected points come closer
#: than.
DEFAULT_SEPARATION = 11

#: Cell size used to reduce the candidates before the suppression walk, as a
#: fraction of the separation. Half was measured against the exact walk: a third
#: recovers another 12% of the yield for 40% more time, and using the whole
#: separation is 25% faster again but throws away a third of the points.
CANDIDATE_CELL_FRACTION = 2

#: Thresholding rules. ``quality`` is the default and the one to reach for.
#:
#: ``percentile`` looks the most natural and is the least useful on a dense
#: score image, because it ranks *pixels* and pixels are overwhelmingly
#: background: on a typical frame the 90th percentile of the score is under
#: 1/500th of the best feature, so nine tenths of the slider's travel is spent
#: inside the featureless area and the whole selection collapses the moment you
#: leave the top percentile. It is kept because it is the right rule for a
#: lattice, where the candidates have already been spaced out.
#:
#: A third rule, a fraction of the literal maximum, was dropped: it is the same
#: rule as ``quality`` with a reference that one dust mote can move, so on any
#: frame worth using it is indistinguishable and on a bad one it is worse.
THRESHOLD_MODES = ('quality', 'percentile')


def threshold_value(score, mask=None, mode='percentile', value=DEFAULT_THRESHOLD):
    """The absolute score a candidate must exceed.

    :param score: score image, ``NaN`` where invalid
    :type score: numpy.ndarray
    :param mask: boolean array restricting which scores are considered; ``None``
        considers the whole image
    :type mask: numpy.ndarray or None
    :param mode: ``'quality'`` (``value`` in 0..1, as a fraction of the robust
        maximum) or ``'percentile'`` (``value`` in 0..100)
    :type mode: str
    :param value: the threshold in the units ``mode`` implies
    :type value: float
    :return: the absolute threshold; ``inf`` when nothing is eligible, so that
        no candidate can pass
    :rtype: float
    :raises ValueError: if ``mode`` is not a known threshold mode
    """
    if mode not in THRESHOLD_MODES:
        raise ValueError(f"mode must be one of {THRESHOLD_MODES}, got {mode!r}.")

    finite = np.isfinite(score)
    eligible = finite if mask is None else (finite & mask)
    if not eligible.any():
        return np.inf

    values = score[eligible]
    if mode == 'percentile':
        if value <= 0:
            # A slider at zero should keep everything. Taking the 0th percentile
            # literally would return the smallest score, which the strict
            # comparison below then excludes -- so a uniform score image would
            # select nothing at all at the loosest setting.
            return -np.inf
        return float(np.percentile(values, value))
    if value <= 0:
        return -np.inf         # as for a zero percentile: keep everything
    return float(value) * float(np.percentile(values, ROBUST_MAXIMUM_PERCENTILE))


def _disc(radius):
    """A boolean disc of the given radius, as a ``(2r+1, 2r+1)`` array."""
    offsets = np.arange(-radius, radius + 1)
    return offsets[:, None] ** 2 + offsets[None, :] ** 2 <= radius * radius


def _mask_window(mask, cell):
    """The slices bounding the mask's area, snapped back to a whole cell.

    Everything the selection does is linear in the pixels it is handed, and a
    region drawn on a large frame is usually a small part of it. Nothing outside
    the mask can be selected, so the work outside its bounding box -- a
    full-frame threshold comparison, a full-frame block reduction, a full-frame
    occupancy grid -- is spent on pixels that were never in the running.

    The start is floored to a multiple of ``cell`` so that the block grid of
    :func:`_block_best` falls exactly where it would have on the whole frame.
    The reduction has to pick the same winners as before, not merely similar
    ones: a grid offset by a few pixels answers a slightly different question.

    :param mask: boolean array restricting where points may be placed
    :type mask: numpy.ndarray
    :param cell: the block size the reduction will use
    :type cell: int
    :return: ``(row_slice, col_slice)``, or ``None`` if the mask is empty
    :rtype: tuple[slice, slice] or None
    """
    rows = np.flatnonzero(mask.any(axis=1))
    if not rows.size:
        return None
    cols = np.flatnonzero(mask.any(axis=0))
    return (slice((rows[0] // cell) * cell, rows[-1] + 1),
            slice((cols[0] // cell) * cell, cols[-1] + 1))


def _block_best(score, eligible, cell):
    """The best eligible pixel in each ``cell`` x ``cell`` block of the image.

    A cheap way to cut the candidate list down before the suppression walk. Cells
    with nothing eligible in them contribute nothing, so a blank area costs no
    points -- unlike a lattice, which places one wherever the grid falls.

    :param score: score image, ``NaN`` where invalid
    :type score: numpy.ndarray
    :param eligible: boolean array of candidate positions
    :type eligible: numpy.ndarray
    :param cell: block size in pixels; must be at least 2
    :type cell: int
    :return: ``(rows, cols)`` of the winners, in row-major block order
    :rtype: tuple[numpy.ndarray, numpy.ndarray]
    """
    height, width = score.shape
    down, across = -(-height // cell), -(-width // cell)
    # Pad to a whole number of blocks with -inf, so the padding can never win a
    # block and the reshape needs no special case at the right and bottom edges.
    padded = np.full((down * cell, across * cell), -np.inf, dtype=np.float32)
    padded[:height, :width] = np.where(eligible, score, -np.inf)
    blocks = padded.reshape(down, cell, across, cell).transpose(0, 2, 1, 3)
    blocks = blocks.reshape(down, across, cell * cell)

    within = blocks.argmax(-1)
    best = np.take_along_axis(blocks, within[..., None], -1)[..., 0]
    occupied = np.isfinite(best)
    block_rows, block_cols = np.nonzero(occupied)
    offset = within[occupied]
    return block_rows * cell + offset // cell, block_cols * cell + offset % cell


def _ordered_candidates(score, eligible, keep=None):
    """Candidate coordinates, best first, ties broken by row then column.

    :param score: score image
    :type score: numpy.ndarray
    :param eligible: boolean array of candidate positions
    :type eligible: numpy.ndarray
    :param keep: sort only this many of the best candidates. Only safe when the
        caller will accept them all, since a candidate below the cut can still
        be accepted once suppression has rejected the ones above it.
    :type keep: int or None
    :return: ``(rows, cols)`` arrays in acceptance order
    """
    rows, cols = np.nonzero(eligible)
    if not rows.size:
        return rows, cols
    values = score[rows, cols]

    if keep is not None and 0 < keep < rows.size:
        # A loose threshold leaves hundreds of thousands of candidates and the
        # sort dominates the whole selection, yet all but `keep` of them are
        # discarded straight afterwards. Partitioning is linear, and taking
        # everything tied with the worst survivor makes it exact: without that
        # the tiebreak at the cut would fall to numpy's internal partition
        # order instead of the row/column rule below.
        cut = values[np.argpartition(-values, keep - 1)[:keep]].min()
        head = np.flatnonzero(values >= cut)
        rows, cols, values = rows[head], cols[head], values[head]

    # np.nonzero returns its indices in row-major order, so the candidates
    # arrive sorted by row and then by column already. A *stable* sort by
    # descending score therefore leaves ties in that order -- the same result a
    # three-key lexsort gives, for a third of the work, and just as independent
    # of numpy's internal ordering.
    order = np.argsort(-values, kind='stable')
    return rows[order], cols[order]


def suppress(rows, cols, shape, radius, max_points=None, occupied=None):
    """Greedy non-maximum suppression over candidates already in priority order.

    :param rows: candidate row coordinates, best first
    :type rows: numpy.ndarray
    :param cols: candidate column coordinates, best first
    :type cols: numpy.ndarray
    :param shape: ``(rows, cols)`` of the image
    :type shape: tuple[int, int]
    :param radius: no accepted point comes within this distance of another; 0
        accepts every candidate
    :type radius: float
    :param max_points: stop after this many; ``None`` for no limit
    :type max_points: int or None
    :param occupied: boolean array of positions already taken, e.g. by
        hand-picked points. Copied, not modified.
    :type occupied: numpy.ndarray or None
    :return: accepted ``(row, col)`` coordinates, in acceptance order
    :rtype: list[tuple[int, int]]
    """
    radius = int(radius)
    if radius <= 0:
        # Nothing suppresses anything, so the walk is just "drop what is already
        # taken, then cut to the cap" -- two array operations instead of twenty
        # thousand trips round a Python loop.
        if occupied is not None:
            free = ~np.asarray(occupied, dtype=bool)[rows, cols]
            rows, cols = rows[free], cols[free]
        if max_points is not None and len(rows) > max_points:
            rows, cols = rows[:max_points], cols[:max_points]
        return list(zip(rows.tolist(), cols.tolist()))

    taken = np.zeros(shape, dtype=bool) if occupied is None else np.asarray(occupied, dtype=bool).copy()
    disc = _disc(radius)
    height, width = shape

    accepted = []
    for start in range(0, rows.size, SUPPRESS_CHUNK):
        chunk_rows = rows[start:start + SUPPRESS_CHUNK]
        chunk_cols = cols[start:start + SUPPRESS_CHUNK]
        # Occupancy only ever grows, so a candidate that is already covered now
        # would still be covered when its turn came: dropping the whole batch of
        # them in one vectorised read is exact, and it is what keeps the Python
        # loop off the great majority of candidates. A tight separation on
        # a dense score image rejects better than nine in ten.
        free = ~taken[chunk_rows, chunk_cols]
        for row, col in zip(chunk_rows[free].tolist(), chunk_cols[free].tolist()):
            if taken[row, col]:
                continue                # taken by an earlier point in this chunk
            accepted.append((row, col))
            if max_points is not None and len(accepted) >= max_points:
                return accepted
            r0, r1 = max(0, row - radius), min(height, row + radius + 1)
            c0, c1 = max(0, col - radius), min(width, col + radius + 1)
            taken[r0:r1, c0:c1] |= disc[r0 - row + radius:r1 - row + radius,
                                        c0 - col + radius:c1 - col + radius]
    return accepted


def select_peaks(score, mask=None, separation=DEFAULT_SEPARATION, threshold=DEFAULT_THRESHOLD,
                 threshold_mode='quality', max_points=DEFAULT_MAX_POINTS, occupied=None):
    """Pick the strongest subsets from a score image, no two closer than ``separation``.

    :param score: score image, ``NaN`` where invalid
    :type score: numpy.ndarray
    :param mask: boolean array restricting where points may be placed; ``None``
        allows the whole image
    :type mask: numpy.ndarray or None
    :param separation: the distance, in pixels, no two selected points may come
        closer than. 1 keeps every pixel above the threshold.
    :type separation: int
    :param threshold: threshold in the units ``threshold_mode`` implies
    :type threshold: float
    :param threshold_mode: one of :data:`THRESHOLD_MODES`
    :type threshold_mode: str
    :param max_points: keep at most this many, highest-scoring first
    :type max_points: int or None
    :param occupied: positions already taken, which no selected point may fall on
    :type occupied: numpy.ndarray or None
    :return: ``(row, col)`` coordinates, best first
    :rtype: list[tuple[int, int]]
    """
    separation = max(1, int(separation))
    # Two is the smallest cell that reduces anything, and the walk over an
    # unreduced megapixel frame is 300 ms.
    cell = 1 if separation <= 1 else max(2, separation // CANDIDATE_CELL_FRACTION)

    # Everything below runs on the mask's bounding box rather than the frame.
    # The answer is the same -- nothing outside the mask was ever eligible, and
    # `occupied` already carries the area blocked by points outside it.
    offset_r = offset_c = 0
    if mask is not None:
        window = _mask_window(mask, cell)
        if window is None:
            return []
        offset_r, offset_c = window[0].start, window[1].start
        score, mask = score[window], mask[window]
        if occupied is not None:
            occupied = np.asarray(occupied, dtype=bool)[window]

    limit = threshold_value(score, mask, threshold_mode, threshold)
    # NaN compares False against any threshold, so the invalid border is
    # excluded here without a separate test.
    eligible = score > limit
    if mask is not None:
        eligible &= mask

    if separation <= 1:
        # Nothing to suppress, so there is no walk to feed and no reason to
        # reduce anything: it is the pixels above the threshold, capped. What is
        # already taken is dropped here rather than inside the walk, so that
        # every remaining candidate is one that will be accepted -- which is
        # what makes it safe to sort only the best `max_points` of them.
        if occupied is not None:
            eligible = eligible & ~np.asarray(occupied, dtype=bool)
        rows, cols = _ordered_candidates(score, eligible, max_points)
        points = suppress(rows, cols, score.shape, 0, max_points)
    else:
        rows, cols = _block_best(score, eligible, cell)
        order = np.argsort(-score[rows, cols], kind='stable')
        points = suppress(rows[order], cols[order], score.shape, separation, max_points, occupied)

    if points and (offset_r or offset_c):
        shifted = np.asarray(points) + (offset_r, offset_c)
        points = list(zip(shifted[:, 0].tolist(), shifted[:, 1].tolist()))
    return points


def select_lattice(score, mask=None, pitch=12, threshold=DEFAULT_THRESHOLD,
                   threshold_mode='quality', max_points=DEFAULT_MAX_POINTS, occupied=None):
    """Pick subsets on a regular grid, optionally dropping the weak ones.

    Uniform sampling is not always the wrong answer -- for full-field work the
    point is to cover the surface evenly, not to find the best features. This
    reproduces that behaviour inside the same pipeline, so a regular grid is a
    choice of selector rather than a separate code path.

    :param score: score image, ``NaN`` where invalid
    :type score: numpy.ndarray
    :param mask: boolean array restricting where points may be placed
    :type mask: numpy.ndarray or None
    :param pitch: grid step, as a scalar or a ``(rows, cols)`` pair
    :type pitch: int or tuple
    :param threshold: threshold in the units ``threshold_mode`` implies. Use 0
        to keep every grid position with a finite score.
    :type threshold: float
    :param threshold_mode: one of :data:`THRESHOLD_MODES`
    :type threshold_mode: str
    :param max_points: keep at most this many
    :type max_points: int or None
    :param occupied: positions already taken
    :type occupied: numpy.ndarray or None
    :return: ``(row, col)`` coordinates, in row-major grid order
    :rtype: list[tuple[int, int]]
    """
    pitch_r, pitch_c = _as_size_pair(pitch)
    pitch_r, pitch_c = max(1, int(pitch_r)), max(1, int(pitch_c))
    height, width = score.shape

    lattice = np.zeros(score.shape, dtype=bool)
    lattice[::pitch_r, ::pitch_c] = True

    limit = threshold_value(score, mask, threshold_mode, threshold)
    eligible = lattice & (score > limit)
    if mask is not None:
        eligible &= mask

    rows, cols = np.nonzero(eligible)
    taken = None if occupied is None else np.asarray(occupied, dtype=bool)
    points = []
    for row, col in zip(rows.tolist(), cols.tolist()):
        if taken is not None and taken[row, col]:
            continue
        points.append((row, col))
        if max_points is not None and len(points) >= max_points:
            break
    return points


#: Selectors by name, for the same reason evaluators are registered by name: a
#: selection entry stores which one it wants, and the interface builds its
#: controls from the signature rather than hard-coding a panel per selector.
SELECTORS = {
    'peaks': select_peaks,
    'lattice': select_lattice,
}


def select(score, mask=None, selector='peaks', occupied=None, **params):
    """Run a named selector.

    :param score: score image, ``NaN`` where invalid
    :type score: numpy.ndarray
    :param mask: boolean array restricting where points may be placed
    :type mask: numpy.ndarray or None
    :param selector: ``'peaks'`` or ``'lattice'``
    :type selector: str
    :param occupied: positions already taken
    :type occupied: numpy.ndarray or None
    :param params: selector-specific parameters. Parameters this selector does
        not take are ignored rather than raising, so that one set of defaults --
        ``separation`` for ``peaks``, ``pitch`` for ``lattice`` -- can be carried
        around and handed to either.
    :return: ``(n_points, 2)`` integer array of ``(row, col)`` coordinates
    :rtype: numpy.ndarray
    :raises ValueError: if ``selector`` is not a known selector
    """
    if selector not in SELECTORS:
        known = ', '.join(sorted(SELECTORS))
        raise ValueError(f"Unknown selector {selector!r}. Known selectors: {known}.")
    function = SELECTORS[selector]
    accepted = set(inspect.signature(function).parameters)
    kwargs = {k: v for k, v in params.items() if k in accepted}
    points = function(score, mask=mask, occupied=occupied, **kwargs)
    return as_point_array(points)


def as_point_array(points):
    """Normalise a list of ``(row, col)`` pairs to an ``(n, 2)`` integer array.

    :param points: the coordinates, possibly empty
    :type points: sequence
    :return: ``(n, 2)`` array of ``intp``; ``(0, 2)`` when empty, so that callers
        can index columns without a special case
    :rtype: numpy.ndarray
    """
    if len(points) == 0:
        return np.empty((0, 2), dtype=np.intp)
    return np.asarray(points, dtype=np.intp).reshape(-1, 2)


def decimate(points, stride=None, count=None):
    """Thin a list of coordinates, keeping their order.

    This drops points that were already chosen. It is not the same as asking
    for fewer points up front: a wider separation re-selects, moving every point,
    whereas decimation leaves the survivors exactly where they were and simply
    keeps fewer of them. That is what you want when the selection is right and
    only the count is too high for the computation you are about to run -- and it
    is why the two are separate controls. It is also why it is not the separation
    control: it thins a list, and a list thinned by score puts most of what
    survives back-to-back on the same feature.

    ``select_peaks`` returns its points best first, so a stride keeps an even
    sample across the whole quality range rather than the top slice of it.

    :param points: ``(row, col)`` coordinates
    :type points: sequence
    :param stride: keep every ``stride``-th point; ignored when ``None``
    :type stride: int or None
    :param count: keep at most this many, spread evenly through the sequence;
        applied after ``stride``
    :type count: int or None
    :return: the kept coordinates, in the original order
    :rtype: list
    """
    points = list(points)
    if stride is not None and stride > 1:
        points = points[::int(stride)]
    if count is not None and 0 <= count < len(points):
        if count == 0:
            return []
        keep = np.linspace(0, len(points) - 1, int(count)).round().astype(int)
        points = [points[i] for i in dict.fromkeys(keep.tolist())]
    return points


def merge_points(literal, picked):
    """Combine hand-picked and automatically selected points, literals first.

    Duplicates are dropped, keeping the first occurrence, so a hand-picked point
    that the selector would also have chosen appears exactly once. Crowding is
    not handled here -- it is prevented earlier, by stamping the literal points
    into the occupancy array before selection runs.

    :param literal: ``(row, col)`` coordinates contributed by ``points``-role entries
    :type literal: sequence
    :param picked: ``(row, col)`` coordinates from the selector
    :type picked: sequence
    :return: ``(n_points, 2)`` integer array
    :rtype: numpy.ndarray
    """
    combined = np.vstack([as_point_array(literal), as_point_array(picked)])
    if not len(combined):
        return combined
    # Deduplicating through a Python set costs more than the selection itself at
    # twenty thousand points. Folding each coordinate pair into one integer makes
    # it a single `unique`, and sorting the indices it returns puts the survivors
    # back in the order they arrived -- which is the part `unique` alone loses.
    # The fold is taken relative to the lowest coordinate, so that it stays
    # one-to-one even if a caller hands in a negative one.
    low_row, low_col = int(combined[:, 0].min()), int(combined[:, 1].min())
    span = int(combined[:, 1].max()) - low_col + 1
    flat = (combined[:, 0] - low_row) * span + (combined[:, 1] - low_col)
    keep = np.sort(np.unique(flat, return_index=True)[1])
    return combined[keep]


def occupancy(points, shape, radius):
    """Positions blocked by an existing set of points.

    Used to give hand-picked points precedence: the selector is handed this
    array and cannot place anything within ``radius`` of one of them.

    :param points: ``(row, col)`` coordinates
    :type points: sequence
    :param shape: ``(rows, cols)`` of the image
    :type shape: tuple[int, int]
    :param radius: radius blocked around each point
    :type radius: float
    :return: boolean array indexed ``[row, col]``
    :rtype: numpy.ndarray
    """
    taken = np.zeros(shape, dtype=bool)
    radius = int(radius)
    height, width = shape
    points = as_point_array(points)
    if radius <= 0:
        # Nothing to stamp but the points themselves, so the whole thing is one
        # indexed assignment rather than a Python step per point.
        rows, cols = points[:, 0], points[:, 1]
        inside = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
        taken[rows[inside], cols[inside]] = True
        return taken

    disc = _disc(radius)
    for point in points:
        row, col = int(point[0]), int(point[1])
        if not (0 <= row < height and 0 <= col < width):
            continue
        r0, r1 = max(0, row - radius), min(height, row + radius + 1)
        c0, c1 = max(0, col - radius), min(width, col + radius + 1)
        taken[r0:r1, c0:c1] |= disc[r0 - row + radius:r1 - row + radius,
                                    c0 - col + radius:c1 - col + radius]
    return taken
