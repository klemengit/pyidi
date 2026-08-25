"""Turning a score image and a mask into the points to track.

A threshold on its own is not a selection. On the sparse grid the old filter
scored, "keep everything above 0.2 of the maximum" was a reasonable answer
because the grid had already spaced the candidates out. On a dense score image
it returns a solid blob of adjacent pixels around every strong corner -- a
thousand subsets stacked on top of each other, all tracking the same feature.

So selection is threshold *plus* non-maximum suppression: walk the candidates
from best to worst and accept one only if nothing already accepted is within
``min_distance`` of it. This is what ``goodFeaturesToTrack`` does, and it is
what makes the result look like a set of features rather than a heat map.

The suppression is done with a boolean occupancy array rather than pairwise
distances: accepting a point stamps a disc into the array, and a candidate is
rejected by a single lookup. That is what makes the exact greedy walk
affordable -- roughly 20 ms over the 10^5 candidates a 90th-percentile
threshold leaves on a megapixel frame, because the expensive stamping happens
only for the few thousand points actually accepted. There is a cheaper
approximation (keep only local maxima, then suppress), and it is deliberately
not used: it is *not* equivalent, since a candidate dominated by a neighbour
that is itself suppressed can legitimately survive.

The occupancy array also gives the merge with hand-picked points for free --
stamp those into it before picking starts and no automatic point can ever crowd
one.
"""

import inspect

import numpy as np

from ..selection_geometry import _as_size_pair

#: Cap on the number of points a single selection returns. Finite by default:
#: a dense score image with a small minimum distance can otherwise produce tens
#: of thousands of subsets from one careless slider drag.
DEFAULT_MAX_POINTS = 20000

#: Default threshold, as a percentile of the finite in-mask scores. Percentile
#: rather than fraction-of-maximum because the maximum over a megapixel is far
#: more extreme than over a few hundred grid subsets, so a single specular
#: highlight compresses every useful fraction into the bottom of the slider.
DEFAULT_THRESHOLD = 90.0

THRESHOLD_MODES = ('percentile', 'fraction')


def threshold_value(score, mask=None, mode='percentile', value=DEFAULT_THRESHOLD):
    """The absolute score a candidate must exceed.

    :param score: score image, ``NaN`` where invalid
    :type score: numpy.ndarray
    :param mask: boolean array restricting which scores are considered; ``None``
        considers the whole image
    :type mask: numpy.ndarray or None
    :param mode: ``'percentile'`` (``value`` in 0..100) or ``'fraction'``
        (``value`` in 0..1, as a fraction of the maximum)
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
    return float(value) * float(values.max())


def _disc(radius):
    """A boolean disc of the given radius, as a ``(2r+1, 2r+1)`` array."""
    offsets = np.arange(-radius, radius + 1)
    return offsets[:, None] ** 2 + offsets[None, :] ** 2 <= radius * radius


def _ordered_candidates(score, eligible):
    """Candidate coordinates, best first, ties broken by row then column.

    :return: ``(rows, cols)`` arrays in acceptance order
    """
    rows, cols = np.nonzero(eligible)
    if not rows.size:
        return rows, cols
    values = score[rows, cols]
    # lexsort's last key is primary: descending score, then ascending row,
    # then ascending column -- so the result never depends on numpy's
    # internal ordering and repeated calls agree exactly.
    order = np.lexsort((cols, rows, -values))
    return rows[order], cols[order]


def suppress(rows, cols, shape, min_distance, max_points=None, occupied=None):
    """Greedy non-maximum suppression over candidates already in priority order.

    :param rows: candidate row coordinates, best first
    :type rows: numpy.ndarray
    :param cols: candidate column coordinates, best first
    :type cols: numpy.ndarray
    :param shape: ``(rows, cols)`` of the image
    :type shape: tuple[int, int]
    :param min_distance: minimum Euclidean distance between accepted points; 0
        accepts every candidate
    :type min_distance: float
    :param max_points: stop after this many; ``None`` for no limit
    :type max_points: int or None
    :param occupied: boolean array of positions already taken, e.g. by
        hand-picked points. Copied, not modified.
    :type occupied: numpy.ndarray or None
    :return: accepted ``(row, col)`` coordinates, in acceptance order
    :rtype: list[tuple[int, int]]
    """
    radius = int(min_distance)
    taken = np.zeros(shape, dtype=bool) if occupied is None else np.asarray(occupied, dtype=bool).copy()
    disc = _disc(radius) if radius > 0 else None
    height, width = shape

    accepted = []
    for row, col in zip(rows.tolist(), cols.tolist()):
        if taken[row, col]:
            continue
        accepted.append((row, col))
        if max_points is not None and len(accepted) >= max_points:
            break
        if disc is None:
            taken[row, col] = True
            continue
        r0, r1 = max(0, row - radius), min(height, row + radius + 1)
        c0, c1 = max(0, col - radius), min(width, col + radius + 1)
        taken[r0:r1, c0:c1] |= disc[r0 - row + radius:r1 - row + radius,
                                    c0 - col + radius:c1 - col + radius]
    return accepted


def select_peaks(score, mask=None, min_distance=10, threshold=DEFAULT_THRESHOLD,
                 threshold_mode='percentile', max_points=DEFAULT_MAX_POINTS, occupied=None):
    """Pick the strongest well-separated subsets from a score image.

    :param score: score image, ``NaN`` where invalid
    :type score: numpy.ndarray
    :param mask: boolean array restricting where points may be placed; ``None``
        allows the whole image
    :type mask: numpy.ndarray or None
    :param min_distance: minimum Euclidean distance between selected points
    :type min_distance: float
    :param threshold: threshold in the units ``threshold_mode`` implies
    :type threshold: float
    :param threshold_mode: ``'percentile'`` or ``'fraction'``
    :type threshold_mode: str
    :param max_points: keep at most this many, highest-scoring first
    :type max_points: int or None
    :param occupied: positions already taken, which no selected point may fall on
    :type occupied: numpy.ndarray or None
    :return: ``(row, col)`` coordinates, best first
    :rtype: list[tuple[int, int]]
    """
    limit = threshold_value(score, mask, threshold_mode, threshold)
    # NaN compares False against any threshold, so the invalid border is
    # excluded here without a separate test.
    eligible = score > limit
    if mask is not None:
        eligible &= mask

    rows, cols = _ordered_candidates(score, eligible)
    return suppress(rows, cols, score.shape, min_distance, max_points, occupied)


def select_lattice(score, mask=None, pitch=12, threshold=DEFAULT_THRESHOLD,
                   threshold_mode='percentile', max_points=DEFAULT_MAX_POINTS, occupied=None):
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
    :param threshold: threshold in the units ``threshold_mode`` implies. Use a
        percentile of 0 to keep every grid position with a finite score.
    :type threshold: float
    :param threshold_mode: ``'percentile'`` or ``'fraction'``
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
        ``min_distance`` for ``peaks``, ``pitch`` for ``lattice`` -- can be
        carried around and handed to either.
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
    """Thin a list of unscored coordinates, keeping their order.

    Scored points are better thinned by the minimum distance in
    :func:`select_peaks`, which keeps the *best* point in each neighbourhood.
    This is for coordinates that carry no score -- a hand-placed line, say --
    where there is nothing to rank by and even spacing through the sequence is
    the only sensible answer.

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
    seen = set()
    merged = []
    for point in list(literal) + list(picked):
        key = (int(point[0]), int(point[1]))
        if key in seen:
            continue
        seen.add(key)
        merged.append(key)
    return as_point_array(merged)


def occupancy(points, shape, min_distance):
    """Positions blocked by an existing set of points.

    Used to give hand-picked points precedence: the selector is handed this
    array and cannot place anything within ``min_distance`` of one of them.

    :param points: ``(row, col)`` coordinates
    :type points: sequence
    :param shape: ``(rows, cols)`` of the image
    :type shape: tuple[int, int]
    :param min_distance: radius blocked around each point
    :type min_distance: float
    :return: boolean array indexed ``[row, col]``
    :rtype: numpy.ndarray
    """
    taken = np.zeros(shape, dtype=bool)
    radius = int(min_distance)
    height, width = shape
    disc = _disc(radius) if radius > 0 else None
    for point in points:
        row, col = int(point[0]), int(point[1])
        if not (0 <= row < height and 0 <= col < width):
            continue
        if disc is None:
            taken[row, col] = True
            continue
        r0, r1 = max(0, row - radius), min(height, row + radius + 1)
        c0, c1 = max(0, col - radius), min(width, col + radius + 1)
        taken[r0:r1, c0:c1] |= disc[r0 - row + radius:r1 - row + radius,
                                    c0 - col + radius:c1 - col + radius]
    return taken
