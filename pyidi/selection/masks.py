"""Selection entries, and the masks and literal points they contribute.

The central idea of this package: a region drawn on the image defines an *area*,
not a set of points. Where the points go is decided later, by the selection
step, from the score image. That is what lets a filter find features the region
never sampled -- the failure mode of a regular grid on a random speckle pattern.

Not every entry wants that treatment, though. A hand-clicked point or a line of
points placed by eye is a statement about exactly where a subset belongs, and
running it through a threshold would be perverse. So every entry carries a
**role**:

``mask``
    the entry contributes its area to the combined mask, and contributes no
    coordinates of its own;
``points``
    the entry contributes coordinates directly, bypassing evaluation and
    selection entirely, and contributes nothing to the mask.

Polygons and brush strokes default to ``mask``; manual points and polylines
default to ``points``. Any entry's role can be changed after the fact, which is
what makes a hand-drawn region usable either way without redrawing it.

Coordinate convention: ``(row, col)`` throughout, and masks are boolean arrays
indexed ``[row, col]``. The helpers in :mod:`pyidi.selection_geometry` are
reused rather than reimplemented, with a flip at the call site for the three
that work in ``(x, y)``.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Set

import numpy as np
from matplotlib.path import Path

from ..selection_geometry import points_along_polygon, rois_inside_mask, rois_inside_polygon

#: Every entry kind, and the role it takes unless told otherwise. A polygon or a
#: painted area is a statement about a region; a clicked point or a line placed
#: by eye is a statement about specific locations.
DEFAULT_ROLE = {
    'polygon': 'mask',
    'brush': 'mask',
    'polyline': 'points',
    'points': 'points',
}

ROLES = ('mask', 'points')


@dataclass
class Entry:
    """One row of the selection: a piece of geometry plus how to use it.

    :param kind: ``'polygon'``, ``'brush'``, ``'polyline'`` or ``'points'``
    :type kind: str
    :param geometry: ``(row, col)`` vertices for ``polygon``/``polyline``,
        ``(row, col)`` coordinates for ``points``, or a boolean array indexed
        ``[row, col]`` for ``brush``
    :type geometry: list or numpy.ndarray
    :param label: the name shown in the selections list
    :type label: str
    :param role: ``'mask'`` or ``'points'``; defaults per :data:`DEFAULT_ROLE`
    :type role: str or None
    :param visible: whether the entry contributes at all
    :type visible: bool
    :param score_name: which named score this entry is filtered against; ``None``
        means the pipeline's default. Ignored when ``role == 'points'``.
    :type score_name: str or None
    :param selector: registry name of the selector to pick this entry's points
        with; ``None`` means the pipeline's default
    :type selector: str or None
    :param selector_params: parameters for that selector; empty means the
        pipeline's defaults
    :type selector_params: dict
    :param mask_width: width, in pixels, of the stroke a ``polyline`` rasterises
        to when its role is ``mask``. Ignored for every other kind.
    :type mask_width: int
    :param erased: area subtracted from this entry by the deselect brush, as a
        boolean array indexed ``[row, col]``, or ``None``. Kept separate from
        ``geometry`` so the original shape stays intact and undo is a matter of
        dropping this array.

        **Replace it, never write into it.** Both the pipeline's rasterisation
        cache and the interface's undo stack identify this array by object, so
        that neither has to read a frame's worth of booleans to notice a change
        -- and an in-place write is a change neither of them can see. Growing an
        erasure means ``entry.erased = entry.erased | more``, not ``|=``.
    :type erased: numpy.ndarray or None
    :param removed: ``(row, col)`` coordinates removed individually from a
        ``points``-role entry
    :type removed: set
    """

    kind: str
    geometry: Any
    label: str = ''
    role: Optional[str] = None
    visible: bool = True
    score_name: Optional[str] = None
    selector: Optional[str] = None
    selector_params: Dict[str, Any] = field(default_factory=dict)
    mask_width: int = 1
    erased: Optional[np.ndarray] = None
    removed: Set = field(default_factory=set)

    def __post_init__(self):
        if self.kind not in DEFAULT_ROLE:
            known = ', '.join(sorted(DEFAULT_ROLE))
            raise ValueError(f'Unknown entry kind {self.kind!r}. Known kinds: {known}.')
        if self.role is None:
            self.role = DEFAULT_ROLE[self.kind]
        if self.role not in ROLES:
            raise ValueError(f"role must be one of {ROLES}, got {self.role!r}.")


def _stamp_discs(mask, centres, radius):
    """Set a filled disc of ``radius`` in ``mask`` around each ``(row, col)`` centre."""
    h, w = mask.shape
    r = int(radius)
    if r <= 0:
        for row, col in centres:
            row, col = int(round(row)), int(round(col))
            if 0 <= row < h and 0 <= col < w:
                mask[row, col] = True
        return

    offsets = np.arange(-r, r + 1)
    disc = offsets[:, None] ** 2 + offsets[None, :] ** 2 <= r * r
    for row, col in centres:
        row, col = int(round(row)), int(round(col))
        r0, r1 = max(0, row - r), min(h, row + r + 1)
        c0, c1 = max(0, col - r), min(w, col + r + 1)
        if r0 >= r1 or c0 >= c1:
            continue
        sub = disc[r0 - (row - r):r1 - (row - r), c0 - (col - r):c1 - (col - r)]
        mask[r0:r1, c0:c1] |= sub


def _polyline_pixels(vertices):
    """Every ``(row, col)`` pixel along an open polyline, densely sampled."""
    pixels = []
    for start, end in zip(vertices[:-1], vertices[1:]):
        start = np.asarray(start, dtype=float)
        end = np.asarray(end, dtype=float)
        n = max(2, int(np.ceil(np.linalg.norm(end - start))) + 1)
        t = np.linspace(0.0, 1.0, n)[:, None]
        pixels.append(start + t * (end - start))
    if not pixels:
        return np.empty((0, 2))
    return np.vstack(pixels)


def _polygon_mask(vertices, shape):
    """Fill a polygon given by ``(row, col)`` vertices into a boolean array.

    Only the polygon's bounding box is tested, which matters when a small
    polygon is drawn on a large frame -- ``contains_points`` over a full
    megapixel grid is otherwise the slowest thing in the mask step.
    """
    mask = np.zeros(shape, dtype=bool)
    vertices = np.asarray(vertices, dtype=float)
    if len(vertices) < 3:
        return mask

    r0 = max(0, int(np.floor(vertices[:, 0].min())))
    r1 = min(shape[0], int(np.ceil(vertices[:, 0].max())) + 1)
    c0 = max(0, int(np.floor(vertices[:, 1].min())))
    c1 = min(shape[1], int(np.ceil(vertices[:, 1].max())) + 1)
    if r0 >= r1 or c0 >= c1:
        return mask

    rows, cols = np.mgrid[r0:r1, c0:c1]
    inside = Path(vertices).contains_points(
        np.column_stack([rows.ravel(), cols.ravel()])
    )
    mask[r0:r1, c0:c1] = inside.reshape(r1 - r0, c1 - c0)
    return mask


def rasterize(entry, shape):
    """The area an entry covers, as a boolean array indexed ``[row, col]``.

    The entry's ``erased`` area, if any, is subtracted here rather than being
    baked into the geometry, so that a deselection outlives a change of subset
    size or selection parameters without the original shape being lost.

    :param entry: the entry to rasterise
    :type entry: Entry
    :param shape: ``(rows, cols)`` of the image
    :type shape: tuple[int, int]
    :return: the covered area
    :rtype: numpy.ndarray
    :raises ValueError: if a ``brush`` entry's mask does not match ``shape``
    """
    if entry.kind == 'brush':
        mask = np.asarray(entry.geometry, dtype=bool)
        if mask.shape != tuple(shape):
            raise ValueError(f'brush mask has shape {mask.shape}, expected {tuple(shape)}.')
        mask = mask.copy()
    elif entry.kind == 'polygon':
        mask = _polygon_mask(entry.geometry, shape)
    elif entry.kind == 'polyline':
        mask = np.zeros(shape, dtype=bool)
        _stamp_discs(mask, _polyline_pixels(np.asarray(entry.geometry, dtype=float)),
                     max(0, (int(entry.mask_width) - 1) // 2))
    else:  # 'points'
        mask = np.zeros(shape, dtype=bool)
        _stamp_discs(mask, entry.geometry, 0)

    if entry.erased is not None:
        mask &= ~np.asarray(entry.erased, dtype=bool)
    return mask


def combined_mask(entries, shape):
    """Union of the areas of every visible entry whose role is ``mask``.

    :param entries: the selection entries
    :type entries: iterable[Entry]
    :param shape: ``(rows, cols)`` of the image
    :type shape: tuple[int, int]
    :return: the combined area; all-``False`` when nothing contributes
    :rtype: numpy.ndarray
    """
    mask = np.zeros(shape, dtype=bool)
    for entry in entries:
        if entry.visible and entry.role == 'mask':
            mask |= rasterize(entry, shape)
    return mask


def literal_points(entry, subset_size, spacing=0):
    """The coordinates a ``points``-role entry contributes.

    Each kind places its points the way that kind always has: a polyline spaces
    them along its segments, a polygon fills itself with a grid, a brush fills
    its painted area with a grid, and a point list is taken literally.

    The entry's ``removed`` coordinates and ``erased`` area are both applied, so
    a point deselected by hand stays deselected across a change of subset size
    or spacing -- the reason those are recorded on the entry rather than being
    deleted from the derived list.

    :param entry: the entry to read
    :type entry: Entry
    :param subset_size: ``(height, width)`` subset size, or a scalar
    :type subset_size: int or tuple
    :param spacing: extra spacing added to the subset size to get the step
        between neighbouring points
    :type spacing: int
    :return: ``(row, col)`` coordinates
    :rtype: list[tuple[int, int]]
    """
    geom = entry.geometry
    if entry.kind == 'points':
        points = [tuple(p) for p in geom]
    elif entry.kind == 'polyline':
        flipped = [(c, r) for r, c in geom]
        points = [(int(round(y)), int(round(x)))
                  for x, y in points_along_polygon(flipped, subset_size, spacing)]
    elif entry.kind == 'polygon':
        flipped = [(c, r) for r, c in geom]
        points = [(int(round(y)), int(round(x)))
                  for x, y in rois_inside_polygon(flipped, subset_size, spacing)]
    else:  # 'brush'
        mask = np.asarray(geom, dtype=bool)
        points = [(int(r), int(c)) for r, c in rois_inside_mask(mask, subset_size, spacing)]

    if entry.removed:
        points = [p for p in points if p not in entry.removed]
    if entry.erased is not None:
        erased = np.asarray(entry.erased, dtype=bool)
        h, w = erased.shape
        points = [p for p in points
                  if not (0 <= p[0] < h and 0 <= p[1] < w and erased[p[0], p[1]])]
    return points


def all_literal_points(entries, subset_size, spacing=0):
    """Every coordinate contributed by visible ``points``-role entries, in order.

    :param entries: the selection entries
    :type entries: iterable[Entry]
    :param subset_size: ``(height, width)`` subset size, or a scalar
    :type subset_size: int or tuple
    :param spacing: extra spacing added to the subset size
    :type spacing: int
    :return: ``(row, col)`` coordinates
    :rtype: list[tuple[int, int]]
    """
    points = []
    for entry in entries:
        if entry.visible and entry.role == 'points':
            points.extend(literal_points(entry, subset_size, spacing))
    return points


def apply_deselection(entries, stroke, shape, subset_size=11, spacing=0, area=None):
    """Subtract a deselect-brush stroke from every entry it touches.

    Mask-role entries record the stroke in their ``erased`` array; the geometry
    itself is left alone, so only the painted part is lost and the rest of the
    region survives. An entry whose area is wiped out entirely is reported for
    deletion rather than left as an empty row.

    Only entries the stroke actually reaches are given an ``erased`` array. A
    stroke covers a few hundred pixels and an ``erased`` array covers the frame,
    so handing one to every mask entry would make a single dab cost a megabyte
    per region -- and cost it again in every undo snapshot.

    Point-role entries lose the covered coordinates. For a ``points`` entry
    those are deleted from the geometry outright, the same way a click-to-remove
    does: recording them as ``removed`` instead would make a later click on that
    very pixel silently do nothing.

    :param entries: the selection entries, modified in place
    :type entries: list[Entry]
    :param stroke: the painted area, a boolean array indexed ``[row, col]``
    :type stroke: numpy.ndarray
    :param shape: ``(rows, cols)`` of the image
    :type shape: tuple[int, int]
    :param subset_size: ``(height, width)`` subset size, needed to know where a
        point-role entry currently places its points
    :type subset_size: int or tuple
    :param spacing: extra spacing added to the subset size
    :type spacing: int
    :param area: ``f(entry) -> ndarray`` giving an entry's covered area;
        :func:`rasterize` when omitted. A caller holding a rasterisation cache
        passes it here rather than filling every polygon a second time.
    :type area: callable or None
    :return: the entries left with nothing, in the order they appear
    :rtype: list[Entry]
    """
    stroke = np.asarray(stroke, dtype=bool)
    if area is None:
        def area(entry):
            return rasterize(entry, shape)
    emptied = []

    for entry in entries:
        if entry.role == 'mask':
            if not (area(entry) & stroke).any():
                continue
            # `|` rather than `|=`: `erased` is replaced wholesale and never
            # written into, which is what lets both the rasterisation cache and
            # an undo snapshot identify it by object rather than by reading a
            # frame's worth of booleans.
            entry.erased = stroke.copy() if entry.erased is None else (entry.erased | stroke)
            if not area(entry).any():
                emptied.append(entry)
            continue

        covered = {p for p in literal_points(entry, subset_size, spacing)
                   if 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1] and stroke[p[0], p[1]]}
        if not covered:
            continue
        if entry.kind == 'points':
            entry.geometry = [p for p in entry.geometry if tuple(p) not in covered]
            if not entry.geometry:
                emptied.append(entry)
        else:
            entry.removed.update(covered)

    return emptied
