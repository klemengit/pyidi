"""The mask -> evaluate -> select pipeline, end to end and without Qt.

Three steps, in the vocabulary settled in issue #51:

1. **mask** -- regions drawn on the image say *where* points may go;
2. **evaluate** -- an evaluator scores every pixel of the image at once;
3. **select** -- a selector turns score plus mask into the points to track.

Only step 2 is expensive, and it depends on nothing but the frame, the
evaluator and the subset size. So editing a mask, dragging a threshold or
changing the separation re-derives the points from a cached array and costs
nothing, which is what makes the interface on top of this able to update while
a slider is still moving.

The entries are the pipeline. Each one carries not just its geometry but which
score it is filtered against and with what parameters, so two regions can be
treated differently without any of this changing shape. An interface that
offers only one global set of controls simply writes the same values into every
entry -- which is what the first version of the GUI does.
"""

import numpy as np

from .masks import Entry, all_literal_points, apply_deselection, literal_points, rasterize
from .scores import ScoreStore, _freeze
from .select import (DEFAULT_MAX_POINTS, DEFAULT_SEPARATION, DEFAULT_THRESHOLD, as_point_array, decimate,
                     merge_points, occupancy, select)

#: Prefix used to label each kind of entry in the selections list.
PRETTY = {
    'polygon': 'Polygon',
    'brush': 'Brush',
    'polyline': 'Line',
    'points': 'Points',
}

#: Selector parameters used by any entry that does not override them.
DEFAULT_SELECTOR_PARAMS = {
    'separation': DEFAULT_SEPARATION,
    'threshold': DEFAULT_THRESHOLD,
    'threshold_mode': 'quality',
    'max_points': DEFAULT_MAX_POINTS,
    'decimation': 1,
}


class SelectionPipeline:
    """Entries, scores and selector settings, and the points they produce.

    :param image: 2-D reference frame, indexed ``[row, col]``
    :type image: numpy.ndarray
    :param subset_size: scalar or ``(height, width)`` pair
    :type subset_size: int or tuple
    :param spacing: extra spacing added to the subset size when a
        ``points``-role entry lays its points out
    :type spacing: int
    """

    def __init__(self, image, subset_size=11, spacing=0):
        self.store = ScoreStore(image, subset_size)
        self.entries = []
        self.spacing = spacing
        #: Score every mask entry is filtered against unless it names another.
        self.default_score = None
        #: Selector every mask entry uses unless it names another.
        self.selector = 'peaks'
        #: Selector parameters every mask entry uses unless it overrides them.
        self.selector_params = dict(DEFAULT_SELECTOR_PARAMS)
        self._label_counters = {kind: 0 for kind in PRETTY}
        #: The whole-frame selection, kept alive across mask edits so that the
        #: interface can show what the mask is leaving out without re-selecting
        #: on every brush stroke. Holds the score array it was computed from,
        #: which is what tells it it is stale.
        self._candidate_cache = None
        #: Rasterised area per entry, keyed by identity and validated against a
        #: fingerprint of the geometry. Filling a polygon is a point-in-path test
        #: over its bounding box, which on a full-frame region is the single most
        #: expensive thing in a redraw -- and a threshold drag does not move a
        #: single vertex.
        self._raster_cache = {}

    # -- image and sizes ---------------------------------------------------

    @property
    def image(self):
        """The reference frame.

        :rtype: numpy.ndarray
        """
        return self.store.image

    @property
    def shape(self):
        """``(rows, cols)`` of the reference frame.

        :rtype: tuple[int, int]
        """
        return self.store.image.shape

    @property
    def subset_size(self):
        """The ``(height, width)`` subset size.

        :rtype: tuple[int, int]
        """
        return self.store.subset_size

    def set_subset_size(self, subset_size):
        """Change the subset size, invalidating every cached score.

        :param subset_size: scalar or ``(height, width)`` pair
        :type subset_size: int or tuple
        """
        self.store.set_subset_size(subset_size)

    def set_image(self, image):
        """Change the reference frame, invalidating every cached score.

        :param image: 2-D frame, indexed ``[row, col]``
        :type image: numpy.ndarray
        """
        self.store.set_image(image)

    # -- scores ------------------------------------------------------------

    def define_score(self, name, evaluator='shi_tomasi', **params):
        """Declare a named score. The first one declared becomes the default.

        :param name: the name entries refer to this score by
        :type name: str
        :param evaluator: registry name of the evaluator
        :type evaluator: str
        :param params: evaluator-specific parameters
        :return: the spec now bound to ``name``
        :rtype: ScoreSpec
        """
        spec = self.store.define(name, evaluator, **params)
        if self.default_score is None:
            self.default_score = name
        return spec

    def ensure_default_score(self):
        """The default score's name, declaring a Shi-Tomasi one if none exists.

        :rtype: str
        """
        if self.default_score is None:
            self.define_score('shi_tomasi', 'shi_tomasi')
        return self.default_score

    # -- entries -----------------------------------------------------------

    def add_entry(self, kind, geometry, label=None, **kwargs):
        """Append an entry and return it.

        Labels are never reused: deleting ``Polygon 2`` and adding another
        polygon gives ``Polygon 4``, not a second ``Polygon 3``, so a label in a
        note or a screenshot always refers to the same thing.

        :param kind: ``'polygon'``, ``'brush'``, ``'polyline'`` or ``'points'``
        :type kind: str
        :param geometry: the entry's geometry; see :class:`~pyidi.selection.masks.Entry`
        :param label: explicit label; generated from a per-kind counter if omitted
        :type label: str or None
        :param kwargs: any other :class:`~pyidi.selection.masks.Entry` field
        :return: the new entry
        :rtype: Entry
        """
        if label is None:
            self._label_counters[kind] += 1
            label = f'{PRETTY[kind]} {self._label_counters[kind]}'
        entry = Entry(kind=kind, geometry=geometry, label=label, **kwargs)
        self.entries.append(entry)
        return entry

    def remove_entry(self, entry):
        """Delete an entry, by identity.

        :param entry: the entry to remove
        :type entry: Entry
        """
        self.entries = [e for e in self.entries if e is not entry]

    def deselect(self, stroke):
        """Subtract a deselect-brush stroke, dropping entries it wipes out.

        :param stroke: the painted area, indexed ``[row, col]``
        :type stroke: numpy.ndarray
        :return: the entries that were removed
        :rtype: list[Entry]
        """
        emptied = apply_deselection(self.entries, stroke, self.shape, self.subset_size,
                                    self.spacing, area=self.area)
        for entry in emptied:
            self.remove_entry(entry)
        return emptied

    @property
    def mask(self):
        """The union of every visible ``mask``-role entry's area.

        The same answer as :func:`~pyidi.selection.masks.combined_mask`, built
        from the rasterised areas this pipeline has already cached rather than
        by filling every polygon again -- this is asked for on every redraw.

        :rtype: numpy.ndarray
        """
        covered = np.zeros(self.shape, dtype=bool)
        for entry in self.entries:
            if entry.visible and entry.role == 'mask':
                covered |= self.area(entry)
        return covered

    def _fingerprint(self, entry):
        """A cheap value that changes whenever an entry's area would.

        Vertex lists are mutated in place, so they are compared by value; a
        brush mask and an erased area are only ever replaced wholesale, so those
        are compared by identity rather than by reading a megabyte of booleans.
        """
        geometry = entry.geometry
        shape = tuple(getattr(geometry, 'shape', ()))
        return (entry.kind, entry.mask_width, self.shape, id(entry.erased),
                id(geometry) if shape else tuple(map(tuple, geometry)))

    def area(self, entry):
        """The area an entry covers, rasterised at most once per change.

        :param entry: the entry to rasterise
        :type entry: Entry
        :return: the covered area, as a boolean array indexed ``[row, col]``.
            Owned by the cache, so callers must not modify it in place.
        :rtype: numpy.ndarray
        """
        fingerprint = self._fingerprint(entry)
        cached = self._raster_cache.get(id(entry))
        if cached is not None and cached[1] == fingerprint:
            return cached[2]
        area = rasterize(entry, self.shape)
        self._raster_cache = {id(e): self._raster_cache[id(e)]
                              for e in self.entries if id(e) in self._raster_cache}
        # The entry itself is held alongside its area. The key is its `id`, and
        # an `id` is only unique among live objects: without a reference here, a
        # deleted entry could be collected and a new one allocated at the same
        # address, which would then read the dead entry's area out of the cache.
        self._raster_cache[id(entry)] = (entry, fingerprint, area)
        return area

    # -- the pipeline ------------------------------------------------------

    def entry_settings(self, entry):
        """The score, selector and parameters an entry is actually run with.

        An entry that names none of them falls back to the pipeline's defaults,
        which is what makes a single global control panel a special case of the
        per-entry model rather than a different one.

        :param entry: the entry to resolve
        :type entry: Entry
        :return: ``(score name, selector name, parameters)``
        :rtype: tuple[str, str, dict]
        """
        params = dict(self.selector_params)
        params.update(entry.selector_params)
        return (entry.score_name or self.ensure_default_score(),
                entry.selector or self.selector,
                params)

    def _mask_groups(self):
        """Visible mask entries grouped by the settings they share.

        Grouping matters: two regions filtered the same way must compete for the
        same separation, or a point in one could land right next to a point in
        the other.

        :return: ``[((score, selector, params), mask), ...]`` in first-seen order
        :rtype: list
        """
        groups = {}
        for entry in self.entries:
            if not (entry.visible and entry.role == 'mask'):
                continue
            score_name, selector, params = self.entry_settings(entry)
            key = (score_name, selector, tuple(sorted((k, _freeze(v)) for k, v in params.items())))
            if key not in groups:
                groups[key] = [(score_name, selector, params), np.zeros(self.shape, dtype=bool)]
            groups[key][1] |= self.area(entry)
        return [tuple(value) for value in groups.values()]

    def in_frame(self, points):
        """The coordinates that lie inside the reference frame.

        Applied to every hand-picked coordinate before it goes anywhere. A click
        lands wherever the interface lets it land, and a subset centred outside
        the frame is not a thing that can be tracked -- so it is dropped here,
        once, rather than being caught by whichever array it is used to index
        first.

        :param points: ``(row, col)`` coordinates
        :type points: sequence
        :return: those inside the frame, in the order given
        :rtype: list[tuple[int, int]]
        """
        array = as_point_array(points)
        if not len(array):
            return []
        height, width = self.shape
        inside = ((array[:, 0] >= 0) & (array[:, 0] < height)
                  & (array[:, 1] >= 0) & (array[:, 1] < width))
        return [(int(row), int(col)) for row, col in array[inside]]

    def remove_point(self, entry, point):
        """Take one displayed point away, and keep it away.

        How depends on where the point came from. A hand-picked coordinate is
        simply deleted, so clicking that same pixel again puts it back.

        A selected point is not stored anywhere -- it is re-derived from the
        score on every redraw -- so the only way to remove one is to take the
        ground it stands on out of the mask. Erasing the single pixel does not
        do it: the selector promotes the next-best pixel of the same block and
        the point reappears a pixel or two away, which reads as the click having
        nudged it rather than removed it. What is erased is the disc the point
        was reserving, its separation, so nothing can land nearer to it than a
        neighbouring point legitimately could have.

        :param entry: the entry the point is credited to
        :type entry: Entry
        :param point: the ``(row, col)`` coordinate to remove
        :type point: tuple[int, int]
        """
        row, col = int(point[0]), int(point[1])
        if entry.role == 'points':
            if entry.kind == 'points':
                entry.geometry = [p for p in entry.geometry if tuple(p) != (row, col)]
            else:
                entry.removed = set(entry.removed) | {(row, col)}
            return
        radius = max(0, int(self.entry_settings(entry)[2].get('separation', 0)))
        blocked = occupancy([(row, col)], self.shape, radius)
        # A new array rather than a write into the old one: `erased` is compared
        # by object identity, both by the rasterisation cache and by undo.
        entry.erased = blocked if entry.erased is None else (entry.erased | blocked)

    def literal_points(self):
        """Every coordinate contributed by visible ``points``-role entries.

        :return: ``(row, col)`` coordinates inside the frame, in entry order
        :rtype: list[tuple[int, int]]
        """
        return self.in_frame(all_literal_points(self.entries, self.subset_size, self.spacing))

    def picked_points(self, literal=None):
        """The points the selector produces, group by group.

        Hand-picked points are stamped into the occupancy array before any group
        runs, so they take precedence: nothing automatic can land within the
        separation of one, and none of them can be displaced.

        :param literal: the hand-picked points to keep clear of; read from the
            entries when omitted
        :type literal: sequence or None
        :return: ``(n_points, 2)`` integer array of ``(row, col)`` coordinates,
            best first within each group
        :rtype: numpy.ndarray
        """
        if literal is None:
            literal = self.literal_points()

        taken = np.zeros(self.shape, dtype=bool)
        literal_taken = occupancy(literal, self.shape, 0) if len(literal) else None
        picked = []
        groups = self._mask_groups()
        for index, ((score_name, selector, params), mask) in enumerate(groups):
            if not mask.any():
                continue
            radius = max(0, int(params.get('separation', 0)))
            occupied = taken
            if literal_taken is not None:
                occupied = taken | (literal_taken if radius <= 0
                                    else occupancy(literal, self.shape, radius))
            points = select(self.store.get(score_name), mask=mask, selector=selector,
                            occupied=occupied, **params)
            # Everything selected is stamped, including what decimation is about
            # to drop, so thinning one group leaves gaps rather than inviting the
            # next group to fill them in. Only what a later group will read: the
            # stamp is a Python loop over every selected point, which is 84 ms at
            # seventeen thousand of them and is usually thrown away unread, since
            # one set of settings for the whole image is one group.
            if index + 1 < len(groups):
                taken |= occupancy(points, self.shape, radius)
            picked.append(as_point_array(decimate(points, stride=params.get('decimation'))))
        return np.vstack(picked) if picked else as_point_array([])

    def candidate_points(self):
        """The points these settings would select over the whole frame.

        What the mask is leaving out, in other words -- an interface can show
        the difference between "there is nothing there" and "you have masked it
        away", which the selection alone cannot say.

        This is the whole frame every time, not the unmasked part, deliberately:
        the answer then depends only on the score and the selector settings, so
        it survives every mask edit and painting a region does not re-select
        anything. The caller filters by the mask, which costs one lookup per
        point.

        :return: ``(n_points, 2)`` integer array of ``(row, col)`` coordinates
        :rtype: numpy.ndarray
        """
        score_name = self.ensure_default_score()
        score = self.store.get(score_name)
        params = dict(self.selector_params)
        key = (self.selector, tuple(sorted((k, _freeze(v)) for k, v in params.items())))
        cached = self._candidate_cache
        # `is`, not `==`: the store replaces the array whenever the evaluator,
        # its parameters or the subset size change, so identity is the version.
        if cached is not None and cached[0] is score and cached[1] == key:
            return cached[2]
        points = select(score, mask=np.ones(self.shape, dtype=bool),
                        selector=self.selector, **params)
        points = as_point_array(decimate(points, stride=params.get('decimation')))
        self._candidate_cache = (score, key, points)
        return points

    def points_and_credits(self):
        """The points, and which entry each one is credited to, from one pass.

        Anything drawing the selection needs both -- the total to plot and the
        per-row counts to label the list with -- and asking for them separately
        runs the whole selection twice for the same answer.

        A selected point is credited to the first visible mask entry whose area
        covers it. With overlapping regions the attribution is arbitrary but
        stable; the total is unaffected.

        :return: ``((n_points, 2) array, one (n, 2) array per entry)``
        :rtype: tuple[numpy.ndarray, list[numpy.ndarray]]
        """
        credited = [as_point_array([]) for _ in self.entries]
        literal = []
        for index, entry in enumerate(self.entries):
            if entry.visible and entry.role == 'points':
                own = self.in_frame(literal_points(entry, self.subset_size, self.spacing))
                credited[index] = as_point_array(own)
                literal.extend(own)

        picked = self.picked_points(literal)
        masks = [(index, self.area(entry))
                 for index, entry in enumerate(self.entries)
                 if entry.visible and entry.role == 'mask']
        if len(picked) and masks:
            # One lookup per mask over the whole point array, rather than one
            # Python step per point: `inside` is (n_masks, n_points), and the
            # first True down each column is the entry that gets the credit.
            rows, cols = picked[:, 0], picked[:, 1]
            inside = np.array([mask[rows, cols] for _, mask in masks])
            owner = inside.argmax(axis=0)
            owned = inside.any(axis=0)
            for position, (index, _) in enumerate(masks):
                credited[index] = picked[owned & (owner == position)]
        return merge_points(literal, picked), credited

    def points_by_entry(self):
        """Which points each entry accounts for, aligned with :attr:`entries`.

        :return: one ``(n, 2)`` array of ``(row, col)`` coordinates per entry,
            in :attr:`entries` order
        :rtype: list[numpy.ndarray]
        """
        return self.points_and_credits()[1]

    def get_points(self):
        """Run the whole pipeline and return the points.

        :return: ``(n_points, 2)`` integer array of ``(row, col)`` coordinates,
            hand-picked points first
        :rtype: numpy.ndarray
        """
        literal = self.literal_points()
        return merge_points(literal, self.picked_points(literal))

    @property
    def points(self):
        """The pipeline's points, as :meth:`get_points` returns them.

        :rtype: numpy.ndarray
        """
        return self.get_points()


def select_points(image, entries=(), subset_size=11, evaluator='shi_tomasi',
                  selector='peaks', spacing=0, evaluator_params=None, **selector_params):
    """Run mask, evaluate and select in one call.

    The headless entry point: no Qt, no interface, just an image and some
    regions in, points out.

    :param image: 2-D reference frame, indexed ``[row, col]``
    :type image: numpy.ndarray
    :param entries: the selection entries; an empty collection masks nothing and
        so selects nothing
    :type entries: iterable[Entry]
    :param subset_size: scalar or ``(height, width)`` pair
    :type subset_size: int or tuple
    :param evaluator: registry name of the evaluator to score with
    :type evaluator: str
    :param selector: registry name of the selector to pick with
    :type selector: str
    :param spacing: extra spacing used when a ``points``-role entry lays out its
        points
    :type spacing: int
    :param evaluator_params: parameters for the evaluator
    :type evaluator_params: dict or None
    :param selector_params: parameters for the selector
    :return: ``(n_points, 2)`` integer array of ``(row, col)`` coordinates
    :rtype: numpy.ndarray
    """
    pipeline = SelectionPipeline(image, subset_size=subset_size, spacing=spacing)
    pipeline.define_score('score', evaluator, **(evaluator_params or {}))
    pipeline.selector = selector
    pipeline.selector_params.update(selector_params)
    pipeline.entries = list(entries)
    return pipeline.get_points()


__all__ = ['SelectionPipeline', 'select_points', 'as_point_array', 'PRETTY',
           'DEFAULT_SELECTOR_PARAMS']
