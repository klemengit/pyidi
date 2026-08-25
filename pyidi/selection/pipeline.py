"""The mask -> evaluate -> select pipeline, end to end and without Qt.

Three steps, in the vocabulary settled in issue #51:

1. **mask** -- regions drawn on the image say *where* points may go;
2. **evaluate** -- an evaluator scores every pixel of the image at once;
3. **select** -- a selector turns score plus mask into the points to track.

Only step 2 is expensive, and it depends on nothing but the frame, the
evaluator and the subset size. So editing a mask, dragging a threshold or
changing a minimum distance re-derives the points from a cached array and costs
nothing, which is what makes the interface on top of this able to update while
a slider is still moving.

The entries are the pipeline. Each one carries not just its geometry but which
score it is filtered against and with what parameters, so two regions can be
treated differently without any of this changing shape. An interface that
offers only one global set of controls simply writes the same values into every
entry -- which is what the first version of the GUI does.
"""

import numpy as np

from .masks import Entry, all_literal_points, apply_deselection, combined_mask, literal_points, rasterize
from .scores import ScoreStore, _freeze
from .select import DEFAULT_MAX_POINTS, DEFAULT_THRESHOLD, as_point_array, merge_points, occupancy, select

#: Prefix used to label each kind of entry in the selections list.
PRETTY = {
    'polygon': 'Polygon',
    'brush': 'Brush',
    'polyline': 'Line',
    'points': 'Points',
}

#: Selector parameters used by any entry that does not override them.
DEFAULT_SELECTOR_PARAMS = {
    'min_distance': 10,
    'threshold': DEFAULT_THRESHOLD,
    'threshold_mode': 'percentile',
    'max_points': DEFAULT_MAX_POINTS,
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
        emptied = apply_deselection(self.entries, stroke, self.shape, self.subset_size, self.spacing)
        for entry in emptied:
            self.remove_entry(entry)
        return emptied

    @property
    def mask(self):
        """The union of every visible ``mask``-role entry's area.

        :rtype: numpy.ndarray
        """
        return combined_mask(self.entries, self.shape)

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
        same minimum distance, or a point in one could land right next to a
        point in the other.

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
            groups[key][1] |= rasterize(entry, self.shape)
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
        do it: the selector promotes the next-best pixel nearby and the point
        reappears a pixel or two along, which reads as the click having nudged
        it rather than removed it. What is erased is the disc the point was
        reserving, its minimum distance, so nothing can land nearer to it than a
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
        radius = max(0, int(self.entry_settings(entry)[2].get('min_distance', 0)))
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
        minimum distance of one, and none of them can be displaced.

        :param literal: the hand-picked points to keep clear of; read from the
            entries when omitted
        :type literal: sequence or None
        :return: ``(row, col)`` coordinates, best first within each group
        :rtype: list[tuple[int, int]]
        """
        if literal is None:
            literal = self.literal_points()

        taken = np.zeros(self.shape, dtype=bool)
        picked = []
        for (score_name, selector, params), mask in self._mask_groups():
            if not mask.any():
                continue
            radius = params.get('min_distance', 0)
            occupied = taken | occupancy(literal, self.shape, radius)
            points = select(self.store.get(score_name), mask=mask, selector=selector,
                            occupied=occupied, **params)
            picked.extend((int(r), int(c)) for r, c in points)
            taken |= occupancy(points, self.shape, radius)
        return picked

    def points_by_entry(self):
        """Which points each entry accounts for, aligned with :attr:`entries`.

        A selected point is credited to the first visible mask entry whose area
        covers it, which is what the selections list needs to show a per-row
        count. With overlapping regions the attribution is arbitrary but stable;
        the total is what it always was.

        :return: one list of ``(row, col)`` coordinates per entry, in
            :attr:`entries` order
        :rtype: list[list[tuple[int, int]]]
        """
        credited = [[] for _ in self.entries]
        literal = []
        for index, entry in enumerate(self.entries):
            if entry.visible and entry.role == 'points':
                credited[index] = self.in_frame(
                    literal_points(entry, self.subset_size, self.spacing))
                literal.extend(credited[index])

        masks = [(index, rasterize(entry, self.shape))
                 for index, entry in enumerate(self.entries)
                 if entry.visible and entry.role == 'mask']
        for point in self.picked_points(literal):
            for index, mask in masks:
                if mask[point[0], point[1]]:
                    credited[index].append(point)
                    break
        return credited

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
