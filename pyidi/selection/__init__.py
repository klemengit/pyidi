"""Automatic feature selection: mask, evaluate, select.

Three steps, none of which needs a GUI:

- **mask** (:mod:`~pyidi.selection.masks`) -- regions drawn on the image define
  *where* points may go, and hand-picked entries name exact locations;
- **evaluate** (:mod:`~pyidi.selection.evaluate`) -- an evaluator scores every
  pixel of the image at once, cached by name in
  :mod:`~pyidi.selection.scores`;
- **select** (:mod:`~pyidi.selection.select`) -- a selector turns score plus
  mask into points, with a threshold and a minimum distance between them.

The quick way in::

    from pyidi.selection import Entry, select_points

    region = Entry('polygon', [(20, 20), (20, 200), (180, 200), (180, 20)])
    points = select_points(image, [region], subset_size=11, min_distance=15)

and the stateful way, when scores should be reused across many parameter
changes::

    from pyidi.selection import SelectionPipeline

    pipeline = SelectionPipeline(image, subset_size=11)
    pipeline.add_entry('polygon', [(20, 20), (20, 200), (180, 200), (180, 20)])
    pipeline.selector_params['threshold'] = 95
    points = pipeline.points

This module imports without PyQt6. The interactive interface built on it lives
in :mod:`pyidi.GUIs`.
"""

from .evaluate import (
    Evaluator,
    Parameter,
    available_evaluators,
    evaluate,
    get_evaluator,
    gradient_direction,
    half_window,
    register_evaluator,
    shi_tomasi,
    window_size,
)
from .masks import (
    DEFAULT_ROLE,
    ROLES,
    Entry,
    all_literal_points,
    apply_deselection,
    combined_mask,
    literal_points,
    rasterize,
)
from .pipeline import DEFAULT_SELECTOR_PARAMS, PRETTY, SelectionPipeline, select_points
from .scores import ScoreSpec, ScoreStore
from .select import (
    DEFAULT_MAX_POINTS,
    DEFAULT_THRESHOLD,
    SELECTORS,
    as_point_array,
    decimate,
    merge_points,
    occupancy,
    select,
    select_lattice,
    select_peaks,
    suppress,
    threshold_value,
)

__all__ = [
    'DEFAULT_MAX_POINTS',
    'DEFAULT_ROLE',
    'DEFAULT_SELECTOR_PARAMS',
    'DEFAULT_THRESHOLD',
    'Entry',
    'Evaluator',
    'PRETTY',
    'Parameter',
    'ROLES',
    'SELECTORS',
    'ScoreSpec',
    'ScoreStore',
    'SelectionPipeline',
    'all_literal_points',
    'apply_deselection',
    'as_point_array',
    'available_evaluators',
    'combined_mask',
    'decimate',
    'evaluate',
    'get_evaluator',
    'gradient_direction',
    'half_window',
    'literal_points',
    'merge_points',
    'occupancy',
    'rasterize',
    'register_evaluator',
    'select',
    'select_lattice',
    'select_peaks',
    'select_points',
    'shi_tomasi',
    'suppress',
    'threshold_value',
    'window_size',
]
