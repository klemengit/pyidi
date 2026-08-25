"""Manual test-drive of the automatic feature selection.

Run this CELL BY CELL in an interactive session (molten.nvim, IPython, VS Code),
not with ``python examples/feature_selection_demo.py``.

Why: ``FeatureSelectionGUI.__init__`` ends with ``sys.exit(app.exec())`` whenever
``sys.ps1`` is absent, which is the case for a plain script run. The window would
open, and closing it would terminate the interpreter before any of the checks
below could run. In an interactive session it calls a bare ``app.exec()`` instead,
which blocks until you close the window and then hands control back.

Either way the GUI is modal: execution stops at the ``FeatureSelectionGUI(...)``
line until you close the window.

Requires the Qt extras:  pip install pyidi[qt]
"""

# %%

import numpy as np
import pyidi

print(f'pyidi {pyidi.__version__}')

VIDEO = 'data/data_synthetic.cih'      # 128 x 256, 101 frames - general purpose
# VIDEO = 'data/data_showcase.cih'     # 40 x 640 beam - good for a single region

video = pyidi.VideoReader(VIDEO)
print(f'{video.N} frames, {video.image_height} x {video.image_width} (height x width)')

# %%
# ---------------------------------------------------------------------------
# 1. The three steps.
#
# This is a DIFFERENT interface from SelectionGUI, not a replacement. The old
# one places subsets and then filters them; this one scores the whole image and
# lets the selection find the features inside the region you drew.
#
# --- MASK ------------------------------------------------------------------
#   Polygon       - click corners; the enclosed AREA becomes a mask
#   Brush         - hold Ctrl and drag to paint an area
#   Line          - click vertices; points spaced along the segments
#   Points        - click to drop individual points
#   Remove point  - click near a point to delete it
#
#   * A REGION PLACES NO POINTS BY ITSELF. Draw a polygon and nothing appears
#     until you have a score. That is the point of the redesign.
#   * EVERY ROW HAS A ROLE, shown in the list: 'mask' or 'points'. Polygons and
#     brush strokes start as 'mask'; lines and clicked points start as 'points'.
#     'Use as points' / 'Use as mask' switches a row over WITHOUT redrawing it.
#   * DESELECT BRUSH. Toggle 'Deselect painted area' and paint over part of a
#     region: only the painted part goes, and it stays gone when you change the
#     subset size afterwards.
#   * Ctrl+Z undoes a vertex add, a vertex move, a stroke, a deletion, a point
#     removal, and a deselection.
#
# --- EVALUATE --------------------------------------------------------------
#   * TURN ON 'Show score overlay'. This is the whole image scored at once. It
#     is worth looking at before you pick anything - you can see where the
#     trackable content actually is.
#   * SWITCH TO 'Gradient in direction' and hit the X / Y preset buttons. The
#     heatmap should change character completely.
#   * SWITCH BACK to Shi-Tomasi. It should be instant: the array is cached.
#
# --- SELECT ----------------------------------------------------------------
#   * DRAG THE THRESHOLD. The points update as you drag, with no recomputation.
#     Same for 'Minimum distance' and 'Maximum points'.
#   * RAISE 'Minimum distance'. Points thin out but stay on the strongest spots
#     - unlike a grid, which thins out wherever the grid happens to land.
#   * SWITCH THE SELECTOR to 'lattice'. That is the old regular-grid behaviour,
#     reproduced inside the same pipeline.
#   * NOW GO BACK TO 'Mask' AND CHANGE THE SUBSET SIZE. This one DOES recompute
#     - the score has to answer "how well would THIS subset track".
#
# CLOSE THE WINDOW to continue.
# ---------------------------------------------------------------------------

gui = pyidi.FeatureSelectionGUI(video, subset_size=11)

# %%

points = np.asarray(gui.points)

print(f'selected {len(points)} points')
print(f'dtype {points.dtype}, shape {points.shape}')
if len(points):
    print(f'row (y) range: {points[:, 0].min()} .. {points[:, 0].max()}  (image height {video.image_height})')
    print(f'col (x) range: {points[:, 1].min()} .. {points[:, 1].max()}  (image width  {video.image_width})')

# EXPECT an (N, 2) integer array in (row, col) order, with the row values bounded
# by the image HEIGHT and the column values by the image WIDTH.

# %%
# ---------------------------------------------------------------------------
# 2. Hand the points to a method. Nothing special is needed - the pipeline
#    returns exactly what set_points() wants.
# ---------------------------------------------------------------------------

lk = pyidi.LucasKanade(video)
lk.set_points(points)
print(f'{len(lk.points)} points, dtype {lk.points.dtype}')

# %%
# ---------------------------------------------------------------------------
# 3. The same thing without a GUI. This imports without Qt.
# ---------------------------------------------------------------------------

from pyidi.selection import Entry, select_points  # noqa: E402

frame = video.get_frame(0)
h, w = frame.shape

region = Entry('polygon', [(10, 10), (10, w - 10), (h - 10, w - 10), (h - 10, 10)])
scripted = select_points(frame, [region], subset_size=11, min_distance=15, threshold=90)

print(f'{len(scripted)} points from the whole frame')

# %%
# ---------------------------------------------------------------------------
# 4. Sweeping a parameter. SelectionPipeline keeps the score cache alive, so
#    only the FIRST of these actually evaluates anything.
# ---------------------------------------------------------------------------

from pyidi.selection import SelectionPipeline  # noqa: E402

pipeline = SelectionPipeline(frame, subset_size=11)
pipeline.add_entry('polygon', region.geometry)

for threshold in (70, 80, 90, 95, 99):
    pipeline.selector_params['threshold'] = threshold
    print(f'threshold {threshold:>3}  ->  {len(pipeline.points):>4} points')

print(f'\nevaluator runs: {pipeline.store.n_evaluations}')
# EXPECT 1. If this printed 5, the score cache is not doing its job.

# %%
# ---------------------------------------------------------------------------
# 5. Mixing automatic and hand-picked points. The literal points win: they
#    survive any threshold, and nothing automatic is placed next to them.
# ---------------------------------------------------------------------------

pipeline.selector_params.update({'threshold': 95, 'min_distance': 12})
automatic = len(pipeline.points)

pipeline.add_entry('points', [(h // 2, w // 2)])
mixed = pipeline.points

print(f'{automatic} automatic -> {len(mixed)} with one hand-picked point added')
assert (mixed == np.array([h // 2, w // 2])).all(axis=1).any(), 'the hand-picked point was dropped!'

distances = np.hypot(mixed[:, 0] - h // 2, mixed[:, 1] - w // 2)
print(f'nearest automatic point is {np.sort(distances)[1]:.1f} px away (min_distance is 12)')

# %%
# ---------------------------------------------------------------------------
# 6. Score images are named and cached per (evaluator, parameters, subset size),
#    so two criteria can be live at once without either discarding the other.
# ---------------------------------------------------------------------------

pipeline.define_score('sideways', 'gradient_direction', direction=(0, 1))
pipeline.define_score('upright', 'gradient_direction', direction=(1, 0))

for name in (pipeline.default_score, 'sideways', 'upright'):
    array = pipeline.store.get(name)
    print(f'{name:>10}: max {np.nanmax(array):.3g}')

print(f'\nevaluator runs: {pipeline.store.n_evaluations}')

# %%
# ---------------------------------------------------------------------------
# 7. Adding an evaluator. No GUI code is involved - register a function and
#    its parameter descriptors, and it turns up in the Evaluate menu.
# ---------------------------------------------------------------------------

from scipy.ndimage import uniform_filter  # noqa: E402

from pyidi.selection import Evaluator, register_evaluator  # noqa: E402


def local_variance(image, window):
    """Variance of the image inside the subset window."""
    img = np.asarray(image, dtype=np.float64)
    mean = uniform_filter(img, size=window, mode='constant')
    mean_of_squares = uniform_filter(img * img, size=window, mode='constant')
    return mean_of_squares - mean * mean


register_evaluator(Evaluator(
    name='variance',
    display_name='Local variance',
    function=local_variance,
    parameters=(),
    description='Contrast inside the subset.',
))

variance_points = select_points(frame, [region], subset_size=11,
                                evaluator='variance', min_distance=15)
print(f'{len(variance_points)} points from the new evaluator')

# %%
# Open the GUI again - 'Local variance' should now be in the evaluator menu.

gui2 = pyidi.FeatureSelectionGUI(video, subset_size=11)
