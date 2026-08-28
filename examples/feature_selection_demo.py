"""Manual test-drive of the automatic feature selection.

Run this CELL BY CELL in an interactive session (molten.nvim, IPython, VS Code),
not with ``python examples/feature_selection_demo.py``.

Why: ``SelectionGUI.__init__`` ends with ``sys.exit(app.exec())`` whenever
``sys.ps1`` is absent, which is the case for a plain script run. The window would
open, and closing it would terminate the interpreter before any of the checks
below could run. In an interactive session it calls a bare ``app.exec()`` instead,
which blocks until you close the window and then hands control back.

Either way the GUI is modal: execution stops at the ``SelectionGUI(...)``
line until you close the window.

Requires the Qt extras:  pip install pyidi[qt]
"""

# %%

import numpy as np
import pyidi

print(f'pyidi {pyidi.__version__}')

VIDEO = 'data/data_synthetic.cih'      # 128 x 256, 101 frames - general purpose
# VIDEO = "/media/klemenzaletelj/My Passport/ETRA_2026/20260618/CTC_02/CTC_02.cihx"
# VIDEO = 'data/data_showcase.cih'     # 40 x 640 beam - good for a single region

video = pyidi.VideoReader(VIDEO)
print(f'{video.N} frames, {video.image_height} x {video.image_width} (height x width)')

# %%
# ---------------------------------------------------------------------------
# 1. Find, then trim.
#
# This is what SelectionGUI names as of 1.4. The window it replaced -- now
# SelectionGUIOld, deprecated -- placed subsets and then filtered them; this one
# scores the whole image and lets the selection find the features, which you
# then trim.
#
# The window OPENS WITH POINTS ALREADY ON IT. The selections list starts with a
# 'Whole image' mask row, so the candidates are there to look at before you have
# drawn anything.
#
# Two tabs, deliberately unnumbered - evaluation does not depend on the mask, so
# there is no step 1.
#
# --- EVALUATE + SELECT -----------------------------------------------------
#   * TURN ON 'Show score overlay'. This is the whole image scored at once. It
#     is worth looking at before you pick anything - you can see where the
#     trackable content actually is.
#   * SWITCH TO 'Gradient in direction' and hit the X / Y preset buttons. The
#     heatmap should change character completely. Then hit 'Draw' and DRAG A
#     LINE on the image - the direction follows the drag, and a red line shows
#     the direction in force.
#   * SWITCH BACK to Shi-Tomasi. It should be instant: the array is cached.
#   * DRAG THE THRESHOLD. The points update as you drag, with no recomputation.
#     Same for 'Separation' and 'Maximum points'. The score overlay is on
#     the same panel, so you can see whether a thin patch is a tight threshold
#     or an area with nothing to offer.
#   * RAISE 'Separation'. It is the one density control: no two points come
#     closer than it, so lowering it is how you ask for more. Points thin out
#     but stay on the strongest spots - unlike a grid, which thins out wherever
#     the grid happens to land, and unlike 'Keep every n-th', which would leave
#     most of the survivors back-to-back on the same feature.
#   * DROP 'Maximum points' TO SOMETHING SMALL and lower the separation until
#     the panel says the cap stopped it. A cap has no other symptom - it just
#     stops adding points, which reads as though the threshold did it.
#   * SWITCH THE SELECTOR to 'lattice'. That is the old regular-grid behaviour,
#     reproduced inside the same pipeline.
#   * NOTE THE PANEL. The selections list is not here: every row in it belongs
#     to the Mask tab. The subset size is, because both tabs read it.
#
# --- MASK ------------------------------------------------------------------
#   Polygon         - click corners; the enclosed AREA becomes a mask
#   Brush           - hold Ctrl and drag to paint an area
#   Line            - click vertices; points spaced along the segments
#   Points          - click to drop individual points
#   Remove point    - click near a point to delete it
#   Remove w/ brush - hold Ctrl and drag to take away what the stroke covers
#
#   * TRIM THE 'Whole image' ROW. Pick 'Remove w/ brush' and paint over
#     the parts you do not want. That is the workflow this ordering exists for:
#     find the candidates, then edit them.
#   * OR DELETE IT and draw your own region. Deleting it selects NOTHING - it
#     does not silently fall back to the whole frame.
#   * A DRAWN REGION STANDS THE 'Whole image' ROW DOWN. Mask rows combine as a
#     union, so a polygon on top of it would otherwise change nothing at all.
#     Drawing one unchecks it and says so in the status bar; tick it again, or
#     Ctrl+Z, to bring the whole frame back.
#   * 'Show score overlay' is HERE TOO, and ganged to the one on the other tab.
#   * EVERY ROW HAS A ROLE, shown in the list: 'mask' or 'points'. Polygons and
#     brush strokes start as 'mask'; lines and clicked points start as 'points'.
#     'Use as points' / 'Use as mask' switches a row over WITHOUT redrawing it.
#   * DESELECTION SURVIVES A PARAMETER CHANGE. Paint part of a region away,
#     then change the subset size: it stays gone.
#   * THREE TIERS OF POINT, and only on this tab. Red is being taken, dim blue
#     is a feature the mask is leaving out, and a magenta ring marks what the
#     row selected in the list accounts for. Draw a small polygon and watch the
#     rest of the frame go blue: that is the difference between "nothing there"
#     and "you masked it away". The blue ones do not move as you paint - they
#     are the whole-frame selection, so a stroke turns them red where it lands.
#   * WATCH THE POINTS AS YOU DESELECT. The ones under the stroke are crossed
#     out while you are still painting, and gone when you let go. Nothing else
#     moves until then: the stroke costs the points it has reached, and the
#     selection is re-run once, when you let go.
#   * 'Clear all' STARTS OVER rather than clearing to nothing: the whole frame
#     comes back, as when the window opened. Deleting the 'Whole image' row on
#     its own still selects nothing - a different act.
#   * Ctrl+Z undoes a vertex add, a vertex move, a stroke, a deletion, a point
#     removal, and a deselection.
#   * CHANGE THE SUBSET SIZE (below the tabs, so it is there on both - the
#     scoring window follows it, and so does the rectangle drawn round each
#     point). This is the one interaction that DOES recompute the score.
#
# CLOSE THE WINDOW to continue.
# ---------------------------------------------------------------------------

gui = pyidi.SelectionGUI(video, subset_size=11)

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
scripted = select_points(frame, [region], subset_size=11, separation=15, threshold=0.02)

print(f'{len(scripted)} points from the whole frame')

# %%
# ---------------------------------------------------------------------------
# 4. Sweeping a parameter. SelectionPipeline keeps the score cache alive, so
#    only the FIRST of these actually evaluates anything.
# ---------------------------------------------------------------------------

from pyidi.selection import SelectionPipeline  # noqa: E402

pipeline = SelectionPipeline(frame, subset_size=11)
pipeline.add_entry('polygon', region.geometry)

for threshold in (0.3, 0.1, 0.03, 0.01, 0.003):
    pipeline.selector_params['threshold'] = threshold
    print(f'quality {threshold:>5}  ->  {len(pipeline.points):>5} points')

print(f'\nevaluator runs: {pipeline.store.n_evaluations}')
# EXPECT 1. If this printed 5, the score cache is not doing its job.

# %%
# ---------------------------------------------------------------------------
# 5. Mixing automatic and hand-picked points. The literal points win: they
#    survive any threshold, and nothing automatic is placed next to them.
# ---------------------------------------------------------------------------

pipeline.selector_params.update({'threshold': 0.02, 'separation': 12})
automatic = len(pipeline.points)

pipeline.add_entry('points', [(h // 2, w // 2)])
mixed = pipeline.points

print(f'{automatic} automatic -> {len(mixed)} with one hand-picked point added')
assert (mixed == np.array([h // 2, w // 2])).all(axis=1).any(), 'the hand-picked point was dropped!'

distances = np.hypot(mixed[:, 0] - h // 2, mixed[:, 1] - w // 2)
print(f'nearest automatic point is {np.sort(distances)[1]:.1f} px away (separation is 12)')

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
                                evaluator='variance', separation=15)
print(f'{len(variance_points)} points from the new evaluator')

# %%
# Open the GUI again - 'Local variance' should now be in the evaluator menu.

gui2 = pyidi.SelectionGUI(video, subset_size=11)
