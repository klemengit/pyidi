"""Manual test-drive of ``SelectionGUIOld``, the deprecated selection window.

Kept so the 1.3 interface can still be exercised while it is around. For the
interface ``SelectionGUI`` names today, see ``feature_selection_demo.py``.

Run this CELL BY CELL in an interactive session (molten.nvim, IPython, VS Code),
not with ``python examples/point_selection_demo.py``.

Why: ``SelectionGUIOld.__init__`` ends with ``sys.exit(app.exec())`` whenever
``sys.ps1`` is absent, which is the case for a plain script run. The window would
open, and closing it would terminate the interpreter before any of the checks
below could run. In an interactive session it calls a bare ``app.exec()`` instead,
which blocks until you close the window and then hands control back.

Either way the GUI is modal: execution stops at the ``SelectionGUIOld(...)`` line
until you close the window. Select your points, then close it to continue.

Requires the Qt extras:  pip install pyidi[qt]
"""

# %%

import numpy as np
import pyidi

print(f'pyidi {pyidi.__version__}')

VIDEO = 'data/data_synthetic.cih'      # 128 x 256, 101 frames - general purpose
# VIDEO = 'data/data_showcase.cih'     # 40 x 640 beam - good for 'Along the line'

video = pyidi.VideoReader(VIDEO)
print(f'{video.N} frames, {video.image_height} x {video.image_width} (height x width)')

# %%
# ---------------------------------------------------------------------------
# 1. Open the GUI and select some points.
#
# Try each of the five selection methods on the right-hand panel:
#   Grid            - click >=3 polygon corners; a grid fills the polygon
#   Manual          - click to drop individual points
#   Along the line  - click polyline vertices; points spaced along the segments
#   Brush           - hold Ctrl and drag to paint; toggle 'Deselect painted area'
#   Remove point    - click near a point to delete it
#
# Then switch to Filter mode (top toolbar) and try Shi-Tomasi / gradient
# direction filtering on top of the selection.
#
# --- things to check specifically ------------------------------------------
#
# The right-hand panel now shows ONE 'selections' list, visible in every mode,
# with one row per grid, per line, per brush stroke, and a single 'Manual' row
# collecting every individually-clicked point.
#
#   * CLICK A ROW. It should become the active selection, the tool should
#     switch to match its type (e.g. clicking a Grid row switches to Grid
#     mode), and its points should highlight in the image.
#   * TOGGLE A ROW'S CHECKBOX. Unchecking it should remove its points from
#     gui.points immediately, without deleting the row. Re-checking it should
#     bring them back.
#   * DELETE A BRUSH STROKE. Select a Brush row and hit 'Delete selected' -
#     it should go. Previously a brush stroke could only be removed via
#     'Clear selections'.
#   * DELETE THE MANUAL ROW. Same as above - previously not possible either.
#   * Ctrl+Z AFTER DELETING A BRUSH STROKE. It should come back at its
#     ORIGINAL row with its ORIGINAL label. Undo now covers deleting any row
#     (grid, line, brush stroke, or the Manual row), not just grid/line as
#     before.
#
# In Grid or 'Along the line' mode:
#   * DRAG A VERTEX. Press within ~10 px of a corner you already placed and
#     drag - it should follow the cursor, and the subsets should re-fill the
#     new shape when you release. Dragging from empty space still pans.
#   * CLICK EXACTLY ON A VERTEX. Nothing should happen. It used to drop a
#     duplicate vertex on top of the existing one.
#   * Ctrl+Z. Still undoes a vertex add and a vertex move, one step at a time.
#   * DELETE THE ONLY GRID/LINE. With exactly one entry in the list, select it
#     and hit delete - it should go. Previously this silently did nothing and
#     you had to create a second one first.
#   * The button above the list should read 'Start new grid' in Grid mode and
#     'Start new line' in 'Along the line' mode.
#
# In Brush mode:
#   * The painted area should now be centred ON the cursor. It used to land
#     about 9 px up and to the left, because the drag handlers were reading
#     ViewBox-local coordinates instead of scene coordinates.
#   * REMOVED POINTS SURVIVE A SPACING CHANGE. Paint a brush stroke (or draw a
#     grid/line), use 'Remove point' to delete a couple of its points, then
#     change 'Distance between subsets' or the subset size. The removed
#     points should stay gone rather than reappearing - previously they were
#     regenerated from the source geometry and silently came back.
#
# CLOSE THE WINDOW to continue.
# ---------------------------------------------------------------------------

gui = pyidi.SelectionGUIOld(video, subset_size=11, subset_overlap=0)

# %%

points = gui.points          # equivalently: gui.get_points()
points = np.asarray(points)

print(f'selected {len(points)} points')
print(f'dtype {points.dtype}, shape {points.shape}')
if len(points):
    print(f'row (y) range: {points[:, 0].min()} .. {points[:, 0].max()}  (image height {video.image_height})')
    print(f'col (x) range: {points[:, 1].min()} .. {points[:, 1].max()}  (image width  {video.image_width})')
    print('\nfirst few points (row, col):')
    print(points[:5])

# EXPECT: an (N, 2) array in (y, x) = (row, column) order, with the row values
# bounded by the image HEIGHT and the column values by the image WIDTH. If those
# two look swapped, the GUI's internal x/y reversal is wrong.

# %%
# ---------------------------------------------------------------------------
# 2. Hand the points to a method. Both forms should work now.
# ---------------------------------------------------------------------------

lk = pyidi.LucasKanade(video)

# (a) pass the GUI object itself - this is the duck-typing fix; it used to fail
#     with an opaque error because only the old SubsetSelection was recognised.
lk.set_points(gui)
print(f'(a) passing the GUI object      -> {len(lk.points)} points, dtype {lk.points.dtype}')

# (b) pass the array
lk.set_points(points)
print(f'(b) passing the array           -> {len(lk.points)} points, dtype {lk.points.dtype}')

assert np.array_equal(lk.points, np.asarray(gui.points)), 'the two forms disagree!'
print('\nboth forms agree.')

# %%
# ---------------------------------------------------------------------------
# 3. Validation. Every one of these used to be accepted silently or to fail
#    with an unhelpful IndexError. They should now raise a clear ValueError.
# ---------------------------------------------------------------------------

sof = pyidi.SimplifiedOpticalFlow(video)

bad_inputs = {
    'empty':                 [],
    '1-D array':             np.array([1, 2]),
    'three columns':         np.array([[1, 2, 3]]),
    'outside the image':     np.array([[9999, 9999]]),
    'negative coordinates':  np.array([[-4, -4]]),
}

for name, value in bad_inputs.items():
    try:
        sof.set_points(value)
        print(f'  {name:22s} -> NOT REJECTED  <-- unexpected')
    except ValueError as e:
        print(f'  {name:22s} -> ValueError: {e}')

# %%
# ---------------------------------------------------------------------------
# 4. Sub-pixel points are rounded to nearest, with a warning.
#    Previously: SimplifiedOpticalFlow crashed on these, while LucasKanade and
#    DirectionalLucasKanade silently truncated TOWARD ZERO (1.7 -> 1).
# ---------------------------------------------------------------------------

import warnings

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter('always')
    sof.set_points(np.array([[1.7, 2.3], [10.5, 20.5]]))
    for w in caught:
        print(f'warning: {w.message}')

print(f'stored: {sof.points.tolist()}  dtype {sof.points.dtype}')
# EXPECT [[2, 2], [10, 20]] as an integer dtype. The point of the check is that
# 1.7 -> 2, NOT 1: the old LucasKanade path truncated toward zero. The 10.5 -> 10
# is numpy's round-half-to-even (np.rint), which is expected, not a bug.

# %%
# ---------------------------------------------------------------------------
# 5. SelectionGUIOld now accepts a plain numpy array, as its docstring always
#    claimed. This raised AttributeError before.
#    CLOSE THE WINDOW to continue.
# ---------------------------------------------------------------------------

frames = video.get_frames()               # (n_frames, height, width)
print(f'passing a 3-D array {frames.shape}')

gui_arr = pyidi.SelectionGUIOld(frames, subset_size=9)
print(f'frame used: {gui_arr.frame.shape}  (should be 2-D: {frames.shape[1:]})')

# %%
# A single 2-D image should work too. CLOSE THE WINDOW to continue.

gui_img = pyidi.SelectionGUIOld(frames[0], subset_size=9)
print(f'frame used: {gui_img.frame.shape}')

# %%
# And an unusable input should now say so clearly, rather than dying on a
# missing attribute deep in the constructor.

try:
    pyidi.SelectionGUIOld('not a video')
except TypeError as e:
    print(f'TypeError: {e}')

# %%
# ---------------------------------------------------------------------------
# 6. The retired widget. Scripts written against SubsetSelection should fail
#    with a message that names the replacement.
# ---------------------------------------------------------------------------

try:
    pyidi.SubsetSelection(video, roi_size=(21, 21), noverlap=0)
except RuntimeError as e:
    print(f'RuntimeError: {e}')

# %%
# ---------------------------------------------------------------------------
# 7. Optional: run a short analysis on the selected points, to confirm the
#    whole chain works end to end. Uses whatever you picked in step 1.
#
# NOTE: points whose ROI reaches past the image edge are clipped with a warning,
# and a point that cannot be tracked comes back as NaN rather than raising
# (behaviour introduced in 1.4.0) - so check for NaN, do not assume success.
# ---------------------------------------------------------------------------

lk.set_points(points)
lk.configure(roi_size=(11, 11), int_order=3, processes=1)
disp = lk.get_displacements()

print(f'displacements shape {disp.shape}  (n_points, n_frames, 2)')
n_failed = int(np.isnan(disp).any(axis=(1, 2)).sum())
print(f'{n_failed} of {len(points)} points failed to track')
if getattr(lk, 'failed_points', None):      # a dict, empty when everything tracked
    print(f'failed_points: {lk.failed_points}')

# %%
# ---------------------------------------------------------------------------
# 8. Optional: the napari GUI, a separate interface again.
#    Its point selection now routes through set_points too, so an out-of-bounds
#    pick warns instead of being accepted silently.
#
#    Pick a method, then 'Set points', then 'Configure', then 'Calculate'.
# ---------------------------------------------------------------------------

# napari_gui = pyidi.GUI(video)
# print(napari_gui.method.points)
