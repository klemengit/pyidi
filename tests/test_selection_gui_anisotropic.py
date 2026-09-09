"""Tests for anisotropic (non-square) subset sizes in ``SelectionGUIOld``.

``SelectionGUIOld`` is a full Qt application, but it can be constructed headlessly
for testing -- the same recipe used by
``docs/source/quick_start/make_selection_animation.py``:

* ``QT_QPA_PLATFORM=offscreen`` must be set before Qt is imported, so Qt
  renders to its software framebuffer instead of opening a real display.
* ``sys.ps1`` must be set before constructing ``SelectionGUIOld``, so its
  constructor takes the "interactive" branch instead of ``sys.exit(...)``.
* ``sys.ps1`` alone is not enough: the "interactive" branch still calls
  ``app.exec()``, which would block in the Qt event loop. So
  ``QtWidgets.QApplication.exec`` is also monkeypatched to a no-op, letting
  construction return immediately with a fully built (but not event-loop-
  driven) window.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import sys  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402

pytest.importorskip("PyQt6")

from PyQt6 import QtCore, QtGui, QtWidgets  # noqa: E402

sys.ps1 = getattr(sys, "ps1", ">>> ")  # Make SelectionGUIOld think it's running interactively.
QtWidgets.QApplication.exec = lambda self=None: 0  # Neutralise the blocking event loop.

from pyidi.GUIs.subset_selection import SelectionGUIOld  # noqa: E402
from pyidi.selection_geometry import rois_inside_polygon  # noqa: E402

# Every test here constructs the deprecated interface on purpose, so its own
# DeprecationWarning is noise rather than a signal. The warning itself is
# covered in tests/test_feature_selection_gui.py.
pytestmark = pytest.mark.filterwarnings(
    "ignore:SelectionGUIOld is deprecated:DeprecationWarning")


def make_image():
    """A synthetic grayscale image, large enough for the polygons used below."""
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, size=(200, 300), dtype=np.uint8)


def make_gui(**kwargs):
    """Construct a headless ``SelectionGUIOld`` on a fresh synthetic image."""
    return SelectionGUIOld(make_image(), **kwargs)


def make_image_128x256():
    """A 128 (height) x 256 (width) synthetic image -- the size used in the bug report."""
    rng = np.random.default_rng(1)
    return rng.integers(0, 255, size=(128, 256), dtype=np.uint8)


def overlay_extent(overlay):
    """Return (extent along axis 0, extent along axis 1) of the non-zero region.

    ``overlay`` is the RGBA array from ``roi_overlay.image``; "non-zero" is
    judged from the alpha channel.
    """
    mask = overlay[..., 3] != 0
    axis0_idx = np.where(mask.any(axis=1))[0]
    axis1_idx = np.where(mask.any(axis=0))[0]
    assert axis0_idx.size and axis1_idx.size, "overlay is entirely empty"
    extent0 = int(axis0_idx.max() - axis0_idx.min() + 1)
    extent1 = int(axis1_idx.max() - axis1_idx.min() + 1)
    return extent0, extent1


def test_constructor_normalizes_scalar_subset_size():
    gui = make_gui(subset_size=11)
    try:
        assert gui.subset_size == (11, 11)
        assert isinstance(gui.subset_size[0], int)
        assert isinstance(gui.subset_size[1], int)
    finally:
        gui.close()


def test_constructor_normalizes_pair_subset_size():
    gui = make_gui(subset_size=(5, 21))
    try:
        assert gui.subset_size == (5, 21)
    finally:
        gui.close()


def test_get_subset_size_reflects_spinboxes():
    gui = make_gui(subset_size=(5, 21))
    try:
        # An anisotropic pair starts with Square unchecked, so both spinboxes reflect
        # the values they were constructed with.
        assert not gui.square_subsets_checkbox.isChecked()
        assert gui.get_subset_size() == (5, 21)

        gui.subset_width_spinbox.setValue(25)
        assert gui.get_subset_size() == (5, 25)

        gui.subset_height_spinbox.setValue(9)
        assert gui.get_subset_size() == (9, 25)
    finally:
        gui.close()


def test_square_checkbox_toggle_locks_and_syncs_width():
    gui = make_gui(subset_size=11)
    try:
        assert gui.square_subsets_checkbox.isChecked()
        assert not gui.subset_width_spinbox.isEnabled()
        assert not gui.subset_width_slider.isVisible()

        gui.square_subsets_checkbox.setChecked(False)
        assert gui.subset_width_spinbox.isEnabled()

        gui.subset_width_spinbox.setValue(31)
        assert gui.get_subset_size() == (11, 31)

        # Toggling square back on snaps width back to height.
        gui.square_subsets_checkbox.setChecked(True)
        assert not gui.subset_width_spinbox.isEnabled()
        assert gui.get_subset_size() == (11, 11)
    finally:
        gui.close()


def test_anisotropic_grid_roi_points_have_different_row_and_column_spacing():
    h, w, overlap = 5, 21, 2
    gui = make_gui(subset_size=(h, w), subset_overlap=overlap)
    try:
        # A rectangular polygon covering most of the synthetic image.
        entry = gui.add_selection('grid', geometry=[(0, 0), (280, 0), (280, 180), (0, 180)])
        gui.recompute_roi_points()

        roi_points = entry['roi_points']
        assert len(roi_points) > 4, "expected a genuine grid of points, not a degenerate case"

        xs = sorted(set(p[0] for p in roi_points))
        ys = sorted(set(p[1] for p in roi_points))
        x_steps = set(round(b - a) for a, b in zip(xs, xs[1:]))
        y_steps = set(round(b - a) for a, b in zip(ys, ys[1:]))

        assert x_steps == {w + overlap}
        assert y_steps == {h + overlap}
        assert x_steps != y_steps
    finally:
        gui.close()


def test_square_subset_size_matches_pre_change_geometry_reference():
    """Regression: a square subset_size must still produce exactly the old ROI points."""
    subset_size, overlap = 15, 3
    gui = make_gui(subset_size=subset_size, subset_overlap=overlap)
    try:
        polygon = [(10, 10), (250, 10), (250, 150), (10, 150)]
        entry = gui.add_selection('grid', geometry=polygon)
        gui.recompute_roi_points()

        expected = rois_inside_polygon(polygon, subset_size, overlap)
        assert entry['roi_points'] == expected
    finally:
        gui.close()


# ---------------------------------------------------------------------------
# Rectangle-drawing / filter-ROI orientation (the "drawn transposed" bug)
# ---------------------------------------------------------------------------

def test_rectangle_overlay_extent_matches_subset_size_axes():
    """subset_size=(5, 25) must draw 25 px along x (axis 0) and 5 px along y (axis 1).

    A manual selection entry's ``geometry`` uses the internal (x, y) convention: (128, 64) sits well
    inside the 128 (height) x 256 (width) image used here, whereas (64, 128)
    would not (x=64 < 256 is fine, but as a first coordinate it would be
    read as a row and 128 is out of the 0..127 row range) -- see the class
    docstring / bug report for why the ordering matters.
    """
    gui = SelectionGUIOld(make_image_128x256(), subset_size=(5, 25))
    try:
        entry = gui.add_selection('manual', geometry=[(128, 64)])
        gui.recompute_entry(entry, gui.get_subset_size(), gui.distance_spinbox.value())
        gui.update_selected_points()

        extent0, extent1 = overlay_extent(gui.roi_overlay.image)
        assert extent0 == 25, f"expected 25 px along axis 0 (x/width), got {extent0}"
        assert extent1 == 5, f"expected 5 px along axis 1 (y/height), got {extent1}"
    finally:
        gui.close()


def test_square_subset_overlay_matches_independent_reference():
    """Regression: a square subset_size must fill byte-identical overlay pixels.

    Only the translucent interior lives in ``roi_overlay``; the border is a separate
    vector path (see ``test_subset_border_path_traces_the_filled_area``).
    """
    subset_size = 15
    half = subset_size // 2
    gui = SelectionGUIOld(make_image_128x256(), subset_size=subset_size)
    try:
        px, py = 128, 64
        entry = gui.add_selection('manual', geometry=[(px, py)])
        gui.recompute_entry(entry, gui.get_subset_size(), gui.distance_spinbox.value())
        gui.update_selected_points()

        n_x, n_y = gui.image_item.image.shape[:2]
        expected = np.zeros((n_x, n_y, 4), dtype=np.uint8)
        ix0, iy0, ix1, iy1 = px - half, py - half, px + half + 1, py + half + 1
        expected[ix0:ix1, iy0:iy1, 1] = 180
        expected[ix0:ix1, iy0:iy1, 3] = 40

        np.testing.assert_array_equal(gui.roi_overlay.image, expected)
    finally:
        gui.close()


def _path_subpaths(path):
    """Split a QPainterPath into a list of (x, y) vertex arrays, one per sub-path."""
    subpaths, current = [], []
    for i in range(path.elementCount()):
        el = path.elementAt(i)
        if el.type == QtGui.QPainterPath.ElementType.MoveToElement and current:
            subpaths.append(np.array(current))
            current = []
        current.append((el.x, el.y))
    if current:
        subpaths.append(np.array(current))
    return subpaths


def test_subset_border_path_traces_the_filled_area():
    """The border path must outline exactly the pixels the overlay fills.

    The border is stroked with a cosmetic pen so it stays a hairline at any zoom;
    that only reads as a subset boundary if its corners sit on the fill's corners.
    """
    subset_size = 15
    half = subset_size // 2
    gui = SelectionGUIOld(make_image_128x256(), subset_size=subset_size)
    try:
        px, py = 128, 64
        entry = gui.add_selection('manual', geometry=[(px, py)])
        gui.recompute_entry(entry, gui.get_subset_size(), gui.distance_spinbox.value())
        gui.update_selected_points()

        assert gui.roi_outline.pen().isCosmetic()

        subpaths = _path_subpaths(gui.roi_outline.path())
        assert len(subpaths) == 1, f"expected one rectangle, got {len(subpaths)}"
        corners = subpaths[0]
        # Pixel ix is the view-coordinate band [ix, ix + 1), so the rectangle spanning
        # pixels ix0..ix1-1 runs from ix0 to ix1 in view coordinates.
        assert corners[:, 0].min() == px - half
        assert corners[:, 0].max() == px + half + 1
        assert corners[:, 1].min() == py - half
        assert corners[:, 1].max() == py + half + 1
    finally:
        gui.close()


def test_subset_borders_are_dropped_together_with_the_fill():
    """A subset whose rectangle runs off the image edge gets neither fill nor border."""
    gui = SelectionGUIOld(make_image_128x256(), subset_size=15)
    try:
        entry = gui.add_selection('manual', geometry=[(128, 64), (2, 2)])
        gui.recompute_entry(entry, gui.get_subset_size(), gui.distance_spinbox.value())
        gui.update_selected_points()

        assert len(_path_subpaths(gui.roi_outline.path())) == 1
        assert overlay_extent(gui.roi_overlay.image) == (15, 15)
    finally:
        gui.close()


def _sobel_shapes(gui, monkeypatch):
    """Patch ``scipy.ndimage.sobel`` to record every ROI shape it is called with."""
    import scipy.ndimage as ndi

    shapes = []
    original_sobel = ndi.sobel

    def spy_sobel(roi, axis):
        shapes.append(roi.shape)
        return original_sobel(roi, axis=axis)

    monkeypatch.setattr(ndi, "sobel", spy_sobel)
    return shapes


def test_shi_tomasi_roi_shape_matches_subset_size_axes(monkeypatch):
    """The ROI sliced inside compute_candidate_points_shi_tomasi must be (2w+1, 2h+1)."""
    gui = SelectionGUIOld(make_image_128x256(), subset_size=(5, 25))
    try:
        entry = gui.add_selection('manual', geometry=[(128, 64)])
        gui.recompute_entry(entry, gui.get_subset_size(), gui.distance_spinbox.value())
        gui.update_selected_points()

        shapes = _sobel_shapes(gui, monkeypatch)
        gui.compute_candidate_points_shi_tomasi()

        half_h, half_w = 5 // 2, 25 // 2
        assert shapes, "sobel was never called -- point was skipped by the bounds check"
        for shape in shapes:
            assert shape == (2 * half_w + 1, 2 * half_h + 1), shape
    finally:
        gui.close()


def test_gradient_direction_roi_shape_matches_subset_size_axes(monkeypatch):
    """The ROI sliced inside compute_candidate_points_gradient_direction must be (2w+1, 2h+1)."""
    gui = SelectionGUIOld(make_image_128x256(), subset_size=(5, 25))
    try:
        entry = gui.add_selection('manual', geometry=[(128, 64)])
        gui.recompute_entry(entry, gui.get_subset_size(), gui.distance_spinbox.value())
        gui.update_selected_points()
        gui.gradient_direction = (1.0, 0.0)

        shapes = _sobel_shapes(gui, monkeypatch)
        gui.compute_candidate_points_gradient_direction()

        half_h, half_w = 5 // 2, 25 // 2
        assert shapes, "sobel was never called -- point was skipped by the bounds check"
        for shape in shapes:
            assert shape == (2 * half_w + 1, 2 * half_h + 1), shape
    finally:
        gui.close()


def test_square_subset_filter_roi_shape_unchanged(monkeypatch):
    """Regression: a square subset_size must still produce a square filter ROI."""
    gui = SelectionGUIOld(make_image_128x256(), subset_size=15)
    try:
        entry = gui.add_selection('manual', geometry=[(128, 64)])
        gui.recompute_entry(entry, gui.get_subset_size(), gui.distance_spinbox.value())
        gui.update_selected_points()

        shapes = _sobel_shapes(gui, monkeypatch)
        gui.compute_candidate_points_shi_tomasi()

        assert shapes
        for shape in shapes:
            assert shape == (15, 15), shape
    finally:
        gui.close()


def test_gradient_direction_dx_dy_convention_matches_real_axes():
    """The gradient-direction filter must respond to the real image gradient.

    ``compute_candidate_points_gradient_direction`` unpacks
    ``dy, dx = self.gradient_direction`` (names swapped relative to the real
    values) and treats ``sobel(roi, axis=1)`` as "gx" (also swapped, since
    roi axis 1 is the real y/height axis). Algebraically the two swaps
    cancel in ``|gx * dx| + |gy * dy|``, so this is NOT a bug -- pinned here
    with an image that has a gradient in only one real axis, checked in both
    directions. Uses a square subset_size so this is independent of the
    anisotropic ROI-extent fix.
    """
    width, height = 100, 60
    xs = np.arange(width)
    frame = np.tile(xs, (height, 1)).astype(np.uint8)  # frame[y, x] = x: varies only in x.

    gui = SelectionGUIOld(frame, subset_size=15)
    try:
        entry = gui.add_selection('manual', geometry=[(50, 30)])
        gui.recompute_entry(entry, gui.get_subset_size(), gui.distance_spinbox.value())
        gui.update_selected_points()

        gui.gradient_direction = (1.0, 0.0)  # real x direction
        gui.compute_candidate_points_gradient_direction()
        strength_x = gui.candidates_grad_dir[0][2]

        gui.gradient_direction = (0.0, 1.0)  # real y direction
        gui.compute_candidate_points_gradient_direction()
        strength_y = gui.candidates_grad_dir[0][2]

        assert strength_x > 0
        assert strength_y == pytest.approx(0, abs=1e-6)
        assert strength_x > strength_y
    finally:
        gui.close()


# ---------------------------------------------------------------------------
# Unified selection-entry list (self.selections) -- see pyidi/GUIs/subset_selection.py
# ---------------------------------------------------------------------------

class _FakeMouseEvent:
    """Minimal stand-in for a pyqtgraph mouse-click event: just ``.scenePos()``."""

    def __init__(self, scene_pos):
        self._scene_pos = scene_pos

    def scenePos(self):
        return self._scene_pos


def _lay_out(gui):
    """Give the view a sane data range so simulated clicks land inside it.

    Until the view has been ranged, ``mapViewToScene`` sends image coordinates far
    outside ``sceneBoundingRect()`` and every click handler bails out as "outside the
    view". ``autoRange()`` does this synchronously. Pumping the event loop
    (``QApplication.processEvents()``) has the same effect but intermittently aborts
    the interpreter under the offscreen platform once the multiprocessing tests
    earlier in the suite have run, so it is deliberately avoided here.
    """
    gui.view.autoRange()


def _click(gui, x, y):
    """Simulate a click at data coordinates (x, y) through the real on_mouse_click path."""
    _lay_out(gui)
    scene_pos = gui.view.mapViewToScene(QtCore.QPointF(x, y))
    gui.on_mouse_click(_FakeMouseEvent(scene_pos))


def _set_method(gui, name):
    """Check `name`'s method button and run the real method_selected handler."""
    button = gui.method_buttons[name]
    button.setChecked(True)
    gui.method_selected(gui.button_group.id(button))


def test_manual_selection_is_singleton():
    """Two manual clicks must land in ONE entry with two points, not two entries."""
    gui = make_gui()
    try:
        _set_method(gui, "Manual")
        _click(gui, 10, 10)
        _click(gui, 20, 20)

        manual_entries = [e for e in gui.selections if e['kind'] == 'manual']
        assert len(manual_entries) == 1
        assert len(manual_entries[0]['geometry']) == 2
        assert len(gui.selections) == gui.selection_list.count()
    finally:
        gui.close()


def _deselect(gui, mask):
    """Run a deselect-mode brush stroke covering `mask` through the real handler."""
    gui.brush_deselect_mode = True
    gui._paint_mask = mask
    gui.handle_brush_end(_FakeMouseEvent(gui.view.mapViewToScene(QtCore.QPointF(0.0, 0.0))))


def test_brush_deselect_only_removes_the_painted_part_of_a_stroke():
    """Deselecting part of a brush stroke must not discard the whole stroke.

    The stroke is subtracted from the painted mask, so the untouched part survives --
    and, because the mask itself is edited, the removal outlasts a spacing change.
    """
    gui = make_gui()
    try:
        _lay_out(gui)
        shape = gui.image_item.image.shape[:2]
        mask = np.zeros(shape, bool)
        mask[20:280, 20:180] = True
        brush = gui.add_selection('brush', geometry=mask)
        gui.recompute_roi_points()
        before = len(gui.entry_points(brush))

        deselected = np.zeros(shape, bool)
        deselected[20:70, 20:70] = True
        _deselect(gui, deselected.copy())

        assert len(gui.selections) == gui.selection_list.count() == 1, "the whole stroke was discarded"
        after = len(gui.entry_points(brush))
        assert 0 < after < before, "expected a partial removal, not all-or-nothing"

        # The mask edit -- not a derived-point filter -- is what makes this stick.
        gui.distance_spinbox.setValue(gui.distance_spinbox.value() + 5)
        still_inside = [p for p in gui.entry_points(brush) if deselected[int(p[0]), int(p[1])]]
        assert not still_inside, "the deselected area came back after a recompute"
    finally:
        gui.close()


def test_brush_deselect_drops_a_stroke_only_once_nothing_is_left_painted():
    """A fully-covered brush row goes away, which shifts every later index.

    The active entry must then be re-derived from the entry object rather than left
    as a stale index, and points of other kinds under the stroke must be removed.
    """
    gui = make_gui()
    try:
        _lay_out(gui)
        shape = gui.image_item.image.shape[:2]
        grid = gui.add_selection('grid', geometry=[(20, 20), (280, 20), (280, 180), (20, 180)])
        first_mask = np.zeros(shape, bool)
        first_mask[30:80, 30:80] = True
        gui.add_selection('brush', geometry=first_mask)
        second_mask = np.zeros(shape, bool)
        second_mask[200:260, 100:160] = True
        second_brush = gui.add_selection('brush', geometry=second_mask)
        gui.recompute_roi_points()

        gui.selection_list.setCurrentRow(2)
        gui.on_entry_selected(2)

        covers_first = np.zeros(shape, bool)
        covers_first[25:85, 25:85] = True          # strictly contains first_mask
        _deselect(gui, covers_first)

        assert len(gui.selections) == gui.selection_list.count() == 2, "the fully-covered brush row should be gone"
        assert gui.selections[gui.active_index] is second_brush, "active entry was not preserved"
        assert grid['removed'], "grid points under the deselect stroke were not removed"
    finally:
        gui.close()


def test_highlight_tracks_the_active_row():
    """The highlight scatter must show exactly the active entry's points, and nothing
    when that entry is hidden via its checkbox."""
    gui = make_gui()

    def n_highlighted():
        data = gui.highlight_scatter.getData()[0]
        return 0 if data is None else len(data)

    try:
        first = gui.add_selection('grid', geometry=[(20, 20), (120, 20), (120, 90)])
        second = gui.add_selection('grid', geometry=[(150, 20), (250, 20), (250, 90)])
        gui.recompute_roi_points()

        # add_selection made the second entry active.
        assert n_highlighted() == len(gui.entry_points(second))

        gui.selection_list.setCurrentRow(0)
        gui.on_entry_selected(0)
        assert n_highlighted() == len(gui.entry_points(first))

        gui.selection_list.item(0).setCheckState(QtCore.Qt.CheckState.Unchecked)
        assert n_highlighted() == 0
    finally:
        gui.close()


def test_switching_tool_away_and_back_continues_the_same_grid():
    """Regression: leaving Grid mode and returning must not silently start a new grid.

    The pre-list code kept a separate ``active_grid_index``/``active_polygon_index``
    per kind; a single ``active_index`` loses that unless the tool switch
    re-activates the most recent entry of the kind being switched to.
    """
    gui = make_gui()
    try:
        _set_method(gui, "Grid")
        for vertex in [(20, 20), (200, 20), (200, 150)]:
            _click(gui, *vertex)

        _set_method(gui, "Manual")       # step away...
        _click(gui, 50, 50)
        _set_method(gui, "Grid")         # ...and back
        _click(gui, 20, 150)

        grids = [e for e in gui.selections if e['kind'] == 'grid']
        assert len(grids) == 1, "a second grid was started instead of continuing the first"
        assert len(grids[0]['geometry']) == 4
    finally:
        gui.close()


def test_clicking_a_row_overrides_the_most_recent_entry_of_that_kind():
    """Selecting a specific row must win over the 'continue the latest one' rule."""
    gui = make_gui()
    try:
        _set_method(gui, "Grid")
        for vertex in [(20, 20), (120, 20), (120, 90)]:
            _click(gui, *vertex)
        gui.start_new_line()
        for vertex in [(150, 20), (250, 20), (250, 90)]:
            _click(gui, *vertex)

        gui.selection_list.setCurrentRow(0)
        gui.on_entry_selected(0)
        _click(gui, 20, 90)

        assert len(gui.selections[0]['geometry']) == 4, "the explicitly selected row was overridden"
        assert len(gui.selections[1]['geometry']) == 3
    finally:
        gui.close()


def test_each_stroke_creates_its_own_entry_and_stays_in_sync():
    """Every grid/line click sequence and every brush stroke gets its own entry."""
    gui = make_gui()
    try:
        _set_method(gui, "Grid")
        for (x, y) in [(20, 20), (150, 20), (150, 120), (20, 120)]:
            _click(gui, x, y)
        assert len(gui.selections) == gui.selection_list.count() == 1

        _set_method(gui, "Along the line")
        for (x, y) in [(30, 30), (200, 30)]:
            _click(gui, x, y)
        assert len(gui.selections) == gui.selection_list.count() == 2

        _set_method(gui, "Brush")
        _lay_out(gui)
        for cx, cy in [(60, 60), (220, 150)]:
            gui.handle_brush_start(_FakeMouseEvent(gui.view.mapViewToScene(QtCore.QPointF(cx, cy))))
            gui.handle_brush_end(_FakeMouseEvent(gui.view.mapViewToScene(QtCore.QPointF(cx, cy))))
        assert len(gui.selections) == gui.selection_list.count() == 4

        assert [e['kind'] for e in gui.selections] == ['grid', 'line', 'brush', 'brush']
    finally:
        gui.close()


def test_labels_are_monotonic_never_reused_after_delete():
    """Deleting Grid 2 and adding a new grid must give Grid 4, never a duplicate Grid 3."""
    gui = make_gui()
    try:
        gui.add_selection('grid')
        gui.add_selection('grid')
        gui.add_selection('grid')
        assert [e['label'] for e in gui.selections] == ['Grid 1', 'Grid 2', 'Grid 3']

        gui.selection_list.setCurrentRow(1)
        gui.delete_selected_entry()

        gui.add_selection('grid')
        labels = [e['label'] for e in gui.selections]
        assert labels == ['Grid 1', 'Grid 3', 'Grid 4']
        assert len(set(labels)) == len(labels), "no two rows should share a label"
        assert len(gui.selections) == gui.selection_list.count()
    finally:
        gui.close()


def test_unchecking_row_removes_and_rechecking_restores_its_points():
    """Toggling a row's visibility checkbox removes/restores exactly that entry's points."""
    gui = make_gui()
    try:
        entry = gui.add_selection('manual', geometry=[(10, 10), (20, 20)])
        gui.recompute_entry(entry)
        gui.update_selected_points()

        n_before = len(gui.points)
        assert n_before == 2

        item = gui.selection_list.item(0)
        item.setCheckState(QtCore.Qt.CheckState.Unchecked)
        assert not gui.selections[0]['visible']
        assert len(gui.points) == 0

        item.setCheckState(QtCore.Qt.CheckState.Checked)
        assert gui.selections[0]['visible']
        assert len(gui.points) == n_before
        assert len(gui.selections) == gui.selection_list.count()
    finally:
        gui.close()


def test_removed_point_survives_recompute():
    """The bug fix: a point removed via `removed` must not reappear after a recompute.

    Before the fix, ``handle_remove_point`` deleted the point straight out of
    ``roi_points``, which ``recompute_roi_points`` regenerates from scratch -- so a
    later spacing/subset-size change silently un-removed it. Now the removal is
    recorded in ``entry['removed']`` and applied only at read time by
    ``entry_points``, so it survives.
    """
    gui = make_gui(subset_size=15, subset_overlap=2)
    try:
        polygon = [(10, 10), (250, 10), (250, 150), (10, 150)]
        entry = gui.add_selection('grid', geometry=polygon)
        gui.recompute_roi_points()
        assert len(entry['roi_points']) > 1

        target = tuple(entry['roi_points'][0])
        entry['removed'].add(target)
        gui.update_selected_points()
        assert target not in gui.selected_points

        # Recompute with the same subset size/spacing regenerates `roi_points`
        # identically (`target` is back in it), but `removed` must still filter it
        # out of the effective points -- this is the actual bug being fixed.
        gui.recompute_roi_points()
        assert target in entry['roi_points'], "sanity check: recompute regenerates the same point"
        assert target not in gui.entry_points(entry)
        assert target not in gui.selected_points

        # A genuine spacing change (as in the bug report) must not resurrect it either.
        gui.distance_spinbox.setValue(gui.distance_spinbox.value() + 3)
        assert target not in gui.entry_points(entry)
        assert target not in gui.selected_points
    finally:
        gui.close()


def test_undo_restores_deleted_entry_at_original_row_and_label():
    """Delete + Ctrl+Z (undo()) must restore an entry at its original row/label.

    Checked for a `grid` entry and, since delete is now generic, a `brush` entry
    too -- brush deletions were not undoable before this refactor.
    """
    gui = make_gui()
    try:
        gui.add_selection('grid', geometry=[(10, 10), (50, 10), (50, 50)])
        gui.add_selection('brush', geometry=np.zeros(gui.image_item.image.shape[:2], dtype=bool))
        gui.add_selection('manual', geometry=[(5, 5)])

        grid_label = gui.selections[0]['label']
        gui.selection_list.setCurrentRow(0)
        gui.delete_selected_entry()
        gui.undo()
        assert gui.selections[0]['kind'] == 'grid'
        assert gui.selections[0]['label'] == grid_label
        assert len(gui.selections) == gui.selection_list.count() == 3

        brush_label = gui.selections[1]['label']
        gui.selection_list.setCurrentRow(1)
        gui.delete_selected_entry()
        gui.undo()
        assert gui.selections[1]['kind'] == 'brush'
        assert gui.selections[1]['label'] == brush_label
        assert len(gui.selections) == gui.selection_list.count() == 3
    finally:
        gui.close()


def test_selected_points_order_matches_entry_creation_order():
    """`selected_points` order is creation order across kinds (manual/grid/line mixed)."""
    gui = make_gui()
    try:
        manual_entry = gui.add_selection('manual', geometry=[(5, 5)])
        gui.recompute_entry(manual_entry)

        grid_entry = gui.add_selection('grid', geometry=[(10, 10), (60, 10), (60, 60), (10, 60)])
        gui.recompute_entry(grid_entry, subset_size=10, spacing=0)

        line_entry = gui.add_selection('line', geometry=[(70, 70), (150, 70)])
        gui.recompute_entry(line_entry, subset_size=10, spacing=0)

        gui.update_selected_points()

        expected = gui.entry_points(manual_entry) + gui.entry_points(grid_entry) + gui.entry_points(line_entry)
        assert gui.selected_points == expected
    finally:
        gui.close()


def test_brush_roi_points_step_matches_subset_size_axes():
    """The brush anisotropic-spacing fix: (h=5, w=21) must step 21 px along x, 5 px along y.

    This fails before the ``_brush_points`` transpose fix and passes after it.
    """
    h, w = 5, 21
    gui = make_gui(subset_size=(h, w), subset_overlap=0)
    try:
        n_x, n_y = gui.image_item.image.shape[:2]
        mask = np.ones((n_x, n_y), dtype=bool)  # paint the whole image
        entry = gui.add_selection('brush', geometry=mask)
        gui.recompute_entry(entry)

        roi_points = entry['roi_points']
        assert len(roi_points) > 4, "expected a genuine grid of points, not a degenerate case"

        xs = sorted(set(p[0] for p in roi_points))
        ys = sorted(set(p[1] for p in roi_points))
        x_steps = set(round(b - a) for a, b in zip(xs, xs[1:]))
        y_steps = set(round(b - a) for a, b in zip(ys, ys[1:]))

        assert x_steps == {w}
        assert y_steps == {h}
    finally:
        gui.close()


# ---------------------------------------------------------------------------
# Filter candidates following the selection
# ---------------------------------------------------------------------------

def _run_shi_tomasi(gui, threshold=1):
    """Run the Shi-Tomasi filter over the current selection and keep nearly everything."""
    gui.switch_mode("filter")
    gui.compute_candidate_points_shi_tomasi()
    gui.threshold_slider.setValue(threshold)
    gui.update_threshold_and_show_shi_tomsi()
    gui.switch_mode("selection")


def _candidates_outside_selection(gui):
    selected = {(int(round(px)), int(round(py))) for px, py in gui.selected_points}
    return [p for p in gui.candidate_points if (int(round(p[0])), int(round(p[1]))) not in selected]


def _brushed_gui():
    """A GUI with one brush stroke covering most of the image, already filtered."""
    gui = make_gui()
    _lay_out(gui)
    mask = np.zeros(gui.image_item.image.shape[:2], bool)
    mask[20:280, 20:180] = True
    entry = gui.add_selection('brush', geometry=mask)
    gui.recompute_roi_points()
    _run_shi_tomasi(gui)
    assert gui.candidate_points, "the filter produced no candidates to test with"
    return gui, entry


def test_brush_deselect_drops_the_filter_candidates_it_removes():
    """Deselected subsets must leave the candidates too, not just the selection.

    ``get_points()`` returns the candidates once a filter has been run, so a candidate
    left behind by a deselect stays in the returned points.
    """
    gui, _ = _brushed_gui()
    try:
        before = len(gui.candidate_points)

        deselected = np.zeros(gui.image_item.image.shape[:2], bool)
        deselected[20:150, 20:100] = True
        _deselect(gui, deselected.copy())

        assert _candidates_outside_selection(gui) == []
        assert len(gui.candidate_points) < before, "no candidate was dropped"
        assert len(gui.get_points()) == len(gui.candidate_points)

        # The threshold slider re-derives the candidates from the cached scores, so it
        # is the obvious way for a dropped candidate to come back.
        gui.update_threshold_and_show_shi_tomsi()
        assert _candidates_outside_selection(gui) == []
    finally:
        gui.close()


def test_unchecking_a_row_hides_its_candidates_and_rechecking_restores_them():
    """The row checkbox is a "try it in and out" control, so it must not be one-way."""
    gui, entry = _brushed_gui()
    try:
        before = len(gui.candidate_points)

        entry['visible'] = False
        gui.update_selected_points()
        assert gui.candidate_points == []

        entry['visible'] = True
        gui.update_selected_points()
        assert len(gui.candidate_points) == before, "the filter result was not restored"
    finally:
        gui.close()


def test_clear_candidates_is_not_undone_by_the_next_selection_change():
    """Clearing must stick: the cached scores are still there to be re-derived from."""
    gui, _ = _brushed_gui()
    try:
        gui.clear_candidates()
        assert gui.candidate_points == []

        gui.update_selected_points()
        assert gui.candidate_points == [], "the cleared candidates came back"
    finally:
        gui.close()
