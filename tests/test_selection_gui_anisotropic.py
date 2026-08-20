"""Tests for anisotropic (non-square) subset sizes in ``SelectionGUI``.

``SelectionGUI`` is a full Qt application, but it can be constructed headlessly
for testing -- the same recipe used by
``docs/source/quick_start/make_selection_animation.py``:

* ``QT_QPA_PLATFORM=offscreen`` must be set before Qt is imported, so Qt
  renders to its software framebuffer instead of opening a real display.
* ``sys.ps1`` must be set before constructing ``SelectionGUI``, so its
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

from PyQt6 import QtWidgets  # noqa: E402

sys.ps1 = getattr(sys, "ps1", ">>> ")  # Make SelectionGUI think it's running interactively.
QtWidgets.QApplication.exec = lambda self=None: 0  # Neutralise the blocking event loop.

from pyidi.GUIs.subset_selection import SelectionGUI  # noqa: E402
from pyidi.selection_geometry import rois_inside_polygon  # noqa: E402


def make_image():
    """A synthetic grayscale image, large enough for the polygons used below."""
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, size=(200, 300), dtype=np.uint8)


def make_gui(**kwargs):
    """Construct a headless ``SelectionGUI`` on a fresh synthetic image."""
    return SelectionGUI(make_image(), **kwargs)


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
        gui.grid_polygons[0]['points'] = [(0, 0), (280, 0), (280, 180), (0, 180)]
        gui.recompute_roi_points()

        roi_points = gui.grid_polygons[0]['roi_points']
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
        gui.grid_polygons[0]['points'] = polygon
        gui.recompute_roi_points()

        expected = rois_inside_polygon(polygon, subset_size, overlap)
        assert gui.grid_polygons[0]['roi_points'] == expected
    finally:
        gui.close()


# ---------------------------------------------------------------------------
# Rectangle-drawing / filter-ROI orientation (the "drawn transposed" bug)
# ---------------------------------------------------------------------------

def test_rectangle_overlay_extent_matches_subset_size_axes():
    """subset_size=(5, 25) must draw 25 px along x (axis 0) and 5 px along y (axis 1).

    ``manual_points`` uses the internal (x, y) convention: (128, 64) sits well
    inside the 128 (height) x 256 (width) image used here, whereas (64, 128)
    would not (x=64 < 256 is fine, but as a first coordinate it would be
    read as a row and 128 is out of the 0..127 row range) -- see the class
    docstring / bug report for why the ordering matters.
    """
    gui = SelectionGUI(make_image_128x256(), subset_size=(5, 25))
    try:
        gui.manual_points = [(128, 64)]
        gui.update_selected_points()

        extent0, extent1 = overlay_extent(gui.roi_overlay.image)
        assert extent0 == 25, f"expected 25 px along axis 0 (x/width), got {extent0}"
        assert extent1 == 5, f"expected 5 px along axis 1 (y/height), got {extent1}"
    finally:
        gui.close()


def test_square_subset_overlay_matches_independent_reference():
    """Regression: a square subset_size must draw byte-identical overlay pixels."""
    subset_size = 15
    half = subset_size // 2
    gui = SelectionGUI(make_image_128x256(), subset_size=subset_size)
    try:
        px, py = 128, 64
        gui.manual_points = [(px, py)]
        gui.update_selected_points()

        n_x, n_y = gui.image_item.image.shape[:2]
        expected = np.zeros((n_x, n_y, 4), dtype=np.uint8)
        ix0, iy0, ix1, iy1 = px - half, py - half, px + half + 1, py + half + 1
        expected[ix0:ix1, iy0:iy1, 1] = 180
        expected[ix0:ix1, iy0:iy1, 3] = 40
        expected[ix0, iy0:iy1, 1] = 255
        expected[ix1 - 1, iy0:iy1, 1] = 255
        expected[ix0:ix1, iy0, 1] = 255
        expected[ix0:ix1, iy1 - 1, 1] = 255
        expected[ix0, iy0:iy1, 3] = 150
        expected[ix1 - 1, iy0:iy1, 3] = 150
        expected[ix0:ix1, iy0, 3] = 150
        expected[ix0:ix1, iy1 - 1, 3] = 150

        np.testing.assert_array_equal(gui.roi_overlay.image, expected)
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
    gui = SelectionGUI(make_image_128x256(), subset_size=(5, 25))
    try:
        gui.manual_points = [(128, 64)]
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
    gui = SelectionGUI(make_image_128x256(), subset_size=(5, 25))
    try:
        gui.manual_points = [(128, 64)]
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
    gui = SelectionGUI(make_image_128x256(), subset_size=15)
    try:
        gui.manual_points = [(128, 64)]
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

    gui = SelectionGUI(frame, subset_size=15)
    try:
        gui.manual_points = [(50, 30)]
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
