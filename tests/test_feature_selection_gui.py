"""Tests for ``FeatureSelectionGUI``, the interface over the selection pipeline.

Constructed headlessly, by the same recipe as
``tests/test_selection_gui_anisotropic.py``:

* ``QT_QPA_PLATFORM=offscreen`` before Qt is imported, so Qt renders to its
  software framebuffer instead of opening a display;
* ``sys.ps1`` set, so the constructor takes its "interactive" branch rather
  than ``sys.exit(...)``;
* ``QApplication.exec`` neutralised, since that branch still calls it and would
  otherwise block in the event loop.

What is worth testing here is not the widgets but the *contract between the
interface and the pipeline*: that a threshold or mask edit re-derives points
from a cached score while a subset-size change pays for a recomputation, that
roles and visibility do what the rows say they do, and that undo puts things
back. Those are the properties the whole design rests on.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import sys  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402

pytest.importorskip("PyQt6")

from PyQt6 import QtCore, QtWidgets  # noqa: E402

sys.ps1 = getattr(sys, "ps1", ">>> ")
QtWidgets.QApplication.exec = lambda self=None: 0

from pyidi.GUIs.feature_selection import FeatureSelectionGUI  # noqa: E402


def make_image():
    """A speckled frame, so every subset has something to score."""
    rng = np.random.default_rng(4)
    return rng.integers(0, 255, size=(160, 240), dtype=np.uint8)


def make_gui(**kwargs):
    """A headless window on a fresh synthetic frame."""
    return FeatureSelectionGUI(make_image(), **kwargs)


def rect(r0, c0, r1, c1):
    """A rectangular polygon as ``(row, col)`` vertices."""
    return [(r0, c0), (r0, c1), (r1, c1), (r1, c0)]


def gui_with_region(**kwargs):
    """A window with one polygon mask covering most of the frame."""
    gui = make_gui(**kwargs)
    gui.pipeline.add_entry('polygon', rect(20, 20, 140, 220))
    gui.refresh()
    return gui


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_constructor_accepts_a_2d_image():
    gui = make_gui()
    try:
        assert gui.frame.shape == (160, 240)
        assert gui.pipeline.subset_size == (11, 11)
    finally:
        gui.close()


def test_constructor_accepts_a_frame_stack():
    stack = np.stack([make_image(), make_image()])
    gui = FeatureSelectionGUI(stack)
    try:
        np.testing.assert_array_equal(gui.frame, stack[0])
    finally:
        gui.close()


def test_constructor_rejects_an_unusable_input():
    with pytest.raises(TypeError, match='VideoReader'):
        FeatureSelectionGUI('not a video')


def test_constructor_normalises_an_anisotropic_subset_size():
    gui = make_gui(subset_size=(21, 7))
    try:
        assert gui.pipeline.subset_size == (21, 7)
        assert not gui.square_check.isChecked()
        assert gui.width_spin.isEnabled()
    finally:
        gui.close()


def test_the_image_is_not_transposed():
    """The whole module works in (row, col); nothing should flip the frame."""
    gui = make_gui()
    try:
        np.testing.assert_array_equal(gui.image_item.image, gui.frame)
    finally:
        gui.close()


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------

def test_the_three_steps_are_separately_reachable():
    gui = make_gui()
    try:
        for name in ('Mask', 'Evaluate', 'Select'):
            gui.select_step(name)
            assert gui.step == name
            assert gui.step_stack.currentWidget() is gui.step_pages[name]
    finally:
        gui.close()


def test_the_highlight_is_a_mask_step_cue():
    gui = gui_with_region()
    try:
        gui.select_step('Mask')
        assert gui.highlight_scatter.isVisible()
        gui.select_step('Select')
        assert not gui.highlight_scatter.isVisible()
    finally:
        gui.close()


def test_selecting_without_a_mask_says_so_rather_than_failing():
    gui = make_gui()
    try:
        gui.select_step('Select')
        assert 'mask' in gui.select_note.text().lower()
        assert gui.get_points().shape == (0, 2)
    finally:
        gui.close()


# ---------------------------------------------------------------------------
# The contract that makes the interface feel live
# ---------------------------------------------------------------------------

def test_a_threshold_change_does_not_re_evaluate():
    gui = gui_with_region()
    try:
        evaluations = gui.pipeline.store.n_evaluations
        assert evaluations >= 1
        before = len(gui.get_points())
        gui.threshold_slider.setValue(990)
        assert gui.pipeline.store.n_evaluations == evaluations
        assert len(gui.get_points()) < before
    finally:
        gui.close()


def test_a_minimum_distance_change_does_not_re_evaluate():
    gui = gui_with_region()
    try:
        evaluations = gui.pipeline.store.n_evaluations
        before = len(gui.get_points())
        gui.min_distance_spin.setValue(30)
        assert gui.pipeline.store.n_evaluations == evaluations
        assert len(gui.get_points()) < before
    finally:
        gui.close()


def test_a_mask_edit_does_not_re_evaluate():
    gui = gui_with_region()
    try:
        evaluations = gui.pipeline.store.n_evaluations
        gui.pipeline.entries[0].geometry = rect(30, 30, 90, 120)
        gui.refresh()
        assert gui.pipeline.store.n_evaluations == evaluations
    finally:
        gui.close()


def test_a_subset_size_change_does_re_evaluate():
    gui = gui_with_region()
    try:
        evaluations = gui.pipeline.store.n_evaluations
        gui.height_spin.setValue(21)
        gui.get_points()
        assert gui.pipeline.store.n_evaluations > evaluations
    finally:
        gui.close()


def test_changing_the_evaluator_re_evaluates_and_keeps_the_old_score_cached():
    gui = gui_with_region()
    try:
        gui.get_points()
        evaluations = gui.pipeline.store.n_evaluations
        gui.evaluator_combo.setCurrentIndex(gui.evaluator_combo.findData('gradient_direction'))
        gui.get_points()
        assert gui.pipeline.store.n_evaluations == evaluations + 1

        # Switching back must be free: the Shi-Tomasi array is still in the cache.
        gui.evaluator_combo.setCurrentIndex(gui.evaluator_combo.findData('shi_tomasi'))
        gui.get_points()
        assert gui.pipeline.store.n_evaluations == evaluations + 1
    finally:
        gui.close()


def test_the_parameter_panel_follows_the_evaluator():
    gui = make_gui()
    try:
        assert gui.param_widgets == {}          # Shi-Tomasi takes no parameters
        gui.evaluator_combo.setCurrentIndex(gui.evaluator_combo.findData('gradient_direction'))
        assert set(gui.param_widgets) == {'direction'}
        assert gui.param_widgets['direction']() == (0.0, 1.0)
    finally:
        gui.close()


# ---------------------------------------------------------------------------
# Rows: visibility, roles, deletion
# ---------------------------------------------------------------------------

def test_a_row_shows_its_label_role_and_count():
    gui = gui_with_region()
    try:
        text = gui.entry_list.item(0).text()
        assert 'Polygon 1' in text
        assert 'mask' in text
        assert 'pts' in text
    finally:
        gui.close()


def test_unchecking_a_row_drops_its_contribution_without_deleting_it():
    gui = gui_with_region()
    try:
        assert len(gui.get_points()) > 0
        gui.entry_list.item(0).setCheckState(QtCore.Qt.CheckState.Unchecked)
        assert gui.get_points().shape == (0, 2)
        assert len(gui.pipeline.entries) == 1

        gui.entry_list.item(0).setCheckState(QtCore.Qt.CheckState.Checked)
        assert len(gui.get_points()) > 0
    finally:
        gui.close()


def test_switching_a_role_changes_what_the_row_contributes():
    gui = gui_with_region()
    try:
        gui.active_index = 0
        automatic = gui.get_points()
        gui.toggle_role()
        assert gui.pipeline.entries[0].role == 'points'
        literal = gui.get_points()
        # As a points row the polygon lays out its own grid instead of being
        # filtered, so the two results are different sets of points.
        assert len(literal) > 0
        assert not np.array_equal(np.sort(literal, axis=0), np.sort(automatic, axis=0))
        assert not gui.pipeline.mask.any()
    finally:
        gui.close()


def test_deleting_a_row_removes_it():
    gui = gui_with_region()
    try:
        gui.active_index = 0
        gui.delete_active()
        assert gui.pipeline.entries == []
        assert gui.entry_list.count() == 0
    finally:
        gui.close()


def test_clear_all_empties_the_list():
    gui = gui_with_region()
    try:
        gui.pipeline.add_entry('points', [(50, 50)])
        gui.refresh()
        gui.clear_all()
        assert gui.pipeline.entries == []
        assert gui.get_points().shape == (0, 2)
    finally:
        gui.close()


# ---------------------------------------------------------------------------
# Undo
# ---------------------------------------------------------------------------

def test_undo_restores_a_deleted_row_at_its_original_position():
    gui = gui_with_region()
    try:
        first = gui.pipeline.entries[0]
        gui.pipeline.add_entry('points', [(50, 50)])
        gui.refresh()
        gui.active_index = 0
        gui.delete_active()
        assert gui.pipeline.entries[0].kind == 'points'

        gui.undo()
        assert gui.pipeline.entries[0] is first
        assert gui.pipeline.entries[0].label == 'Polygon 1'
    finally:
        gui.close()


def test_undo_reverses_a_vertex_add():
    gui = make_gui()
    try:
        gui.select_tool('polygon')
        for position in [(20, 20), (20, 100), (100, 100)]:
            gui.add_vertex(position)
        entry = gui.pipeline.entries[0]
        assert len(entry.geometry) == 3
        gui.undo()
        assert len(entry.geometry) == 2
    finally:
        gui.close()


def test_undo_reverses_a_brush_stroke():
    gui = make_gui()
    try:
        gui.select_tool('brush')
        gui.brush_start()
        gui.brush_move((80, 120))
        gui.brush_end()
        assert len(gui.pipeline.entries) == 1
        gui.undo()
        assert gui.pipeline.entries == []
    finally:
        gui.close()


def test_undo_reverses_a_deselection():
    gui = make_gui()
    try:
        gui.select_tool('brush')
        gui.brush_start()
        gui.brush_move((80, 120))
        gui.brush_end()
        painted = gui.pipeline.mask.sum()
        assert painted > 0

        gui.deselect_button.setChecked(True)
        gui.brush_start()
        gui.brush_move((80, 120))
        gui.brush_end()
        assert gui.pipeline.mask.sum() < painted

        gui.undo()
        assert gui.pipeline.mask.sum() == painted
    finally:
        gui.close()


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def test_the_subset_border_pen_is_cosmetic():
    """Cosmetic means the width is in screen pixels, so it survives a zoom."""
    gui = gui_with_region()
    try:
        assert gui.roi_outline.pen().isCosmetic()
        assert not gui.roi_outline.path().isEmpty()
    finally:
        gui.close()


def test_the_subset_overlay_covers_the_expected_area():
    gui = make_gui(subset_size=(21, 7))
    try:
        gui.pipeline.add_entry('points', [(80, 120)])
        gui.refresh()
        overlay = gui.roi_overlay.image
        covered = overlay[..., 3] != 0
        rows = np.flatnonzero(covered.any(axis=1))
        cols = np.flatnonzero(covered.any(axis=0))
        assert rows.max() - rows.min() + 1 == 21
        assert cols.max() - cols.min() + 1 == 7
    finally:
        gui.close()


def test_hiding_the_subsets_clears_the_overlay():
    gui = gui_with_region()
    try:
        assert gui.roi_overlay.image is not None
        gui.show_subsets.setChecked(False)
        assert gui.roi_overlay.image is None
        assert gui.roi_outline.path().isEmpty()
    finally:
        gui.close()


def test_the_score_overlay_is_transparent_on_the_invalid_border():
    gui = gui_with_region()
    try:
        gui.show_score.setChecked(True)
        assert gui.score_overlay.isVisible()
        rgba = gui.score_overlay.image
        assert (rgba[:5, :, 3] == 0).all()
        assert (rgba[80, 100:140, 3] > 0).all()
    finally:
        gui.close()


def test_a_large_selection_redraws_quickly():
    """Thousands of subsets must stay interactive, hence the raster-plus-path split."""
    import time

    gui = make_gui(subset_size=5)
    try:
        rng = np.random.default_rng(11)
        rows = rng.integers(10, 150, 5000)
        cols = rng.integers(10, 230, 5000)
        points = np.column_stack([rows, cols])
        start = time.perf_counter()
        gui.draw_subset_rectangles(points, 2, 2)
        elapsed = time.perf_counter() - start
        assert elapsed < 0.5, f'redraw of 5000 subsets took {elapsed:.3f} s'
    finally:
        gui.close()


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

def test_points_are_row_col_integers_inside_the_frame():
    gui = gui_with_region()
    try:
        points = gui.points
        assert points.dtype.kind == 'i'
        assert points.shape[1] == 2
        assert (points[:, 0] < gui.frame.shape[0]).all()
        assert (points[:, 1] < gui.frame.shape[1]).all()
        np.testing.assert_array_equal(points, gui.get_points())
    finally:
        gui.close()


def test_points_are_accepted_by_a_method_class(tmp_path):
    import warnings

    from pyidi import SimplifiedOpticalFlow, VideoReader

    gui = gui_with_region()
    try:
        points = gui.points
        assert len(points)
        video = VideoReader(np.stack([gui.frame, gui.frame]), root=str(tmp_path))
        method = SimplifiedOpticalFlow(video)
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            method.set_points(points)
        np.testing.assert_array_equal(method.points, points)
    finally:
        gui.close()


def test_hand_picked_points_survive_a_high_threshold():
    gui = gui_with_region()
    try:
        gui.pipeline.add_entry('points', [(80, 120)])
        gui.threshold_slider.setValue(999)
        points = gui.get_points()
        assert (points == np.array([80, 120])).all(axis=1).any()
    finally:
        gui.close()


def test_the_count_label_follows_the_points():
    gui = gui_with_region()
    try:
        assert gui.count_label.text() == f'{len(gui.get_points())} points'
        gui.threshold_slider.setValue(995)
        assert gui.count_label.text() == f'{len(gui.get_points())} points'
    finally:
        gui.close()


def test_the_lattice_selector_gives_regular_spacing():
    gui = gui_with_region()
    try:
        gui.selector_combo.setCurrentIndex(gui.selector_combo.findData('lattice'))
        gui.pitch_spin.setValue(20)
        gui.threshold_slider.setValue(0)
        points = gui.get_points()
        assert len(points) > 4
        assert set(np.diff(sorted(set(points[:, 0].tolist())))) <= {20}
    finally:
        gui.close()


# ---------------------------------------------------------------------------
# The existing interface is untouched
# ---------------------------------------------------------------------------

def test_both_interfaces_are_importable():
    from pyidi import SelectionGUI

    assert SelectionGUI is not FeatureSelectionGUI


# ---------------------------------------------------------------------------
# Removing a point by clicking it
# ---------------------------------------------------------------------------

def test_the_remove_tool_takes_a_point_away_every_time():
    gui = gui_with_region()
    try:
        gui.select_tool('remove')
        for _ in range(4):
            target = tuple(int(v) for v in gui._points[0])
            gui.remove_nearest_point(target)
            gui.refresh()
            assert not any(tuple(p) == target for p in gui._points)
    finally:
        gui.close()


def test_removing_a_point_is_undoable():
    gui = gui_with_region()
    try:
        gui.select_tool('remove')
        before = len(gui._points)
        target = tuple(int(v) for v in gui._points[0])

        gui.remove_nearest_point(target)
        gui.refresh()
        assert not any(tuple(p) == target for p in gui._points)

        gui.undo()
        assert len(gui._points) == before
        assert any(tuple(p) == target for p in gui._points)
    finally:
        gui.close()


def test_a_click_far_from_every_point_removes_nothing():
    gui = gui_with_region()
    try:
        before = gui.pipeline.entries[0].erased
        gui.remove_nearest_point((0, 0))
        assert gui.pipeline.entries[0].erased is before
    finally:
        gui.close()


# ---------------------------------------------------------------------------
# Clicks that land off the image
#
# The aspect is locked, so one axis always has a margin, and zooming out adds
# more. A subset centred off the frame is not something that can be tracked.
# ---------------------------------------------------------------------------

def test_a_click_outside_the_image_adds_no_point():
    gui = make_gui()
    try:
        gui.select_tool('points')
        rows, cols = gui.pipeline.shape
        gui.add_point((rows + 40.0, 20.0))
        gui.add_point((20.0, -12.0))
        assert gui.pipeline.entries == []
    finally:
        gui.close()


def test_a_click_on_the_image_still_adds_one():
    gui = make_gui()
    try:
        gui.select_tool('points')
        gui.add_point((80.0, 120.0))
        assert gui.pipeline.entries[0].geometry == [(80, 120)]
    finally:
        gui.close()


def test_an_off_frame_point_cannot_reach_the_result():
    """Whatever route it arrived by -- it is an index error in every array."""
    gui = make_gui()
    try:
        gui.pipeline.add_entry('points', [(80, 120), (999, 120)])
        gui.refresh()
        assert [tuple(p) for p in gui.get_points()] == [(80, 120)]
    finally:
        gui.close()


# ---------------------------------------------------------------------------
# The score overlay is part of the redraw
# ---------------------------------------------------------------------------

def test_the_score_overlay_follows_a_change_of_subset_size():
    """It is drawn from the score, so it has to be part of the redraw."""
    gui = gui_with_region()
    try:
        gui.show_score.setChecked(True)
        before = gui.score_overlay.image.copy()
        gui.height_spin.setValue(31)
        assert not np.array_equal(gui.score_overlay.image, before)
    finally:
        gui.close()


# ---------------------------------------------------------------------------
# Ctrl, and what a stroke costs to remember
# ---------------------------------------------------------------------------

class FakeDrag:
    """The parts of a pyqtgraph drag event the canvas actually reads."""

    def __init__(self, ctrl=False, start=False, finish=False):
        self._ctrl = ctrl
        self._start = start
        self._finish = finish
        self.accepted = False

    def modifiers(self):
        return (QtCore.Qt.KeyboardModifier.ControlModifier if self._ctrl
                else QtCore.Qt.KeyboardModifier.NoModifier)

    def isStart(self):
        return self._start

    def isFinish(self):
        return self._finish

    def accept(self):
        self.accepted = True

    def scenePos(self):
        return QtCore.QPointF(0.0, 0.0)

    def buttonDownScenePos(self):
        return QtCore.QPointF(0.0, 0.0)


def test_ctrl_is_read_off_the_event_not_tracked():
    """A panel widget with focus can swallow the key; a tracked flag then lies."""
    gui = make_gui()
    try:
        gui.select_tool('brush')
        assert not gui.view._handle_brush_drag(FakeDrag(ctrl=False, start=True))
        assert gui.view._handle_brush_drag(FakeDrag(ctrl=True, start=True))
        assert gui.painting
    finally:
        gui.close()


def test_letting_go_of_ctrl_mid_stroke_still_finishes_the_stroke():
    """Otherwise the drag stops being handled and the stroke is silently lost."""
    gui = make_gui()
    try:
        gui.select_tool('brush')
        gui.view._handle_brush_drag(FakeDrag(ctrl=True, start=True))
        gui.brush_move((80.0, 120.0))

        assert gui.view._handle_brush_drag(FakeDrag(ctrl=False))
        assert gui.view._handle_brush_drag(FakeDrag(ctrl=False, finish=True))
        assert not gui.painting
        assert gui.pipeline.mask.sum() > 0
    finally:
        gui.close()


def test_an_undo_snapshot_does_not_copy_the_erased_arrays():
    """One frame of booleans per region, in each of fifty undo slots.

    Safe to hold by reference because ``erased`` is always replaced, never
    written into -- the contract on :class:`~pyidi.selection.masks.Entry`.
    """
    gui = gui_with_region()
    try:
        gui.select_tool('brush')
        gui.deselect_button.setChecked(True)
        gui.brush_start()
        gui.brush_move((80.0, 120.0))
        gui.brush_end()

        entry = gui.pipeline.entries[0]
        assert entry.erased is not None
        held = {id(state[1]) for state in gui._snapshot()['state']}
        assert id(entry.erased) in held
    finally:
        gui.close()
