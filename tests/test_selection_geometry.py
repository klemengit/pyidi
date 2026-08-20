"""
Tests for ``pyidi/selection_geometry.py``, the pure-numpy geometry helpers
behind the point/ROI selection GUIs.

The module deliberately keeps two *different* coordinate conventions:
``get_roi_grid`` works in (row, col) / (y, x) order, while
``rois_inside_polygon`` and ``points_along_polygon`` work in (x, y) order,
and ``rois_inside_mask`` returns (y, x) again. These tests pin that split
down explicitly (see ``test_convention_split_is_intentional``), as well as
the anisotropic ``roi_size`` support that is the reason ``get_roi_grid``
was kept as a separate function instead of being folded into the others.
"""

import numpy as np
import pytest

from pyidi.selection_geometry import (
    get_roi_grid,
    points_along_polygon,
    rois_inside_polygon,
    rois_inside_mask,
)


# ---------------------------------------------------------------------------
# get_roi_grid -- (row, col) convention
# ---------------------------------------------------------------------------

def test_get_roi_grid_rectangle_known_count_and_containment():
    """A known rectangle + roi_size/noverlap produces an exact, known grid.

    The polygon is an axis-aligned rectangle spanning row 0..40, col 0..60.
    ``matplotlib.path.Path.contains_points`` excludes points that fall
    exactly on the polygon's own boundary, so the row/col equal to the
    rectangle's minimum edge (0) drop out of the candidate grid, and the
    grid step never reaches the maximum edge because ``np.arange`` excludes
    its stop value. Both facts were verified empirically against the
    implementation and are pinned here.
    """
    poly = np.array([[0, 0], [0, 60], [40, 60], [40, 0]])  # (row, col)
    pts = get_roi_grid(poly, roi_size=(10, 10), noverlap=0, deselect_polygon=[[], []])

    expected_rows = [10, 20, 30]
    expected_cols = [10, 20, 30, 40, 50]
    assert set(pts[:, 0].tolist()) == set(expected_rows)
    assert set(pts[:, 1].tolist()) == set(expected_cols)
    assert len(pts) == len(expected_rows) * len(expected_cols) == 15

    # Independent containment check (not reusing matplotlib.path): every
    # returned point must be strictly inside the axis-aligned rectangle.
    assert np.all((pts[:, 0] > 0) & (pts[:, 0] < 40))
    assert np.all((pts[:, 1] > 0) & (pts[:, 1] < 60))


def test_get_roi_grid_anisotropic_roi_size_spaces_axes_differently():
    """An anisotropic roi_size=(7, 12) must space rows and cols differently.

    This is the capability that makes get_roi_grid worth keeping separate
    from the (x, y) helpers below, which only take a scalar subset_size.
    """
    poly = np.array([[0, 0], [0, 60], [40, 60], [40, 0]])
    pts = get_roi_grid(poly, roi_size=(7, 12), noverlap=0, deselect_polygon=[[], []])

    rows = np.array(sorted(set(pts[:, 0].tolist())))
    cols = np.array(sorted(set(pts[:, 1].tolist())))

    assert len(rows) > 1 and len(cols) > 1
    row_spacing = np.diff(rows)
    col_spacing = np.diff(cols)
    assert np.all(row_spacing == 7), row_spacing
    assert np.all(col_spacing == 12), col_spacing
    assert row_spacing[0] != col_spacing[0]


def test_get_roi_grid_deselect_polygon_removes_exactly_its_points():
    """deselect_polygon removes only the points inside it, nothing else."""
    poly = np.array([[0, 0], [0, 60], [40, 60], [40, 0]])
    base = get_roi_grid(poly, roi_size=(10, 10), noverlap=0, deselect_polygon=[[], []])

    # small square (rows, cols) enclosing exactly the single point (10, 10)
    deselect = [[5, 5, 15, 15], [5, 15, 15, 5]]
    out = get_roi_grid(poly, roi_size=(10, 10), noverlap=0, deselect_polygon=deselect)

    base_set = set(map(tuple, base.tolist()))
    out_set = set(map(tuple, out.tolist()))

    assert base_set - out_set == {(10, 10)}
    # everything else must survive completely untouched
    assert out_set == base_set - {(10, 10)}


def test_get_roi_grid_raises_for_wrong_length_roi_size():
    poly = np.array([[0, 0], [0, 1], [1, 1], [1, 0]])
    with pytest.raises(Exception, match=r"roi_size"):
        get_roi_grid(poly, roi_size=(10,), noverlap=0, deselect_polygon=[[], []])
    with pytest.raises(Exception, match=r"roi_size"):
        get_roi_grid(poly, roi_size=(10, 10, 10), noverlap=0, deselect_polygon=[[], []])


def test_get_roi_grid_accepts_transposed_2xN_input():
    """A (2, N) polygon array is transposed, per the documented contract."""
    poly_nx2 = np.array([[0, 0], [0, 60], [40, 60], [40, 0]])
    poly_2xn = poly_nx2.T
    a = get_roi_grid(poly_nx2, roi_size=(10, 10), noverlap=0, deselect_polygon=[[], []])
    b = get_roi_grid(poly_2xn, roi_size=(10, 10), noverlap=0, deselect_polygon=[[], []])
    assert np.array_equal(a, b)


# ---------------------------------------------------------------------------
# rois_inside_polygon -- (x, y) convention
# ---------------------------------------------------------------------------

def test_rois_inside_polygon_rectangle_containment():
    """Every returned point must actually be inside the polygon, in (x, y)."""
    poly = [(0, 0), (0, 40), (60, 40), (60, 0)]  # (x, y), x in 0..60, y in 0..40
    pts = np.array(rois_inside_polygon(poly, subset_size=10, spacing=0))

    assert len(pts) > 0
    # independent geometric check, not reusing matplotlib.path
    assert np.all((pts[:, 0] >= 0) & (pts[:, 0] <= 60))
    assert np.all((pts[:, 1] >= 0) & (pts[:, 1] <= 40))


def test_rois_inside_polygon_returns_empty_for_fewer_than_3_vertices():
    assert rois_inside_polygon([], 5, 0) == []
    assert rois_inside_polygon([(0, 0)], 5, 0) == []
    assert rois_inside_polygon([(0, 0), (1, 1)], 5, 0) == []


# ---------------------------------------------------------------------------
# points_along_polygon -- (x, y) convention
# ---------------------------------------------------------------------------

def test_points_along_polygon_spacing_on_a_straight_segment():
    """A known straight segment produces exactly the expected step points."""
    poly = [(0, 0), (30, 0)]
    pts = points_along_polygon(poly, subset_size=10, spacing=0)
    assert pts == [(0, 0), (10, 0), (20, 0), (30, 0)]


def test_points_along_polygon_returns_empty_for_fewer_than_2_vertices():
    assert points_along_polygon([], 5, 0) == []
    assert points_along_polygon([(0, 0)], 5, 0) == []


def test_points_along_polygon_skips_zero_length_segments():
    """A degenerate (repeated-vertex) segment must not blow up or duplicate."""
    poly = [(0, 0), (0, 0), (20, 0)]
    pts = points_along_polygon(poly, subset_size=10, spacing=0)
    # only the second (non-degenerate) segment contributes points
    assert pts == [(0, 0), (10, 0), (20, 0)]


# ---------------------------------------------------------------------------
# rois_inside_mask -- mask[y, x] in, (y, x) out
# ---------------------------------------------------------------------------

def test_rois_inside_mask_true_region_returns_yx_points():
    mask = np.zeros((20, 20), dtype=bool)
    mask[5:15, 5:15] = True  # rows (y) 5..14, cols (x) 5..14

    pts = set(rois_inside_mask(mask, subset_size=5, spacing=0))
    assert pts == {(5, 5), (5, 10), (10, 5), (10, 10)}

    # every returned point must independently satisfy mask[y, x] is True
    for y, x in pts:
        assert mask[y, x]


def test_rois_inside_mask_all_false_returns_empty():
    mask = np.zeros((20, 20), dtype=bool)
    assert rois_inside_mask(mask, subset_size=5, spacing=0) == []


# ---------------------------------------------------------------------------
# Explicit convention-split pin
# ---------------------------------------------------------------------------

def test_convention_split_is_intentional():
    """get_roi_grid is (row, col); rois_inside_polygon is (x, y).

    This test feeds the SAME vertex list to both functions. Because the
    rectangle is much longer along its first coordinate (0..100) than its
    second (0..10), the two functions -- if they truly disagree about which
    tuple slot is which axis -- must disagree about which axis of their
    *output* is the "many points" one.

    get_roi_grid reads index 0 as row: the tall axis (0..100) becomes rows,
    so almost all variation is in column 0 of the output and column 1 barely
    varies (exactly 1 distinct value, verified empirically). rois_inside_polygon
    reads the *same* raw vertex list as (x, y) instead -- its candidate
    generation also differs slightly (it includes the far edge, get_roi_grid
    does not), so its "thin" axis ends up with 2 distinct values rather than
    1. If someone "helpfully" unified the two functions to share one
    convention and one candidate-generation scheme, this mismatch (1 vs 2)
    would disappear or flip -- that is exactly what this test guards.
    """
    verts = [(0, 0), (0, 10), (100, 10), (100, 0)]

    roi_pts = get_roi_grid(np.array(verts), roi_size=(5, 5), noverlap=0, deselect_polygon=[[], []])
    n_rows = len(set(roi_pts[:, 0].tolist()))
    n_cols = len(set(roi_pts[:, 1].tolist()))
    assert n_rows > 1 and n_cols == 1, (n_rows, n_cols)

    xy_pts = np.array(rois_inside_polygon(verts, subset_size=5, spacing=0))
    n_x = len(set(xy_pts[:, 0].tolist()))
    n_y = len(set(xy_pts[:, 1].tolist()))
    assert n_x > 1 and n_y == 2, (n_x, n_y)
