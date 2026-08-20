"""Pure-numpy geometry helpers for ROI/point selection GUIs.

This module has no GUI-toolkit dependencies (no tkinter, no PyQt6) so it can
be imported from anywhere without pulling in optional GUI extras. It only
depends on ``numpy`` and ``matplotlib.path.Path``.

The functions here were moved, unchanged in behaviour, out of the selection
GUIs, so that the geometry can be tested without a GUI toolkit and shared
between the napari GUI and ``SelectionGUI``.

They do **not** share a common coordinate convention -- each docstring below
states explicitly which convention that function uses, since this differs
between ``get_roi_grid`` (row/column) and the rest (x/y). The two families
also differ in boundary handling: ``get_roi_grid`` steps with
``np.arange(low, high, step)`` and so excludes the far edge, while
``rois_inside_polygon`` uses ``np.arange(min, max + 1, step)`` and can
include it. They are therefore *not* axis-swapped versions of each other and
can return structurally different grids for the same geometry. Preserve this
when editing -- ``tests/test_selection_geometry.py`` pins it deliberately.
"""

import numpy as np
from matplotlib.path import Path


def get_roi_grid(polygon_points, roi_size, noverlap, deselect_polygon):
    """Generate a regular grid of ROI centre points inside a polygon.

    Coordinate convention: this function works in (y, x) / (row, column)
    order throughout. ``polygon_points`` is expected as an array of
    (row, col) points (or, if given as a 2-row array, it is transposed so
    that rows become points), ``roi_size`` is ``(roi_size_y, roi_size_x)``,
    and the returned candidate points are ``(row, col)`` pairs. This is the
    convention used by ``pyidi/GUIs/gui.py``.

    Parameters
    ----------
    polygon_points : array_like
        Vertices of the selection polygon, as an array of shape ``(N, 2)``
        with ``(row, col)`` points, or shape ``(2, N)`` (will be
        transposed).
    roi_size : tuple of int
        ``(roi_size_y, roi_size_x)``, i.e. the ROI size in the (row,
        column) directions. Must have length 2 -- the two entries may
        differ (anisotropic ROI size).
    noverlap : int
        Overlap, in pixels, between neighbouring ROIs along each axis. The
        centre-to-centre spacing along axis ``i`` is ``roi_size[i] -
        noverlap``.
    deselect_polygon : sequence of two sequences
        ``(rows, cols)`` coordinates of a polygon whose interior should be
        excluded from the returned grid. Pass two empty sequences (e.g.
        ``[[], []]``) to disable deselection.

    Returns
    -------
    numpy.ndarray
        Integer array of shape ``(M, 2)`` with the ``(row, col)`` centre
        points of the ROIs that fall inside ``polygon_points`` and outside
        ``deselect_polygon``.
    """
    if len(roi_size) != 2:
        raise ValueError(f'roi_size must be a tuple of length 2, got length {len(roi_size)}.')

    cent_dist_0 = roi_size[0] - noverlap
    cent_dist_1 = roi_size[1] - noverlap

    points = np.array(polygon_points)
    if points.shape[0] == 2:
        points = points.T

    low_0 = np.min(points[:, 0])
    high_0 = np.max(points[:, 0])
    low_1 = np.min(points[:, 1])
    high_1 = np.max(points[:, 1])

    candidates_0 = np.arange(low_0, high_0, cent_dist_0)
    candidates_1 = np.arange(low_1, high_1, cent_dist_1)
    candidates = np.concatenate([_.flatten()[:, None] for _ in np.meshgrid(candidates_0, candidates_1)], axis=1)

    path = Path(points)
    mask = path.contains_points(candidates)

    if len(deselect_polygon[0]) and len(deselect_polygon[1]):
        path_deselect = Path(np.array(deselect_polygon).T)
        mask_deselect = path_deselect.contains_points(candidates)
        mask = np.logical_and(mask, np.logical_not(mask_deselect))

    return np.round(candidates[mask]).astype(int)


def points_along_polygon(polygon, subset_size, spacing=0):
    """Place evenly-spaced points along the segments of an open polygon.

    Coordinate convention: (x, y) throughout -- ``polygon`` is a sequence
    of ``(x, y)`` vertices, as stored by ``SelectionGUI``, and the returned
    points are ``(x, y)`` pairs (each shifted by -0.5 and rounded to the
    nearest integer, to align with pixel centres).

    Parameters
    ----------
    polygon : sequence of (x, y)
        Vertices of an open polyline; points are generated along each
        consecutive segment ``polygon[i] -> polygon[i + 1]``.
    subset_size : float
        Size of the subset/ROI; combined with ``spacing`` this sets the
        step between consecutive points along each segment.
    spacing : float, optional
        Extra spacing added to ``subset_size`` to get the step between
        points. Default is 0.

    Returns
    -------
    list of tuple
        ``(x, y)`` points along the polygon, rounded to the nearest
        integer (after a -0.5 pixel-centre shift). Returns an empty list
        if ``polygon`` has fewer than 2 vertices.
    """
    if len(polygon) < 2:
        return []

    step = subset_size + spacing
    if step <= 0:
        step = 1

    result_points = []

    for i in range(len(polygon) - 1):
        p1 = np.array(polygon[i])
        p2 = np.array(polygon[i + 1])
        segment = p2 - p1
        length = np.linalg.norm(segment)

        if length == 0:
            continue

        direction = segment / length
        n_points = int(length // step)

        for j in range(n_points + 1):
            pt = p1 + j * step * direction
            result_points.append((round(pt[0] - 0.5), round(pt[1] - 0.5)))

    return result_points


def rois_inside_polygon(polygon, subset_size, spacing):
    """Generate a regular grid of points inside a closed polygon.

    Coordinate convention: (x, y) throughout -- ``polygon`` is a sequence
    of ``(x, y)`` vertices, as stored by ``SelectionGUI``, and the returned
    points are ``(x, y)`` pairs.

    Parameters
    ----------
    polygon : sequence of (x, y)
        Vertices of a closed polygon. Must contain at least 3 points.
    subset_size : float
        Size of the subset/ROI; combined with ``spacing`` this sets the
        grid step along both x and y.
    spacing : float
        Extra spacing added to ``subset_size`` to get the grid step.

    Returns
    -------
    list of tuple
        ``(x, y)`` grid points that fall inside ``polygon``. Returns an
        empty list if ``polygon`` has fewer than 3 vertices.
    """
    if len(polygon) < 3:
        return []

    polygon = np.array(polygon)
    min_x, max_x = int(np.floor(np.min(polygon[:, 0]))), int(np.ceil(np.max(polygon[:, 0])))
    min_y, max_y = int(np.floor(np.min(polygon[:, 1]))), int(np.ceil(np.max(polygon[:, 1])))

    step = subset_size + spacing
    if step <= 0:
        step = 1  # minimum step to avoid infinite loop
    xs = np.arange(min_x, max_x+1, step)
    ys = np.arange(min_y, max_y+1, step)

    grid_x, grid_y = np.meshgrid(xs, ys)
    points = np.vstack([grid_x.ravel(), grid_y.ravel()]).T

    mask = Path(polygon).contains_points(points)
    return [tuple(p) for p in points[mask]]


def rois_inside_mask(mask, subset_size, spacing):
    """Generate a regular grid of points inside a boolean mask.

    Coordinate convention: the input ``mask`` is indexed as ``mask[y, x]``
    (row, col), and the returned points are ``(y, x)`` pairs -- the
    opposite convention from ``points_along_polygon`` and
    ``rois_inside_polygon``.

    Parameters
    ----------
    mask : numpy.ndarray
        2D boolean array of shape ``(h, w)``, indexed as ``mask[y, x]``.
    subset_size : float
        Size of the subset/ROI; combined with ``spacing`` this sets the
        grid step along both axes.
    spacing : float
        Extra spacing added to ``subset_size`` to get the grid step.

    Returns
    -------
    list of tuple
        ``(y, x)`` grid points for which ``mask`` is True.
    """
    step = subset_size + spacing
    if step <= 0:
        step = 1

    h, w = mask.shape
    xs = np.arange(0, w, step)
    ys = np.arange(0, h, step)
    grid_x, grid_y = np.meshgrid(xs, ys)

    candidate_points = np.vstack([grid_y.ravel(), grid_x.ravel()]).T  # (y, x)

    # Only keep points where the mask is True
    selected = [tuple(p) for p in candidate_points if mask[p[0], p[1]]]
    return selected
