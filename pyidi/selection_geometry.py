"""Pure-numpy geometry helpers for ROI/point selection GUIs.

This module has no GUI-toolkit dependencies (no tkinter, no PyQt6) so it can
be imported from anywhere without pulling in optional GUI extras. It only
depends on ``numpy`` and ``matplotlib.path.Path``.

The functions here were moved, unchanged in behaviour, out of the selection
GUIs, so that the geometry can be tested without a GUI toolkit and shared
between the napari GUI, ``SelectionGUI`` and ``SelectionGUIOld``.

They do **not** share a common coordinate convention -- each docstring below
states explicitly which convention that function uses, since this differs
between ``get_roi_grid`` (row/column) and the rest (x/y). The two families
also differ in boundary handling: ``get_roi_grid`` steps with
``np.arange(low, high, step)`` and so excludes the far edge, while
``rois_inside_polygon`` uses ``np.arange(min, max + 1, step)`` and can
include it. They are therefore *not* axis-swapped versions of each other and
can return structurally different grids for the same geometry. Preserve this
when editing -- ``tests/test_selection_geometry.py`` pins it deliberately.

All four functions accept an anisotropic ``subset_size``/``roi_size``, i.e. a
``(height, width)`` pair instead of a single scalar. ``get_roi_grid`` already
had this (``roi_size=(roi_size_y, roi_size_x)``) and its signature is
unchanged here. ``points_along_polygon``, ``rois_inside_polygon`` and
``rois_inside_mask`` now normalize a scalar or a ``(height, width)`` pair
through the private ``_as_size_pair`` helper; a scalar still produces
byte-identical output to before.
"""

import numpy as np
from matplotlib.path import Path


def _as_size_pair(subset_size):
    """Normalize a scalar or (height, width) subset size to a (h, w) pair.

    Integrality is preserved rather than always casting to float: if
    ``subset_size`` is a scalar integer, or a pair of integers, ``h`` and
    ``w`` are returned as Python ``int``; otherwise (any float involved)
    they are returned as ``float``. This matters downstream --
    ``IDIMethod.set_points()`` treats non-integer point coordinates as
    sub-pixel and warns about them, and GUI callers always pass integer
    sizes (``QSpinBox`` values), so they must keep getting integer grid
    coordinates out, not a spurious sub-pixel warning on every selection.

    Parameters
    ----------
    subset_size : int, float, or a length-2 sequence of int/float
        A scalar (broadcast to both axes) or a ``(height, width)`` pair,
        i.e. ``(y_extent, x_extent)``. May be a plain Python number or a
        0-d/1-d numpy array or scalar.

    Returns
    -------
    tuple of int or tuple of float
        ``(h, w)``, as ``int`` if ``subset_size`` was integral, ``float``
        otherwise.

    Raises
    ------
    ValueError
        If ``subset_size`` is a sequence whose length is not 2.
    """
    arr = np.asarray(subset_size)

    if arr.ndim != 0 and arr.shape != (2,):
        raise ValueError(f'subset_size must be a scalar or a (height, width) pair, got shape {arr.shape}.')

    cast = int if np.issubdtype(arr.dtype, np.integer) else float

    if arr.ndim == 0:
        s = cast(arr)
        return s, s

    return cast(arr[0]), cast(arr[1])


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
    of ``(x, y)`` vertices, as stored by ``SelectionGUIOld``, and the returned
    points are ``(x, y)`` pairs (each shifted by -0.5 and rounded to the
    nearest integer, to align with pixel centres).

    Parameters
    ----------
    polygon : sequence of (x, y)
        Vertices of an open polyline; points are generated along each
        consecutive segment ``polygon[i] -> polygon[i + 1]``.
    subset_size : float or (height, width)
        Size of the subset/ROI, as a scalar or a ``(height, width)`` pair
        (``height`` is the vertical/y extent, ``width`` the
        horizontal/x extent). Combined with ``spacing``, this sets the
        step between consecutive points along each segment: the step is
        the extent of the subset projected along the segment's direction
        (see implementation note below), which reduces to
        ``subset_size + spacing`` for a square/scalar subset, regardless
        of the segment's angle.
    spacing : float, optional
        Extra spacing added to the projected subset extent to get the
        step between points. Default is 0.

    Returns
    -------
    list of tuple
        ``(x, y)`` points along the polygon, rounded to the nearest
        integer (after a -0.5 pixel-centre shift). Returns an empty list
        if ``polygon`` has fewer than 2 vertices.
    """
    if len(polygon) < 2:
        return []

    h, w = _as_size_pair(subset_size)

    result_points = []

    for i in range(len(polygon) - 1):
        p1 = np.array(polygon[i])
        p2 = np.array(polygon[i + 1])
        segment = p2 - p1
        length = np.linalg.norm(segment)

        if length == 0:
            continue

        direction = segment / length

        # Step = the elliptical extent of the (h, w) subset projected along
        # the unit segment direction (dx, dy). For h == w == s this is
        # s * sqrt(dx**2 + dy**2) == s for every angle (dx, dy is a unit
        # vector), i.e. exactly the old isotropic behaviour -- do not
        # "simplify" this to |dx|*w + |dy|*h, which is NOT equivalent (it
        # gives 1.414*s on a 45-degree segment instead of s).
        dx, dy = direction[0], direction[1]
        extent = np.sqrt((dx * w) ** 2 + (dy * h) ** 2)
        step = extent + spacing
        if step <= 0:
            step = 1

        n_points = int(length // step)

        for j in range(n_points + 1):
            pt = p1 + j * step * direction
            result_points.append((round(pt[0] - 0.5), round(pt[1] - 0.5)))

    return result_points


def rois_inside_polygon(polygon, subset_size, spacing):
    """Generate a regular grid of points inside a closed polygon.

    Coordinate convention: (x, y) throughout -- ``polygon`` is a sequence
    of ``(x, y)`` vertices, as stored by ``SelectionGUIOld``, and the returned
    points are ``(x, y)`` pairs.

    Parameters
    ----------
    polygon : sequence of (x, y)
        Vertices of a closed polygon. Must contain at least 3 points.
    subset_size : float or (height, width)
        Size of the subset/ROI, as a scalar or a ``(height, width)`` pair
        (``height`` is the vertical/y extent, ``width`` the
        horizontal/x extent). Combined with ``spacing`` this sets the
        grid step along x (from ``width``) and y (from ``height``).
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

    h, w = _as_size_pair(subset_size)

    polygon = np.array(polygon)
    min_x, max_x = int(np.floor(np.min(polygon[:, 0]))), int(np.ceil(np.max(polygon[:, 0])))
    min_y, max_y = int(np.floor(np.min(polygon[:, 1]))), int(np.ceil(np.max(polygon[:, 1])))

    step_x = w + spacing
    if step_x <= 0:
        step_x = 1  # minimum step to avoid infinite loop
    step_y = h + spacing
    if step_y <= 0:
        step_y = 1  # minimum step to avoid infinite loop
    xs = np.arange(min_x, max_x+1, step_x)
    ys = np.arange(min_y, max_y+1, step_y)

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
    subset_size : float or (height, width)
        Size of the subset/ROI, as a scalar or a ``(height, width)`` pair
        (``height`` is the vertical/y extent, ``width`` the
        horizontal/x extent). Combined with ``spacing`` this sets the
        grid step along y (from ``height``) and x (from ``width``).
    spacing : float
        Extra spacing added to ``subset_size`` to get the grid step.

    Returns
    -------
    list of tuple
        ``(y, x)`` grid points for which ``mask`` is True.
    """
    size_h, size_w = _as_size_pair(subset_size)

    step_x = size_w + spacing
    if step_x <= 0:
        step_x = 1
    step_y = size_h + spacing
    if step_y <= 0:
        step_y = 1

    h, w = mask.shape
    # Defensive cast to int: these are pixel indices used directly to index
    # `mask` below. For integer subset_size/spacing (the normal case --
    # _as_size_pair now preserves integrality) step_x/step_y and thus xs/ys
    # are already int, so this is a no-op. But subset_size may legitimately
    # be a float per the docstring above, in which case np.arange yields a
    # float array that mask[...] cannot be indexed with -- cast defensively
    # so a float subset_size degrades to truncated pixel indices instead of
    # crashing.
    xs = np.arange(0, w, step_x).astype(int)
    ys = np.arange(0, h, step_y).astype(int)
    grid_x, grid_y = np.meshgrid(xs, ys)

    candidate_points = np.vstack([grid_y.ravel(), grid_x.ravel()]).T  # (y, x)

    # Only keep points where the mask is True
    selected = [tuple(p) for p in candidate_points if mask[p[0], p[1]]]
    return selected
