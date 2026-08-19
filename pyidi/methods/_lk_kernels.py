"""
Compiled (numba) kernels for the Lucas-Kanade family of methods.

This module contains fused implementations of the Lucas-Kanade inner loop, both
for the two-parameter translation model (``_lucas_kanade.py``) and for the
one-parameter directional model (``_directional_lucas_kanade.py``).
The pure-Python/NumPy implementations evaluate a
``scipy.interpolate.RectBivariateSpline`` and compute the least-squares update
through separate NumPy calls, which costs roughly 20 Python-level calls per
point and time step. Here the gradient computation, the pseudo-inverse of the
gradient matrix, the spline evaluation and the iteration loop are fused into a
single compiled function, and the loop over points is parallelised with
``numba.prange``.

The spline machinery (:func:`_basis`, :func:`_eval_spline_grid`) is shared by
both methods; only the normal equations differ, so each method has its own
per-point optimiser (:func:`_optimize_translation`, :func:`_optimize_directional`)
and its own per-frame driver.

The spline evaluation reproduces FITPACK's ``bispev`` (the routine behind
``RectBivariateSpline.__call__``) by evaluating the tensor-product B-spline
directly with de Boor's algorithm, so results agree with the reference
implementation to within floating-point round-off.

:note: The spline coefficients are still produced by ``RectBivariateSpline``;
    only the *evaluation* is reimplemented here.

:warning: Do not add ``fastmath=True`` to these kernels. It lets LLVM assume no
    NaN or infinity is ever produced, which silently turns the ``np.isfinite``
    divergence guards below into dead code (they constant-fold to True). It also
    makes the summation order platform dependent. Measured gain was about 8 per
    cent, which does not pay for either problem.
"""

import os

import numpy as np

try:
    import numba as nb
    NUMBA_AVAILABLE = True
except ImportError:                      # pragma: no cover - environment dependent
    nb = None
    NUMBA_AVAILABLE = False

if NUMBA_AVAILABLE:
    njit = nb.njit
    prange = nb.prange

    # numba's OpenMP threading layer is not safe across ``fork``. If the thread
    # pool has been started in the parent process (by any single-process
    # analysis) and the user then runs with ``processes=N``, the forked workers
    # die with BrokenProcessPool. Ask numba for a layer that is safe under fork,
    # unless the user has explicitly chosen one through the environment.
    if 'NUMBA_THREADING_LAYER' not in os.environ:
        nb.config.THREADING_LAYER = 'forksafe'
else:                                    # pragma: no cover - environment dependent
    def njit(*args, **kwargs):
        """No-op stand-in for ``numba.njit`` when numba is not installed.

        The kernels below are then plain Python and far too slow to use, so
        ``LucasKanade`` never selects them; this only keeps the module
        importable so the rest of pyidi works without numba.
        """
        def decorator(func):
            return func

        if args and callable(args[0]):
            return args[0]
        return decorator

    prange = range

# Cubic splines are the only degree used by the fused kernel. The reference
# implementation supports arbitrary ``int_order``, so ``LucasKanade`` falls back
# to the NumPy path when ``int_order != 3``.
_K = 3

# Outcome of a single region-of-interest optimisation.
STATUS_OK = 0
STATUS_SINGULAR = 1     # gradient matrix is singular: flat or edge-only ROI
STATUS_DIVERGED = 2     # the iteration ran away to a non-finite displacement


@njit(cache=True, inline="always")
def _is_sane(value, limit):
    """Check that a displacement is finite and smaller than the image.

    :param value: displacement component in pixels
    :param limit: corresponding image dimension in pixels
    :return: False if the value is non-finite or physically impossible
    :rtype: bool
    """
    return np.isfinite(value) and abs(value) <= limit


@njit(cache=True, inline="always")
def _basis(t, x, n_coef, out):
    """
    Evaluate the ``_K + 1`` non-zero cubic B-spline basis functions at ``x``.

    Uses de Boor's recurrence, matching the FITPACK ``fpbspl`` routine.

    :param t: knot vector
    :type t: ndarray, float64
    :param x: evaluation coordinate
    :type x: float
    :param n_coef: number of B-spline coefficients along this axis
    :type n_coef: int
    :param out: buffer of length ``_K + 1`` receiving the basis values
    :type out: ndarray, float64
    :return: index of the knot span containing ``x``
    :rtype: int
    """
    # FITPACK's ``bispev`` clamps the evaluation coordinate into the knot range
    # rather than extrapolating, so a spline evaluated past its domain returns
    # the boundary value. Reproduce that here: without it the iteration silently
    # disagrees with ``RectBivariateSpline`` as soon as it wanders more than
    # ``pad`` pixels outside the region of interest.
    if x < t[_K]:
        x = t[_K]
    elif x > t[n_coef]:
        x = t[n_coef]

    # Locate the knot span: t[span] <= x < t[span + 1].
    span = _K
    while span < n_coef - 1 and x >= t[span + 1]:
        span += 1

    out[0] = 1.0
    out[1] = 0.0
    out[2] = 0.0
    out[3] = 0.0

    for j in range(1, _K + 1):
        saved = 0.0
        for r in range(j):
            left = x - t[span + 1 - j + r]
            right = t[span + 1 + r] - x
            denom = right + left
            temp = out[r] / denom if denom != 0.0 else 0.0
            out[r] = saved + right * temp
            saved = left * temp
        out[j] = saved

    return span


@njit(cache=True, inline="always")
def _eval_spline_grid(tx, ty, C, dy, dx, F, span_y, span_x, basis_y, basis_x, buf):
    """
    Evaluate a tensor-product cubic spline on the shifted regular grid
    ``(arange(h) + dy, arange(w) + dx)``.

    The basis functions are separable, so they are evaluated once per row and
    once per column rather than once per grid point.

    :param tx: knot vector along the first (row) axis
    :param ty: knot vector along the second (column) axis
    :param C: spline coefficients, shape ``(n_cy, n_cx)``
    :param dy: sub-pixel offset along the first axis
    :param dx: sub-pixel offset along the second axis
    :param F: output buffer, shape ``(h, w)``
    :param span_y: int buffer of length ``h`` (scratch)
    :param span_x: int buffer of length ``w`` (scratch)
    :param basis_y: float buffer, shape ``(h, _K + 1)`` (scratch)
    :param basis_x: float buffer, shape ``(w, _K + 1)`` (scratch)
    :param buf: float buffer of length ``_K + 1`` (scratch)
    """
    h = F.shape[0]
    w = F.shape[1]
    n_cy = C.shape[0]
    n_cx = C.shape[1]

    for a in range(h):
        span_y[a] = _basis(tx, a + dy, n_cy, buf)
        for p in range(_K + 1):
            basis_y[a, p] = buf[p]

    for b in range(w):
        span_x[b] = _basis(ty, b + dx, n_cx, buf)
        for p in range(_K + 1):
            basis_x[b, p] = buf[p]

    for a in range(h):
        row0 = span_y[a] - _K
        for b in range(w):
            col0 = span_x[b] - _K
            acc = 0.0
            for p in range(_K + 1):
                row = row0 + p
                acc += basis_y[a, p] * (
                    basis_x[b, 0] * C[row, col0]
                    + basis_x[b, 1] * C[row, col0 + 1]
                    + basis_x[b, 2] * C[row, col0 + 2]
                    + basis_x[b, 3] * C[row, col0 + 3]
                )
            F[a, b] = acc


@njit(cache=True)
def _optimize_translation(G, tx, ty, C, maxiter, tol, dy_init, dx_init):
    """
    Fused Lucas-Kanade optimisation for a single region of interest.

    Computes the image gradients of ``G``, the inverse of the gradient matrix
    and then iterates the least-squares translation update, all without
    returning to the Python interpreter.

    :param G: current image subset, shape ``(h + 2, w + 2)``. The one-pixel
        border is consumed by the central-difference gradient.
    :param tx: knot vector along the first axis of the reference spline
    :param ty: knot vector along the second axis of the reference spline
    :param C: reference spline coefficients, shape ``(n_cy, n_cx)``
    :param maxiter: maximum number of iterations
    :param tol: convergence tolerance on the update norm
    :param dy_init: initial sub-pixel guess along the first axis
    :param dx_init: initial sub-pixel guess along the second axis
    :return: ``(dy, dx, status)``, where status is ``STATUS_OK``,
        ``STATUS_SINGULAR`` (flat or edge-only region of interest) or
        ``STATUS_DIVERGED`` (the update ran away to a non-finite value)
    :rtype: tuple of (float, float, int)
    """
    h = G.shape[0] - 2
    w = G.shape[1] - 2

    Gx = np.empty((h, w), dtype=np.float64)
    Gy = np.empty((h, w), dtype=np.float64)
    Gc = np.empty((h, w), dtype=np.float64)

    # Central-difference gradients, matching ``tools.get_gradient``.
    sum_xx = 0.0
    sum_yy = 0.0
    sum_xy = 0.0
    for i in range(h):
        for j in range(w):
            gx = (np.float64(G[i + 1, j + 2]) - np.float64(G[i + 1, j])) * 0.5
            gy = (np.float64(G[i + 2, j + 1]) - np.float64(G[i, j + 1])) * 0.5
            Gx[i, j] = gx
            Gy[i, j] = gy
            Gc[i, j] = np.float64(G[i + 1, j + 1])
            sum_xx += gx * gx
            sum_yy += gy * gy
            sum_xy += gx * gy

    det = sum_xy * sum_xy - sum_xx * sum_yy
    if abs(det) < 1e-10:
        return 0.0, 0.0, STATUS_SINGULAR

    # Inverse of [[sum_yy, sum_xy], [sum_xy, sum_xx]] as written in the
    # reference implementation.
    a00 = sum_xy / det
    a01 = -sum_xx / det
    a10 = -sum_yy / det
    a11 = sum_xy / det

    # Scratch buffers, allocated once and reused across iterations.
    F = np.empty((h, w), dtype=np.float64)
    span_y = np.empty(h, dtype=np.int64)
    span_x = np.empty(w, dtype=np.int64)
    basis_y = np.empty((h, _K + 1), dtype=np.float64)
    basis_x = np.empty((w, _K + 1), dtype=np.float64)
    buf = np.empty(_K + 1, dtype=np.float64)

    disp_y = dy_init
    disp_x = dx_init
    delta_y = dy_init
    delta_x = dx_init
    off_y = 0.0
    off_x = 0.0

    for _ in range(maxiter):
        off_y += delta_y
        off_x += delta_x

        _eval_spline_grid(tx, ty, C, off_y, off_x, F,
                          span_y, span_x, basis_y, basis_x, buf)

        b0 = 0.0
        b1 = 0.0
        for i in range(h):
            for j in range(w):
                residual = Gc[i, j] - F[i, j]
                b0 += Gx[i, j] * residual
                b1 += Gy[i, j] * residual

        delta_y = a00 * b0 + a01 * b1
        delta_x = a10 * b0 + a11 * b1
        disp_y += delta_y
        disp_x += delta_x

        if np.sqrt(delta_y * delta_y + delta_x * delta_x) < tol:
            break

    # An ill-conditioned ROI can make the update run away instead of hitting an
    # exactly singular matrix. Report that instead of returning a silent NaN.
    if not (np.isfinite(disp_y) and np.isfinite(disp_x)):
        return 0.0, 0.0, STATUS_DIVERGED

    # The roles of F and G are switched, hence the sign change.
    return -disp_y, -disp_x, STATUS_OK


@njit(cache=True, parallel=True)
def optimize_frame(frame, points, tx, ty, C_all, previous, out, status, clamped,
                   roi_h, roi_w, pad, maxiter, tol):
    """
    Run the Lucas-Kanade optimisation for every point of a single frame.

    The loop over points is parallelised with ``numba.prange``. The number of
    threads is controlled by the caller through ``numba.set_num_threads``.

    :param frame: the full current frame, any numeric dtype
    :param points: point coordinates, shape ``(n_points, 2)``, int64
    :param tx: knot vector along the first axis (shared by all points)
    :param ty: knot vector along the second axis (shared by all points)
    :param C_all: spline coefficients per point, shape ``(n_points, n_cy, n_cx)``
    :param previous: displacements at the previous time step, shape
        ``(n_points, 2)``
    :param out: buffer receiving the new displacements, shape ``(n_points, 2)``
    :param status: int buffer of length ``n_points`` receiving ``STATUS_OK``,
        ``STATUS_SINGULAR`` or ``STATUS_DIVERGED`` per point
    :param clamped: bool buffer of length ``n_points``, True where the region
        of interest had to be shifted to stay inside the image
    :param roi_h: region of interest height
    :param roi_w: region of interest width
    :param pad: padding used when slicing the current frame
    :param maxiter: maximum number of iterations
    :param tol: convergence tolerance
    """
    n_points = points.shape[0]
    height = frame.shape[0]
    width = frame.shape[1]

    half_h = roi_h // 2 + pad
    half_w = roi_w // 2 + pad

    for p in prange(n_points):
        # Start the optimisation from the previous optimal parameters: the
        # integer part re-centres the slice, the remainder is the sub-pixel
        # initial guess.
        prev_y = previous[p, 0]
        prev_x = previous[p, 1]
        if not (_is_sane(prev_y, height) and _is_sane(prev_x, width)):
            # A previous time step already diverged. Casting such a value to
            # int64 would overflow, so stop here instead.
            status[p] = STATUS_DIVERGED
            clamped[p] = False
            out[p, 0] = prev_y
            out[p, 1] = prev_x
            continue
        int_y = np.int64(np.round(prev_y))
        int_x = np.int64(np.round(prev_x))
        res_y = prev_y - int_y
        res_x = prev_x - int_x

        # Bounds checking, matching ``LucasKanade._padded_slice``.
        y_raw = points[p, 0] + int_y
        x_raw = points[p, 1] + int_x
        y = y_raw
        x = x_raw
        if y < half_h:
            y = half_h
        elif y > height - half_h - 1:
            y = height - half_h - 1
        if x < half_w:
            x = half_w
        elif x > width - half_w - 1:
            x = width - half_w - 1
        clamped[p] = (y != y_raw) or (x != x_raw)

        G = frame[y - half_h:y + half_h + 1, x - half_w:x + half_w + 1]

        dy, dx, point_status = _optimize_translation(
            G, tx, ty, C_all[p], maxiter, tol, -res_y, -res_x
        )

        new_y = dy + int_y
        new_x = dx + int_x

        # An ill-conditioned ROI (a single straight edge, say) makes the update
        # run away without ever hitting an exactly singular matrix. A
        # displacement larger than the image itself is not physical, so treat it
        # as divergence rather than letting it overflow the int64 cast on the
        # next time step.
        if point_status == STATUS_OK and not (
                _is_sane(new_y, height) and _is_sane(new_x, width)):
            point_status = STATUS_DIVERGED

        status[p] = point_status
        out[p, 0] = new_y
        out[p, 1] = new_x


@njit(cache=True)
def _optimize_directional(G, tx, ty, C, maxiter, tol, dy_init, dx_init, di, dj):
    """
    Fused directional Lucas-Kanade optimisation for a single region of interest.

    Identical in structure to :func:`_optimize_translation`, but the motion is
    constrained to a single known direction ``(di, dj)``. The two-parameter
    normal equations therefore collapse to a scalar: the image gradient is
    projected onto the direction and only the signed distance along it is
    solved for.

    :param G: current image subset, shape ``(h + 2, w + 2)``. The one-pixel
        border is consumed by the central-difference gradient.
    :param tx: knot vector along the first axis of the reference spline
    :param ty: knot vector along the second axis of the reference spline
    :param C: reference spline coefficients, shape ``(n_cy, n_cx)``
    :param maxiter: maximum number of iterations
    :param tol: convergence tolerance, compared against the squared scalar
        residual (this is what the reference implementation compares)
    :param dy_init: initial sub-pixel guess along the first axis
    :param dx_init: initial sub-pixel guess along the second axis
    :param di: first (row) component of the unit direction vector
    :param dj: second (column) component of the unit direction vector
    :return: ``(dy, dx, status)``, where status is ``STATUS_OK``,
        ``STATUS_SINGULAR`` (no gradient along the direction at all) or
        ``STATUS_DIVERGED`` (the update ran away to a non-finite value)
    :rtype: tuple of (float, float, int)
    """
    h = G.shape[0] - 2
    w = G.shape[1] - 2

    Gd = np.empty((h, w), dtype=np.float64)
    Gc = np.empty((h, w), dtype=np.float64)

    # Central-difference gradients, matching ``tools.get_gradient``, projected
    # onto the direction as ``Gd = Gx*dij[1] + Gy*dij[0]``.
    sum_dd = 0.0
    for i in range(h):
        for j in range(w):
            gx = (np.float64(G[i + 1, j + 2]) - np.float64(G[i + 1, j])) * 0.5
            gy = (np.float64(G[i + 2, j + 1]) - np.float64(G[i, j + 1])) * 0.5
            gd = gx * dj + gy * di
            Gd[i, j] = gd
            Gc[i, j] = np.float64(G[i + 1, j + 1])
            sum_dd += gd * gd

    # The reference implementation tests for exact zero here, so the compiled
    # kernel does too. A small but non-zero value makes the iteration run away
    # instead, which the sanity check at the end catches.
    if sum_dd == 0.0:
        return 0.0, 0.0, STATUS_SINGULAR

    inv_dd = 1.0 / sum_dd

    # Scratch buffers, allocated once and reused across iterations.
    F = np.empty((h, w), dtype=np.float64)
    span_y = np.empty(h, dtype=np.int64)
    span_x = np.empty(w, dtype=np.int64)
    basis_y = np.empty((h, _K + 1), dtype=np.float64)
    basis_x = np.empty((w, _K + 1), dtype=np.float64)
    buf = np.empty(_K + 1, dtype=np.float64)

    disp_y = dy_init
    disp_x = dx_init
    delta_y = dy_init
    delta_x = dx_init
    off_y = 0.0
    off_x = 0.0

    for _ in range(maxiter):
        off_y += delta_y
        off_x += delta_x

        _eval_spline_grid(tx, ty, C, off_y, off_x, F,
                          span_y, span_x, basis_y, basis_x, buf)

        acc = 0.0
        for i in range(h):
            for j in range(w):
                acc += Gd[i, j] * (Gc[i, j] - F[i, j])
        error = acc * inv_dd

        delta_y = di * error
        delta_x = dj * error
        disp_y += delta_y
        disp_x += delta_x

        if error * error < tol:
            break

    # An ill-conditioned ROI can make the update run away instead of leaving no
    # gradient along the direction at all. Report that instead of returning a
    # silent NaN.
    if not (np.isfinite(disp_y) and np.isfinite(disp_x)):
        return 0.0, 0.0, STATUS_DIVERGED

    # The roles of F and G are switched, hence the sign change.
    return -disp_y, -disp_x, STATUS_OK


@njit(cache=True, parallel=True)
def optimize_frame_directional(frame, points, directions, tx, ty, C_all, previous,
                               out, status, clamped, roi_h, roi_w, pad_y, pad_x,
                               maxiter, tol, rbm_int_y, rbm_int_x,
                               rbm_res_y, rbm_res_x):
    """
    Run the directional Lucas-Kanade optimisation for every point of a frame.

    The loop over points is parallelised with ``numba.prange``. The number of
    threads is controlled by the caller through ``numba.set_num_threads``.

    :param frame: the full current frame, any numeric dtype
    :param points: point coordinates, shape ``(n_points, 2)``, int64
    :param directions: unit direction per point, shape ``(n_points, 2)``, float64
    :param tx: knot vector along the first axis (shared by all points)
    :param ty: knot vector along the second axis (shared by all points)
    :param C_all: spline coefficients per point, shape ``(n_points, n_cy, n_cx)``
    :param previous: displacements at the previous time step, shape
        ``(n_points, 2)``
    :param out: buffer receiving the new displacements, shape ``(n_points, 2)``
    :param status: int buffer of length ``n_points`` receiving ``STATUS_OK``,
        ``STATUS_SINGULAR`` or ``STATUS_DIVERGED`` per point
    :param clamped: bool buffer of length ``n_points``, True where the region
        of interest had to be shifted to stay inside the image
    :param roi_h: region of interest height
    :param roi_w: region of interest width
    :param pad_y: padding used when slicing the current frame, first axis
    :param pad_x: padding used when slicing the current frame, second axis
    :param maxiter: maximum number of iterations
    :param tol: convergence tolerance
    :param rbm_int_y: integer part of the prescribed rigid-body motion, first axis
    :param rbm_int_x: integer part of the prescribed rigid-body motion, second axis
    :param rbm_res_y: sub-pixel remainder of the rigid-body motion, first axis
    :param rbm_res_x: sub-pixel remainder of the rigid-body motion, second axis
    """
    n_points = points.shape[0]
    height = frame.shape[0]
    width = frame.shape[1]

    half_h = roi_h // 2 + pad_y
    half_w = roi_w // 2 + pad_x

    for p in prange(n_points):
        # Start the optimisation from the previous optimal parameters: the
        # integer part re-centres the slice, the remainder is the sub-pixel
        # initial guess.
        prev_y = previous[p, 0]
        prev_x = previous[p, 1]
        if not (_is_sane(prev_y, height) and _is_sane(prev_x, width)):
            # A previous time step already diverged. Casting such a value to
            # int64 would overflow, so stop here instead.
            status[p] = STATUS_DIVERGED
            clamped[p] = False
            out[p, 0] = prev_y
            out[p, 1] = prev_x
            continue
        int_y = np.int64(np.round(prev_y))
        int_x = np.int64(np.round(prev_x))
        res_y = prev_y - int_y
        res_x = prev_x - int_x

        # Bounds checking, matching ``DirectionalLucasKanade._padded_slice``.
        y_raw = points[p, 0] + int_y + rbm_int_y
        x_raw = points[p, 1] + int_x + rbm_int_x
        y = y_raw
        x = x_raw
        if y < half_h:
            y = half_h
        elif y > height - half_h - 1:
            y = height - half_h - 1
        if x < half_w:
            x = half_w
        elif x > width - half_w - 1:
            x = width - half_w - 1
        clamped[p] = (y != y_raw) or (x != x_raw)

        G = frame[y - half_h:y + half_h + 1, x - half_w:x + half_w + 1]

        di = directions[p, 0]
        dj = directions[p, 1]

        dy, dx, point_status = _optimize_directional(
            G, tx, ty, C_all[p], maxiter, tol,
            -res_y + rbm_res_y, -res_x + rbm_res_x, di, dj
        )

        # Split the sub-pixel rigid-body motion into the part along the search
        # direction and the part perpendicular to it, as the reference
        # implementation does.
        dot = rbm_res_y * di + rbm_res_x * dj
        perp_y = rbm_res_y - dot * di
        perp_x = rbm_res_x - dot * dj

        new_y = dy + int_y - rbm_res_y - perp_y
        new_x = dx + int_x - rbm_res_x - perp_x

        # An ill-conditioned ROI (a single straight edge, say) makes the update
        # run away without the projected gradient ever vanishing exactly. A
        # displacement larger than the image itself is not physical, so treat it
        # as divergence rather than letting it overflow the int64 cast on the
        # next time step.
        if point_status == STATUS_OK and not (
                _is_sane(new_y, height) and _is_sane(new_x, width)):
            point_status = STATUS_DIVERGED

        status[p] = point_status
        out[p, 0] = new_y
        out[p, 1] = new_x


def warn_if_fork_unsafe():
    """Warn if numba's threading layer cannot survive ``fork``.

    Only reachable when the user has explicitly selected a threading layer
    through ``NUMBA_THREADING_LAYER``; otherwise this module asks numba for a
    fork-safe layer at import time.
    """
    import multiprocessing
    import warnings

    if multiprocessing.get_start_method() != 'fork':
        return

    try:
        from numba.np.ufunc import parallel as nb_parallel
        layer = nb_parallel.threading_layer()
    except Exception:
        # The thread pool has not been started, so there is nothing to inherit.
        return

    if layer == 'omp':
        warnings.warn(
            "numba's OpenMP threading layer is active and is not safe across fork. "
            "The worker processes may die with BrokenProcessPool. Unset "
            "NUMBA_THREADING_LAYER, or set it to 'tbb' or 'workqueue', to avoid this."
        )
