"""
Tests for the compiled (numba) Lucas-Kanade kernel.

The kernel is a fused reimplementation of the inner optimization loop. These
tests pin it to the pure NumPy implementation, which is kept reachable through
``configure(use_compiled_kernel=False)`` and acts as the reference.
"""

import numpy as np
import pytest
import sys
import os
import warnings

my_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, my_path + '/../')

import pyidi
from pyidi.methods import _lk_kernels

DATA = os.path.join(my_path, '..', 'data', 'data_synthetic.cih')

# Points sitting on strong image features, so that the gradient matrix is well
# conditioned and the two paths are compared on a meaningful signal.
POINTS = np.array([
    [31,  35],
    [31, 215],
    [31, 126],
    [95,  71],
    [91,  35],
    [66, 191],
])


def _run(video, use_compiled_kernel, points=POINTS, **kwargs):
    lk = pyidi.LucasKanade(video)
    lk.set_points(points)
    lk.configure(verbose=0, show_pbar=False, processes=1,
                 use_compiled_kernel=use_compiled_kernel, **kwargs)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return lk.get_displacements(autosave=False)


def test_numba_matches_numpy():
    """The compiled kernel reproduces the NumPy path to within round-off."""
    video = pyidi.VideoReader(input_file=DATA)

    reference = _run(video, use_compiled_kernel=False)
    compiled = _run(video, use_compiled_kernel=True)

    assert compiled.shape == reference.shape
    # The two paths sum the same terms in a different order, so they agree to
    # float64 round-off rather than bit-for-bit.
    np.testing.assert_allclose(compiled, reference, atol=1e-9, rtol=0)


def test_numba_matches_numpy_non_square_roi():
    """Equivalence holds when the ROI is not square."""
    video = pyidi.VideoReader(input_file=DATA)

    reference = _run(video, use_compiled_kernel=False, roi_size=(11, 7))
    compiled = _run(video, use_compiled_kernel=True, roi_size=(11, 7))

    np.testing.assert_allclose(compiled, reference, atol=1e-9, rtol=0)


def test_numba_is_the_default():
    """``use_compiled_kernel`` defaults to True and is stored as a class attribute."""
    video = pyidi.VideoReader(input_file=DATA)
    lk = pyidi.LucasKanade(video)
    lk.configure(verbose=0, show_pbar=False)

    assert lk.use_compiled_kernel is True


def test_falls_back_for_non_cubic_interpolation():
    """``int_order != 3`` warns and falls back to the NumPy implementation."""
    video = pyidi.VideoReader(input_file=DATA)
    lk = pyidi.LucasKanade(video)
    lk.set_points(POINTS)
    lk.configure(verbose=0, show_pbar=False, use_compiled_kernel=True, int_order=1)

    with pytest.warns(UserWarning, match='int_order'):
        displacements = lk.get_displacements(autosave=False)

    assert displacements.shape == (len(POINTS), video.N, 2)


def test_untrackable_point_becomes_nan_and_warns():
    """A point that cannot be tracked is NaN and warns, it does not abort.

    Both a flat ROI (singular gradient matrix) and a runaway iteration are
    handled this way, in both implementations.
    """
    video = pyidi.VideoReader(input_file=DATA)
    # This part of the synthetic image is uniform, so the gradient matrix there
    # is singular.
    flat = np.array([[121, 60]])

    for use_compiled_kernel in (False, True):
        lk = pyidi.LucasKanade(video)
        lk.set_points(flat)
        lk.configure(verbose=0, show_pbar=False, processes=1, use_compiled_kernel=use_compiled_kernel)

        with pytest.warns(UserWarning, match='could not be tracked'):
            displacements = lk.get_displacements(autosave=False)

        assert displacements.shape == (1, video.N, 2)
        assert np.isnan(displacements).any(), 'the lost point should be NaN'
        assert 0 in lk.failed_points


def test_one_bad_point_does_not_lose_the_good_ones():
    """The whole analysis must survive a single untrackable point.

    This is the case that matters in practice: a few of several hundred points
    are badly placed, and losing every good point with them is not acceptable.
    """
    video = pyidi.VideoReader(input_file=DATA)
    flat = [121, 60]                       # uniform region, cannot be tracked
    points = np.vstack([POINTS, [flat]])

    lk = pyidi.LucasKanade(video)
    lk.set_points(points)
    lk.configure(verbose=0, show_pbar=False, processes=1, use_compiled_kernel=True)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        displacements = lk.get_displacements(autosave=False)

    good = displacements[:len(POINTS)]
    bad = displacements[len(POINTS):]

    assert np.isfinite(good).all(), 'the good points must be unaffected'
    assert np.isnan(bad).any(), 'the bad point must be NaN'
    assert set(lk.failed_points) == {len(POINTS)}

    # and they must match a run without the bad point at all
    clean = _run(video, use_compiled_kernel=True)
    np.testing.assert_allclose(good, clean, atol=1e-9, rtol=0)


def test_failed_points_records_frame_and_reason():
    """``failed_points`` carries enough detail to diagnose the failure."""
    video = pyidi.VideoReader(input_file=DATA)

    lk = pyidi.LucasKanade(video)
    lk.set_points(np.array([[121, 60]]))
    lk.configure(verbose=0, show_pbar=False, processes=1, use_compiled_kernel=True)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        lk.get_displacements(autosave=False)

    record = lk.failed_points[0]
    assert 'frame' in record and 'status' in record
    assert record['status'] in (_lk_kernels.STATUS_SINGULAR,
                                _lk_kernels.STATUS_DIVERGED)


def test_isfinite_guard_is_live():
    """The divergence guard must not be compiled away.

    ``fastmath=True`` lets LLVM assume no NaN or infinity is produced, which
    turns ``np.isfinite`` into a constant True and silently disables the guard.
    This pins the kernels to non-fastmath compilation.
    """
    assert _lk_kernels._optimize_translation.targetoptions.get('fastmath') in (None, False)
    assert _lk_kernels.optimize_frame.targetoptions.get('fastmath') in (None, False)


def test_spline_kernel_matches_scipy():
    """The de Boor evaluator reproduces ``RectBivariateSpline.__call__``."""
    from scipy.interpolate import RectBivariateSpline

    rng = np.random.default_rng(0)
    pad, roi = 2, 9
    grid = np.arange(-pad, roi + pad)
    values = rng.random((len(grid), len(grid))) * 255

    spline = RectBivariateSpline(grid, grid, values, kx=3, ky=3, s=0)
    tx, ty, coeffs = spline.tck
    n_c = len(tx) - 4
    coeffs = np.ascontiguousarray(coeffs.reshape(n_c, n_c))

    out = np.empty((roi, roi))
    scratch = (np.empty(roi, np.int64), np.empty(roi, np.int64),
               np.empty((roi, 4)), np.empty((roi, 4)), np.empty(4))

    for dy, dx in [(0.0, 0.0), (0.3, 0.2), (-0.5, 0.5), (0.9, -0.9)]:
        expected = spline(np.arange(roi) + dy, np.arange(roi) + dx)
        _lk_kernels._eval_spline_grid(tx, ty, coeffs, dy, dx, out, *scratch)
        np.testing.assert_allclose(out, expected, atol=1e-10, rtol=0)


def test_spline_kernel_clamps_outside_its_domain():
    """Past its knot range the kernel clamps, exactly as FITPACK does.

    ``RectBivariateSpline.__call__`` clamps the evaluation coordinate into the
    knot range rather than extrapolating. The kernel used to extrapolate with the
    boundary polynomial, so the two silently disagreed once the iteration wandered
    more than ``pad`` pixels outside the region of interest, which is exactly when
    a point is losing track.
    """
    from scipy.interpolate import RectBivariateSpline

    rng = np.random.default_rng(3)
    pad, roi = 2, 9
    grid = np.arange(-pad, roi + pad)
    values = rng.random((len(grid), len(grid))) * 255

    spline = RectBivariateSpline(grid, grid, values, kx=3, ky=3, s=0)
    tx, ty, coeffs = spline.tck
    n_c = len(tx) - 4
    coeffs = np.ascontiguousarray(coeffs.reshape(n_c, n_c))

    out = np.empty((roi, roi))
    scratch = (np.empty(roi, np.int64), np.empty(roi, np.int64),
               np.empty((roi, 4)), np.empty((roi, 4)), np.empty(4))

    # The knot range is [-pad, roi + pad - 1]; every offset beyond about 2 px
    # pushes part of the grid outside it.
    for offset in (2.5, 3.0, 5.0, 20.0, -3.0, -8.0):
        expected = spline(np.arange(roi) + offset, np.arange(roi) + offset)
        _lk_kernels._eval_spline_grid(tx, ty, coeffs, offset, offset, out, *scratch)
        np.testing.assert_allclose(out, expected, atol=1e-10, rtol=0)


def test_matches_numpy_for_large_previous_displacements():
    """The two paths agree even when the iteration leaves the spline domain.

    With a previous displacement of several pixels the optimization starts far
    from the solution and the iterate can leave the region of interest. This
    pins the clamping fix at the level of a whole frame.
    """
    video = pyidi.VideoReader(input_file=DATA)
    frame = np.asarray(video.get_frame(1))
    rng = np.random.default_rng(11)

    def prepare():
        lk = pyidi.LucasKanade(video)
        lk.set_points(POINTS)
        lk.configure(verbose=0, show_pbar=False, processes=1)
        lk.image_size = (video.image_height, video.image_width)
        lk.displacements = np.zeros((len(POINTS), 2, 2))
        lk.failed_points = {}
        lk.warnings = []
        lk._interpolate_reference(video)
        return lk

    for scale in (1.0, 2.5, 5.0):
        previous = rng.normal(0, scale, (len(POINTS), 2))

        reference = prepare()
        reference.displacements[:, 0, :] = previous
        compiled = prepare()
        compiled.displacements[:, 0, :] = previous
        compiled._prepare_numba_reference()

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            reference._optimize_frame_numpy(frame, 1, 1)
            compiled._optimize_frame_numba(frame, 1, 1)

        a = reference.displacements[:, 1, :]
        b = compiled.displacements[:, 1, :]
        assert np.array_equal(np.isnan(a), np.isnan(b))
        finite = ~np.isnan(a)
        np.testing.assert_allclose(a[finite], b[finite], atol=1e-9, rtol=0)


def test_multiprocessing_matches_single_process():
    """Splitting points across processes gives the same result."""
    video = pyidi.VideoReader(input_file=DATA)

    single = _run(video, use_compiled_kernel=True)

    lk = pyidi.LucasKanade(video)
    lk.set_points(POINTS)
    lk.configure(verbose=0, show_pbar=False, processes=2, use_compiled_kernel=True)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        multi = lk.get_displacements(autosave=False)

    np.testing.assert_allclose(multi, single, atol=1e-9, rtol=0)


def test_threading_layer_is_fork_safe():
    """The kernel module must request a fork-safe numba threading layer.

    numba's OpenMP layer cannot survive ``fork``. Once the thread pool has been
    started by any single-process analysis, a later ``processes=N`` run would
    kill its workers with BrokenProcessPool on platforms that fork (Linux, and
    macOS if the start method is changed back).
    """
    if 'NUMBA_THREADING_LAYER' in os.environ:
        pytest.skip('threading layer explicitly configured by the environment')

    import numba
    assert numba.config.THREADING_LAYER == 'forksafe'


def test_fork_after_threaded_run_does_not_break_workers():
    """A threaded run followed by a multiprocessing run must not break the pool.

    This is the notebook workflow: run once with processes=1, then re-run with
    processes=N in the same session.
    """
    video = pyidi.VideoReader(input_file=DATA)

    single = _run(video, use_compiled_kernel=True)          # starts the numba thread pool

    lk = pyidi.LucasKanade(video)
    lk.set_points(POINTS)
    lk.configure(verbose=0, show_pbar=False, processes=2, use_compiled_kernel=True)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        forked = lk.get_displacements(autosave=False)

    np.testing.assert_allclose(forked, single, atol=1e-9, rtol=0)


def test_warm_up_matches_the_real_call_signature():
    """The pre-fork warm-up must compile the signature the real loop uses.

    numba specializes on dtype *and memory layout*. If the warm-up call differs
    in either, it compiles a specialization nothing else uses and every worker
    still compiles the kernel itself, making the warm-up pointless. This pins
    the two together.
    """
    from pyidi.methods._lucas_kanade import _warm_up_kernels

    video = pyidi.VideoReader(input_file=DATA)

    _warm_up_kernels(video, {'roi_size': np.array([9, 9]), 'pad': 2,
                             'int_order': 3, 'tol': 1e-8, 'use_compiled_kernel': True})
    warmed = set(_lk_kernels.optimize_frame.signatures)
    assert warmed, 'the warm-up did not compile anything'

    _run(video, use_compiled_kernel=True)

    new = set(_lk_kernels.optimize_frame.signatures) - warmed
    assert not new, f'the real call compiled a signature the warm-up missed: {new}'
