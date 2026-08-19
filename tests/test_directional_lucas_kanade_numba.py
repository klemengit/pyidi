"""
Tests for the compiled (numba) directional Lucas-Kanade kernel.

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
from pyidi.methods import _directional_lucas_kanade as dlk

DATA = os.path.join(my_path, '..', 'data', 'data_synthetic.cih')

# Points sitting on strong image features, so that the projected gradient is
# well conditioned and the two paths are compared on a meaningful signal.
POINTS = np.array([
    [31,  35],
    [31, 215],
    [31, 126],
    [95,  71],
    [91,  35],
    [66, 191],
])

# A direction along each axis, a diagonal and an oblique one, so the projection
# is exercised with and without zero components.
DIRECTIONS = [(1, 0), (0, 1), (1, 1), (0.6, -0.8)]


def _run(video, use_compiled_kernel, points=POINTS, dij=(1, 0), rbm=None, **kwargs):
    method = pyidi.DirectionalLucasKanade(video)
    method.set_points(points)
    method.configure(verbose=0, show_pbar=False, processes=1, dij=dij,
                     use_compiled_kernel=use_compiled_kernel, **kwargs)
    if rbm is not None:
        method.set_rigid_body_motion(rbm)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return method.get_displacements(autosave=False)


def _assert_same(a, b, atol=1e-9):
    """Displacements agree, and agree about which points were lost."""
    assert np.array_equal(np.isnan(a), np.isnan(b))
    finite = ~np.isnan(a)
    np.testing.assert_allclose(a[finite], b[finite], atol=atol, rtol=0)


@pytest.mark.parametrize('dij', DIRECTIONS)
def test_numba_matches_numpy(dij):
    """The compiled kernel reproduces the NumPy path for every direction."""
    video = pyidi.VideoReader(input_file=DATA)
    _assert_same(_run(video, False, dij=dij), _run(video, True, dij=dij))


def test_numba_matches_numpy_non_square_roi():
    """The two axes are not interchangeable; a non-square ROI catches a swap."""
    video = pyidi.VideoReader(input_file=DATA)
    _assert_same(_run(video, False, roi_size=(11, 7)),
                 _run(video, True, roi_size=(11, 7)))


def test_numba_matches_numpy_per_point_directions():
    """A different search direction for every point."""
    video = pyidi.VideoReader(input_file=DATA)
    dij = [(1, 0), (0, 1), (1, 1), (1, 0), (0.5, 0.5), (0, 1)]
    _assert_same(_run(video, False, dij=dij), _run(video, True, dij=dij))


def test_numba_matches_numpy_with_rigid_body_motion():
    """The prescribed rigid-body motion is split the same way in both paths."""
    video = pyidi.VideoReader(input_file=DATA)
    n = video.N
    t = np.arange(n + 1)
    # Small enough that the points stay locked; a phantom rigid-body motion the
    # video does not contain makes the optimization run away in both paths.
    rbm = np.stack([0.05 * np.sin(2 * np.pi * t / 37),
                    0.05 * np.cos(2 * np.pi * t / 53)], axis=-1)
    _assert_same(_run(video, False, dij=(0.6, 0.8), rbm=rbm),
                 _run(video, True, dij=(0.6, 0.8), rbm=rbm))


def test_numba_matches_numpy_frame_by_frame_with_offsets():
    """Single frame, identical inputs, no feedback loop.

    A whole-video comparison only ever exercises small previous displacements.
    Here the previous displacement and the rigid-body motion are set directly, so
    the integer re-centring and the sub-pixel remainder are exercised across
    their whole range.
    """
    video = pyidi.VideoReader(input_file=DATA)
    frame = np.asarray(video.get_frame(1))
    rng = np.random.default_rng(5)

    def prepare(dij, **kwargs):
        method = pyidi.DirectionalLucasKanade(video)
        method.set_points(POINTS)
        method.configure(verbose=0, show_pbar=False, processes=1, dij=dij, **kwargs)
        method.set_rigid_body_motion(None)
        method.image_size = (video.image_height, video.image_width)
        method.displacements = np.zeros((len(POINTS), 2, 2))
        method.failed_points = {}
        method.warnings = []
        method._interpolate_reference(video)
        method.set_directions(method.dij)
        return method

    for dij in DIRECTIONS:
        for _ in range(4):
            previous = rng.normal(0, 2.5, (len(POINTS), 2))
            rbm_int = rng.integers(-2, 3, 2)
            rbm_res = rng.uniform(-0.5, 0.5, 2)

            reference = prepare(dij)
            reference.displacements[:, 0, :] = previous
            compiled = prepare(dij)
            compiled.displacements[:, 0, :] = previous
            compiled._prepare_numba_reference()

            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                reference._optimize_frame_numpy(frame, 1, 1, rbm_int, rbm_res)
                compiled._optimize_frame_numba(frame, 1, 1, rbm_int, rbm_res)

            _assert_same(reference.displacements[:, 1, :],
                         compiled.displacements[:, 1, :])


def test_numba_is_the_default():
    """``use_compiled_kernel`` defaults to True and is stored as an attribute."""
    video = pyidi.VideoReader(input_file=DATA)
    method = pyidi.DirectionalLucasKanade(video)
    method.set_points(POINTS)
    method.configure(verbose=0, show_pbar=False)

    assert method.use_compiled_kernel is True
    assert method._compiled_kernel_available() is True


def test_falls_back_for_non_cubic_interpolation():
    """``int_order != 3`` is not supported by the kernel, so it must fall back."""
    video = pyidi.VideoReader(input_file=DATA)
    method = pyidi.DirectionalLucasKanade(video)
    method.set_points(POINTS)
    method.configure(verbose=0, show_pbar=False, use_compiled_kernel=True, int_order=1)

    with pytest.warns(UserWarning, match='int_order=3 only'):
        assert method._compiled_kernel_available() is False


def test_int_order_fallback_warns_only_once():
    """The fallback warning must not fire once per frame."""
    video = pyidi.VideoReader(input_file=DATA)
    method = pyidi.DirectionalLucasKanade(video)
    method.set_points(POINTS)
    method.configure(verbose=0, show_pbar=False, use_compiled_kernel=True, int_order=1)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        for _ in range(5):
            method._compiled_kernel_available()

    assert sum('int_order' in str(w.message) for w in caught) == 1


def test_pad_accepts_a_scalar_and_a_pair():
    """``pad`` is a pair here, but a scalar must not break."""
    video = pyidi.VideoReader(input_file=DATA)
    _assert_same(_run(video, True, pad=2), _run(video, True, pad=(2, 2)))


def test_untrackable_point_becomes_nan_and_warns():
    """A point with no gradient along the direction is NaN, not an exception."""
    video = pyidi.VideoReader(input_file=DATA)
    frames = np.asarray(video.get_frames()).copy()
    # A uniform patch has no gradient in any direction.
    frames[:, 100:140, 100:140] = 50
    flat = pyidi.VideoReader(input_file=frames, root=video.root)

    points = np.array([[120, 120], [31, 35]])
    for use_compiled_kernel in (False, True):
        result = _run(flat, use_compiled_kernel, points=points)
        # Frame 0 is the reference, so the point is NaN from frame 1 onwards.
        assert np.isnan(result[0][1:]).all()
        # The good point is unaffected.
        assert not np.isnan(result[1]).any()


def test_failed_points_records_frame_and_reason():
    """``failed_points`` names the point, the frame and the reason."""
    video = pyidi.VideoReader(input_file=DATA)
    frames = np.asarray(video.get_frames()).copy()
    frames[:, 100:140, 100:140] = 50
    flat = pyidi.VideoReader(input_file=frames, root=video.root)

    method = pyidi.DirectionalLucasKanade(flat)
    method.set_points(np.array([[120, 120], [31, 35]]))
    method.configure(verbose=0, show_pbar=False, processes=1, dij=(1, 0))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        method.get_displacements(autosave=False)

    assert 0 in method.failed_points
    assert 1 not in method.failed_points
    detail = method.failed_points[0]
    assert detail['status'] in (_lk_kernels.STATUS_SINGULAR, _lk_kernels.STATUS_DIVERGED)
    assert isinstance(detail['frame'], int)


def test_multiprocessing_matches_single_process():
    """Splitting the points over processes does not change the result."""
    video = pyidi.VideoReader(input_file=DATA)
    single = _run(video, True)

    method = pyidi.DirectionalLucasKanade(video)
    method.set_points(POINTS)
    method.configure(verbose=0, show_pbar=False, processes=2, dij=(1, 0),
                     use_compiled_kernel=True)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        multi = method.get_displacements(autosave=False)

    _assert_same(single, multi)


def test_failed_points_survive_multiprocessing():
    """Worker-local point indices are mapped back onto the full point list."""
    video = pyidi.VideoReader(input_file=DATA)
    frames = np.asarray(video.get_frames()).copy()
    frames[:, 100:140, 100:140] = 50
    flat = pyidi.VideoReader(input_file=frames, root=video.root)

    # The bad point is last, so a worker reports it under a local index of its own.
    points = np.array([[31, 35], [31, 215], [95, 71], [120, 120]])

    method = pyidi.DirectionalLucasKanade(flat)
    method.set_points(points)
    method.configure(verbose=0, show_pbar=False, processes=2, dij=(1, 0),
                     use_compiled_kernel=True)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        result = method.get_displacements(autosave=False)

    assert 3 in method.failed_points
    assert np.isnan(result[3][1:]).all()


def test_warm_up_matches_the_real_call_signature():
    """The pre-fork warm-up must compile the signature the real call uses.

    Warming with a signature the analysis never uses compiles the kernel twice
    and the workers gain nothing, silently. The frames are cast to a dtype no
    other test uses, so the kernel is genuinely cold for this signature and the
    check does not depend on the order the tests run in.
    """
    if not _lk_kernels.NUMBA_AVAILABLE:
        pytest.skip('numba is not installed')

    source = pyidi.VideoReader(input_file=DATA)
    frames = np.asarray(source.get_frames()).astype(np.int32)
    video = pyidi.VideoReader(input_file=frames, root=source.root)

    before = len(_lk_kernels.optimize_frame_directional.signatures)

    dlk._warm_up_kernels(video, {'roi_size': np.array([9, 9]), 'pad': np.array([2, 2]),
                                 'int_order': 3, 'tol': 1e-8,
                                 'use_compiled_kernel': True})
    after_warm_up = len(_lk_kernels.optimize_frame_directional.signatures)
    assert after_warm_up == before + 1, 'the warm-up compiled nothing'

    _run(video, True)
    assert len(_lk_kernels.optimize_frame_directional.signatures) == after_warm_up, (
        'the real call needed a signature the warm-up did not compile'
    )


def test_numpy_fallback_helpers_match_the_compiled_ones():
    """The vectorised helper used without numba matches the compiled one."""
    rng = np.random.default_rng(0)
    F = rng.random((9, 9)) * 255
    G = rng.random((9, 9)) * 255
    Gd = rng.random((9, 9)) * 10
    dij = np.array([0.6, 0.8])
    inv = 1.0 / np.sum(Gd ** 2)

    a_delta, a_error = dlk._delta_vectorised(F, G, Gd, inv, dij)
    b_delta, b_error = dlk.compute_delta(F, G, Gd, inv, dij)

    np.testing.assert_allclose(a_delta, b_delta, atol=1e-12, rtol=0)
    np.testing.assert_allclose(a_error, b_error, atol=1e-12, rtol=0)


def test_analysis_runs_on_the_numpy_fallback_helpers(monkeypatch):
    """Without numba the NumPy path still produces the same displacements."""
    video = pyidi.VideoReader(input_file=DATA)
    reference = _run(video, True)

    monkeypatch.setattr(dlk, 'compute_delta', dlk._delta_vectorised)
    fallback = _run(video, False)

    _assert_same(reference, fallback)


def test_compute_delta_numba_alias_still_exists():
    """The historical module-level name stays importable."""
    assert dlk.compute_delta_numba is dlk.compute_delta


def test_missing_numba_falls_back_and_warns(monkeypatch):
    """Without numba the method degrades to the NumPy path instead of failing."""
    monkeypatch.setattr(dlk, 'NUMBA_AVAILABLE', False)

    video = pyidi.VideoReader(input_file=DATA)
    method = pyidi.DirectionalLucasKanade(video)
    method.set_points(POINTS)
    method.configure(verbose=0, show_pbar=False, use_compiled_kernel=True)

    with pytest.warns(UserWarning, match='numba is not installed'):
        assert method._compiled_kernel_available() is False
