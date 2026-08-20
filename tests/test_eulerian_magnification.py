import numpy as np
import sys, os
my_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, my_path + '/../')

import pytest
import imageio.v3 as iio

import pyidi
from pyidi.postprocessing import EulerianMagnifier, eulerian_magnification
from pyidi.postprocessing import _eulerian_magnification as evm_mod
from pyidi.postprocessing._eulerian_magnification import _level_amplifications


def _make_moving_edge_video(n_frames, fps, freq, amplitude, height=64, width=64, dtype=np.uint8):
    """Synthetic video: a soft vertical edge shifting sub-pixel sinusoidally in x.

    Returns a ``(n_frames, height, width)`` stack in ``dtype``. The edge is a smooth
    (tanh) transition centered near the middle column, displaced by
    ``amplitude * sin(2*pi*freq*t)`` pixels.
    """
    x = np.arange(width, dtype=np.float32)
    t = np.arange(n_frames, dtype=np.float32) / fps
    shift = amplitude * np.sin(2.0 * np.pi * freq * t)

    frames = np.zeros((n_frames, height, width), dtype=np.float32)
    center = width / 2.0
    for i in range(n_frames):
        profile = 0.5 * (1.0 + np.tanh((x - center - shift[i]) / 2.0))
        frames[i, :, :] = profile[np.newaxis, :]

    frames = (frames * 200.0 + 20.0)  # keep away from 0/255 clipping
    return frames.astype(dtype)


def _edge_position_amplitude(video):
    """Std of the intensity-gradient centroid (edge x-position) over time.

    Measured in a central crop to avoid pyramid border artifacts.
    """
    return _signed_edge_position(video).std()


def _signed_edge_position(video):
    """Per-frame intensity-gradient centroid (edge x-position) time series.

    Unlike a magnitude-only statistic (e.g. its std), this keeps the sign/phase
    information needed to tell "magnified in phase with the true motion" apart
    from "magnified but sign-inverted" or otherwise decorrelated from the truth.
    Measured in a central crop to avoid pyramid border artifacts.
    """
    stack = np.asarray(video, dtype=np.float32)
    n_frames, height, width = stack.shape
    rows = slice(height // 2 - 4, height // 2 + 4)
    cols = slice(8, width - 8)

    positions = np.zeros(n_frames)
    x = np.arange(width)[cols]
    for i in range(n_frames):
        row = stack[i, rows, :].mean(axis=0)
        grad = np.abs(np.gradient(row))[cols]
        positions[i] = np.sum(x * grad) / np.sum(grad)
    return positions


def _correlation(a, b):
    """Pearson correlation between two 1D signals."""
    a_c = a - a.mean()
    b_c = b - b.mean()
    denom = np.sqrt(np.sum(a_c ** 2) * np.sum(b_c ** 2))
    return float(np.sum(a_c * b_c) / denom)


def test_shape_and_dtype_preserved():
    fps = 100
    frames = _make_moving_edge_video(n_frames=128, fps=fps, freq=10, amplitude=0.2)
    root = os.path.join(my_path, 'tmp_evm_shape')
    video = pyidi.VideoReader(frames, root=root, fps=fps)

    mag = EulerianMagnifier(video)
    mag.configure(freq_band=(8, 12), amplification=10, levels=3, show_progress=False)
    out = mag.get_magnified_video()

    assert out.shape == frames.shape
    assert out.dtype == frames.dtype


def test_in_band_motion_is_amplified():
    fps = 100
    freq = 10
    frames = _make_moving_edge_video(n_frames=128, fps=fps, freq=freq, amplitude=0.2)
    root = os.path.join(my_path, 'tmp_evm_inband')
    video = pyidi.VideoReader(frames, root=root, fps=fps)

    out = eulerian_magnification(
        video, freq_band=(8, 12), amplification=10, levels=3, filter_type='ideal',
        show_progress=False
    )

    amp_in = _edge_position_amplitude(frames)
    amp_out = _edge_position_amplitude(out)

    # Linear EVM multiplies in-band motion by ~ (1 + alpha); require a clear gain.
    assert amp_out > 4.0 * amp_in


def test_out_of_band_motion_is_not_amplified():
    """Quantitative out-of-band rejection.

    The old version of this test compared only the std of an abs-gradient
    centroid, which stays roughly flat even when the band-pass is disabled
    (gain applied at all frequencies), because that motion saturates the
    centroid estimator rather than showing up as a clean amplitude increase.
    Comparing per-frame absolute pixel differences against the original is far
    more sensitive: a disabled band-pass amplifies the strong 30 Hz motion by
    ~(1 + alpha) everywhere, producing large frame-to-frame differences even
    though the ratio-of-stds metric can look deceptively close to 1.
    """
    fps = 100
    freq = 30  # motion at 30 Hz, band-pass at 8-12 Hz -> should be untouched
    frames = _make_moving_edge_video(n_frames=128, fps=fps, freq=freq, amplitude=0.2)
    root = os.path.join(my_path, 'tmp_evm_outband')
    video = pyidi.VideoReader(frames, root=root, fps=fps)

    out = eulerian_magnification(
        video, freq_band=(8, 12), amplification=10, levels=3, filter_type='ideal',
        show_progress=False
    )

    amp_in = _edge_position_amplitude(frames)
    amp_out = _edge_position_amplitude(out)
    mean_abs_diff = np.mean(np.abs(out.astype(np.float32) - frames.astype(np.float32)))

    # A disabled/no-op band-pass amplifies this strong out-of-band motion by
    # ~11x everywhere, which shows up as a large mean pixel difference (empirically
    # ~24 intensity levels for this fixture, vs. ~0.5 for the real band-pass).
    assert mean_abs_diff < 2.0
    assert amp_out < 2.0 * amp_in


def test_mask_restricts_amplification():
    fps = 100
    frames = _make_moving_edge_video(n_frames=128, fps=fps, freq=10, amplitude=0.2)
    root = os.path.join(my_path, 'tmp_evm_mask')
    video = pyidi.VideoReader(frames, root=root, fps=fps)
    height, width = frames.shape[1], frames.shape[2]
    amp_in = _edge_position_amplitude(frames)

    # A zero mask must gate off all amplification -> output equals the original.
    zero_mask = np.zeros((height, width), dtype=np.float32)
    out0 = eulerian_magnification(
        video, freq_band=(8, 12), amplification=10, levels=3,
        mask=zero_mask, show_progress=False
    )
    assert _edge_position_amplitude(out0) < 1.5 * amp_in
    assert np.allclose(out0.astype(np.float32), frames.astype(np.float32), atol=1.0)

    # A full mask must behave like no mask -> clearly amplified.
    full_mask = np.ones((height, width), dtype=np.float32)
    out1 = eulerian_magnification(
        video, freq_band=(8, 12), amplification=10, levels=3,
        mask=full_mask, show_progress=False
    )
    assert _edge_position_amplitude(out1) > 4.0 * amp_in


def test_mask_shape_validation():
    fps = 100
    frames = _make_moving_edge_video(n_frames=64, fps=fps, freq=10, amplitude=0.2)
    root = os.path.join(my_path, 'tmp_evm_maskshape')
    video = pyidi.VideoReader(frames, root=root, fps=fps)

    mag = EulerianMagnifier(video)
    mag.configure(freq_band=(8, 12), mask=np.ones((10, 10)), show_progress=False)
    try:
        mag.get_magnified_video()
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_nyquist_validation():
    fps = 100
    frames = _make_moving_edge_video(n_frames=64, fps=fps, freq=10, amplitude=0.2)
    root = os.path.join(my_path, 'tmp_evm_nyquist')
    video = pyidi.VideoReader(frames, root=root, fps=fps)

    mag = EulerianMagnifier(video)
    mag.configure(freq_band=(40, 60), show_progress=False)  # 60 Hz >= Nyquist (50 Hz)
    try:
        mag.get_magnified_video()
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_zero_amplification_is_identity():
    """amplification=0 must reproduce the input exactly (up to rounding).

    This exercises the full build -> collapse round trip with no amplification
    added, so any pyramid defect (e.g. silently dropping the finest Laplacian
    band during collapse) shows up as a nonzero difference here even though it
    would otherwise be invisible in the amplification-focused tests above.
    """
    fps = 100
    frames = _make_moving_edge_video(n_frames=64, fps=fps, freq=10, amplitude=0.2)
    root = os.path.join(my_path, 'tmp_evm_identity')
    video = pyidi.VideoReader(frames, root=root, fps=fps)

    out = eulerian_magnification(
        video, freq_band=(8, 12), amplification=0, levels=3, filter_type='ideal',
        show_progress=False
    )

    assert np.allclose(out.astype(np.float32), frames.astype(np.float32), atol=1.0)


def test_signed_amplification_is_in_phase():
    """Magnified motion must track the TRUE motion's sign, not just its magnitude.

    ``np.std`` of a magnitude-only centroid cannot distinguish "amplified in
    phase" from "amplified with inverted sign" -- both produce a larger std. Using
    the signed centroid time series and checking correlation (not just amplitude)
    closes that hole.
    """
    fps = 100
    freq = 10
    frames = _make_moving_edge_video(n_frames=128, fps=fps, freq=freq, amplitude=0.2)
    root = os.path.join(my_path, 'tmp_evm_signed')
    video = pyidi.VideoReader(frames, root=root, fps=fps)

    out = eulerian_magnification(
        video, freq_band=(8, 12), amplification=10, levels=3, filter_type='ideal',
        show_progress=False
    )

    pos_in = _signed_edge_position(frames)
    pos_out = _signed_edge_position(out)

    corr = _correlation(pos_in, pos_out)
    ratio = pos_out.std() / pos_in.std()

    assert corr > 0.9
    assert 3.0 < ratio < 9.0


def test_sign_inverted_amplification_would_fail_signed_check(monkeypatch):
    """Self-check: a sign-inverted amplification anti-correlates with the truth.

    This monkeypatches ``_level_amplifications`` to return negated factors and
    verifies the signed-correlation metric used above actually goes negative,
    proving that metric (unlike a magnitude-only std) can catch a sign bug. It
    does not exercise the real (correct) implementation's factors.
    """
    fps = 100
    freq = 10
    frames = _make_moving_edge_video(n_frames=128, fps=fps, freq=freq, amplitude=0.2)
    root = os.path.join(my_path, 'tmp_evm_signed_mutant')
    video = pyidi.VideoReader(frames, root=root, fps=fps)

    original = evm_mod._level_amplifications

    def negated(*args, **kwargs):
        return [-f for f in original(*args, **kwargs)]

    monkeypatch.setattr(evm_mod, "_level_amplifications", negated)

    out = eulerian_magnification(
        video, freq_band=(8, 12), amplification=10, levels=3, filter_type='ideal',
        show_progress=False
    )

    pos_in = _signed_edge_position(frames)
    pos_out = _signed_edge_position(out)
    corr = _correlation(pos_in, pos_out)

    assert corr < -0.5


def test_level_amplifications_lambda_c_reference_values():
    """Pinned regression values for the lambda_c spatial-attenuation ramp.

    Locks in the direction and magnitude of the ramp (attenuated at the finest
    band, approaching full amplification toward the coarsest band, zero at the
    low-pass residual) so a reintroduced sign/direction bug is caught exactly.
    """
    factors = _level_amplifications(
        n_levels=4, amplification=10.0, lambda_c=200.0, image_shape=(256, 256)
    )
    expected = [0.0, 0.6593, 2.3187, 5.6374, 0.0]
    assert np.allclose(factors, expected, rtol=1e-3)


def test_level_amplifications_all_zero_warns():
    """A lambda_c/amplification/levels combo that zeroes every level must warn."""
    with pytest.warns(UserWarning, match="lambda_c"):
        factors = _level_amplifications(
            n_levels=1, amplification=10.0, lambda_c=200.0, image_shape=(256, 256)
        )
    assert factors == [0.0, 0.0]


def test_save_gif_roundtrip(tmp_path):
    fps = 50
    frames = _make_moving_edge_video(n_frames=8, fps=fps, freq=10, amplitude=0.2, height=32, width=32)
    root = os.path.join(my_path, 'tmp_evm_save')
    video = pyidi.VideoReader(frames, root=root, fps=fps)

    mag = EulerianMagnifier(video)
    mag.configure(freq_band=(5, 15), amplification=5, levels=2, show_progress=False)
    mag.get_magnified_video()

    out_path = os.path.join(str(tmp_path), "magnified")
    mag.save(out_path, output_format="gif")

    uri = out_path + ".gif"
    assert os.path.exists(uri)

    read_back = iio.imread(uri, index=None)
    assert read_back.dtype == np.uint8
    assert read_back.shape == (8, 32, 32, 3)


def test_save_without_fps_raises(tmp_path):
    fps = 50
    frames = _make_moving_edge_video(n_frames=8, fps=fps, freq=10, amplitude=0.2, height=32, width=32)
    root = os.path.join(str(tmp_path), 'tmp_evm_save_nofps')
    video = pyidi.VideoReader(frames, root=root, fps=fps)

    mag = EulerianMagnifier(video)
    mag.configure(freq_band=(5, 15), amplification=5, levels=2, show_progress=False)
    mag.get_magnified_video()

    # Simulate fps becoming unresolvable after computation (the dedicated guard
    # in save(), distinct from the fps check in get_magnified_video()).
    mag.fps = None
    with pytest.raises(ValueError):
        mag.save(os.path.join(str(tmp_path), "out_nofps"), output_format="gif")


def test_butterworth_filter_amplifies_in_band_motion():
    fps = 100
    freq = 10
    frames = _make_moving_edge_video(n_frames=128, fps=fps, freq=freq, amplitude=0.2)
    root = os.path.join(my_path, 'tmp_evm_butter')
    video = pyidi.VideoReader(frames, root=root, fps=fps)

    out = eulerian_magnification(
        video, freq_band=(8, 12), amplification=10, levels=3, filter_type='butter',
        show_progress=False
    )

    pos_in = _signed_edge_position(frames)
    pos_out = _signed_edge_position(out)

    # Looser bounds than the ideal filter: filtfilt has edge (transient) effects
    # near the start/end of a short stack that the FFT brick-wall filter doesn't.
    assert _correlation(pos_in, pos_out) > 0.8
    assert pos_out.std() / pos_in.std() > 2.0


def test_odd_dimensions_identity():
    fps = 100
    frames = _make_moving_edge_video(n_frames=48, fps=fps, freq=10, amplitude=0.2, height=65, width=63)
    root = os.path.join(my_path, 'tmp_evm_odd')
    video = pyidi.VideoReader(frames, root=root, fps=fps)

    out = eulerian_magnification(
        video, freq_band=(8, 12), amplification=0, levels=3, filter_type='ideal',
        show_progress=False
    )

    assert out.shape == frames.shape
    assert np.allclose(out.astype(np.float32), frames.astype(np.float32), atol=1.0)


def test_float32_input_not_clipped():
    """Non-integer source dtype must stay float, and overshoot must not be clipped.

    The implementation only clips to the source dtype range when it is an
    integer type; a float32 input with a large amplification can legitimately
    overshoot the original [min, max] range in the returned array.
    """
    fps = 100
    freq = 10
    frames = _make_moving_edge_video(
        n_frames=64, fps=fps, freq=freq, amplitude=0.5, dtype=np.float32
    )
    root = os.path.join(my_path, 'tmp_evm_float')
    video = pyidi.VideoReader(frames, root=root, fps=fps)

    out = eulerian_magnification(
        video, freq_band=(8, 12), amplification=20, levels=3, filter_type='ideal',
        show_progress=False
    )

    assert out.dtype == np.float32
    assert out.max() > frames.max()
    assert out.min() < frames.min()
