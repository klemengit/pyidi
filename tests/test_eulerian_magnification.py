import numpy as np
import sys, os
my_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, my_path + '/../')

import pyidi
from pyidi.postprocessing import EulerianMagnifier, eulerian_magnification


def _make_moving_edge_video(n_frames, fps, freq, amplitude, height=64, width=64):
    """Synthetic video: a soft vertical edge shifting sub-pixel sinusoidally in x.

    Returns a ``(n_frames, height, width)`` uint8 stack. The edge is a smooth
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
    return frames.astype(np.uint8)


def _edge_position_amplitude(video):
    """Std of the intensity-gradient centroid (edge x-position) over time.

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
    return np.std(positions)


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
