"""
Linear Eulerian Video Magnification (EVM) for pre-test motion visualization.

This module magnifies subtle, sub-pixel motion directly from raw video, *before*
any displacement identification is performed. It is intended as a qualitative
pre-test aid: to reveal whether (and where) a structure is moving, to help pick
regions of interest / seed points, and to isolate a single mode by band-passing
around a suspected natural frequency.

.. warning::
    The output is for **visualization only**. Eulerian magnification distorts
    motion amplitudes non-linearly and must **not** be used as a measurement of
    displacement. Use the identification methods (Lucas-Kanade, Simplified Optical
    Flow, ...) for quantitative results.

The implementation follows the linear EVM of Wu et al., "Eulerian Video
Magnification for Revealing Subtle Changes in the World", ACM SIGGRAPH 2012:
a spatial Laplacian pyramid, a temporal band-pass filter applied per level, and
linear amplification of the band-passed signal added back to the original.
"""

import warnings

import numpy as np
import scipy.fft
import scipy.signal
import scipy.ndimage
import imageio.v3 as iio
from tqdm import tqdm

from ..video_reader import VideoReader


def _pyr_down(image):
    """Blur-and-halve one level of a Gaussian pyramid."""
    blurred = scipy.ndimage.gaussian_filter(image, sigma=1.0, mode="reflect")
    return blurred[::2, ::2]


def _pyr_up(image, dst_shape):
    """Upsample ``image`` to ``dst_shape`` (``(H, W)``) with a smoothing blur.

    The exact same operator is used in both pyramid construction and collapse, so
    reconstruction of the un-amplified signal is exact by telescoping.
    """
    zoom = (dst_shape[0] / image.shape[0], dst_shape[1] / image.shape[1])
    upsampled = scipy.ndimage.zoom(image, zoom, order=1)
    return scipy.ndimage.gaussian_filter(upsampled, sigma=1.0, mode="reflect")


def _validate_freq_band(freq_band):
    """Validate a ``(low, high)`` pass-band and return it as a float tuple."""
    if len(freq_band) != 2 or freq_band[0] >= freq_band[1]:
        raise ValueError("'freq_band' must be (low, high) with low < high.")
    return (float(freq_band[0]), float(freq_band[1]))


def _validate_filter_type(filter_type):
    """Validate the temporal filter type."""
    if filter_type not in ("ideal", "butter"):
        raise ValueError("'filter_type' must be 'ideal' or 'butter'.")
    return filter_type


def _max_pyramid_levels(height, width, min_size=4):
    """Largest number of pyramid levels keeping the smallest dimension >= min_size."""
    smallest = min(height, width)
    levels = 0
    while smallest // 2 >= min_size:
        smallest //= 2
        levels += 1
    return max(levels, 1)


def _laplacian_pyramid_single(image, levels):
    """Decompose a single 2D float image into a Laplacian pyramid.

    :param image: input image, 2D float32 array
    :param levels: number of Laplacian (band-pass) levels; the returned list has
        ``levels + 1`` entries, the last being the low-pass residual
    :return: list of 2D arrays, finest band first, residual last
    """
    gauss = [image]
    for _ in range(levels):
        gauss.append(_pyr_down(gauss[-1]))

    lap = []
    for i in range(levels):
        up = _pyr_up(gauss[i + 1], gauss[i].shape)
        lap.append(gauss[i] - up)
    lap.append(gauss[-1])  # low-pass residual
    return lap


def _build_laplacian_pyramid(frames, levels, progress=False):
    """Build a Laplacian pyramid for every frame.

    :param frames: video stack, ``(N, H, W)`` float32
    :param levels: number of band-pass levels
    :param progress: show a per-frame progress bar
    :return: list of length ``levels + 1``; each entry is a ``(N, h_l, w_l)`` stack
    """
    n_frames = frames.shape[0]
    per_level = [[] for _ in range(levels + 1)]
    for n in tqdm(range(n_frames), desc="Building pyramid", disable=not progress):
        lap = _laplacian_pyramid_single(frames[n], levels)
        for lvl in range(levels + 1):
            per_level[lvl].append(lap[lvl])
    return [np.stack(stack, axis=0) for stack in per_level]


def _collapse_laplacian_pyramid(pyramid, progress=False):
    """Collapse a Laplacian pyramid (list of ``(N, h_l, w_l)`` stacks) back to frames.

    :param pyramid: list of per-level stacks, finest first, residual last
    :param progress: show a per-frame progress bar
    :return: reconstructed video stack ``(N, H, W)`` float32
    """
    n_frames = pyramid[0].shape[0]
    height, width = pyramid[0].shape[1], pyramid[0].shape[2]
    out = np.zeros((n_frames, height, width), dtype=np.float32)

    for n in tqdm(range(n_frames), desc="Reconstructing", disable=not progress):
        image = pyramid[-1][n]
        for lvl in range(len(pyramid) - 2, -1, -1):
            shape = (pyramid[lvl].shape[1], pyramid[lvl].shape[2])
            image = _pyr_up(image, shape) + pyramid[lvl][n]
        out[n] = image
    return out


def _temporal_bandpass(stack, fps, freq_band, filter_type):
    """Band-pass filter a level stack along the time axis (axis 0).

    :param stack: ``(N, h, w)`` float32 stack
    :param fps: sampling rate in Hz
    :param freq_band: ``(low, high)`` pass-band in Hz
    :param filter_type: ``"ideal"`` (FFT brick-wall) or ``"butter"`` (Butterworth)
    :return: filtered stack, same shape as input
    """
    low, high = freq_band
    if filter_type == "ideal":
        n_frames = stack.shape[0]
        spectrum = scipy.fft.rfft(stack, axis=0)
        freqs = scipy.fft.rfftfreq(n_frames, d=1.0 / fps)
        mask = (freqs >= low) & (freqs <= high)
        spectrum[~mask] = 0.0
        return scipy.fft.irfft(spectrum, n=n_frames, axis=0).astype(np.float32)

    nyquist = 0.5 * fps
    b, a = scipy.signal.butter(1, [low / nyquist, high / nyquist], btype="band")
    return scipy.signal.filtfilt(b, a, stack, axis=0).astype(np.float32)


def _level_amplifications(n_levels, amplification, lambda_c, image_shape):
    """Per-level amplification factors.

    With ``lambda_c=None`` every band-pass level gets the full ``amplification``
    and the low-pass residual gets 0. When ``lambda_c`` is given, the spatial
    wavelength attenuation of Wu et al. is applied (experimental) so that high
    spatial frequencies are amplified less, reducing ringing artifacts.

    :param n_levels: number of band-pass levels (pyramid has ``n_levels + 1`` entries)
    :param amplification: target amplification factor ``alpha``
    :param lambda_c: spatial wavelength cutoff in pixels, or None
    :param image_shape: ``(H, W)`` of the full-resolution frame
    :return: list of length ``n_levels + 1`` of amplification factors
    """
    if lambda_c is None:
        return [amplification] * n_levels + [0.0]

    height, width = image_shape
    delta = lambda_c / 8.0 / (1.0 + amplification)
    lam = np.sqrt(height ** 2 + width ** 2) / 3.0

    factors = []
    for lvl in range(n_levels + 1):
        if lvl == 0 or lvl == n_levels:
            factors.append(0.0)  # ignore finest band and low-pass residual
        else:
            curr = lam / delta / 8.0 - 1.0
            factors.append(float(np.clip(curr, 0.0, amplification)))
        lam /= 2.0
    return factors


class EulerianMagnifier:
    """Linear Eulerian Video Magnification for pre-test motion visualization.

    Reveals subtle, sub-pixel motion directly from raw video by band-passing the
    per-pixel intensity in time and amplifying it. Intended as a qualitative
    pre-test aid only (see the module-level warning): do not measure displacement
    from the magnified video.

    Typical use::

        from pyidi import VideoReader
        from pyidi.postprocessing import EulerianMagnifier

        video = VideoReader("recording.cih")
        mag = EulerianMagnifier(video)
        mag.configure(freq_band=(90, 110), amplification=20)
        magnified = mag.get_magnified_video()
        mag.save("magnified.mp4", output_format="mp4")

    :param video: the video to magnify
    :type video: VideoReader
    """

    def __init__(self, video, *args, **kwargs):
        if not isinstance(video, VideoReader):
            raise TypeError("Expected 'video' to be a pyidi.VideoReader instance.")
        self.video = video

        # Defaults
        self.freq_band = None
        self.amplification = 10.0
        self.levels = 4
        self.filter_type = "ideal"
        self.lambda_c = None
        self.fps = getattr(video, "fps", None)
        self.mask = None
        self.show_progress = True

        self.magnified = None
        self._display_range = None

        self.configure(*args, **kwargs)

    def configure(self, freq_band=None, amplification=None, levels=None,
                  filter_type=None, lambda_c=None, fps=None, mask=None,
                  show_progress=None):
        """Store magnification settings as attributes.

        Only arguments that are not ``None`` overwrite the current setting, so a
        subset of parameters can be updated on repeated calls. To use the
        ``lambda_c`` spatial attenuation, set it here; it stays disabled otherwise.

        :param freq_band: temporal pass-band ``(low, high)`` in Hz
        :type freq_band: tuple or list
        :param amplification: amplification factor ``alpha``, defaults to 10
        :type amplification: int or float
        :param levels: number of Laplacian pyramid levels, defaults to 4
        :type levels: int
        :param filter_type: ``"ideal"`` (FFT band-pass, best for a narrow band around
            a mode) or ``"butter"`` (Butterworth), defaults to ``"ideal"``
        :type filter_type: str
        :param lambda_c: spatial wavelength cutoff in pixels for artifact-reducing
            attenuation (experimental); ``None`` uses a constant amplification
        :type lambda_c: int, float or None
        :param fps: sampling rate in Hz; if ``None`` the value from the VideoReader
            is used
        :type fps: int, float or None
        :param mask: region-of-interest mask, a 2D array matching the frame size
            ``(height, width)``. Amplification is applied only where the mask is
            non-zero, so motion outside stays as in the original. Values may be
            boolean or floats in ``[0, 1]`` (soft edges reduce seams). ``None``
            magnifies the whole frame.
        :type mask: numpy.ndarray or None
        :param show_progress: show progress bars while processing, defaults to True
        :type show_progress: bool
        """
        if freq_band is not None:
            self.freq_band = _validate_freq_band(freq_band)
        if filter_type is not None:
            self.filter_type = _validate_filter_type(filter_type)
        if amplification is not None:
            self.amplification = float(amplification)
        if levels is not None:
            self.levels = int(levels)
        if lambda_c is not None:
            self.lambda_c = float(lambda_c)
        if fps is not None:
            self.fps = float(fps)
        if mask is not None:
            self.mask = np.asarray(mask, dtype=np.float32)
        if show_progress is not None:
            self.show_progress = bool(show_progress)

    def _resolve_settings(self, frames):
        """Validate settings against the loaded frames and return ``(fps, levels)``."""
        if self.freq_band is None:
            raise ValueError("'freq_band' is not set. Call configure(freq_band=...).")
        if self.fps is None:
            raise ValueError(
                "Sampling rate 'fps' is unknown. Set it via configure(fps=...) or on "
                "the VideoReader."
            )

        low, high = self.freq_band
        if high >= 0.5 * self.fps:
            raise ValueError(
                f"Upper band edge {high} Hz must be below Nyquist "
                f"({0.5 * self.fps} Hz for fps={self.fps})."
            )
        if self.filter_type == "butter" and low <= 0:
            raise ValueError("'butter' filter requires a strictly positive lower band edge.")

        height, width = frames.shape[1], frames.shape[2]
        max_levels = _max_pyramid_levels(height, width)
        levels = self.levels
        if levels > max_levels:
            warnings.warn(
                f"Requested {levels} pyramid levels but image is too small; "
                f"using {max_levels}."
            )
            levels = max_levels
        return self.fps, levels

    def get_magnified_video(self, frame_range=None):
        """Compute and return the motion-magnified video.

        :param frame_range: passed to ``VideoReader.get_frames``; ``None`` uses all
            frames, an ``int`` uses frames ``0..int``, a ``(start, stop)`` tuple uses
            that slice. Subset large recordings to bound memory.
        :type frame_range: int, tuple, list or None
        :return: magnified video, ``(N, H, W)``, same dtype as the source frames
        :rtype: numpy.ndarray
        """
        frames = self.video.get_frames(frame_range)
        source_dtype = frames.dtype
        frames = np.asarray(frames, dtype=np.float32)
        intensity_min, intensity_max = float(frames.min()), float(frames.max())

        fps, levels = self._resolve_settings(frames)
        mask = self._resolve_mask(frames.shape[1:])
        progress = self.show_progress

        pyramid = _build_laplacian_pyramid(frames, levels, progress=progress)
        factors = _level_amplifications(
            levels, self.amplification, self.lambda_c, frames.shape[1:]
        )

        active = [lvl for lvl in range(len(pyramid)) if factors[lvl] != 0.0]
        for lvl in tqdm(active, desc="Temporal filtering", disable=not progress):
            filtered = _temporal_bandpass(
                pyramid[lvl], fps, self.freq_band, self.filter_type
            )
            pyramid[lvl] += factors[lvl] * filtered

        magnified = _collapse_laplacian_pyramid(pyramid, progress=progress)

        # Restrict amplification to the region of interest: keep only the added
        # (amplified) part inside the mask, leaving the rest as the original.
        if mask is not None:
            magnified = frames + mask * (magnified - frames)

        # Clip to the source dtype range only, to avoid integer wraparound on cast
        # (this preserves the amplified signal). Display contrast is handled at save
        # time by mapping the original intensity range to 0-255, so a few overshoot
        # pixels do not wash out the result.
        if np.issubdtype(source_dtype, np.integer):
            info = np.iinfo(source_dtype)
            magnified = np.clip(magnified, info.min, info.max)
        self._display_range = (intensity_min, intensity_max)
        self.magnified = magnified.astype(source_dtype)
        return self.magnified

    def _resolve_mask(self, frame_shape):
        """Validate the ROI mask against the frame size and return it (or None)."""
        if self.mask is None:
            return None
        if self.mask.shape != tuple(frame_shape):
            raise ValueError(
                f"mask shape {self.mask.shape} does not match the frame size "
                f"{tuple(frame_shape)}. The mask must be a 2D (height, width) array "
                "matching the frames the magnifier sees (e.g. after any downscaling)."
            )
        return self.mask

    def save(self, filename, fps=None, output_format="mp4", frame_range=None):
        """Write the magnified video to a file.

        Frames are contrast-normalized to 8-bit for playback. If the video has not
        been computed yet (or ``frame_range`` is given), it is computed first.

        :param filename: output path without extension
        :type filename: str
        :param fps: playback frame rate; defaults to the configured sampling rate
        :type fps: int, float or None
        :param output_format: one of ``"mp4"``, ``"avi"``, ``"mov"``, ``"gif"``
        :type output_format: str
        :param frame_range: optional frame range to (re)compute before saving
        :type frame_range: int, tuple, list or None
        """
        if output_format not in ("mp4", "avi", "mov", "gif"):
            raise ValueError("'output_format' must be one of 'mp4', 'avi', 'mov', 'gif'.")

        if self.magnified is None or frame_range is not None:
            self.get_magnified_video(frame_range=frame_range)

        write_fps = int(fps if fps is not None else (self.fps or 30))
        frames_8bit = self._to_uint8_rgb(self.magnified, self._display_range)

        uri = f"{filename}.{output_format}"
        if output_format == "gif":
            iio.imwrite(uri, frames_8bit, plugin="pillow",
                        duration=1000.0 / write_fps, loop=0)
        else:
            iio.imwrite(uri, frames_8bit, plugin="pyav",
                        codec="libx264", fps=write_fps)

        print(f"Magnified video saved to: {uri}")

    @staticmethod
    def _to_uint8_rgb(video, display_range=None):
        """Map a ``(N, H, W)`` stack to ``(N, H, W, 3)`` uint8 for playback.

        The intensity window ``display_range = (lo, hi)`` is mapped to ``0..255``
        (values outside are clipped), so amplification overshoot does not wash out
        contrast. If ``None``, the stack's own min/max are used.
        """
        vid = video.astype(np.float32)
        if display_range is None:
            lo, hi = float(vid.min()), float(vid.max())
        else:
            lo, hi = display_range
        if hi > lo:
            vid = np.clip((vid - lo) / (hi - lo), 0.0, 1.0) * 255.0
        else:
            vid = np.zeros_like(vid)
        vid = vid.astype(np.uint8)
        return np.repeat(vid[..., np.newaxis], 3, axis=-1)


def eulerian_magnification(video, freq_band, amplification=10, levels=4,
                           filter_type="ideal", lambda_c=None, fps=None,
                           mask=None, show_progress=True, frame_range=None):
    """Functional wrapper around :class:`EulerianMagnifier`.

    See :class:`EulerianMagnifier` and its ``configure`` method for the meaning of
    the parameters. Returns the magnified video (visualization only — not a
    displacement measurement).

    :param video: the video to magnify
    :type video: VideoReader
    :param freq_band: temporal pass-band ``(low, high)`` in Hz
    :type freq_band: tuple or list
    :param amplification: amplification factor ``alpha``, defaults to 10
    :type amplification: int or float
    :param levels: number of Laplacian pyramid levels, defaults to 4
    :type levels: int
    :param filter_type: ``"ideal"`` or ``"butter"``, defaults to ``"ideal"``
    :type filter_type: str
    :param lambda_c: spatial wavelength cutoff in pixels (experimental), or None
    :type lambda_c: int, float or None
    :param fps: sampling rate in Hz; if None the VideoReader value is used
    :type fps: int, float or None
    :param mask: region-of-interest mask ``(height, width)``; amplifies only where
        non-zero. ``None`` magnifies the whole frame.
    :type mask: numpy.ndarray or None
    :param show_progress: show progress bars while processing, defaults to True
    :type show_progress: bool
    :param frame_range: frame range passed to ``VideoReader.get_frames``
    :type frame_range: int, tuple, list or None
    :return: magnified video, ``(N, H, W)``, same dtype as the source frames
    :rtype: numpy.ndarray
    """
    magnifier = EulerianMagnifier(video)
    magnifier.configure(
        freq_band=freq_band, amplification=amplification, levels=levels,
        filter_type=filter_type, lambda_c=lambda_c, fps=fps, mask=mask,
        show_progress=show_progress,
    )
    return magnifier.get_magnified_video(frame_range=frame_range)
