.. _eulerian-magnification:

Eulerian video magnification
============================

Eulerian Video Magnification (EVM) amplifies subtle, often sub-pixel motion
directly in a raw recording. It answers the question you have *before* running
an identification: **is this structure moving at all, where, and at which
frequency?**

.. warning::

    **This is qualitative visualization, not a measurement.** Eulerian
    magnification distorts motion amplitudes non-linearly. Never read
    displacement off a magnified video — use
    :doc:`../quick_start/disp_id_methods` for that.

The implementation follows the linear EVM of Wu et al. [1]_: a spatial
Laplacian pyramid, a temporal band-pass filter applied per pyramid level, and
linear amplification of the band-passed signal added back onto the original.

What it is good for
-------------------

* **A pre-test sanity check.** Confirm the excitation is reaching the
  structure and the camera is seeing it, before committing to a long analysis.
* **Finding where to put points.** The magnified video shows which parts of
  the frame move, which is where the tracked points belong.
* **Isolating one mode.** Band-pass around a suspected natural frequency and
  everything else — rigid-body drift, higher modes, flicker — is suppressed,
  so the deflection shape of that one mode becomes visible.

Quick start
-----------

.. code:: python

    from pyidi import VideoReader
    from pyidi.postprocessing import EulerianMagnifier

    video = VideoReader('measurement.cih')

    evm = EulerianMagnifier(video)
    evm.configure(
        freq_band=(45.0, 55.0),   # Hz, around the mode of interest
        amplification=25,
        levels=4,
    )

    magnified = evm.get_magnified_video()   # (n_frames, height, width)
    evm.save('mode_50Hz', output_format='mp4')

There is also a one-shot functional form, if you only want the array:

.. code:: python

    from pyidi.postprocessing import eulerian_magnification

    magnified = eulerian_magnification(video, freq_band=(45.0, 55.0), amplification=25)

Parameters
----------

Settings are stored on the object by ``configure()``. Only arguments that are
not ``None`` overwrite the current value, so you can adjust one knob at a time
and re-run.

.. list-table::
   :header-rows: 1
   :widths: 18 12 70

   * - Parameter
     - Default
     - Meaning
   * - ``freq_band``
     - *required*
     - Temporal pass-band ``(low, high)`` in Hz. Must satisfy
       ``low < high``, and ``high`` must be below Nyquist (``0.5 * fps``).
   * - ``amplification``
     - ``10.0``
     - The gain :math:`\alpha` applied to the band-passed signal.
   * - ``levels``
     - ``4``
     - Laplacian pyramid depth. More levels means coarser spatial detail is
       amplified too. Reduced automatically (with a warning) if the frames are
       too small.
   * - ``filter_type``
     - ``"ideal"``
     - ``"ideal"`` is an FFT brick-wall filter — the sharpest band, and the
       right choice for isolating a single mode. ``"butter"`` is a Butterworth
       filter and needs a strictly positive lower band edge.
   * - ``lambda_c``
     - ``None``
     - Spatial-wavelength cutoff in pixels (experimental). See
       :ref:`evm-lambda-c`.
   * - ``fps``
     - from the video
     - Sampling rate in Hz. Taken from the
       :class:`~pyidi.video_reader.VideoReader` if it knows it.
   * - ``mask``
     - ``None``
     - 2-D region-of-interest array matching the frame size. See
       :ref:`evm-mask`.
   * - ``show_progress``
     - ``True``
     - Progress bars while processing.

Choosing a band and a gain
--------------------------

The band is the important setting; the gain is the one you tune afterwards.

**Band.** Pick it around a frequency you already suspect — from a
preliminary identification, an accelerometer, or an FE model. A narrow band
around a single peak gives the cleanest result. A band that is too wide lets
several modes through at once and the video becomes hard to read.

**Gain.** Start around 10-20 and increase until motion is visible. Too much
gain produces halos and ringing at strong edges: that is the linear
approximation breaking down, not a real feature of the structure.

**Frame count.** The temporal filter needs enough periods of the frequency you
are isolating to resolve it. A 0.4 Hz mode at 60 fps needs on the order of a
thousand frames; a 500 Hz mode at 10 kHz needs far fewer. Rule of thumb: aim
for at least five to ten periods inside the analysed range.

**Memory.** The whole range is held in memory as ``float32`` and a pyramid is
built on top of it. Bound this with ``frame_range``:

.. code:: python

    magnified = evm.get_magnified_video(frame_range=(0, 1200))

``frame_range`` is passed through to
:meth:`VideoReader.get_frames <pyidi.video_reader.VideoReader.get_frames>`:
``None`` means all frames, an ``int`` means frames ``0..int``, and a
``(start, stop)`` tuple means that slice. For long or high-resolution
recordings, spatially downscaling the video before magnifying is the other
lever.

.. _evm-mask:

Restricting to a region of interest
-----------------------------------

A ``mask`` limits amplification to part of the frame. Everything outside stays
exactly as recorded, which keeps a busy background from being amplified into
noise and makes the moving component easier to see.

.. code:: python

    import numpy as np

    mask = np.zeros((video.image_height, video.image_width))
    mask[200:600, 300:900] = 1.0

    evm.configure(mask=mask)

The mask may be boolean or float. Float values in ``[0, 1]`` are honoured as a
soft blend, so feathering the edge (for example with a Gaussian blur) avoids a
visible seam at the mask boundary.

.. _evm-lambda-c:

``lambda_c`` — spatial attenuation
----------------------------------

``lambda_c`` is the spatial-wavelength cutoff from Wu et al.: pyramid levels
whose spatial wavelength is below the cutoff get progressively less
amplification. The intent is to damp the amplification of fine, noisy detail
while leaving broad structural motion at full gain.

It is experimental and off by default (``None`` applies a constant
amplification to every band-pass level, and none to the low-pass residual).
Reach for it when a magnified video is dominated by high-frequency speckle
rather than by the motion you are after.

.. note::

    If ``lambda_c`` is small enough to attenuate every level to zero, the
    output equals the input and a warning is raised — the video is not broken,
    the cutoff is simply too aggressive.

Saving the result
-----------------

.. code:: python

    evm.save('mode_50Hz', output_format='mp4', fps=120)

* ``output_format`` is one of ``"mp4"``, ``"avi"``, ``"mov"``, ``"gif"``. The
  extension is appended to ``filename``.
* ``fps`` is the *playback* rate. It defaults to the sampling rate. For a
  high-speed recording, playing back at the capture rate is unwatchable — pass
  a slower rate to get the slow-motion effect.
* ``save()`` computes the video first if it has not been computed yet, or if a
  ``frame_range`` is given.
* The intensity range is mapped to 8-bit using the range of the *source*
  frames, so amplification overshoot does not wash out the whole video.

A full worked script, including streaming a long recording in and downscaling
it, is in `examples/eulerian_magnification_varcila.py
<https://github.com/ladisk/pyidi/blob/master/examples/eulerian_magnification_varcila.py>`_.

Reference
---------

.. [1] Wu, H.-Y., Rubinstein, M., Shih, E., Guttag, J., Durand, F., & Freeman, W.
   (2012). Eulerian Video Magnification for Revealing Subtle Changes in the
   World. *ACM Transactions on Graphics (Proc. SIGGRAPH 2012)*, 31(4).
   https://doi.org/10.1145/2185520.2185561

See :ref:`the API reference <api-eulerian>` for the full signatures.
