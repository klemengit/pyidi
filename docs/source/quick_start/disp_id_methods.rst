.. _implemented_disp_id_methods:

Displacement identification methods
===================================

Four methods are implemented. They share the same interface — construct with a
:class:`~pyidi.video_reader.VideoReader`, ``set_points()``, ``configure()``,
``get_displacements()`` — and differ in what they solve for and at what cost.

Choosing a method
-----------------

.. list-table::
   :header-rows: 1
   :widths: 22 16 20 20 22

   * - Method
     - Solves for
     - Displacement range
     - Speed
     - Typical use
   * - :ref:`Simplified Optical Flow <method-sof>`
     - 2 translations, from the image gradient
     - well below one pixel
     - fastest
     - a first look at a dense field of points
   * - :ref:`Lucas-Kanade <method-lk>`
     - 2 translations, iteratively
     - several pixels
     - fast (compiled)
     - the default choice
   * - :ref:`Directional Lucas-Kanade <method-dlk>`
     - 1 translation along a known direction
     - several pixels
     - fast (compiled)
     - motion along a known axis; features defined in one direction only
   * - :ref:`DIC <method-dic>`
     - 6 (affine) or 3 (rigid) warp parameters
     - several pixels
     - slowest
     - strain and in-plane rotation, not just translation

If you are unsure, start with ``LucasKanade``.

.. _method-sof:

Simplified Optical Flow (SOF)
-----------------------------

SOF estimates displacement directly from the image gradient and the intensity
change relative to a reference image. There is no iteration, which makes it
very fast, but it is a linearisation: it is only valid while the motion stays
well below one pixel.

.. code:: python

    from pyidi import VideoReader, SimplifiedOpticalFlow

    video = VideoReader('measurement.cih')

    sof = SimplifiedOpticalFlow(video)
    sof.set_points(points)
    sof.configure(subset_size=3, reference_range=(0, 100))

    displacements = sof.get_displacements()

.. list-table::
   :header-rows: 1
   :widths: 26 14 60

   * - Parameter
     - Default
     - Meaning
   * - ``subset_size``
     - ``3``
     - Size of the averaging subset around each point.
   * - ``reference_range``
     - ``(0, 100)``
     - Frames averaged into the reference image. Averaging suppresses sensor
       noise in the reference.
   * - ``pixel_shift``
     - ``False``
     - Track the integer part of the displacement by shifting the subset,
       extending the usable range beyond a fraction of a pixel.
   * - ``convert_from_px``
     - ``1.``
     - Distance unit per pixel, if you want the result in physical units.
   * - ``mean_n_neighbours``
     - ``0``
     - Average the result over this many neighbouring points, to trade spatial
       resolution for noise.
   * - ``zero_shift``
     - ``False``
     - Shift each signal so its mean is zero.
   * - ``frame_range``
     - ``'all'``
     - Part of the recording to process.

Reference:

    [1] Javh, J., Slavič, J., & Boltežar, M. (2017). The subpixel resolution of optical-flow-based modal analysis. Mechanical Systems and Signal Processing, 88, 89–99. https://doi.org/10.1016/j.ymssp.2016.11.009

.. _method-lk:

Lucas-Kanade (LK)
-----------------

The Lucas-Kanade method iteratively solves for the translation of each subset
between the reference image and the current frame, to sub-pixel accuracy. This
is the workhorse method.

.. code:: python

    from pyidi import VideoReader, LucasKanade

    video = VideoReader('measurement.cih')

    lk = LucasKanade(video)
    lk.set_points(points)
    lk.configure(roi_size=(21, 21), max_nfev=20, tol=1e-8)

    displacements = lk.get_displacements()

.. list-table::
   :header-rows: 1
   :widths: 26 14 60

   * - Parameter
     - Default
     - Meaning
   * - ``roi_size``
     - ``(9, 9)``
     - Subset size in pixels, ``(vertical, horizontal)``. Larger is more
       robust and less local; it must be large enough to contain distinctive
       texture.
   * - ``pad``
     - ``2``
     - Padding around the subset, so the interpolation has data to work with
       at the edges.
   * - ``max_nfev``
     - ``20``
     - Maximum iterations per point per frame.
   * - ``tol``
     - ``1e-8``
     - Convergence threshold on the displacement increment.
   * - ``int_order``
     - ``3``
     - Interpolation spline order. Only ``3`` runs on the compiled kernel.
   * - ``reference_image``
     - ``0``
     - Frame index, a ``(start, stop)`` tuple to average over, or an array.
   * - ``frame_range``
     - ``'full'``
     - Part of the recording to process.
   * - ``processes``
     - ``1``
     - Number of worker processes.
   * - ``resume_analysis``
     - ``False``
     - Continue an interrupted run from its last checkpoint. See
       :doc:`results`.
   * - ``use_compiled_kernel``
     - ``True``
     - Use the compiled numba kernel. See below.

Reference:

    [2] Lucas, B. D., & Kanade, T. (1981). An Iterative Image Registration Technique with an Application to Stereo Vision. In Proceedings of the 7th International Joint Conference on Artificial Intelligence - Volume 2 (pp. 674–679). San Francisco, CA, USA: Morgan Kaufmann Publishers Inc. Retrieved from http://dl.acm.org/citation.cfm?id=1623264.1623280

.. _lk-performance:

Performance
^^^^^^^^^^^

The inner optimization loop is compiled with ``numba`` and parallelized over
points. This is on by default (``use_compiled_kernel=True``) and is typically
one to two orders of magnitude faster than the pure NumPy implementation::

    lk.configure(roi_size=(9, 9), use_compiled_kernel=True)

Measured against 1.3.3 on the same machine, with identical results:

.. list-table::
   :header-rows: 1
   :widths: 52 16 16 16

   * - Case
     - 1.3.3
     - 1.4.0
     - Speed-up
   * - ``data_synthetic.cih``, 200 points, 101 frames
     - 7.94 s
     - 0.10 s
     - 77x
   * - synthetic 512x512, 400 points, 150 frames
     - 21.04 s
     - 0.24 s
     - 89x
   * - ``data_synthetic.mp4``, 60 points, 10 frames
     - 2.16 s
     - 0.06 s
     - 36x

Notes:

* The compiled kernel supports cubic interpolation only. With ``int_order``
  set to anything other than ``3`` it falls back to the NumPy implementation
  and warns once.
* The kernel is compiled the first time it runs, which takes a few seconds.
  The result is cached on disk, so later runs in a fresh session skip it. Set
  ``NUMBA_CACHE_DIR`` if pyIDI is installed somewhere the cache cannot be
  written.
* ``use_compiled_kernel=False`` selects the NumPy implementation. Results
  agree with the compiled kernel to floating-point round-off, so the switch
  affects speed only. If numba is not importable the same fallback happens
  automatically, with a warning. The NumPy path is itself faster than 1.3.3,
  because the frame is no longer re-read for every point.
* Points are parallelized with threads when ``processes=1`` (the default), and
  with processes when ``processes`` is greater than one. The two are never
  combined, so they cannot oversubscribe the CPU.
* On Linux, pyIDI requests numba's fork-safe threading layer at import time,
  and switches the worker pool away from ``fork`` if a GNU OpenMP runtime is
  already loaded — forking after libgomp has been used crashes the children.
  This is automatic; you only need to know about it if you set
  ``NUMBA_THREADING_LAYER`` yourself.

.. _failed-points-lk:

Points that cannot be tracked
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A point placed on a uniform region, or on a single straight edge with no
gradient along it, cannot be tracked. Rather than aborting the analysis, such
a point is set to ``NaN`` from the frame at which it was lost, every other
point is computed normally, and a warning is issued. Which points failed, and
why, is recorded in ``failed_points`` — see :ref:`failed-points`.

.. _method-dlk:

Directional Lucas-Kanade
------------------------

The directional method solves for a *single* translation along a prescribed
direction per point, instead of two independent components. Constraining the
solve this way makes it possible to track features that plain Lucas-Kanade
cannot — most usefully a single straight edge, which carries no information
along its own length but is sharply defined across it.

.. code:: python

    from pyidi import VideoReader, DirectionalLucasKanade

    video = VideoReader('measurement.cih')

    lk1d = DirectionalLucasKanade(video)
    lk1d.set_points(points)
    lk1d.configure(roi_size=(9, 9), dij=(1, 0), pad=(2, 2))

    displacements = lk1d.get_displacements()

``dij`` is the assumed motion direction as ``(di, dj)`` — row and column
components, in the convention *negative is down, positive is right*. It is
normalised automatically. A single ``(2,)`` vector applies to every point; an
``(n_points, 2)`` array gives each point its own direction:

.. code:: python

    lk1d.set_directions(dij)      # (2,) or (n_points, 2)

Per-point directions are what the automatic feature selection in [3]_ produces,
and they are saved alongside the results (``directions.pkl``) so a reloaded
analysis keeps them.

.. note::

    ``pad`` here is a ``(pad_y, pad_x)`` pair, unlike ``LucasKanade.pad`` which
    is a scalar. A bare integer is accepted and broadcast to both axes.

.. _rigid-body-motion:

Prescribed rigid-body motion
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When the whole structure translates — a machine on soft mounts, a specimen on
a shaker, a camera that drifts — the interesting local motion sits on top of a
large common motion. If the rigid-body translation is known independently (for
example from a fiducial marker, see :doc:`../fiducial_marker`), it can be
prescribed:

.. code:: python

    lk1d.configure(roi_size=(9, 9), dij=(1, 0))
    lk1d.set_rigid_body_motion(rbm_ij)      # (n_time_points, 2), in pixels

    displacements = lk1d.get_displacements()

Two things then happen. The tracking window of every point *follows* the
prescribed motion, so the feature stays inside the subset even when the
rigid-body translation is many pixels — the local solve never has to chase it.
And the prescribed motion is subtracted back out of the result, so
``displacements`` reports the local motion **relative to the rigid body**,
not each point's absolute position in the frame.

.. code:: python

    lk1d.set_rigid_body_motion(None)        # back to zero (requires configure() first)

Limitations, as currently implemented:

* Only the component of the rigid-body motion aligned with each point's
  tracking direction ``dij`` is used. A rigid-body translation perpendicular
  to ``dij`` is not compensated.
* The shape of ``rbm_ij`` is not validated at runtime; it must be
  ``(n_time_points, 2)`` for the frame range being processed.
* If ``set_rigid_body_motion`` is never called, it defaults to zero and the
  analysis behaves exactly as before.

Performance
^^^^^^^^^^^

``DirectionalLucasKanade`` uses the same compiled kernel machinery as
Lucas-Kanade, with the two-parameter translation solve replaced by the
one-parameter solve along the prescribed direction. Everything in
:ref:`lk-performance` applies here too: ``use_compiled_kernel=True`` by
default, cubic interpolation only, threads when ``processes=1`` and processes
otherwise::

    lk1d.configure(roi_size=(9, 9), dij=(1, 0), use_compiled_kernel=True)

.. note::

    ``use_compiled_kernel`` replaces the ``use_numba`` argument, which was
    accepted but never did anything. Calls that passed ``use_numba`` need
    updating; the behaviour they asked for is now the default.

Points that cannot be tracked
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The directional method only ever sees the image gradient projected onto the
search direction, so it is easier to lose a point here than with plain
Lucas-Kanade: a feature with a strong gradient across the direction but none
along it is untrackable, however well defined it looks. Such a point is set to
``NaN`` from the frame at which it was lost, the rest of the analysis
continues, and the detail is recorded in ``failed_points``, exactly as for
Lucas-Kanade.

.. [3] Masmeijer T., Habtour E., Zaletelj K. & Slavič J. (2025). Directional DIC method with automatic feature selection. Mechanical Systems and Signal Processing, 224. https://doi.org/10.1016/j.ymssp.2024.112080

.. _method-dic:

Digital Image Correlation (DIC)
-------------------------------

Full-field 2D Digital Image Correlation using Inverse Compositional
Gauss-Newton (IC-GN) optimization with the Zero Normalized Sum of Squared
Differences (ZNSSD) criterion. Unlike the methods above, DIC solves for the
full deformation of each subset, not just its translation.

.. code:: python

    from pyidi import VideoReader, DIC

    video = VideoReader('measurement.cih')

    dic = DIC(video)
    dic.set_points(points)
    dic.configure(roi_size=(21, 21), warp='affine', max_nfev=100, tol=1e-6)

    displacements = dic.get_displacements()

Two warp models are supported:

* ``warp='affine'`` (default, 6 parameters): full first-order shape function
  with translation, normal strains, shear and rotation. Parameter vector
  ``[du/dx, du/dy, u, dv/dx, dv/dy, v]``.
* ``warp='rigid'`` (3 parameters): translation and in-plane rotation.
  Parameter vector ``[u, v, phi]``.

In addition to the standard ``displacements`` array of shape
``(n_points, n_frames, 2)``, the method exposes the full converged warp
parameters as ``self.warp_params`` of shape ``(n_points, n_frames, n_param)``.
From the affine parameters one can directly recover in-plane strains and
rotation::

    eps_xx   = dic.warp_params[..., 0]
    eps_yy   = dic.warp_params[..., 4]
    shear_xy = 0.5 * (dic.warp_params[..., 1] + dic.warp_params[..., 3])
    rotation = 0.5 * (dic.warp_params[..., 3] - dic.warp_params[..., 1])  # rad

``prefilter_gauss=True`` (the default) uses the Gauss-prefiltered finite
difference kernel ``[-0.446, 0, 0.446]`` for the reference gradient, instead
of ``[-0.5, 0, 0.5]``.

This method is a port of the **pyDIC** library by the LADISK research group
(University of Ljubljana, Faculty of Mechanical Engineering) into the pyIDI
multi-point ``IDIMethod`` framework. The original implementation is available
at https://github.com/ladisk/pyDIC and provides the algorithmic basis for this
class (gradient kernel, Jacobians, steepest-descent images, Hessian,
inverse-compositional warp update, ZNSSD error image). Please cite both the
underlying algorithm and the pyDIC repository when using this method.

References:

    [4] Baker, S., & Matthews, I. (2004). Lucas-Kanade 20 Years On: A Unifying Framework. International Journal of Computer Vision, 56(3), 221-255. https://doi.org/10.1023/B:VISI.0000011205.11775.fd

    [5] Pan, B., Qian, K., Xie, H., & Asundi, A. (2009). Two-dimensional digital image correlation for in-plane displacement and strain measurement: a review. Measurement Science and Technology, 20(6), 062001. https://doi.org/10.1088/0957-0233/20/6/062001
