.. _implemented_disp_id_methods:

Displacement identification methods
===============================================

Simplified Optical Flow (SOF)
-----------------------------

    [1] Javh, J., Slavič, J., & Boltežar, M. (2017). The subpixel resolution of optical-flow-based modal analysis. Mechanical Systems and Signal Processing, 88, 89–99. https://doi.org/10.1016/j.ymssp.2016.11.009

Lucas-Kanade (LK)
-----------------

    [2] Lucas, B. D., & Kanade, T. (1981). An Iterative Image Registration Technique with an Application to Stereo Vision. In Proceedings of the 7th International Joint Conference on Artificial Intelligence - Volume 2 (pp. 674–679). San Francisco, CA, USA: Morgan Kaufmann Publishers Inc. Retrieved from http://dl.acm.org/citation.cfm?id=1623264.1623280

Performance
~~~~~~~~~~~

The inner optimization loop is compiled with ``numba`` and parallelized over
points. This is on by default (``use_compiled_kernel=True``) and is typically one to two
orders of magnitude faster than the pure NumPy implementation::

    lk.configure(roi_size=(9, 9), use_compiled_kernel=True)

Notes:

* The compiled kernel supports cubic interpolation only. With ``int_order`` set
  to anything other than ``3`` it falls back to the NumPy implementation and
  warns once.
* The kernel is compiled the first time it runs, which takes a few seconds. The
  result is cached on disk, so later runs in a fresh session skip it. Set
  ``NUMBA_CACHE_DIR`` if pyidi is installed somewhere the cache cannot be
  written.
* ``use_compiled_kernel=False`` selects the NumPy implementation. Results agree
  with the compiled kernel to floating-point round-off, so the switch affects
  speed only. If numba is not importable the same fallback happens
  automatically, with a warning.
* Points are parallelized with threads when ``processes=1`` (the default), and
  with processes when ``processes`` is greater than one. The two are never
  combined, so they cannot oversubscribe the CPU.

Points that cannot be tracked
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A point placed on a uniform region, or on a single straight edge with no gradient
along it, cannot be tracked. Rather than aborting the analysis, such a point is
set to ``NaN`` from the frame at which it was lost, every other point is computed
normally, and a warning is issued. Which points failed, and why, is recorded in
``failed_points``::

    displacements = lk.get_displacements()
    lost = lk.failed_points          # {point_index: {'frame': ..., 'status': ...}}

.. warning::

    The detection is best effort. It catches points whose displacement becomes
    non-finite or larger than the image, but a point can return physically
    implausible values well below that bound. A result without ``NaN`` is not
    proof that every point tracked correctly. Check the displacements against
    what the structure can plausibly do.


Directional DIC
------------------------

    [3] Masmeijer T., Habtour E., Zaletelj K. & Slavič J. (2025). Directional DIC method with automatic feature selection. Mechanical Systems and Signal Processing, 224. https://doi.org/10.1016/j.ymssp.2024.112080

Performance
~~~~~~~~~~~

``DirectionalLucasKanade`` uses the same compiled kernel machinery as
Lucas-Kanade, with the two-parameter translation solve replaced by the
one-parameter solve along the prescribed direction. Everything in the
`Lucas-Kanade performance notes <#performance>`_ above applies here too:
``use_compiled_kernel=True`` by default, cubic interpolation only, threads when
``processes=1`` and processes otherwise::

    lk1d.configure(roi_size=(9, 9), dij=(1, 0), use_compiled_kernel=True)

.. note::

    ``use_compiled_kernel`` replaces the ``use_numba`` argument, which was
    accepted but never did anything. Calls that passed ``use_numba`` need
    updating; the behaviour they asked for is now the default.

Points that cannot be tracked
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The directional method only ever sees the image gradient projected onto the
search direction, so it is easier to lose a point here than with plain
Lucas-Kanade: a feature with a strong gradient across the direction but none
along it is untrackable, however well defined it looks. Such a point is set to
``NaN`` from the frame at which it was lost, the rest of the analysis continues,
and the detail is recorded in ``failed_points``, exactly as for Lucas-Kanade.
The same best-effort warning applies.

Digital Image Correlation (DIC)
-------------------------------

Full-field 2D Digital Image Correlation method using Inverse Compositional Gauss-Newton
(IC-GN) optimization with the Zero Normalized Sum of Squared Differences (ZNSSD) criterion.

This method is a port of the **pyDIC** library by the LADISK research group
(University of Ljubljana, Faculty of Mechanical Engineering) into the pyidi
multi-point ``IDIMethod`` framework. The original implementation is available at
https://github.com/ladisk/pyDIC and provides the algorithmic basis for this
class (gradient kernel, Jacobians, steepest-descent images, Hessian,
inverse-compositional warp update, ZNSSD error image). Please cite both the
underlying algorithm and the pyDIC repository when using this method.

Two warp models are supported:

* ``warp='affine'`` (default, 6 parameters): full first-order shape function with
  translation, normal strains, shear and rotation. Parameter vector
  ``[du/dx, du/dy, u, dv/dx, dv/dy, v]``.
* ``warp='rigid'`` (3 parameters): translation and in-plane rotation. Parameter vector
  ``[u, v, phi]``.

In addition to the standard ``displacements`` array of shape ``(n_points, n_frames, 2)``,
the method exposes the full converged warp parameters as ``self.warp_params`` of shape
``(n_points, n_frames, n_param)``. From the affine parameters one can directly recover
in-plane strains and rotation, e.g.::

    eps_xx   = idi.warp_params[..., 0]
    eps_yy   = idi.warp_params[..., 4]
    shear_xy = 0.5 * (idi.warp_params[..., 1] + idi.warp_params[..., 3])
    rotation = 0.5 * (idi.warp_params[..., 3] - idi.warp_params[..., 1])  # rad

The implementation is a port of the pyDIC algorithm (https://github.com/ladisk/pyDIC)
into the pyidi multi-point method framework.

    [4] Baker, S., & Matthews, I. (2004). Lucas-Kanade 20 Years On: A Unifying Framework. International Journal of Computer Vision, 56(3), 221-255. https://doi.org/10.1023/B:VISI.0000011205.11775.fd

    [5] Pan, B., Qian, K., Xie, H., & Asundi, A. (2009). Two-dimensional digital image correlation for in-plane displacement and strain measurement: a review. Measurement Science and Technology, 20(6), 062001. https://doi.org/10.1088/0957-0233/20/6/062001