.. _mode-shape-magnification:

Mode-shape magnification
========================

Mode-shape magnification takes an *identified* mode shape and warps the
reference image by it, scaled up by a chosen factor. Where
:doc:`Eulerian magnification <postprocessing/eulerian_magnification>` works on
the raw video before any identification, this works on the result: the
displacements are already known, and the magnification is a faithful geometric
scaling of them.

The implementation follows Čufar et al. [1]_.

Two functions are provided:

* ``mode_shape_magnification()`` — a single magnified image;
* ``animate()`` — an animation of the mode shape over one or more periods,
  written to a file.

A still image
-------------

.. code:: python

    from pyidi.postprocessing import mode_shape_magnification

    magnified = mode_shape_magnification(
        displacements=mode_shape,     # (n_points, 2), the identified mode shape
        magnification_factor=20,
        idi=lk,                       # the method instance the shape came from
    )

Passing the method instance as ``idi`` is the shortcut: the reference image and
the point coordinates are taken from it. Both can be given explicitly instead,
which is what you do when the shape comes from somewhere else:

.. code:: python

    magnified = mode_shape_magnification(
        displacements=mode_shape,
        magnification_factor=20,
        image=reference_image,        # (height, width)
        points=points,                # (n_points, 2), row/column
    )

Other arguments:

.. list-table::
   :header-rows: 1
   :widths: 26 14 60

   * - Parameter
     - Default
     - Meaning
   * - ``background_brightness``
     - ``0.3``
     - Brightness of the background, in ``[0, 1]``.
   * - ``show_undeformed``
     - ``False``
     - Draw the reference image underneath the magnified shape, so the
       deformation can be read against the original geometry.

An animation
------------

.. code:: python

    from pyidi.postprocessing import animate

    animate(
        displacements=mode_shape,
        magnification_factor=20,
        idi=lk,
        fps=30,
        n_periods=3,
        filename='mode_1',
        output_format='gif',
    )

The mode shape is animated through ``n_periods`` full periods at ``fps``
frames per second and written to ``filename.output_format``.

.. note::

    ``displacements`` here is a *mode shape*: one real displacement vector per
    point, shape ``(n_points, 2)``. It is **not** the ``(n_points, n_frames, 2)``
    time history returned by ``get_displacements()`` — a 2-D array is required
    and anything else raises ``TypeError``. Extract the shape first, for
    example from an FRF-based modal identification of the displacement
    histories.

    ``animate()`` scales that one shape harmonically through
    ``n_periods`` periods; it does not replay the measured time history.

Worked example
--------------

See the `showcase notebook
<https://github.com/ladisk/pyidi/blob/master/examples/Showcase_MS_mag.ipynb>`_
for a full example, from identification through modal analysis to the
magnified animation.

Reference
---------

.. [1] Čufar, K., Slavič, J., & Boltežar, M. (2024). Mode-shape magnification
   in high-speed camera measurements. *Mechanical Systems and Signal
   Processing*, 213, 111336. https://doi.org/10.1016/J.YMSSP.2024.111336
