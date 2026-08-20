.. _migration:

Upgrading
=========

What breaks between versions, and what to write instead. For the full list of
changes see the :doc:`changelog`.

From 1.4.0
----------

``SubsetSelection`` has been removed
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

pyIDI had accumulated five separate point-selection implementations. There is
now one: :ref:`SelectionGUI <point-selection>`. The tkinter ``SubsetSelection``
widget is gone. The name is still importable, but instantiating it raises a
``RuntimeError`` naming the replacement, so old scripts fail with an
actionable message rather than an ``ImportError``.

.. code:: python

    # before
    selection = SubsetSelection(video, roi_size=(21, 21), noverlap=0)

    # now
    selection = SelectionGUI(video, subset_size=21, subset_overlap=0)

Note the sign convention: ``noverlap`` counted overlapping pixels, while
``subset_overlap`` is added to ``subset_size`` to give the step between subset
centres. A positive ``subset_overlap`` therefore spreads subsets *apart*; pass
a negative value to overlap them.

``subset_size`` accepts a scalar or a ``(height, width)`` pair, in the same
``(vertical, horizontal)`` convention as
``LucasKanade.configure(roi_size=...)``.

Also removed, all of it dead code: ``tools.ManualROI``, ``tools.GridOfROI``,
the unreachable ``PickPoints`` class in ``_simplified_optical_flow.py``, and
the ``pyidi.py`` module.

``set_points()`` now validates its input
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Input that was previously accepted silently — or that failed much later with
an opaque error — is now rejected up front. See :ref:`point-conventions`.

.. list-table::
   :header-rows: 1
   :widths: 34 33 33

   * - Input
     - Before
     - Now
   * - Empty, or 1-D
     - ``IndexError: tuple index out of range``
     - ``ValueError`` naming the problem
   * - Wrong number of columns
     - accepted, failed later
     - ``ValueError``
   * - Coordinates outside the image
     - accepted silently, surfaced as ``NaN`` results
     - ``ValueError``
   * - Sub-pixel (float) coordinates
     - crash in ``SimplifiedOpticalFlow``, silent truncation *toward zero*
       elsewhere
     - rounded to the nearest pixel, with a warning

``set_points()`` also accepts any object exposing a ``.points`` attribute, so
a selection GUI instance can be passed straight through:
``lk.set_points(gui)``.

From 1.3.3
----------

``use_numba`` is now ``use_compiled_kernel``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``DirectionalLucasKanade.configure()`` accepted a ``use_numba`` argument that
never did anything. The compiled path is now real, is on by default, and is
selected by ``use_compiled_kernel``:

.. code:: python

    # before (accepted, but had no effect)
    lk1d.configure(roi_size=(9, 9), dij=(1, 0), use_numba=True)

    # now (this is the default; pass False for the NumPy implementation)
    lk1d.configure(roi_size=(9, 9), dij=(1, 0), use_compiled_kernel=True)

Untrackable points return ``NaN`` instead of aborting
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A point on a uniform region, or on a single straight edge with no gradient
along it, used to abort the whole analysis. It now goes to ``NaN`` from the
frame at which it was lost, every other point is computed normally, and the
detail is recorded in ``failed_points``. See :ref:`failed-points`.

The practical consequence for downstream code: **results can contain
``NaN``**. Use ``np.nanmax``, ``np.nanmean`` and friends, or drop the failed
points explicitly.

``compute_inverse_numba`` and ``compute_delta_numba``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The 1.4.0 numba rewrite renamed these two helpers to ``compute_inverse`` and
``compute_delta``, which broke code importing the 1.3.3 names. Both old names
are restored as aliases, so no change is needed.

From 0.x — the pre-1.0 ``pyIDI`` class
--------------------------------------

Version 1.0 replaced the monolithic ``pyIDI`` class with a
:class:`~pyidi.video_reader.VideoReader` plus a separate method class. This
makes autocompletion and inline documentation work properly in VSCode,
PyCharm and similar editors, at the cost of backward compatibility.

.. code:: python

    # before
    from pyidi import pyIDI

    video = pyIDI('video.cih')
    video.set_method('sof')
    video.set_points(points)
    displacements = video.get_displacements()

    # now
    from pyidi import VideoReader, SimplifiedOpticalFlow

    video = VideoReader('video.cih')

    sof = SimplifiedOpticalFlow(video)
    sof.set_points(points)
    displacements = sof.get_displacements()

The methods themselves are unchanged — only the way they are called.

The legacy class is still importable as ``from pyidi import pyIDI``, kept for
compatibility only. It does not offer the full functionality of the current
API and is not being updated. To stay on the old version entirely:

.. code:: bash

    pip install pyidi==0.30.2
