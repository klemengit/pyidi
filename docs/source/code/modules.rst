API reference
=============

Everything below is generated from the docstrings in the source. For a
task-oriented introduction, start from the :doc:`tutorial
<../quick_start/basic_usage>` instead.

Top-level namespace
-------------------

These names are importable directly from ``pyidi``:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Name
     - Purpose
   * - :class:`~pyidi.video_reader.VideoReader`
     - Read a recording of any supported format.
   * - ``SimplifiedOpticalFlow``, ``LucasKanade``, ``DirectionalLucasKanade``,
       ``DIC``
     - The displacement identification methods.
   * - ``SelectionGUI``
     - Interactive point selection (requires the ``[qt]`` extra).
   * - ``GUI``, ``ResultViewer``, ``Viewer``
     - napari and Qt viewers (require the ``[qt]`` extra).
   * - ``load_analysis``
     - Reload a saved analysis from disk.
   * - ``Fiducial``
     - Fiducial-marker tracking and rigid-body compensation.
   * - ``postprocessing``
     - Eulerian video magnification and mode-shape magnification.
   * - ``pyIDI``
     - The legacy pre-1.0 class, kept for compatibility only.

Video reader
------------

.. automodule:: pyidi.video_reader
    :members:

Identification methods
----------------------

IDIMethod base class
^^^^^^^^^^^^^^^^^^^^

Every method inherits from ``IDIMethod``, which provides the shared
configuration handling, multiprocessing, checkpointing and result
persistence.

.. automodule:: pyidi.methods.idi_method
    :members:

Simplified optical flow
^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: pyidi.methods._simplified_optical_flow
    :members:

Lucas-Kanade
^^^^^^^^^^^^

.. automodule:: pyidi.methods._lucas_kanade
    :members:

Directional Lucas-Kanade
^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: pyidi.methods._directional_lucas_kanade
    :members:

Digital Image Correlation (DIC)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: pyidi.methods._dic
    :members:

Post-processing
---------------

.. _api-eulerian:

Eulerian video magnification
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: pyidi.postprocessing._eulerian_magnification
    :members:

Mode-shape magnification
^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: pyidi.postprocessing._motion_magnification
    :members:

Fiducial markers
----------------

.. automodule:: pyidi.fiducial
    :members:

Point selection geometry
------------------------

Pure-numpy ROI-grid geometry, shared by the napari ``GUI`` and
``SelectionGUI``. No GUI toolkit is needed to import or use it.

.. warning::

    These functions do not all share one coordinate convention — each
    docstring states which one it uses.

.. automodule:: pyidi.selection_geometry
    :members:

Saved analyses
--------------

.. automodule:: pyidi.load_analysis
    :members:

Legacy pyIDI class
------------------

.. autoclass:: pyidi.pyidi_legacy.pyIDI
    :members:
