.. _results:

Results, saving and reloading
=============================

The displacement array
----------------------

``get_displacements()`` returns, and stores on the method object as
``.displacements``, an array of shape ``(n_points, n_frames, 2)`` in pixels,
relative to the reference frame:

.. code:: python

    displacements = lk.get_displacements()

    displacements[i, :, 0]     # displacement history of point i, row (y) direction
    displacements[i, :, 1]     # displacement history of point i, column (x) direction

``DIC`` additionally exposes ``.warp_params`` of shape
``(n_points, n_frames, n_param)``, from which strain and rotation are read
directly — see :doc:`disp_id_methods`.

.. _failed-points:

Points that could not be tracked
--------------------------------

A point placed on a uniform region, or on a single straight edge with no
gradient along it, cannot be tracked. Since 1.4.0 this no longer aborts the
analysis: the point is set to ``NaN`` from the frame at which it was lost,
every other point is computed normally, and a warning is issued.

Which points failed, and why, is recorded in ``failed_points``:

.. code:: python

    displacements = lk.get_displacements()

    lk.failed_points          # {point_index: {'frame': ..., 'status': ...}}

``status`` distinguishes a singular gradient matrix (a flat or edge-only
subset — the point never had enough information to track) from a diverged
iteration (the optimizer ran away to a non-finite value).

The practical consequence downstream is that results may contain ``NaN``:

.. code:: python

    import numpy as np

    ok = ~np.isnan(displacements).any(axis=(1, 2))     # points that tracked all the way
    amplitude = np.nanmax(np.abs(displacements), axis=1)

.. warning::

    The detection is best effort. It catches points whose displacement becomes
    non-finite or larger than the image, but a point can return physically
    implausible values well below that bound. **A result without ``NaN`` is
    not proof that every point tracked correctly.** Check the displacements
    against what the structure can plausibly do.

Where results are saved
-----------------------

``get_displacements()`` saves automatically (pass ``autosave=False`` to
suppress it). Results go into a directory next to the recording, one
sub-directory per run:

.. code:: text

    measurement.cih
    measurement_pyidi_analysis/
        analysis_001/
            points.pkl          # the points that were tracked
            results.pkl         # the displacements
            directions.pkl      # only for DirectionalLucasKanade
            warp_params.pkl     # only for DIC
            settings.json       # every configure() argument, plus source and date
        analysis_002/
            ...

``settings.json`` is what makes a run reproducible: it records the input file,
the method, the creation date, the video dimensions, and the full
configuration. This is why every ``configure()`` parameter must be stored as
an attribute of the same name.

Reloading a saved analysis
--------------------------

.. code:: python

    from pyidi import load_analysis

    video, idi, settings = load_analysis('measurement_pyidi_analysis/analysis_001')

    displacements = idi.displacements
    points = idi.points

``load_analysis`` returns three things: a fresh
:class:`~pyidi.video_reader.VideoReader`, the reconstructed method object with
its points, directions and results restored, and the settings dictionary.

.. code:: python

    # the recording has moved since the analysis was run
    video, idi, settings = load_analysis(path, input_file='new/location/measurement.cih')

    # only the points and settings, without reading the results back
    video, idi, settings = load_analysis(path, load_results=False)

Resuming an interrupted analysis
--------------------------------

Long analyses checkpoint as they go, into a ``temp_file`` directory beside the
recording. If a run is interrupted — a crash, a full disk, a closed laptop —
it can pick up from the last completed time point instead of starting over:

.. code:: python

    lk.configure(resume_analysis=True)
    displacements = lk.get_displacements()

The checkpoint is only reused if the settings match the interrupted run; if
they do not, the analysis restarts from the beginning. The temporary files are
removed on successful completion.

.. note::

    ``failed_points`` is rebuilt from scratch on resume rather than being
    restored from the checkpoint. A point that was already lost before the
    interruption stays lost — the previous displacement is checked for
    ``NaN``/``inf`` before it is used — but the recorded frame number refers to
    the resumed run.

Viewing the results
-------------------

``ResultViewer`` animates the identified displacements over the recording
(requires the ``[qt]`` extra):

.. code:: python

    from pyidi import ResultViewer

    viewer = ResultViewer(
        video,
        displacements=displacements,
        points=points,
        fps=30,                # playback rate
        magnification=10,      # visual amplification of the displacement
        point_size=10,
        colormap='cool',
    )

The window opens on construction and blocks until it is closed. ``points``
uses the same ``(row, column)`` convention as everywhere else, and
``displacements`` is the array as returned by ``get_displacements()``. A
2-D ``(n_points, 2)`` array is also accepted and animated as a mode shape.

If the analysis was run through the napari :doc:`GUI <napari>`, the method
object is at ``gui.method`` and the results at ``gui.method.displacements``.
