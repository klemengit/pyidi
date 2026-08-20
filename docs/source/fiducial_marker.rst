.. _fiducial-markers:

Fiducial markers and motion compensation
========================================

When the camera and the structure move relative to each other for reasons that
have nothing to do with the vibration you are measuring — a shaking floor, a
drifting tripod, a specimen on a moving stage — that global motion contaminates
every identified displacement.

The ``Fiducial`` class handles this by tracking markers of known geometry
(ArUco and related types) in the frame, computing the frame-to-frame
transformation they imply, and inverting it. Either the frames themselves are
warped back into the reference frame's coordinate system, or the known
rigid-body motion is fed to
:ref:`DirectionalLucasKanade <rigid-body-motion>` as a prescribed motion.

.. note::

    ``Fiducial`` takes a **numpy array** of frames, not a ``VideoReader``:
    shape ``(n_frames, height, width)``, or ``(n_frames, height, width, 3)``
    for RGB, which is converted to grayscale automatically.

    .. code:: python

        frames = video.get_frames()
        fid = Fiducial(frames)

Workflow
--------

.. code:: python

    from pyidi import VideoReader, Fiducial

    video = VideoReader('measurement.cih')
    fid = Fiducial(video.get_frames())

    # 1. optional: make the markers easier to find
    fid.pre_process(clahe=True, apply_blur=True)

    # 2. detect the markers in every frame
    id_coords = fid.detect_markers(marker_type='aruco')

    # 3. the transformation from each frame to the reference frame
    transformations = fid.compute_transformations(id_coords, reference_index=0,
                                                  transform_type='euclidean')

    # 4a. warp the frames back into the reference coordinate system
    aligned = fid.revert_frames(transformations, transform_type='euclidean')

    # 4b. ...or just the marker coordinates, to check the quality of the fit
    transformed = fid.revert_fiducial(id_coords, transformations)
    stats = fid.uncertainty_analysis(id_coords, transformed, plot=True)

The aligned frames can be handed straight back to a
:class:`~pyidi.video_reader.VideoReader` as an array, and the identification
run on them as usual.

Detection
---------

``detect_markers(video=None, marker_type='aruco', fiducial_dictionary=None,
known_ids=None)`` supports ``"aruco"``, ``"apriltag"``, ``"charuco"`` and
``"artoolkit"``. ``known_ids`` restricts detection to specific marker IDs,
which is worth setting when other marker-like patterns are in the field of
view.

If detection is unreliable, ``pre_process()`` offers clipping and
normalisation, global histogram equalisation, CLAHE (adaptive histogram
equalisation), Gaussian blur, adaptive thresholding, and morphological
opening/closing. Detection quality usually improves more from fixing the
lighting than from any of these.

Transformation types
--------------------

``compute_transformations`` and ``revert_frames`` take the same
``transform_type``:

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Type
     - Degrees of freedom
     - Use when
   * - ``'euclidean'``
     - 3
     - The camera-structure relation is a rigid translation plus rotation.
       This is the default and the right choice for most vibration setups.
   * - ``'affine'``
     - 6
     - Scale and shear are also present, for example if the object distance
       changes.
   * - ``'homography'``
     - 8
     - Full projective mapping, for out-of-plane camera motion looking at a
       planar target.

Prefer the least flexible transformation that fits. A more flexible model
absorbs real structural motion into the "compensation" and quietly removes the
thing you are measuring.

If a transformation cannot be computed for a frame — too few common markers —
its entry is ``None``, and that frame passes through unchanged.

Checking the compensation
-------------------------

``uncertainty_analysis(id_coords, transformed_fiducial, plot=True)`` returns
per-frame and overall Euclidean error statistics between the transformed and
reference marker positions. The residual is the floor on what the compensation
can deliver: displacements smaller than it are not trustworthy after
compensation.

Worked example
--------------

This `showcase notebook
<https://github.com/ladisk/pyidi/blob/master/examples/Showcase_fiducial.ipynb>`_
demonstrates the full workflow. The example dataset originates from
infrared-spectrum measurements; to meet GitHub storage limits the original data
was undersampled to a 25-frame video. The package works equally on
visible-range acquisitions.

Contact
-------

For further details, please contact Dr. Janko Slavič
(`janko.slavic@fs.uni-lj.si <mailto:janko.slavic@fs.uni-lj.si>`_) or
Dr. Lorenzo Capponi (`lorenzo.capponi@fs.uni-lj.si
<mailto:lorenzo.capponi@fs.uni-lj.si>`_).

Acknowledgments
---------------

This work was conducted as part of the **ARTEMIDE** project, funded by the
European Research Agency (ERA) under the Marie Skłodowska-Curie Actions (MSCA),
Grant Agreement No. **101180595**.
