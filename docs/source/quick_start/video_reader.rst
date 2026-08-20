.. _video-reader:

Reading a video
===============

:class:`~pyidi.video_reader.VideoReader` is the single entry point for every
supported recording format. It hides the difference between a Photron header
plus raw file, a proprietary camera container, a folder of TIFFs and an MP4 —
downstream, every identification method sees the same interface.

.. code:: python

    from pyidi import VideoReader

    video = VideoReader('measurement.cih')

Supported formats
-----------------

.. list-table::
   :header-rows: 1
   :widths: 24 26 50

   * - Source
     - Extensions
     - Notes
   * - Photron
     - ``.cih``, ``.cihx``
     - Point at the header file; the ``.mraw`` next to it is memory-mapped.
       Frame rate and bit depth are read from the header.
   * - Phantom
     - ``.cine``
     - Read through the ``cine-handler`` package, which is installed with
       pyIDI. Frame rate comes from the file's setup block.
   * - Pharsighted
     - ``.SLOW``
     - Read by the bundled ``slow_reader``.
   * - Image sequence
     - ``.png``, ``.tif``, ``.tiff``, ``.bmp``, ``.jpg``, ``.jpeg``, ``.gif``
     - See :ref:`image-sequences` below.
   * - Video file
     - ``.avi``, ``.mkv``, ``.mp4``, ``.mov``, ``.m4v``, ``.wmv``, ``.webm``,
       ``.flv``, ``.ogg``, ``.ogv``
     - Decoded with PyAV. Currently 8-bit only.
   * - In memory
     - :class:`numpy.ndarray`
     - Shape ``(n_time_points, image_height, image_width)``. A ``root``
       directory must be given, because that is where results get written.

.. code:: python

    # Photron
    video = VideoReader('data/data_synthetic.cih')

    # Phantom
    video = VideoReader('data/data_small_cine.cine')

    # an image sequence: point at any image in the folder
    video = VideoReader('frames/im_0000.png')

    # a numpy array already in memory
    video = VideoReader(array, root='analysis_output')

.. _image-sequences:

Image sequences
^^^^^^^^^^^^^^^

Give the path to *any* image in the sequence. All images must be in the same
directory and named so that a plain sort puts them in the right order — pad
the numbers: ``im_0000.png ... im_9999.png``, not ``im_0.png ... im_9999.png``.
Multi-image containers (``.gif``, multi-page ``.tif``) are read as a sequence
from the single file.

Frame rate
----------

The frame rate is what turns a displacement history into a frequency, so it is
worth checking rather than assuming.

pyIDI reads it from the file where the format carries it (Photron headers,
``.cine`` setup blocks, and video-container metadata). Where it does not — a
numpy array, most image sequences, and any container whose metadata is missing
or wrong — set it yourself:

.. code:: python

    video = VideoReader('frames/im_0000.png', fps=10000)
    # or later
    video.configure(fps=10000)

.. warning::

    Ordinary video containers frequently report a *playback* rate rather than
    the capture rate — a 6000 fps recording exported to MP4 will often claim
    30 fps. If you did not record the file yourself, verify the rate against
    the acquisition settings.

The value is available as ``video.fps``, and format-specific metadata is in
``video.info``.

Colour and bit depth
--------------------

Frames come back as a 2-D :class:`numpy.ndarray` of shape
``(image_height, image_width)``, ``uint8`` or ``uint16`` depending on the
source. Colour material is converted to grayscale (the luma channel) by
default. To use a single channel or your own weights instead:

.. code:: python

    # one channel
    video.configure(video_format='rgb24', channel='R')

    # custom weights
    video.configure(video_format='rgb24', channel_weights=[0.299, 0.587, 0.114])

``channel`` and ``channel_weights`` only take effect if ``video_format`` is set
to an RGB format matching the source bit depth (``rgb24``, ``rgb48le``,
``rgb48be``); the default formats (``gray``, ``gray16be``, ``gray16le``)
already deliver a monochrome frame.

Reading frames
--------------

.. code:: python

    frame = video.get_frame(0)              # a single frame, (height, width)

    frames = video.get_frames()             # every frame, (n, height, width)
    frames = video.get_frames(500)          # frames 0..500
    frames = video.get_frames((100, 400))   # frames 100..400

Useful attributes:

.. list-table::
   :widths: 30 70

   * - ``video.N``
     - number of frames
   * - ``video.image_width``, ``video.image_height``
     - frame size in pixels
   * - ``video.fps``
     - frame rate in Hz
   * - ``video.info``
     - metadata dictionary from the source file
   * - ``video.root``
     - directory where the analysis results will be written

Call ``video.close()`` when you are done with a ``.cine`` or memory-mapped
source, or let the object go out of scope.

Viewing the recording
---------------------

.. code:: python

    video.gui()

opens the napari viewer on the recording (requires the ``[qt]`` extra — see
:doc:`napari`).

Next
----

* :doc:`points_selection` — choosing where to track.
* :doc:`disp_id_methods` — choosing how to track.
