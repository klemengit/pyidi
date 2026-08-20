.. _point-selection:

Point selection UI
==================

A convenient UI (``SelectionGUI``) is available to make the point selection easier.
It is a PyQt6-based tool, so the ``[qt]`` extra must be installed first:

.. code:: bash

    pip install pyidi[qt]

Without this extra, ``SelectionGUI`` can still be imported, but instantiating
it raises a ``RuntimeError``.

To use the UI, a ``VideoReader`` object must first be created (a plain
``numpy.ndarray`` image also works):

.. code:: python

    from pyidi import VideoReader, SimplifiedOpticalFlow, SelectionGUI

    video = VideoReader(input_file)

where ``input_file`` can be a Photron ``.cih``/``.cihx`` path, an image, a
video file, a numpy array, or a ``.SLOW`` file.

A ``SelectionGUI`` window can then be opened:

.. code:: python

    gui = SelectionGUI(video, subset_size=11, subset_overlap=0)

where ``subset_size`` is the side length (in pixels) of the
Region-Of-Interest/subset drawn around each point, and ``subset_overlap``
sets the spacing between neighbouring subsets (the step between subset
centers is ``subset_size + subset_overlap``, so a positive value spreads the
subsets further apart and a negative value overlaps them).

``subset_size`` can be a single int for a square subset, or a ``(height,
width)`` pair for an anisotropic one -- the same ``(vertical, horizontal)``
convention as ``LucasKanade.configure(roi_size=(vertical, horizontal))``. In
the UI, the ``Subset Configuration`` group has a ``Square subsets`` checkbox
(checked by default) alongside the height/width spinboxes and sliders: while
checked, the width tracks the height and only one size can be set; unchecking
it frees the width spinbox/slider to be set independently.

Selection mode
--------------

The window opens in **Select** mode. Five selection methods are available as
buttons on the right:

- **Grid**: click to place the corners of a polygon; once at least three
  corners are placed, a regular grid of subsets is generated inside the
  polygon. Multiple grids can be created (``Start new line``) and managed
  individually in the list, and deleted with ``Delete selected grid``.
- **Manual**: click on the image to add individual points one at a time.
- **Along the line**: click to place points defining a polyline; subsets are
  placed at regular intervals along its segments. As with ``Grid``, multiple
  lines can be drawn and managed/deleted individually.
- **Brush**: hold Ctrl and drag over the image to paint a region; subsets are
  placed on a regular grid inside the painted area. The brush radius is set
  with a slider, and the ``Deselect painted area`` toggle switches the brush
  to remove already-selected subsets instead of adding new ones.
- **Remove point**: click near an existing point to remove it.

The ``Subset Configuration`` group lets you adjust the subset size, toggle
the subset rectangle overlay (``Show subsets``), and clear the current
selection (``Clear selections``). For ``Grid``, ``Along the line``, and
``Brush``, a ``Distance between subsets`` control sets the spacing described
above.

Filter mode
-----------

Switching to **Filter** mode (top toolbar) applies automatic filtering on top
of the subsets placed in Select mode, to keep only the ones on
strongly-textured image content. Two filter methods are available:

- **Shi-Tomasi**: ranks each subset by the corner strength of the image
  content inside it (the smaller eigenvalue of the local gradient structure
  tensor, in the style of the Shi-Tomasi corner criterion). A threshold
  slider keeps only the subsets above a fraction of the strongest one.
- **Gradient in direction**: ranks each subset by the strength of the image
  gradient projected onto a chosen direction. The direction is set either by
  clicking ``Set direction on image`` and dragging across the image, or with
  the ``X Direction``/``Y Direction`` preset buttons. A threshold slider
  then keeps only the subsets with a strong-enough gradient in that
  direction.

Filtered (candidate) points are shown in green; ``Clear candidates`` resets
the filter back to the full selection from Select mode.

Retrieving the points
----------------------

Once the selection is complete, the points can be retrieved through the
``.points`` property or the ``.get_points()`` method. If a filter has been
applied in Filter mode, the filtered (candidate) points are returned;
otherwise, the points from Select mode are returned.

.. code:: python

    points = gui.points          # or: gui.get_points()

The returned array has shape ``(n_points, 2)``, with points given in
**row/column** (``y``/``x``) image coordinates: ``points[:, 0]`` is the row
(``y``) coordinate and ``points[:, 1]`` is the column (``x``) coordinate.

The points can be passed directly to a method object, either as the GUI
instance itself (``set_points`` duck-types on a ``.points`` attribute) or as
the extracted array:

.. code:: python

    sof = SimplifiedOpticalFlow(video)

    sof.set_points(gui)          # or: sof.set_points(points)

.. image:: selection.gif
    :alt: Animated demo of the SelectionGUI Grid method: a polygon is drawn
        vertex by vertex over the video frame, filling in with the subset
        grid it encloses.
