.. _feature-selection:

Automatic feature selection
===========================

``FeatureSelectionGUI`` finds the points for you. Instead of placing subsets on
a grid and then discarding the poor ones, it scores *every* pixel of the image
and picks the best-separated maxima inside the region you drew. On a random
speckle pattern or an intricate structure that is the difference between
sampling where the features happen to be and sampling where the grid happens to
fall.

It is a separate interface from :doc:`points_selection`, not a replacement.
``SelectionGUI`` is unchanged and still the right tool when you want to place
subsets yourself.

Both need the ``[qt]`` extra:

.. code:: bash

    pip install pyidi[qt]

The three steps
---------------

The window has three steps, and they are the pipeline:

1. **Mask** — draw regions. A region says *where* points may go, not where they
   are. Nothing is placed yet.
2. **Evaluate** — score every subset position in the image at once. This is the
   only expensive step.
3. **Select** — turn the score and the mask into points, with a threshold and a
   minimum distance between them.

Only step 2 depends on the frame, the evaluator and the subset size. Everything
else re-derives the points from the cached score image, so editing a mask or
dragging a threshold updates while you are still moving the control. Changing
the subset size or the evaluator is the only thing that recomputes.

.. code:: python

    from pyidi import VideoReader, FeatureSelectionGUI

    video = VideoReader(input_file)
    gui = FeatureSelectionGUI(video, subset_size=11)

    points = gui.points          # or: gui.get_points()

The window is modal: the call blocks until you close it. ``points`` is an
``(n_points, 2)`` integer array in **row/column** order, ready for
``set_points``.

Mask
----

Five tools, on the right:

- **Polygon** — click to place corners; the enclosed area becomes a mask.
- **Brush** — hold Ctrl and drag to paint an area. ``Deselect painted area``
  switches the brush to erasing, which subtracts only the part you paint over.
- **Line** — click to place vertices; points are spaced along the segments.
- **Points** — click to place individual points.
- **Remove point** — click near a point to remove it. A hand-placed point is
  simply deleted, so clicking that pixel again puts it back. A *selected* point
  is not stored anywhere — it is re-derived from the score every time the
  selection runs — so removing one takes the ground it stands on out of the
  mask: the disc it was reserving, its minimum distance, so that nothing lands
  in its place.

Every region becomes a row in the ``Selections`` list, and **each row has a
role**:

``mask``
    the row contributes its area, and the points inside it are chosen by the
    selection step;
``points``
    the row contributes its coordinates directly, bypassing scoring entirely.

Polygons and brush strokes start as ``mask``; lines and clicked points start as
``points``. ``Use as points`` / ``Use as mask`` switches a row over without
redrawing it. That is how a filtered region and a hand-placed line of points
coexist in one session: hand-picked points always survive, whatever their score,
and no automatic point is placed within the minimum distance of one.

Unchecking a row excludes it without deleting it. Ctrl+Z undoes adding a vertex,
moving a vertex, painting a stroke, deleting a row, removing a point, and a
deselection.

Evaluate
--------

Two evaluators are built in:

- **Shi-Tomasi** — corner strength: high where the subset is constrained in
  both directions, so a subset on a plain edge scores low (it can slide along
  the edge) and one on a corner scores high.
- **Gradient in direction** — gradient strength along one chosen direction, for
  when only one component of the motion matters. The direction is a
  ``(row, col)`` pair, with ``X`` and ``Y`` preset buttons.

``Show score overlay`` draws the score as a heatmap, so you can see where the
features are before committing to any points. The border where the subset window
would leave the image is drawn fully transparent — it is not scored, rather than
scored badly.

The scoring window follows the subset size, so the score always answers the
question "how well would *this* subset track". Scores are cached per evaluator,
per parameter set and per subset size, so switching between two evaluators and
back costs nothing the second time.

Select
------

- **Threshold** — as a percentile of the scores (the default) or as a fraction
  of the maximum. Percentile is the sane default on a dense score image: the
  maximum over a megapixel is far more extreme than over a few hundred grid
  subsets, so one specular highlight would compress every useful
  fraction-of-maximum setting into the bottom of the slider.
- **Minimum distance** — no two points end up closer than this. Without it a
  threshold on a dense score image returns a solid blob of adjacent pixels
  around every strong corner.
- **Maximum points** — a cap, keeping the highest-scoring points.

The ``lattice`` selector places points on a regular grid of a given pitch
instead of at local maxima, optionally dropping cells that score too low. Use it
when you want even coverage rather than the best features — full-field work,
typically. It is a choice of selector, not a different mode.

Without the GUI
---------------

The pipeline is a plain module and imports without Qt, so the same selection can
be scripted:

.. code:: python

    from pyidi.selection import Entry, select_points

    region = Entry('polygon', [(20, 20), (20, 200), (180, 200), (180, 20)])
    points = select_points(image, [region], subset_size=11,
                           evaluator='shi_tomasi', min_distance=15)

``Entry`` geometry is in ``(row, col)``, as is the returned array. For repeated
work on one frame, ``SelectionPipeline`` keeps the score cache alive across
parameter changes:

.. code:: python

    from pyidi.selection import SelectionPipeline

    pipeline = SelectionPipeline(image, subset_size=11)
    pipeline.add_entry('polygon', [(20, 20), (20, 200), (180, 200), (180, 20)])

    for threshold in (80, 90, 95):
        pipeline.selector_params['threshold'] = threshold
        print(threshold, len(pipeline.points))     # scored once, not three times

Scoring can be extended without touching any GUI code — register a function that
turns an image and a window into a score array, together with descriptors for
its parameters, and it appears in the evaluator menu:

.. code:: python

    from pyidi.selection import Evaluator, Parameter, register_evaluator

    register_evaluator(Evaluator(
        name='variance',
        display_name='Local variance',
        function=my_variance_score,      # f(image, window) -> ndarray
        parameters=(),
        description='Contrast inside the subset.',
    ))
