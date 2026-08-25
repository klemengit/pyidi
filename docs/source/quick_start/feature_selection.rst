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

Find, then trim
---------------

Three things happen, and the window shows them as two tabs:

**Evaluate + select** — *evaluate* scores every subset position in the image at
once, and *select* turns that score into points with a threshold, a separation
and a cap. The two are tuned against each other, since changing the evaluator
changes what a threshold means, so they share one panel.

**Mask** — regions say where points are allowed. The window opens with a
``Whole image`` row already in the selections list, so there are candidates over
the whole frame from the start and this tab is where you trim them: paint away
the clamp, drop the background, keep the part you care about.

The selections list sits on the **Mask** tab and only there: every row in it,
and every button under it, acts on a region drawn there. The subset size, by
contrast, sits below the tabs rather than on either of them, because both
steps read it: the scoring window follows it, which is what makes it one of the
few settings that stales the score, and it is also the size of the rectangle
drawn round each point while you mask. It takes **odd values only** — a subset
is centred on its point, so an even extent has no centre to be, and the
pipeline reads one as the odd size below it anyway. ``Show score overlay`` is on
both tabs, and the two controls stay in step.

The tabs are deliberately unnumbered and can be used in any order. Evaluation
does not depend on the mask — the score is always computed for the whole frame
and cached — so masking and scoring are not a sequence. Only the frame, the
evaluator, its parameters and the subset size feed the score; everything else
re-derives the points from the cached array, which is why editing a mask or
dragging a threshold updates while you are still moving the control.

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

The ``Whole image`` row that the window starts with is an ordinary row: uncheck
it, paint it away with the deselect brush, or delete it. Deleting it selects
nothing — it does not silently revert to the whole frame.

Mask rows combine as a **union**, so drawing a region while the whole frame is
still selected would change nothing at all. The first drawn region therefore
unchecks the ``Whole image`` row, with a note in the status bar; tick it again
to bring the whole frame back, or press Ctrl+Z.

``Clear all`` starts over, which means the state the window opens in: every
selection dropped and the ``Whole image`` row seeded again. It is undoable.

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
  mask: the disc it was reserving, its separation, so that nothing lands in its
  place.

Points under a deselect stroke are crossed out while you paint, so you can see
what the stroke is about to take before you let go.

While masking, the points come in three tiers, because an empty patch otherwise
means two different things — nothing to track there, or something you have
masked away:

- **red** — a point the selection is taking;
- **dim blue** — a feature the mask is leaving out;
- **ringed in magenta** — the points the row selected in the list accounts for.

The dim tier is what the current settings would select over the whole frame, so
it does not move while you edit a mask: painting a region turns points from blue
to red where it lands rather than re-selecting underneath you. One consequence
is worth knowing — a blue point right at the edge of a mask need not coincide
exactly with a red one, because a selection inside a region starts its
separation afresh.

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
and no automatic point is placed within the separation of one.

Unchecking a row excludes it without deleting it. Ctrl+Z undoes adding a vertex,
moving a vertex, painting a stroke, deleting a row, removing a point, and a
deselection.

Evaluate
--------

Two evaluators are built in, chosen at the top of the **Evaluate + select**
tab:

- **Shi-Tomasi** — corner strength: high where the subset is constrained in
  both directions, so a subset on a plain edge scores low (it can slide along
  the edge) and one on a corner scores high.
- **Gradient in direction** — gradient strength along one chosen direction, for
  when only one component of the motion matters. The direction is a
  ``(row, col)`` pair, with ``X`` and ``Y`` presets and a ``Draw`` button that
  lets you drag the direction out on the image. Whichever way you set it, a red
  line shows the direction currently in force.

Each evaluator describes itself in the tooltip of the ``Score`` menu.
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

Below the evaluator, on the same tab:

- **Threshold** — by default a **quality**: a fraction of the best feature in
  the region, so ``0.01`` means "at least a hundredth as good as the best one
  here". The slider is logarithmic, because the useful settings span three
  decades — featureless background scores around ``0.001`` of the best feature
  and a strong corner around ``1``.
- **Separation** — the distance no two points may come closer than, and so the
  control for how many you get: lower it for more. This is the one knob that
  decides density.
- **Maximum points** — a safety valve. When it stops the selection the panel
  says so, because a cap has no other symptom: it simply stops adding points,
  and the result reads as though the threshold or the separation did it.
- **Keep every n-th** — decimation. It thins the points that were *already*
  selected, leaving the survivors exactly where they are. Use it when the
  selection is right and only the count is too high for the computation you are
  about to run. Widening the separation instead re-selects and moves every
  point, which is a different thing; hand-placed points are never decimated.

Why decimation is not the density control
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The obvious way to thin a dense selection is to keep every n-th of the pixels
above the threshold, and it does not work. Measured on a 1024×1024 frame with
357 000 pixels above the threshold, thinned to twenty thousand points:

===============================  ==========  ==================
rule                             median gap  pairs under 3 px
===============================  ==========  ==================
every n-th, best first           2.0 px      78 %
every n-th, in scan order        1.0 px      92 %
separation *n*                   ≥ *n* px    0 %
===============================  ==========  ==================

Keeping every n-th by score fails because consecutive ranks are neighbours on
the same feature. Keeping every n-th in scan order fails because the stride
aliases against the row length and lands in columns. Either way most of the
subsets end up on top of another one, which is what the selection step exists to
prevent — so decimation stays what it is good at, thinning a selection that is
already well spread.

The separation is enforced by a greedy walk from best to worst, accepting a
candidate only if nothing already accepted is within the separation of it. That
walk is exact but linear in the *candidates*, and a loose threshold leaves
hundreds of thousands of them — 40 ms to 300 ms, which no slider can drag. So
the candidates are reduced first, to the best pixel in each cell of a grid half
the separation across. What that approximation costs is yield: at a separation
of 11 it finds 1708 points where the exact walk finds 2193, in 9 ms instead of
39. What it does not cost is the guarantee — the walk still runs, so the
separation still holds exactly — and a point count is what the separation
control is for adjusting anyway.

Why quality and not a percentile
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A percentile ranks *pixels*, and on a dense score image the pixels are
overwhelmingly background. On a typical frame the 90th percentile of the score
is under a five-hundredth of the best feature, so nine tenths of a percentile
slider's travel is spent inside the featureless area: lowering it does not relax
the quality bar, it floods the frame with background. The separation then
spreads that background out evenly, which makes it look as though the spacing is
at fault.

Quality is measured against the best feature instead, so the whole slider stays
inside the range that actually distinguishes features. The reference is the
99.9th percentile of the scores rather than their literal maximum, so one dust
mote or specular highlight cannot drag every useful setting into the floor of
the slider.

``percentile of scores`` is still available in the ``Threshold`` menu. It is the
right rule for the ``lattice`` selector, where the candidates have already been
spaced out and there is no background flood to worry about.

A third rule, a fraction of the literal maximum, was offered and then dropped:
it is the same rule as quality with a reference that one bright pixel can move,
so on any frame worth using it is indistinguishable and on a bad one it is
worse.

The ``lattice`` selector places points on a regular grid of a given pitch
instead of at local maxima, optionally dropping cells that score too low. The
settings a selector has no use for are hidden rather than greyed out, so
switching to it swaps ``Separation`` for ``Grid pitch``. Use it
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
                           evaluator='shi_tomasi', separation=15, threshold=0.01)

``Entry`` geometry is in ``(row, col)``, as is the returned array. For repeated
work on one frame, ``SelectionPipeline`` keeps the score cache alive across
parameter changes:

.. code:: python

    from pyidi.selection import SelectionPipeline

    pipeline = SelectionPipeline(image, subset_size=11)
    pipeline.add_entry('polygon', [(20, 20), (20, 200), (180, 200), (180, 20)])

    for threshold in (0.2, 0.05, 0.01):
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
