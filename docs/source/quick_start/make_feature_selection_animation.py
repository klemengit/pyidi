"""Generate ``feature_selection.gif`` for the point-selection docs page.

Produces an animated GIF of ``SelectionGUI`` (``pyidi/GUIs/feature_selection.py``)
working on the ``data/data_synthetic.cih`` demo video, in the order the interface
is meant to be used:

1. the window as it opens -- already scored, with points over the whole frame,
   because the ``Whole image`` mask row is seeded on startup;
2. the ``Mask`` tab, with a polygon drawn corner by corner; the points outside it
   drop to the dim tier as soon as the polygon closes and becomes a real mask;
3. back on ``Evaluate + select``, the separation swept down and up, which is the
   control that decides how many points there are.

That order is the pitch: the score comes first and the region trims it, rather
than a grid being placed and then filtered. ``make_selection_animation.py``
does the same job for the deprecated ``SelectionGUIOld``.

Why the headless setup is needed
--------------------------------
``SelectionGUI`` is a full Qt application whose constructor calls ``show()`` and
then enters the event loop, ending in ``sys.exit(app.exec())`` unless ``sys.ps1``
is set. To build the window, drive it and grab pixels from it in a plain script:

* ``QT_QPA_PLATFORM=offscreen`` must be set *before* Qt is imported, so Qt
  renders into its software framebuffer instead of opening a display;
* ``sys.ps1`` is set before construction, so the constructor takes the
  interactive branch rather than ``sys.exit(...)``;
* ``QtWidgets.QApplication.exec`` is monkeypatched to a no-op, because the
  interactive branch still calls ``app.exec()``, which would block with nothing
  driving it.

The window is then driven through the same calls a real click makes -- see
``on_mouse_click``: ``add_vertex``, ``_retire_whole_image``, ``refresh``.

Run with:

    QT_QPA_PLATFORM=offscreen python docs/source/quick_start/make_feature_selection_animation.py

This (re)writes ``docs/source/quick_start/feature_selection.gif`` in place.
"""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import imageio.v3 as iio  # noqa: E402
import numpy as np  # noqa: E402
from PyQt6 import QtGui, QtWidgets  # noqa: E402

from pyidi.GUIs.feature_selection import STEP_FIND, STEP_MASK, SelectionGUI  # noqa: E402
from pyidi.video_reader import VideoReader  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(HERE, "..", "..", "..", "data", "data_synthetic.cih")
OUT_PATH = os.path.join(HERE, "feature_selection.gif")

#: Polygon corners as ``(row, col)`` -- the convention this GUI uses throughout,
#: unlike ``SelectionGUIOld``, which stores ``(x, y)``. Chosen to sit inside the
#: 128x256 demo frame with margin on all sides.
POLYGON_VERTICES = [
    (100, 40),
    (35, 30),
    (15, 110),
    (40, 215),
    (105, 200),
]

#: Separation values swept on the ``Evaluate + select`` tab, to show that this
#: is the density control. Starts at the default so the first step is visible.
SEPARATIONS = [11, 7, 5, 8, 14]

#: Per-frame display time (ms). The encoder deduplicates identical consecutive
#: frames, so the hold at each end is a long duration rather than repeated frames.
STEP_DURATION_MS = 550
HOLD_DURATION_MS = 1800

#: Frames are rendered large and downscaled, which is cheaper than a small
#: window and keeps the text legible.
OUTPUT_WIDTH = 700


def grab_frame(window):
    """Process pending Qt events and grab the window as an (H, W, 4) uint8 array."""
    QtWidgets.QApplication.processEvents()
    QtWidgets.QApplication.processEvents()
    image = window.grab().toImage().convertToFormat(QtGui.QImage.Format.Format_RGBA8888)
    width, height = image.width(), image.height()
    ptr = image.bits()
    ptr.setsize(height * width * 4)
    return np.frombuffer(ptr, dtype=np.uint8).reshape((height, width, 4)).copy()


def downscale(frame, target_width):
    """Nearest-neighbour downscale of an (H, W, C) array to ``target_width`` columns."""
    h, w = frame.shape[:2]
    if w <= target_width:
        return frame
    scale = target_width / w
    target_height = max(1, int(round(h * scale)))
    col_idx = (np.arange(target_width) / scale).astype(int).clip(0, w - 1)
    row_idx = (np.arange(target_height) / scale).astype(int).clip(0, h - 1)
    return frame[row_idx][:, col_idx]


def main():
    sys.ps1 = ">>> "  # Make SelectionGUI think it's running interactively.
    QtWidgets.QApplication.exec = lambda self=None: 0  # Neutralise the blocking event loop.

    video = VideoReader(DATA_PATH)
    window = SelectionGUI(video, subset_size=15)
    # Fixed rather than merely resized: the status bar and the select-tab note
    # change length as the window is driven, and a resize between two grabs
    # would give the encoder frames of different shapes.
    window.setFixedSize(1250, 820)

    frames = []

    # 1. As opened: the whole frame is masked, so there are points to look at
    # before anything has been drawn.
    window.select_step(STEP_FIND)
    window.refresh()
    frames.append(grab_frame(window))

    # 2. The Mask tab, then the polygon corner by corner. This is exactly what
    # on_mouse_click does for a left click with the polygon tool active.
    window.select_step(STEP_MASK)
    window.select_tool('polygon')
    frames.append(grab_frame(window))

    for vertex in POLYGON_VERTICES:
        window.add_vertex(vertex)
        window._retire_whole_image()
        window.refresh()
        frames.append(grab_frame(window))

    # 3. Back to Evaluate + select, sweeping the separation.
    window.select_step(STEP_FIND)
    window.refresh()
    frames.append(grab_frame(window))

    for separation in SEPARATIONS:
        window.separation_spin.setValue(separation)
        window.refresh()
        print(f"separation {separation:>3} px -> {len(window.get_points()):>4} points")
        frames.append(grab_frame(window))

    # Belt and braces: the offscreen platform plugin does not implement
    # propagateSizeHints(), so trim to the common extent before encoding.
    height = min(f.shape[0] for f in frames)
    width = min(f.shape[1] for f in frames)
    frames = [downscale(f[:height, :width, :3], OUTPUT_WIDTH) for f in frames]
    durations = ([HOLD_DURATION_MS]
                 + [STEP_DURATION_MS] * (len(frames) - 2)
                 + [HOLD_DURATION_MS])
    iio.imwrite(OUT_PATH, frames, duration=durations, loop=0)
    print(f"Wrote {OUT_PATH} ({len(frames)} frames)")


if __name__ == "__main__":
    main()
