"""Generate ``selection.gif`` for the points-selection docs page.

Produces an animated GIF of the PyQt6-based ``SelectionGUIOld``
(``pyidi/GUIs/subset_selection.py``) building up a Grid selection on the
``data/data_synthetic.cih`` demo video: an empty frame, the polygon
vertices of the selection region appearing one at a time, the subset
grid filling in as soon as the polygon closes, and the finished
selection held for a few extra frames.

Why the headless setup is needed
---------------------------------
``SelectionGUIOld`` is a full Qt application (``QtWidgets.QMainWindow``)
built for interactive use: its constructor calls ``self.show()`` and
then starts the Qt event loop, ending with
``sys.exit(app.exec())`` whenever ``sys.ps1`` is not set (i.e. a plain
script run, as opposed to an interactive interpreter). That is fine
when a person is using the GUI, but fatal for a script that wants to
construct the window, poke at its state, and grab pixels from it:

* ``QT_QPA_PLATFORM=offscreen`` must be set *before* Qt is imported, so
  Qt renders to its software framebuffer instead of trying to open a
  real display.
* ``sys.ps1`` is set before constructing ``SelectionGUIOld``, so its
  constructor takes the "interactive" branch instead of
  ``sys.exit(...)``.
* ``sys.ps1`` alone is not enough: the "interactive" branch still calls
  ``app.exec()``, which blocks in the Qt event loop with nothing driving
  it. ``QtWidgets.QApplication.exec`` is therefore also monkeypatched to
  a no-op returning 0, so construction returns immediately with a fully
  built, shown window that this script can then drive by hand (appending
  vertices directly to a selection entry's ``geometry`` and calling the
  GUI's own display/recompute methods, the same calls ``handle_grid_drawing``
  makes on a real click).

Run with:

    QT_QPA_PLATFORM=offscreen python docs/source/quick_start/make_selection_animation.py

This (re)writes ``docs/source/quick_start/selection.gif`` in place.
"""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import imageio.v3 as iio  # noqa: E402
import numpy as np  # noqa: E402
from PyQt6 import QtGui, QtWidgets  # noqa: E402

from pyidi.GUIs.subset_selection import SelectionGUIOld  # noqa: E402
from pyidi.video_reader import VideoReader  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(HERE, "..", "..", "..", "data", "data_synthetic.cih")
OUT_PATH = os.path.join(HERE, "selection.gif")

#: Vertices (x, y) of the polygon used for the Grid selection, chosen to
#: sit comfortably inside the 256x128 demo frame with margin on all sides.
POLYGON_VERTICES = [
    (40, 100),
    (30, 35),
    (110, 15),
    (215, 40),
    (200, 105),
]

#: Per-frame display time (milliseconds). The GIF encoder deduplicates
#: identical consecutive frames, so the "hold" at the end is done by
#: giving the final frame a long duration rather than by repeating it.
STEP_DURATION_MS = 500
HOLD_DURATION_MS = 2500

#: Output size (window is built larger for a crisper render, then frames
#: are downscaled to this width to keep the GIF file size small).
OUTPUT_WIDTH = 700


def grab_frame(window):
    """Process pending Qt events and grab the window as an (H, W, 4) uint8 array."""
    QtWidgets.QApplication.processEvents()
    QtWidgets.QApplication.processEvents()
    pixmap = window.grab()
    image = pixmap.toImage().convertToFormat(QtGui.QImage.Format.Format_RGBA8888)
    width, height = image.width(), image.height()
    ptr = image.bits()
    ptr.setsize(height * width * 4)
    arr = np.frombuffer(ptr, dtype=np.uint8).reshape((height, width, 4)).copy()
    return arr


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
    sys.ps1 = ">>> "  # Make SelectionGUIOld think it's running interactively.
    QtWidgets.QApplication.exec = lambda self=None: 0  # Neutralise the blocking event loop.

    video = VideoReader(DATA_PATH)
    window = SelectionGUIOld(video, subset_size=15, subset_overlap=3)
    window.resize(1200, 800)

    frames = []

    # 1. Empty frame: nothing selected yet.
    frames.append(grab_frame(window))

    # 2. Grid selection, matching what a real click does in handle_grid_drawing:
    # register the "Grid 1" list entry, then add vertices one at a time,
    # recomputing the ROI points (and therefore the filled grid) as soon as
    # the polygon has at least 3 vertices.
    grid = window.add_selection("grid")

    subset_size = window.subset_size_spinbox.value()
    spacing = window.distance_spinbox.value()

    for vertex in POLYGON_VERTICES:
        grid["geometry"].append(vertex)
        window.recompute_entry(grid, subset_size, spacing)
        window.update_geometry_display()
        window.update_selected_points()
        frames.append(grab_frame(window))

    print(f"Selected subsets in final frame: {len(window.selected_points)}")

    frames = [downscale(f[:, :, :3], OUTPUT_WIDTH) for f in frames]
    durations = [STEP_DURATION_MS] * (len(frames) - 1) + [HOLD_DURATION_MS]
    iio.imwrite(OUT_PATH, frames, duration=durations, loop=0)
    print(f"Wrote {OUT_PATH} ({len(frames)} frames)")


if __name__ == "__main__":
    main()
