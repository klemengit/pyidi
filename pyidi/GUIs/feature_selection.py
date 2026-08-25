"""Interactive front end for the mask -> evaluate -> select pipeline.

``FeatureSelectionGUI`` drives :mod:`pyidi.selection`, whose three steps --
mask, evaluate, select, in the vocabulary settled in issue #51 -- are presented
as *two* tabs.

That is not a simplification of the pipeline but a truer picture of it.
Evaluation does not depend on the mask at all: the score store computes the
whole frame and deliberately never crops to a region, because a cropped score
would have to be discarded the moment the region grew. Mask and evaluate are
therefore siblings feeding select, not a sequence, and numbering them 1-2-3
would imply an order the code does not have.

So the interface leads with **Evaluate + select** -- named for the two steps
it holds, and holding both because changing the evaluator changes what
threshold makes sense, so the two are tuned against each other -- and follows
with **Mask**, which is where the candidates get trimmed. The selections list starts with a "Whole image" row so
there is something to trim from the moment the window opens.

Only evaluation is expensive, and it depends on nothing but the frame, the
evaluator and the subset size. Everything else -- painting a mask, dragging a
polygon vertex, moving the threshold slider, changing the separation --
re-derives the points from a cached score image, so it updates while the
control is still moving. Changing the subset size or the evaluator is the only
thing that pays for a recomputation.

Coordinate convention: ``(row, col)`` everywhere, matching the pipeline and
numpy. The image item is set to ``axisOrder='row-major'`` so pyqtgraph's view
coordinates are ``x = column, y = row``, and the only conversion in the whole
module is that swap at the mouse-event boundary. The older ``SelectionGUI``
instead transposes the frame and carries ``(x, y)`` internally, which is where
most of its axis-order bugs came from.

This is a separate interface, not a replacement: ``SelectionGUI`` is untouched
and keeps working.
"""

import sys
import time

import numpy as np
import pyqtgraph as pg
from PyQt6 import QtCore, QtGui, QtWidgets
from pyqtgraph import GraphicsLayoutWidget, ImageItem, ScatterPlotItem

from ..selection import (
    DEFAULT_MAX_POINTS,
    DEFAULT_THRESHOLD,
    SELECTORS,
    SelectionPipeline,
    available_evaluators,
)
from ..selection_geometry import _as_size_pair

#: Grab radius, in screen pixels, within which a drag hits an existing vertex.
#: Screen rather than image pixels so hit-testing feels the same at any zoom.
VERTEX_GRAB_RADIUS_PX = 10

#: The two tabs, in order, named for the pipeline steps each one holds. Also
#: the toolbar button labels and the keys of ``step_pages``. Deliberately
#: unnumbered: mask and evaluate do not depend on each other, so there is no
#: step 1. A ``+`` rather than an ``&`` because Qt reads an ampersand in a
#: button label as a mnemonic and swallows it.
STEP_FIND = 'Evaluate + select'
STEP_MASK = 'Mask'
STEPS = (STEP_FIND, STEP_MASK)

#: What the status bar says on arriving at each tab. The window opens with the
#: whole frame selected, so neither of them is an instruction to draw anything.
STEP_HINTS = {
    STEP_FIND: 'Score the frame, then turn the score into points. '
               'Lower the separation for more points, raise the threshold for better ones.',
    STEP_MASK: 'Red points are selected; grey ones are features the mask leaves out. '
               'The selected row\'s points are ringed.',
}

#: Label of the mask row seeded on startup, covering the whole frame.
WHOLE_IMAGE_LABEL = 'Whole image'

#: How long a redraw may take before the interface stops doing one per control
#: change and starts coalescing them. Roughly a frame: below this the display
#: can follow a dragged slider exactly, above it the requests have to be
#: collapsed or they queue up behind a redraw that is already too slow.
REDRAW_BUDGET_MS = 12.0

#: Region tools available in the mask step, as ``(button label, entry kind)``.
#: ``remove`` is not an entry kind -- it deletes points rather than making any.
TOOLS = (
    ('Polygon', 'polygon'),
    ('Brush', 'brush'),
    ('Line', 'polyline'),
    ('Points', 'points'),
    ('Remove point', 'remove'),
)

#: Kinds whose geometry is a list of draggable vertices.
VERTEX_KINDS = ('polygon', 'polyline')

#: Threshold rules offered, as ``(menu label, mode, slider decades or None)``.
#:
#: ``quality`` is logarithmic because the useful settings span three decades:
#: featureless background sits around 0.001 of the best feature and a strong
#: corner at 1, so a linear slider would spend most of its travel in a range
#: where nothing changes. `percentile` is linear over its natural range.
THRESHOLD_RULES = (
    ('quality of the best', 'quality', (1e-3, 1.0)),
    ('percentile of scores', 'percentile', None),
)

#: Where each rule's slider starts, and the value that position means.
THRESHOLD_DEFAULTS = {'quality': DEFAULT_THRESHOLD, 'percentile': 90.0}


def odd(value):
    """``value`` rounded up to an odd number.

    :param value: a subset extent, in pixels
    :type value: int
    :rtype: int
    """
    value = int(value)
    return value if value % 2 else value + 1


class OddSpinBox(QtWidgets.QSpinBox):
    """A spin box that holds odd numbers only, by stepping and by typing.

    A subset is centred on the pixel it belongs to, so an even extent has no
    centre to be. The pipeline already reads one as the odd size below it --
    a subset size of 10 scores through an 11-pixel window, and draws an
    11-pixel rectangle -- so the even values are a second spelling of the odd
    ones and offering them only invites the question of what they do.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setSingleStep(2)

    def validate(self, text, position):
        state, text, position = super().validate(text, position)
        if state == QtGui.QValidator.State.Acceptable and int(text) % 2 == 0:
            # Intermediate rather than Invalid, so that typing "12" on the way
            # to "121" is not rejected keystroke by keystroke. Anything still
            # even when the box loses focus goes through fixup().
            return QtGui.QValidator.State.Intermediate, text, position
        return state, text, position

    def fixup(self, text):
        try:
            return str(odd(text))
        except ValueError:
            return super().fixup(text)

    def setValue(self, value):
        """Set the value, rounded up to odd. Programmatic writes come here too."""
        super().setValue(odd(value))


#: Length of the segment a dot is drawn as, in image pixels. Only has to be
#: non-zero: Qt draws nothing for a degenerate subpath.
DOT_LENGTH = 1e-3


class DotCloud(QtWidgets.QGraphicsPathItem):
    """Uniform round dots, drawn as one stroked path.

    A ``ScatterPlotItem`` is the obvious way to draw these and the wrong one at
    this scale: it keeps a record per spot and rebuilds a symbol atlas, which is
    17 ms for seventeen thousand points and is paid on every redraw. A path of
    zero-length segments stroked with a round-cap pen draws the same dots -- the
    cap *is* the dot -- and is built by one vectorised call: 2 ms.

    The pen is cosmetic, so the dots keep their size in screen pixels at any
    zoom, which is what the scatter item did too.

    Only the layers that are genuinely uniform use this. Anything with a per-point
    colour, a symbol or a hover behaviour still wants the scatter item.
    """

    def __init__(self, colour, size):
        super().__init__()
        pen = pg.mkPen(*colour, width=size)
        pen.setCapStyle(QtCore.Qt.PenCapStyle.RoundCap)
        pen.setCosmetic(True)
        self.setPen(pen)
        self._pos = np.empty((0, 2))

    def setData(self, pos):
        """Draw a dot at each ``(x, y)`` row of ``pos``."""
        pos = np.asarray(pos, dtype=float)
        self._pos = pos
        if not len(pos):
            self.setPath(QtGui.QPainterPath())
            return
        # Each point twice, joined in pairs: one very short segment per point,
        # whose round cap is the dot. Short rather than zero-length, because Qt
        # drops a degenerate subpath and draws nothing at all; a thousandth of a
        # pixel is far below the width the cap draws at any zoom.
        x = np.repeat(pos[:, 0], 2)
        x[1::2] += DOT_LENGTH
        self.setPath(pg.arrayToQPath(x, np.repeat(pos[:, 1], 2), connect='pairs'))

    def clear(self):
        self.setData(np.empty((0, 2)))

    def getData(self):
        """``(x, y)``, matching :meth:`ScatterPlotItem.getData`."""
        return self._pos[:, 0], self._pos[:, 1]


class CanvasViewBox(pg.ViewBox):
    """The image view, with brush painting and vertex dragging layered on panning.

    :param parent_gui: the window this view belongs to
    :type parent_gui: FeatureSelectionGUI
    """

    def __init__(self, parent_gui, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setMouseMode(self.PanMode)
        self.parent_gui = parent_gui
        self._drag = None
        self._direction_start = None

    def _scene_to_rc(self, pos):
        """A scene position as ``(row, col)`` floats, or ``None`` if off-image."""
        if not self.sceneBoundingRect().contains(pos):
            return None
        point = self.mapSceneToView(pos)
        return point.y(), point.x()

    def _start_vertex_drag(self, ev):
        """Grab a vertex if the drag began near one; otherwise let the view pan."""
        gui = self.parent_gui
        if gui.step != STEP_MASK or gui.tool not in VERTEX_KINDS:
            return False
        position = self._scene_to_rc(ev.buttonDownScenePos())
        if position is None:
            return False
        entry, index = gui.vertex_at(position)
        if entry is None:
            return False
        self._drag = {'entry': entry, 'index': index, 'original': entry.geometry[index]}
        ev.accept()
        return True

    def _continue_vertex_drag(self, ev):
        """Move the grabbed vertex, committing undo and refreshing on release."""
        if self._drag is None:
            return False
        position = self._scene_to_rc(ev.scenePos())
        if position is not None:
            self._drag['entry'].geometry[self._drag['index']] = position
            self.parent_gui.draw_geometry()
        if ev.isFinish():
            self.parent_gui.push_undo({
                'type': 'vertex_move',
                'entry': self._drag['entry'],
                'index': self._drag['index'],
                'original': self._drag['original'],
            })
            self.parent_gui.refresh()
            self._drag = None
        ev.accept()
        return True

    def _handle_direction_drag(self, ev):
        """Drag out the gradient direction while the ``Draw`` button is armed.

        A direction is a thing you point at. Typing two components and checking
        the heatmap afterwards is a slower way of saying the same thing, so this
        reproduces the drag the older ``SelectionGUI`` offered.
        """
        gui = self.parent_gui
        if not gui.drawing_direction:
            return False
        ev.accept()
        if ev.isStart():
            self._direction_start = self._scene_to_rc(ev.buttonDownScenePos())
            return True
        position = self._scene_to_rc(ev.scenePos())
        if self._direction_start is None or position is None:
            return True
        gui.show_direction(self._direction_start, position)
        if ev.isFinish():
            gui.set_direction_from_drag(self._direction_start, position)
            self._direction_start = None
        return True

    @staticmethod
    def _ctrl(ev):
        """Whether Ctrl was down when this event was delivered.

        Read off the event rather than tracked by a key filter on the window: a
        panel widget with focus can swallow the key press, and a Ctrl released
        while the window is not focused is never seen at all, either of which
        leaves a tracked flag stuck at the wrong value.
        """
        return bool(ev.modifiers() & QtCore.Qt.KeyboardModifier.ControlModifier)

    def _handle_brush_drag(self, ev):
        """Paint while Ctrl is held and the brush tool is active.

        A stroke already under way keeps the drag whether Ctrl is still down or
        not, so letting go of the key mid-stroke finishes the stroke instead of
        abandoning it half-painted.
        """
        gui = self.parent_gui
        if not (gui.step == STEP_MASK and gui.tool == 'brush'
                and (self._ctrl(ev) or gui.painting)):
            return False
        ev.accept()
        if ev.isStart():
            gui.brush_start()
        elif ev.isFinish():
            gui.brush_move(self._scene_to_rc(ev.scenePos()))
            gui.brush_end()
        else:
            gui.brush_move(self._scene_to_rc(ev.scenePos()))
        return True

    def mouseClickEvent(self, ev):
        gui = self.parent_gui
        if gui.step == STEP_MASK and gui.tool == 'brush':
            if self._ctrl(ev):
                ev.accept()
                gui.brush_start()
                gui.brush_move(self._scene_to_rc(ev.scenePos()))
                gui.brush_end()
            else:
                ev.ignore()
            return
        super().mouseClickEvent(ev)

    def mouseDragEvent(self, ev, axis=None):
        if self._handle_direction_drag(ev):
            return
        if self._handle_brush_drag(ev):
            return
        if ev.isStart():
            if self._start_vertex_drag(ev):
                return
        elif self._continue_vertex_drag(ev):
            return
        super().mouseDragEvent(ev, axis)


class FeatureSelectionGUI(QtWidgets.QMainWindow):
    """Pick tracking points by masking, scoring and selecting.

    The window is modal: the constructor blocks until it is closed, and the
    points are then available through :attr:`points` or :meth:`get_points`.

    :param video: a ``VideoReader``, a 2-D ``(height, width)`` image, or a 3-D
        ``(n_frames, height, width)`` stack whose first frame is used
    :type video: VideoReader or numpy.ndarray
    :param subset_size: side length of the subset, as a scalar or a
        ``(height, width)`` pair. The scoring window follows it, so the score
        always answers the question "how well would *this* subset track".
    :type subset_size: int or tuple
    :param subset_overlap: extra spacing between the points a ``points``-role
        entry lays out; positive spreads them apart, negative overlaps them
    :type subset_overlap: int
    :raises TypeError: if ``video`` is none of the accepted types
    """

    def __init__(self, video, subset_size=11, subset_overlap=0):
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        super().__init__()
        self.setWindowTitle('Feature Selection')
        self.resize(1250, 820)

        self.frame = self._frame_from(video)
        # Rounded up to odd here rather than only in the spin box, so that the
        # pipeline and the control that shows it never disagree.
        height, width = _as_size_pair(subset_size)
        self.pipeline = SelectionPipeline(self.frame, (odd(height), odd(width)), subset_overlap)
        self.pipeline.define_score('score', 'shi_tomasi')

        self.step = STEP_FIND
        self.tool = 'polygon'
        self._whole_image = None
        self.deselect_mode = False
        self.drawing_direction = False
        self.direction_spins = []
        self.direction_button = None
        self.score_toggles = []
        self._paint = None
        self._stroke_path = None
        self._syncing = False
        self._last_refresh_ms = 0.0
        self._refresh_timer = QtCore.QTimer(self)
        self._refresh_timer.setSingleShot(True)
        self._refresh_timer.timeout.connect(self.refresh)
        self.active_index = None
        self.undo_stack = []
        self.undo_limit = 50

        QtGui.QShortcut(QtGui.QKeySequence.StandardKey.Undo, self).activated.connect(self.undo)

        self._build_ui()
        self.image_item.setImage(self.frame)
        self.add_whole_image_mask()
        self.select_step(STEP_FIND)
        self.refresh()

        self.show()
        if not hasattr(sys, 'ps1'):
            sys.exit(app.exec())
        else:
            app.exec()

    # -- construction ------------------------------------------------------

    @staticmethod
    def _frame_from(video):
        """The single 2-D frame to work on, whatever form the video came in."""
        from ..video_reader import VideoReader

        if isinstance(video, VideoReader):
            return video.get_frame(0)
        if isinstance(video, np.ndarray) and video.ndim == 3:
            return video[0]
        if isinstance(video, np.ndarray) and video.ndim == 2:
            return video
        raise TypeError(
            f'`video` must be a VideoReader, or a 2-D (height, width) or 3-D '
            f'(n_frames, height, width) np.ndarray, got {type(video).__name__!r}.'
        )

    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        layout.addWidget(self._build_step_toolbar())
        self.splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        layout.addWidget(self.splitter, stretch=1)

        self._build_canvas()
        self._build_panel()

        self.status = self.statusBar()

    def _build_step_toolbar(self):
        bar = QtWidgets.QWidget()
        row = QtWidgets.QHBoxLayout(bar)
        row.setContentsMargins(5, 4, 5, 4)
        self.step_buttons = {}
        group = QtWidgets.QButtonGroup(self)
        group.setExclusive(True)
        for name in STEPS:
            button = QtWidgets.QPushButton(name)
            button.setCheckable(True)
            button.setMinimumWidth(120)
            group.addButton(button)
            row.addWidget(button)
            button.clicked.connect(lambda _, n=name: self.select_step(n))
            self.step_buttons[name] = button
        row.addStretch(1)
        bar.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)
        return bar

    def _build_canvas(self):
        self.pg_widget = GraphicsLayoutWidget()
        self.view = CanvasViewBox(parent_gui=self, lockAspect=True, invertY=True)
        self.pg_widget.addItem(self.view)

        # row-major throughout: view x is the column index and y the row index,
        # so nothing in this module has to transpose anything but a mouse event.
        self.image_item = ImageItem(axisOrder='row-major')
        self.score_overlay = ImageItem(axisOrder='row-major')
        self.roi_overlay = ImageItem(axisOrder='row-major')

        # The subset borders live apart from the translucent fill so they can be
        # stroked with a *cosmetic* pen, whose width is in screen pixels: they stay
        # a hairline at any zoom, where a raster border cannot go below one image
        # pixel and becomes a thick band as soon as you zoom in.
        self.roi_outline = QtWidgets.QGraphicsPathItem()
        pen = pg.mkPen(0, 255, 0, 150)
        pen.setCosmetic(True)
        self.roi_outline.setPen(pen)
        self.roi_outline.setBrush(pg.mkBrush(None))

        # The stroke being painted is a path of overlapping discs rather than a
        # raster overlay. A raster one has to be rebuilt and re-uploaded whole on
        # every mouse move -- eight milliseconds a move on a four-megapixel frame,
        # paid while the mouse is moving, which is exactly when it is felt.
        self.brush_overlay = QtWidgets.QGraphicsPathItem()
        self.brush_overlay.setPen(pg.mkPen(None))

        self.geometry_line = pg.PlotDataItem(pen=pg.mkPen('y', width=2))
        self.geometry_vertices = ScatterPlotItem(pen=pg.mkPen(None), brush=pg.mkBrush(255, 255, 0, 200), size=7)
        self.point_scatter = DotCloud((255, 100, 100, 220), 7)
        # The features the mask is leaving out, shown while masking so that an
        # empty patch says which of the two things it is: nothing to track
        # there, or something you have masked away. Smaller and greyer than a
        # selected point, and drawn under it, so the two never read alike.
        self.candidate_scatter = DotCloud((90, 170, 255, 170), 5)
        # Points the deselect brush is about to take away, shown while the stroke
        # is still being painted rather than only after the mouse comes up. White,
        # and drawn over the stroke's red wash rather than under it, which is the
        # only place it is ever seen.
        self.doomed_scatter = ScatterPlotItem(
            pen=pg.mkPen(255, 255, 255, 240, width=1.5), brush=pg.mkBrush(None), size=9, symbol='x')
        self.highlight_scatter = ScatterPlotItem(
            pen=pg.mkPen(255, 0, 255, 230, width=2), brush=pg.mkBrush(None), size=13)
        self.direction_line = pg.PlotDataItem(pen=pg.mkPen('r', width=2))

        for item, z in ((self.image_item, 0), (self.score_overlay, 0.5), (self.roi_overlay, 1),
                        (self.roi_outline, 1), (self.geometry_line, 2), (self.geometry_vertices, 2),
                        (self.candidate_scatter, 2.5), (self.point_scatter, 3),
                        (self.highlight_scatter, 3), (self.direction_line, 3.5),
                        (self.brush_overlay, 4), (self.doomed_scatter, 4.5)):
            item.setZValue(z)
            self.view.addItem(item)
        self.score_overlay.setVisible(False)

        self.pg_widget.scene().sigMouseClicked.connect(self.on_mouse_click)
        self.splitter.addWidget(self.pg_widget)

    def _build_panel(self):
        panel = QtWidgets.QWidget()
        column = QtWidgets.QVBoxLayout(panel)

        self.step_stack = QtWidgets.QStackedLayout()
        self.step_pages = {}
        for name, builder in ((STEP_FIND, self._build_find_page),
                              (STEP_MASK, self._build_mask_page)):
            page = QtWidgets.QWidget()
            builder(QtWidgets.QVBoxLayout(page))
            self.step_stack.addWidget(page)
            self.step_pages[name] = page
        column.addLayout(self.step_stack, stretch=1)

        # Outside the stack, so it shows on both tabs. It is neither tab's
        # setting: the scoring window follows it, and so does the rectangle
        # drawn round every point.
        # The subset size is on both tabs because both read it; the selections
        # list is not, because every row in it and every button under it belongs
        # to the mask step. Nothing on the other tab acts on a row.
        column.addWidget(self._build_subset_group())
        self.selection_box = self._build_selection_list()
        column.addWidget(self.selection_box)

        self.count_label = QtWidgets.QLabel('0 points')
        font = self.count_label.font()
        font.setBold(True)
        self.count_label.setFont(font)
        column.addWidget(self.count_label)

        panel.setMinimumWidth(320)
        panel.setMaximumWidth(600)
        self.splitter.addWidget(panel)
        self.splitter.setStretchFactor(0, 5)
        self.splitter.setStretchFactor(1, 0)
        self.splitter.setSizes([920, 340])

    def _build_mask_page(self, layout):
        tools = QtWidgets.QGroupBox('Region tool')
        grid = QtWidgets.QGridLayout(tools)
        self.tool_buttons = {}
        group = QtWidgets.QButtonGroup(self)
        group.setExclusive(True)
        # Two columns: five full-width buttons stacked was most of the panel's
        # height for something you click once.
        for index, (label, kind) in enumerate(TOOLS):
            button = QtWidgets.QPushButton(label)
            button.setCheckable(True)
            group.addButton(button)
            grid.addWidget(button, index // 2, index % 2)
            button.clicked.connect(lambda _, k=kind: self.select_tool(k))
            self.tool_buttons[kind] = button
        self.tool_buttons['polygon'].setChecked(True)
        layout.addWidget(tools)

        self.new_entry_button = QtWidgets.QPushButton('Start new polygon')
        self.new_entry_button.setToolTip(
            'Begin a second polygon or line instead of adding vertices to the one '
            'already selected in the list.')
        self.new_entry_button.clicked.connect(self.start_new_entry)
        layout.addWidget(self.new_entry_button)

        brush = QtWidgets.QGroupBox('Brush (hold Ctrl and drag to paint)')
        brush_column = QtWidgets.QVBoxLayout(brush)
        self.brush_radius = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.brush_radius.setRange(1, 100)
        self.brush_radius.setValue(12)
        self.brush_radius.setToolTip('Brush radius, in pixels.')
        radius = self._form()
        radius.addRow('Radius', self.brush_radius)
        brush_column.addLayout(radius)
        self.deselect_button = QtWidgets.QPushButton('Deselect painted area')
        self.deselect_button.setCheckable(True)
        self.deselect_button.setToolTip(
            'Paint to subtract instead of adding. It takes away only the part you '
            'paint over, and the points about to go are crossed out while you paint.')
        self.deselect_button.toggled.connect(self._set_deselect_mode)
        brush_column.addWidget(self.deselect_button)
        layout.addWidget(brush)

        self.spacing_spin = QtWidgets.QSpinBox()
        self.spacing_spin.setRange(-500, 500)
        self.spacing_spin.setValue(self.pipeline.spacing)
        self.spacing_spin.valueChanged.connect(self._on_spacing_changed)
        self.spacing_spin.setToolTip(
            'Extra spacing between the points a "points" row lays out. Rows that '
            'act as a mask are unaffected: how far apart their points end up is '
            'the separation, on the other tab.')
        spacing = self._form()
        spacing.addRow('Point spacing', self.spacing_spin)
        layout.addLayout(spacing)

        # The overlay is as useful here as on the other tab: it is what tells you
        # whether the area you are about to keep has anything worth tracking in it.
        self.show_score_mask = self._make_score_toggle()
        layout.addWidget(self.show_score_mask)

        clear = QtWidgets.QPushButton('Clear all')
        clear.setToolTip('Drop every selection and go back to the whole frame.')
        clear.clicked.connect(self.clear_all)
        layout.addWidget(clear)
        layout.addStretch(1)

    def _build_subset_group(self):
        """The subset size, shown on every tab because it belongs to none of them.

        It drives *evaluate* -- the scoring window is the subset size, so this is
        one of only three things that make the score image stale -- and it is
        also what the rectangle drawn round each point measures while you mask.
        Putting it on one tab would make it look like that tab's setting and hide
        it from the other, so it sits below the tabs instead.
        """
        box = QtWidgets.QGroupBox('Subset')
        grid = QtWidgets.QGridLayout(box)
        height, width = self.pipeline.subset_size

        self.square_check = QtWidgets.QCheckBox('Square subsets')
        self.square_check.setToolTip('Untick to set the height and the width separately.')
        self.square_check.setChecked(height == width)
        self.square_check.toggled.connect(self._on_square_toggled)
        grid.addWidget(self.square_check, 0, 0, 1, 2)

        self.height_spin = OddSpinBox()
        self.width_spin = OddSpinBox()
        for spin, value in ((self.height_spin, height), (self.width_spin, width)):
            spin.setToolTip(
                'The size of the subset each point stands for, in pixels. Odd only: '
                'the subset is centred on its point, so an even extent has no centre '
                'to be. The scoring window follows it, so this is one of the few '
                'settings that makes the score stale and pays for a fresh evaluation.')
            spin.setRange(3, 501)
            spin.setValue(value)
            spin.valueChanged.connect(self._on_subset_size_changed)
        self.width_spin.setEnabled(height != width)
        grid.addWidget(QtWidgets.QLabel('Height'), 1, 0)
        grid.addWidget(self.height_spin, 1, 1)
        grid.addWidget(QtWidgets.QLabel('Width'), 2, 0)
        grid.addWidget(self.width_spin, 2, 1)

        self.show_subsets = QtWidgets.QCheckBox('Show subsets')
        self.show_subsets.setToolTip(
            'Draw each point as the subset it stands for, so overlapping subsets are '
            'visible. It changes nothing about the selection.')
        self.show_subsets.setChecked(True)
        self.show_subsets.toggled.connect(lambda _: self.draw_points())
        grid.addWidget(self.show_subsets, 3, 0, 1, 2)
        return box

    def _build_find_page(self, layout):
        """Evaluate and select, in one panel.

        They are tuned against each other -- switching the evaluator changes
        what threshold means -- so splitting them across tabs would only buy a
        tab switch after every change.
        """
        layout.addWidget(self._build_evaluate_group())
        layout.addWidget(self._build_select_group())

        # The only prose left on the panel, and it is empty unless something is
        # actually wrong. Everything the labels used to say -- what an evaluator
        # measures, what gets recomputed -- is a tooltip now: it was permanent
        # screen furniture that you read once.
        self.select_note = QtWidgets.QLabel('')
        self.select_note.setWordWrap(True)
        layout.addWidget(self.select_note)
        layout.addStretch(1)
        self._update_selector_rows()

    def _build_evaluate_group(self):
        """The evaluator and its parameters, in one flat form.

        The parameters used to sit in a group box inside this one. Two nested
        frames cost two sets of margins out of a panel that is already narrow,
        which is what was clipping the values off the right-hand side, and the
        inner title said nothing the rows did not.
        """
        box = QtWidgets.QGroupBox('Evaluate')
        layout = QtWidgets.QVBoxLayout(box)
        self.param_layout = self._form()
        layout.addLayout(self.param_layout)

        self.evaluator_combo = QtWidgets.QComboBox()
        self.evaluators = available_evaluators()
        for name, spec in sorted(self.evaluators.items()):
            self.evaluator_combo.addItem(spec.display_name, name)
        self.evaluator_combo.setCurrentIndex(self.evaluator_combo.findData('shi_tomasi'))
        self.evaluator_combo.currentIndexChanged.connect(self._on_evaluator_changed)
        self.param_layout.addRow('Score', self.evaluator_combo)

        self.param_widgets = {}
        self._rebuild_param_widgets()

        self.show_score = self._make_score_toggle()
        layout.addWidget(self.show_score)
        return box

    @staticmethod
    def _form():
        """A form layout whose fields take the width they are given.

        :rtype: PyQt6.QtWidgets.QFormLayout
        """
        form = QtWidgets.QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        return form

    def _make_score_toggle(self):
        """A ``Show score overlay`` checkbox, ganged to every other one.

        Each tab gets its own, because the overlay answers a different question
        on each -- "is this threshold too tight or is there nothing there" on
        one, "does the area I am keeping have anything in it" on the other --
        and having to switch tabs to turn it on would defeat both.

        :rtype: PyQt6.QtWidgets.QCheckBox
        """
        box = QtWidgets.QCheckBox('Show score overlay')
        box.setToolTip(
            'Draw the score itself as a heatmap, so you can see where the trackable '
            'content is before committing to any points. The border the subset window '
            'cannot reach is left transparent: it is unscored, not scored badly.')
        box.toggled.connect(self._on_show_score_toggled)
        self.score_toggles.append(box)
        return box

    def _build_select_group(self):
        """The selector and its settings, in one flat form.

        Rows that the current selector ignores are hidden rather than greyed
        out: a disabled ``Grid pitch`` is a line of panel spent saying that this
        line does not apply.
        """
        box = QtWidgets.QGroupBox('Select')
        layout = QtWidgets.QVBoxLayout(box)
        self.select_form = self._form()
        layout.addLayout(self.select_form)

        self.selector_combo = QtWidgets.QComboBox()
        for name in sorted(SELECTORS):
            self.selector_combo.addItem(name, name)
        self.selector_combo.setCurrentIndex(self.selector_combo.findData('peaks'))
        self.selector_combo.setToolTip(
            '"peaks" puts a point on each local maximum of the score, which is how '
            'the points end up on the features. "lattice" puts them on a regular '
            'grid instead, for even coverage rather than the best features.')
        self.selector_combo.currentIndexChanged.connect(self._on_selector_changed)
        self.select_form.addRow('Points', self.selector_combo)

        self.threshold_mode = QtWidgets.QComboBox()
        for label, mode, _ in THRESHOLD_RULES:
            self.threshold_mode.addItem(label, mode)
        self.threshold_mode.setToolTip(
            'Quality is a fraction of the best feature in the region, so 0.01 means '
            '"at least a hundredth as good as the best". Percentile ranks pixels '
            'instead, and on a dense score image the pixels are overwhelmingly '
            'background, so it is only really useful with the lattice selector.')
        self.threshold_mode.currentIndexChanged.connect(self._on_threshold_mode_changed)
        self.select_form.addRow('Threshold', self.threshold_mode)

        self.threshold_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.threshold_slider.setRange(0, 1000)
        self.threshold_slider.setValue(self._slider_position('quality', DEFAULT_THRESHOLD))
        self.threshold_slider.setToolTip(
            'How good a subset has to be to be worth tracking. The quality scale is '
            'logarithmic: featureless background sits near 0.001 of the best feature '
            'and a strong corner near 1.')
        self.threshold_slider.valueChanged.connect(self._on_threshold_changed)
        self.threshold_label = QtWidgets.QLabel(f'{DEFAULT_THRESHOLD:.3g}')
        self.threshold_label.setMinimumWidth(40)
        self.threshold_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight
                                          | QtCore.Qt.AlignmentFlag.AlignVCenter)
        slider_row = QtWidgets.QWidget()
        slider_layout = QtWidgets.QHBoxLayout(slider_row)
        slider_layout.setContentsMargins(0, 0, 0, 0)
        slider_layout.addWidget(self.threshold_slider, stretch=1)
        slider_layout.addWidget(self.threshold_label)
        self.select_form.addRow('', slider_row)

        self.separation_spin = QtWidgets.QSpinBox()
        self.separation_spin.setRange(1, 500)
        self.separation_spin.setSuffix(' px')
        self.separation_spin.setValue(self.pipeline.selector_params['separation'])
        self.separation_spin.setToolTip(
            'No two points end up closer together than this, so it is the control for '
            'how many you get: lower it for more. Thinning the pixels above the '
            'threshold any other way puts most of the subsets back-to-back on the '
            'same feature, which is why this is not a "keep every n-th".')
        self.separation_spin.valueChanged.connect(self._on_separation_changed)
        self.select_form.addRow('Separation', self.separation_spin)

        self.pitch_spin = QtWidgets.QSpinBox()
        self.pitch_spin.setRange(1, 500)
        self.pitch_spin.setValue(12)
        self.pitch_spin.setToolTip('Distance between grid positions, for the lattice selector.')
        self.pitch_spin.valueChanged.connect(self._on_pitch_changed)
        self.select_form.addRow('Grid pitch', self.pitch_spin)

        self.max_points_spin = QtWidgets.QSpinBox()
        self.max_points_spin.setRange(1, 200000)
        self.max_points_spin.setValue(DEFAULT_MAX_POINTS)
        self.max_points_spin.setToolTip(
            'A safety valve, not a target: the selection stops here however many '
            'points the threshold and the separation would have given. It says so '
            'below when it is what stopped it.')
        self.max_points_spin.valueChanged.connect(self._on_max_points_changed)
        self.select_form.addRow('Maximum points', self.max_points_spin)

        self.decimation_spin = QtWidgets.QSpinBox()
        self.decimation_spin.setRange(1, 100)
        self.decimation_spin.setValue(1)
        self.decimation_spin.setToolTip(
            'Keep every n-th of the points already selected. Unlike a wider '
            'separation, which re-selects and moves every point, this leaves the '
            'survivors exactly where they are -- for when the selection is right and '
            'only the count is too high for the computation you are about to run.')
        self.decimation_spin.valueChanged.connect(self._on_decimation_changed)
        self.select_form.addRow('Keep every n-th', self.decimation_spin)
        return box

    @staticmethod
    def _slider_position(mode, value):
        """Where a threshold value sits on the 0..1000 slider, for ``mode``."""
        if mode == 'quality':
            low, high = dict((m, r) for _, m, r in THRESHOLD_RULES)['quality']
            span = np.log10(high) - np.log10(low)
            return int(round(1000 * (np.log10(max(value, low)) - np.log10(low)) / span))
        return int(round(value * (10 if mode == 'percentile' else 1000)))

    @staticmethod
    def _slider_value(mode, position):
        """The threshold a slider position means, and how to print it."""
        if mode == 'quality':
            low, high = dict((m, r) for _, m, r in THRESHOLD_RULES)['quality']
            span = np.log10(high) - np.log10(low)
            value = 10 ** (np.log10(low) + span * position / 1000.0)
            return value, f'{value:.3g}'
        if mode == 'percentile':
            return position / 10.0, f'{position / 10.0:.1f}'
        return position / 1000.0, f'{position / 1000.0:.3f}'

    def _build_selection_list(self):
        box = QtWidgets.QGroupBox('Selections')
        column = QtWidgets.QVBoxLayout(box)
        self.entry_list = QtWidgets.QListWidget()
        self.entry_list.currentRowChanged.connect(self._on_row_changed)
        self.entry_list.itemChanged.connect(self._on_item_changed)
        column.addWidget(self.entry_list)

        buttons = QtWidgets.QHBoxLayout()
        self.role_button = QtWidgets.QPushButton('Use as points')
        self.role_button.setToolTip(
            'A "mask" row says where points may go and lets the selection choose '
            'them; a "points" row contributes its own coordinates directly, whatever '
            'they score. Switching does not redraw the region.')
        self.role_button.clicked.connect(self.toggle_role)
        buttons.addWidget(self.role_button)
        delete = QtWidgets.QPushButton('Delete')
        delete.clicked.connect(self.delete_active)
        buttons.addWidget(delete)
        column.addLayout(buttons)
        return box

    # -- steps and tools ---------------------------------------------------

    def select_step(self, name):
        """Switch tab.

        :param name: one of :data:`STEPS`
        :type name: str
        """
        self.step = name
        self.step_buttons[name].setChecked(True)
        self.step_stack.setCurrentWidget(self.step_pages[name])
        self.status.showMessage(STEP_HINTS[name])
        # The magenta ring and the vertex handles mark what is being edited, so
        # they belong to the Mask tab; elsewhere they would decorate geometry
        # nobody is touching.
        self.selection_box.setVisible(name == STEP_MASK)
        self.highlight_scatter.setVisible(name == STEP_MASK)
        self.geometry_line.setVisible(name == STEP_MASK)
        self.geometry_vertices.setVisible(name == STEP_MASK)
        self.refresh()

    def select_tool(self, kind):
        """Make a region tool active.

        :param kind: an entry kind, or ``'remove'``
        :type kind: str
        """
        self.tool = kind
        self.tool_buttons[kind].setChecked(True)
        self.new_entry_button.setEnabled(kind in VERTEX_KINDS)
        if kind in VERTEX_KINDS:
            self.new_entry_button.setText(f'Start new {"polygon" if kind == "polygon" else "line"}')
        self.status.showMessage({
            'polygon': 'Click to place polygon corners. The enclosed area becomes a mask.',
            'brush': 'Hold Ctrl and drag to paint a mask.',
            'polyline': 'Click to place line vertices. Points are spaced along the segments.',
            'points': 'Click to place individual points. They bypass scoring.',
            'remove': 'Click near a point to remove it.',
        }[kind])

    def start_new_entry(self):
        """Begin a fresh polygon or polyline instead of extending the active one."""
        if self.tool in VERTEX_KINDS:
            entry = self.pipeline.add_entry(self.tool, [])
            self.active_index = len(self.pipeline.entries) - 1
            self.push_undo({'type': 'add', 'entry': entry})
            self.refresh()

    # -- entries -----------------------------------------------------------

    def active_entry(self, kind=None):
        """The entry the selections list has selected, if it is of ``kind``.

        :param kind: required entry kind, or ``None`` for any
        :type kind: str or None
        :rtype: Entry or None
        """
        if self.active_index is None or not (0 <= self.active_index < len(self.pipeline.entries)):
            return None
        entry = self.pipeline.entries[self.active_index]
        return entry if kind is None or entry.kind == kind else None

    def _entry_for_tool(self, kind):
        """The entry a click should extend, creating one when there is none."""
        entry = self.active_entry(kind)
        if entry is not None:
            return entry
        for index in range(len(self.pipeline.entries) - 1, -1, -1):
            if self.pipeline.entries[index].kind == kind:
                self.active_index = index
                return self.pipeline.entries[index]
        entry = self.pipeline.add_entry(kind, [])
        self.active_index = len(self.pipeline.entries) - 1
        return entry

    def add_whole_image_mask(self):
        """Seed the selections list with a mask covering the whole frame.

        Without it the window opens showing nothing, and the Mask tab would have
        to be visited before anything happened -- which is the workflow this
        ordering exists to avoid. With it, the candidates are there to look at
        immediately and masking becomes what it should be: trimming them.

        It is a row like any other, so it can be unchecked, painted away with
        the deselect brush, or deleted outright. Deleting it selects nothing,
        which is the same rule as for every other mask row. The geometry is a
        brush mask rather than a four-corner polygon because rasterising it is
        then a copy rather than a point-in-polygon test over every pixel of the
        frame, on every redraw.

        :return: the new entry
        :rtype: Entry
        """
        entry = self.pipeline.add_entry(
            'brush', np.ones(self.pipeline.shape, dtype=bool), label=WHOLE_IMAGE_LABEL)
        self._whole_image = entry
        self.active_index = len(self.pipeline.entries) - 1
        return entry

    def _retire_whole_image(self):
        """Uncheck the seeded whole-image row once a drawn region covers something.

        Mask rows combine as a *union*, so a region drawn while the whole frame
        is still selected changes nothing at all -- you draw a polygon and the
        points do not move, which reads as the drawing being broken. Standing the
        seeded row down as soon as another mask has area makes the drawing do what
        it looks like it does.

        The row is unchecked rather than deleted, so ticking it again in the list
        brings the whole frame back, and the undo stack records the change.
        """
        seeded, others = None, False
        for entry in self.pipeline.entries:
            # By identity, not by label: the label is what the row is called,
            # which is not the same as which row this is.
            if entry is self._whole_image:
                seeded = entry
            elif entry.role == 'mask' and entry.visible and self.pipeline.area(entry).any():
                others = True
        if seeded is None or not seeded.visible or not others:
            return
        seeded.visible = False
        self.push_undo({'type': 'visible', 'entry': seeded, 'value': True})
        self.status.showMessage(
            f'Unchecked "{WHOLE_IMAGE_LABEL}" so the region you drew takes effect. '
            'Tick it again in the list to bring the whole frame back.')

    def toggle_role(self):
        """Flip the active row between contributing an area and contributing points."""
        entry = self.active_entry()
        if entry is None:
            return
        entry.role = 'points' if entry.role == 'mask' else 'mask'
        self.refresh()

    def delete_active(self):
        """Delete the active row."""
        entry = self.active_entry()
        if entry is None:
            return
        index = self.active_index
        self.push_undo({'type': 'delete', 'entry': entry, 'index': index})
        self.pipeline.entries.pop(index)
        self.active_index = min(index, len(self.pipeline.entries) - 1)
        if self.active_index < 0:
            self.active_index = None
        self.refresh()

    def clear_all(self):
        """Start over: drop every selection and seed the whole-image row again.

        "Start over" means the state the window opens in, which has the whole
        frame selected -- not an empty canvas. Clearing back to nothing would
        leave you looking at a blank frame and needing to know that a mask is
        what brings the points back.

        Deleting the whole-image row on its own still selects nothing. That is a
        different act: it says "not this area", where this one says "forget what
        I have done so far".
        """
        self.push_undo({'type': 'restore', 'entries': list(self.pipeline.entries)})
        self.pipeline.entries = []
        self.add_whole_image_mask()
        self.refresh()

    # -- mouse -------------------------------------------------------------

    def on_mouse_click(self, event):
        """Route a click on the image to whichever tool is active."""
        if self.step != STEP_MASK or event.button() != QtCore.Qt.MouseButton.LeftButton:
            return
        if not self.view.sceneBoundingRect().contains(event.scenePos()):
            return
        point = self.view.mapSceneToView(event.scenePos())
        position = (point.y(), point.x())     # view y is the row, x the column

        if self.tool == 'remove':
            self.remove_nearest_point(position)
        elif self.tool in VERTEX_KINDS:
            self.add_vertex(position)
        elif self.tool == 'points':
            self.add_point(position)
        else:
            return
        self._retire_whole_image()
        self.refresh()

    def add_vertex(self, position):
        """Append a vertex to the active polygon or polyline."""
        entry = self._entry_for_tool(self.tool)
        rounded = (float(position[0]), float(position[1]))
        if any(np.hypot(v[0] - rounded[0], v[1] - rounded[1]) < 1e-6 for v in entry.geometry):
            return          # clicking exactly on a vertex must not stack a duplicate
        entry.geometry.append(rounded)
        self.push_undo({'type': 'vertex_add', 'entry': entry})

    def add_point(self, position):
        """Append a hand-picked coordinate, if the click landed on the image.

        The view is larger than the frame -- the aspect is locked, so one axis
        always has a margin, and zooming out adds more -- and a subset centred
        off the frame is not something that can be tracked. So a click outside
        it is ignored rather than recorded as a point nothing downstream can use.
        """
        coordinate = (int(round(position[0])), int(round(position[1])))
        height, width = self.pipeline.shape
        if not (0 <= coordinate[0] < height and 0 <= coordinate[1] < width):
            self.status.showMessage('That click was outside the image, so no point was added.')
            return
        entry = self._entry_for_tool('points')
        if coordinate in entry.geometry:
            return
        entry.geometry.append(coordinate)
        entry.removed.discard(coordinate)
        self.push_undo({'type': 'vertex_add', 'entry': entry})

    def remove_nearest_point(self, position):
        """Remove whichever displayed point is nearest the click, if any is close."""
        credited = getattr(self, '_credited', None) or self.pipeline.points_by_entry()
        best, best_entry, best_distance = None, None, np.inf
        for entry, points in zip(self.pipeline.entries, credited):
            if not len(points):
                continue
            distances = np.hypot(points[:, 0] - position[0], points[:, 1] - position[1])
            nearest = int(distances.argmin())
            if distances[nearest] < best_distance:
                best_distance = float(distances[nearest])
                best_entry = entry
                best = (int(points[nearest, 0]), int(points[nearest, 1]))
        if best is None or best_distance > max(self.pipeline.subset_size):
            return
        self.push_undo(self._snapshot())
        self.pipeline.remove_point(best_entry, best)

    def vertex_at(self, position):
        """The vertex within the grab radius of ``position``, if any.

        The radius is constant in *screen* pixels, so grabbing a vertex feels the
        same however far the view is zoomed.

        :param position: ``(row, col)`` in image coordinates
        :type position: tuple
        :return: ``(entry, vertex index)``, or ``(None, None)``
        :rtype: tuple
        """
        scale = self.view.viewPixelSize()[0] or 1.0
        radius = VERTEX_GRAB_RADIUS_PX * scale
        for entry in self.pipeline.entries:
            if entry.kind != self.tool or not entry.visible:
                continue
            for index, vertex in enumerate(entry.geometry):
                if np.hypot(vertex[0] - position[0], vertex[1] - position[1]) <= radius:
                    return entry, index
        return None, None

    # -- brush -------------------------------------------------------------

    @property
    def painting(self):
        """Whether a brush stroke is currently being laid down.

        :rtype: bool
        """
        return self._paint is not None

    def brush_start(self):
        """Begin a stroke."""
        self._paint = np.zeros(self.pipeline.shape, dtype=bool)
        self._stroke_path = QtGui.QPainterPath()
        self._stroke_path.setFillRule(QtCore.Qt.FillRule.WindingFill)

    def brush_move(self, position):
        """Add a dab at ``position``, given as ``(row, col)`` or ``None``."""
        if self._paint is None or position is None:
            return
        row, col = int(round(position[0])), int(round(position[1]))
        radius = self.brush_radius.value()
        height, width = self._paint.shape
        rows, cols = np.ogrid[max(0, row - radius):min(height, row + radius + 1),
                              max(0, col - radius):min(width, col + radius + 1)]
        dab = (rows - row) ** 2 + (cols - col) ** 2 <= radius ** 2
        self._paint[max(0, row - radius):min(height, row + radius + 1),
                    max(0, col - radius):min(width, col + radius + 1)][dab] = True
        # x is the column and y the row, and the ellipse is inscribed in the
        # square the raster dab fills.
        self._stroke_path.addEllipse(
            QtCore.QRectF(col - radius, row - radius, 2 * radius + 1, 2 * radius + 1))
        self.draw_brush()
        if self.deselect_mode:
            # Cheap: it re-reads the stroke against points already computed,
            # rather than re-running the pipeline on every mouse move.
            self.draw_doomed()

    def brush_end(self):
        """Commit the stroke, either as a new region or as a deselection."""
        if self._paint is None:
            return
        stroke = self._paint
        self._paint = None
        self._stroke_path = QtGui.QPainterPath()
        if not stroke.any():
            self.draw_brush()
            return

        if self.deselect_mode:
            self.push_undo(self._snapshot())
            self.pipeline.deselect(stroke)
        else:
            entry = self.pipeline.add_entry('brush', stroke)
            self.active_index = len(self.pipeline.entries) - 1
            self.push_undo({'type': 'add', 'entry': entry})
            self._retire_whole_image()
        self.draw_brush()
        self.refresh()

    def _set_deselect_mode(self, enabled):
        self.deselect_mode = enabled

    # -- undo --------------------------------------------------------------

    def _snapshot(self):
        """A restorable copy of every entry's mutable state.

        Deselection touches an unpredictable set of entries at once -- erasing
        part of some, emptying others -- so it is undone by restoring the whole
        list rather than by trying to invert each edit.

        The ``erased`` array is held by reference, not copied: it is always
        replaced wholesale rather than written into, so the array a snapshot
        points at still holds what it held when the snapshot was taken. Copying
        it would put a frame's worth of booleans per region into every one of
        the fifty undo slots. The vertex list and the removed set *are* appended
        to in place, so those are copied.
        """
        return {
            'type': 'restore',
            'entries': list(self.pipeline.entries),
            'state': [(entry,
                       entry.erased,
                       list(entry.geometry) if isinstance(entry.geometry, list) else entry.geometry,
                       set(entry.removed))
                      for entry in self.pipeline.entries],
        }

    def push_undo(self, action):
        """Record an undoable action.

        :param action: the action record
        :type action: dict
        """
        self.undo_stack.append(action)
        del self.undo_stack[:-self.undo_limit]

    def undo(self):
        """Reverse the last undoable action."""
        if not self.undo_stack:
            return
        action = self.undo_stack.pop()
        self._REVERSALS[action['type']](self, action)
        self.active_index = min(self.active_index or 0, len(self.pipeline.entries) - 1)
        if self.active_index < 0:
            self.active_index = None
        self.refresh()

    def _undo_restore(self, action):
        """Put the whole entry list, and every entry's state, back as it was."""
        self.pipeline.entries = list(action['entries'])
        for entry, erased, geometry, removed in action.get('state', []):
            entry.erased, entry.geometry, entry.removed = erased, geometry, removed

    #: How each recorded action is reversed. A table rather than a chain of
    #: ``elif``s so adding an undoable action is one entry, not one more branch.
    _REVERSALS = {
        'vertex_add': lambda self, a: a['entry'].geometry and a['entry'].geometry.pop(),
        'vertex_move': lambda self, a: a['entry'].geometry.__setitem__(a['index'], a['original']),
        'add': lambda self, a: self.pipeline.remove_entry(a['entry']),
        'delete': lambda self, a: self.pipeline.entries.insert(a['index'], a['entry']),
        'visible': lambda self, a: setattr(a['entry'], 'visible', a['value']),
        'restore': lambda self, a: self._undo_restore(a),
    }

    # -- settings callbacks ------------------------------------------------

    def _on_square_toggled(self, checked):
        self.width_spin.setEnabled(not checked)
        if checked:
            self.width_spin.setValue(self.height_spin.value())
        self._on_subset_size_changed()

    def _on_subset_size_changed(self, *_):
        if self.square_check.isChecked():
            self._syncing, previous = True, self._syncing
            self.width_spin.setValue(self.height_spin.value())
            self._syncing = previous
        self.pipeline.set_subset_size((self.height_spin.value(), self.width_spin.value()))
        self.request_refresh()

    def _on_spacing_changed(self, value):
        self.pipeline.spacing = value
        self.request_refresh()

    def _on_evaluator_changed(self, *_):
        self._rebuild_param_widgets()
        self._redefine_score()

    def _on_show_score_toggled(self, checked):
        for box in self.score_toggles:
            if box.isChecked() != checked:
                box.blockSignals(True)
                box.setChecked(checked)
                box.blockSignals(False)
        self.score_overlay.setVisible(checked)
        if checked:
            self.draw_score()

    def _on_selector_changed(self, *_):
        self.pipeline.selector = self.selector_combo.currentData()
        self._update_selector_rows()
        self.refresh()

    def _on_threshold_mode_changed(self, *_):
        """Switch rule, and put the slider where that rule's default lives.

        Carrying the position across would be meaningless: the same position is
        a percentile of 90 under one rule and a quality of 0.5 under another.
        """
        mode = self.threshold_mode.currentData()
        self.pipeline.selector_params['threshold_mode'] = mode
        position = self._slider_position(mode, THRESHOLD_DEFAULTS[mode])
        if self.threshold_slider.value() == position:
            self._on_threshold_changed(position)        # setValue would not signal
        else:
            self.threshold_slider.setValue(position)

    def _on_threshold_changed(self, value):
        threshold, text = self._slider_value(self.threshold_mode.currentData(), value)
        self.threshold_label.setText(text)
        self.pipeline.selector_params['threshold'] = threshold
        self.request_refresh()

    def _on_decimation_changed(self, value):
        self.pipeline.selector_params['decimation'] = value
        self.request_refresh()

    def _on_separation_changed(self, value):
        self.pipeline.selector_params['separation'] = value
        self.request_refresh()

    def _on_pitch_changed(self, value):
        self.pipeline.selector_params['pitch'] = value
        self.request_refresh()

    def _on_max_points_changed(self, value):
        self.pipeline.selector_params['max_points'] = value
        self.request_refresh()

    def _update_selector_rows(self):
        """Show only the rows the current selector actually reads."""
        peaks = self.selector_combo.currentData() == 'peaks'
        self.select_form.setRowVisible(self.separation_spin, peaks)
        self.select_form.setRowVisible(self.pitch_spin, not peaks)

    def _rebuild_param_widgets(self):
        """Build the evaluator's parameter controls from its descriptors.

        Nothing here knows what a Shi-Tomasi or a gradient-direction parameter
        is: the registry says a parameter is a float, an int or a direction, and
        that is enough to make a widget for it. Adding an evaluator therefore
        needs no change to this module.
        """
        while self.param_layout.rowCount() > 1:      # row 0 is the evaluator itself
            self.param_layout.removeRow(1)
        self.param_widgets = {}
        self.direction_spins = []
        self.direction_button = None
        # The line describes a parameter that no longer exists once the evaluator
        # has changed, so it goes with the widget that owned it.
        self.drawing_direction = False
        self.direction_line.clear()

        spec = self.evaluators[self.evaluator_combo.currentData()]
        for parameter in spec.parameters:
            if parameter.kind == 'direction':
                widget, getter = self._direction_widget(parameter)
            else:
                widget, getter = self._number_widget(parameter)
            widget.setToolTip(parameter.description)
            self.param_layout.addRow(parameter.name.replace('_', ' ').capitalize(), widget)
            self.param_widgets[parameter.name] = getter
        self.evaluator_combo.setToolTip(spec.description)
        if self.direction_spins:
            self._syncing = True    # the line only, not a fresh evaluation
            self.set_direction(*(spin.value() for spin in self.direction_spins))
            self._syncing = False

    def _number_widget(self, parameter):
        spin = QtWidgets.QDoubleSpinBox() if parameter.kind == 'float' else QtWidgets.QSpinBox()
        spin.setRange(parameter.minimum if parameter.minimum is not None else -1e9,
                      parameter.maximum if parameter.maximum is not None else 1e9)
        spin.setValue(parameter.default)
        spin.valueChanged.connect(lambda _: self._redefine_score())
        return spin, spin.value

    def _direction_widget(self, parameter):
        """The two components, with the presets and the drag button beneath them.

        On one line the five controls squeeze the form's label column until
        "Direction" elides to "Direct", so the buttons get a line of their own.
        """
        widget = QtWidgets.QWidget()
        column = QtWidgets.QVBoxLayout(widget)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(2)
        row = QtWidgets.QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        column.addLayout(row)
        spins = []
        for value in parameter.default:
            spin = QtWidgets.QDoubleSpinBox()
            spin.setRange(-1e6, 1e6)
            # Three decimals rather than Qt's two: a normalised component wants
            # better than the ~0.6 degrees two would round a dragged vector to.
            spin.setDecimals(3)
            spin.setSingleStep(0.05)
            spin.setMaximumWidth(80)
            spin.setValue(float(value))
            spin.valueChanged.connect(lambda _: self._redefine_score())
            row.addWidget(spin)
            spins.append(spin)
        row.addStretch(1)
        self.direction_spins = spins

        buttons = QtWidgets.QHBoxLayout()
        buttons.setContentsMargins(0, 0, 0, 0)
        column.addLayout(buttons)
        for label, vector in (('X', (0.0, 1.0)), ('Y', (1.0, 0.0))):
            button = QtWidgets.QPushButton(label)
            button.setMaximumWidth(52)
            button.setToolTip(f'Along the image {label.lower()} axis.')
            buttons.addWidget(button)
            button.clicked.connect(lambda _, v=vector: self.set_direction(*v))
        self.direction_button = QtWidgets.QPushButton('Draw')
        self.direction_button.setCheckable(True)
        self.direction_button.setToolTip('Drag on the image to point the direction out.')
        self.direction_button.setMaximumWidth(52)
        self.direction_button.toggled.connect(self._set_direction_drawing)
        buttons.addWidget(self.direction_button)
        buttons.addStretch(1)
        return widget, (lambda s=spins: (s[0].value(), s[1].value()))

    # -- gradient direction ------------------------------------------------

    def _set_direction_drawing(self, enabled):
        """Arm or disarm dragging the direction out on the image."""
        self.drawing_direction = enabled
        if enabled:
            self.status.showMessage('Drag on the image to set the gradient direction.')
        else:
            self.status.showMessage('')

    def show_direction(self, start, end):
        """Draw the direction as a line between two ``(row, col)`` points."""
        self.direction_line.setData([start[1], end[1]], [start[0], end[0]])

    def set_direction_from_drag(self, start, end):
        """Adopt a dragged line as the gradient direction.

        The line stays where it was drawn rather than snapping to the middle of
        the frame, because where you dragged it is usually the feature you were
        pointing at.
        """
        self.direction_button.setChecked(False)     # one drag sets it once
        self.set_direction(end[0] - start[0], end[1] - start[1], line=(start, end))

    def set_direction(self, drow, dcol, line=None):
        """Set the gradient direction to a ``(row, col)`` vector, normalised.

        :param drow: row component
        :type drow: float
        :param dcol: column component
        :type dcol: float
        :param line: the two ``(row, col)`` endpoints to draw, or ``None`` to
            draw the vector through the middle of the frame
        :type line: tuple or None
        """
        norm = float(np.hypot(drow, dcol))
        if norm < 1e-9 or len(self.direction_spins) != 2:
            return
        unit = (drow / norm, dcol / norm)

        # Both components are written before the score is redefined, so a
        # direction costs one evaluation rather than one per component.
        previous, self._syncing = self._syncing, True
        for spin, value in zip(self.direction_spins, unit):
            spin.setValue(value)
        self._syncing = previous

        if line is None:
            rows, cols = self.pipeline.shape
            span = min(rows, cols) / 4.0
            centre = ((rows - 1) / 2.0, (cols - 1) / 2.0)
            line = ((centre[0] - unit[0] * span, centre[1] - unit[1] * span),
                    (centre[0] + unit[0] * span, centre[1] + unit[1] * span))
        self.show_direction(*line)
        self._redefine_score()

    def _redefine_score(self):
        """Re-declare the score from the current evaluator and parameters.

        Declaring costs nothing -- the store computes on request -- so this goes
        through the same coalescing every other control uses. That matters more
        here than anywhere else: this is the one control that can make a redraw
        expensive, since a parameter it has not scored before is a whole-frame
        evaluation, and a spin box dragged through sixty values would otherwise
        queue sixty of them.
        """
        if self._syncing:
            return
        params = {name: getter() for name, getter in self.param_widgets.items()}
        self.pipeline.define_score('score', self.evaluator_combo.currentData(), **params)
        self.request_refresh()

    # -- selections list ---------------------------------------------------

    def _on_row_changed(self, row):
        if self._syncing:
            return
        self.active_index = row if row >= 0 else None
        entry = self.active_entry()
        if entry is not None and entry.kind != 'brush':
            self.select_tool(entry.kind)
        self.draw_geometry()
        self.draw_highlight()
        self._update_role_button()

    def _on_item_changed(self, item):
        if self._syncing:
            return
        row = self.entry_list.row(item)
        if 0 <= row < len(self.pipeline.entries):
            self.pipeline.entries[row].visible = item.checkState() == QtCore.Qt.CheckState.Checked
            self.refresh()

    def _update_role_button(self):
        entry = self.active_entry()
        self.role_button.setEnabled(entry is not None)
        if entry is not None:
            self.role_button.setText('Use as mask' if entry.role == 'points' else 'Use as points')

    def _refresh_list(self, credited):
        self._syncing = True
        self.entry_list.clear()
        for entry, points in zip(self.pipeline.entries, credited):
            item = QtWidgets.QListWidgetItem(f'{entry.label} — {entry.role} — {len(points)} pts')
            item.setFlags(item.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(QtCore.Qt.CheckState.Checked if entry.visible
                               else QtCore.Qt.CheckState.Unchecked)
            self.entry_list.addItem(item)
        if self.active_index is not None and 0 <= self.active_index < self.entry_list.count():
            self.entry_list.setCurrentRow(self.active_index)
        self._syncing = False
        self._update_role_button()

    # -- drawing -----------------------------------------------------------

    def request_refresh(self):
        """Redraw, collapsing the flood of requests a dragged control produces.

        A slider emits a change per pixel of travel, and a redraw is not always
        cheap enough to keep up. Two behaviours, chosen by measurement rather
        than by a fixed delay:

        While a redraw costs less than a frame, it happens immediately, so the
        display tracks the control exactly -- which is the whole appeal of a
        live slider and is worth nothing to defer.

        Once it costs more, requests are *coalesced*: the first one schedules a
        redraw for the moment the event queue next drains, and every request
        arriving before then is absorbed into it. Nothing queues up, so a fast
        drag does not repaint every position on the way -- it repaints as often
        as it can and always with the value the control is on *now*, which is
        where it stops. A slow drag drains the queue between steps and still
        redraws at every one.
        """
        if self._last_refresh_ms <= REDRAW_BUDGET_MS:
            self.refresh()
        elif not self._refresh_timer.isActive():
            self._refresh_timer.start(0)

    def flush_refresh(self):
        """Run a coalesced redraw now, if one is pending."""
        if self._refresh_timer.isActive():
            self.refresh()

    def refresh(self):
        """Re-run the pipeline once and redraw everything that depends on it.

        Once, not three times: the total, the per-row counts and the highlight
        all come from the same pass, and at twenty thousand points each pass is
        tens of milliseconds you would otherwise pay three times over on every
        step of a slider drag.
        """
        self._refresh_timer.stop()
        started = time.perf_counter()
        self._points, self._credited = self.pipeline.points_and_credits()
        self._refresh_list(self._credited)
        self.count_label.setText(f'{len(self._points)} points')
        self.draw_points()
        self.draw_candidates()
        self.draw_geometry()
        self.draw_highlight()
        self._refresh_select_note()
        # Part of the redraw, and timed with it: the overlay is drawn from the
        # score, so a change of evaluator has to reach it too -- and its cost is
        # exactly the sort the coalescing above exists to notice.
        if self.show_score.isChecked():
            self.draw_score()
        self._last_refresh_ms = (time.perf_counter() - started) * 1000.0

    def draw_points(self):
        """Draw the selected points and, optionally, their subset rectangles."""
        points = getattr(self, '_points', None)
        if points is None or not len(points):
            self.point_scatter.clear()
            self.doomed_scatter.clear()
            self.clear_subset_rectangles()
            return
        # +0.5 puts the marker at the pixel centre rather than its top-left corner.
        self.point_scatter.setData(pos=points[:, ::-1] + 0.5)
        self.draw_doomed()
        if self.show_subsets.isChecked():
            height, width = self.pipeline.subset_size
            self.draw_subset_rectangles(points, height // 2, width // 2)
        else:
            self.clear_subset_rectangles()

    def draw_doomed(self):
        """Cross out the points the deselect stroke has covered so far.

        Drawn *over* the red points rather than swapped for them, so a stroke in
        progress costs only the handful of points it has reached: replacing the
        cloud would mean handing every one of tens of thousands of positions back
        to the scatter item on every mouse move, which is what made a long stroke
        drag behind the cursor.
        """
        points = getattr(self, '_points', None)
        if points is None or not len(points) or self._paint is None or not self.deselect_mode:
            self.doomed_scatter.clear()
            return
        doomed = self._paint[points[:, 0], points[:, 1]]
        if not doomed.any():
            self.doomed_scatter.clear()
            return
        self.doomed_scatter.setData(pos=points[doomed][:, ::-1] + 0.5)

    def draw_candidates(self):
        """Show what the mask is leaving out, on the mask step.

        Three tiers while masking, because "no point here" is otherwise
        ambiguous: dim grey for a feature the mask excludes, the ordinary red
        for a point that is being taken, and the magenta ring
        (:meth:`draw_highlight`) for the ones the selected row accounts for.

        The candidates are the whole-frame selection, so they do not move while
        a mask is edited -- painting a region turns points from grey to red
        where it lands rather than re-selecting underneath you. The consequence
        is that a grey point near the edge of a mask need not coincide exactly
        with a red one, since a selection inside a region starts its separation
        afresh.
        """
        if self.step != STEP_MASK:
            self.candidate_scatter.clear()
            return
        candidates = self.pipeline.candidate_points()
        outside = (~self.pipeline.mask[candidates[:, 0], candidates[:, 1]]
                   if len(candidates) else np.zeros(0, dtype=bool))
        if not outside.any():
            self.candidate_scatter.clear()
            return
        self.candidate_scatter.setData(pos=candidates[outside][:, ::-1] + 0.5)

    def clear_subset_rectangles(self):
        """Remove both halves of the subset-rectangle display."""
        self.roi_overlay.clear()
        self.roi_outline.setPath(QtGui.QPainterPath())

    def draw_subset_rectangles(self, points, half_h, half_w):
        """Draw each subset as a translucent fill plus a hairline border.

        The two halves are drawn by different means because each is cheap in a
        different way. The fill goes into one RGBA image, which costs a single
        upload however many subsets there are. The borders go into one
        ``QPainterPath`` stroked with a cosmetic pen, whose width is in screen
        pixels -- that is what keeps them a hairline at any zoom, where a raster
        border is pinned to one image pixel and becomes a band when zoomed in.

        Both are built with whole-array numpy rather than a loop over the points,
        which is what keeps the redraw quick for tens of thousands of subsets.

        :param points: subset centres, as an ``(n, 2)`` array of ``(row, col)``
        :type points: numpy.ndarray
        :param half_h: half the subset height, in pixels
        :type half_h: int
        :param half_w: half the subset width, in pixels
        :type half_w: int
        """
        n_rows, n_cols = self.pipeline.shape
        span_r, span_c = 2 * half_h + 1, 2 * half_w + 1

        r0 = points[:, 0].astype(int) - half_h
        c0 = points[:, 1].astype(int) - half_w
        inside = (r0 >= 0) & (c0 >= 0) & (r0 + span_r <= n_rows) & (c0 + span_c <= n_cols)
        r0, c0 = r0[inside], c0[inside]
        if not len(r0):
            self.clear_subset_rectangles()
            return

        # Mark every covered pixel at once by broadcasting the per-subset index
        # ranges against each other, giving an (n, span_r, span_c) fancy index.
        covered = np.zeros((n_rows, n_cols), dtype=bool)
        covered[(r0[:, None] + np.arange(span_r))[:, :, None],
                (c0[:, None] + np.arange(span_c))[:, None, :]] = True
        overlay = np.zeros((n_rows, n_cols, 4), dtype=np.uint8)
        overlay[..., 1] = covered * np.uint8(180)
        overlay[..., 3] = covered * np.uint8(40)
        self.roi_overlay.setImage(overlay, autoLevels=False)

        # Five corners per rectangle (the first repeated to close it) separated by
        # a nan, which is how arrayToQPath is told to start a new sub-path.
        top, left = r0.astype(float), c0.astype(float)
        bottom, right = top + span_r, left + span_c
        xs = np.empty((len(left), 6))
        ys = np.empty((len(top), 6))
        xs[:, 0] = xs[:, 3] = xs[:, 4] = left
        xs[:, 1] = xs[:, 2] = right
        ys[:, 0] = ys[:, 1] = ys[:, 4] = top
        ys[:, 2] = ys[:, 3] = bottom
        xs[:, 5] = ys[:, 5] = np.nan
        self.roi_outline.setPath(pg.arrayToQPath(xs.ravel(), ys.ravel(), connect='finite'))

    def draw_geometry(self):
        """Outline the active vertex-based entry and show its vertices."""
        entry = self.active_entry()
        if self.step != STEP_MASK or entry is None or entry.kind not in VERTEX_KINDS or not entry.geometry:
            self.geometry_line.clear()
            self.geometry_vertices.clear()
            return
        vertices = np.asarray(entry.geometry, dtype=float)
        closed = np.vstack([vertices, vertices[:1]]) if entry.kind == 'polygon' else vertices
        self.geometry_line.setData(closed[:, 1], closed[:, 0])
        self.geometry_vertices.setData(pos=vertices[:, ::-1])

    def draw_highlight(self):
        """Ring the points belonging to the active row."""
        if self.step != STEP_MASK or self.active_index is None:
            self.highlight_scatter.clear()
            return
        credited = getattr(self, '_credited', None) or self.pipeline.points_by_entry()
        points = credited[self.active_index] if self.active_index < len(credited) else []
        if not len(points):
            self.highlight_scatter.clear()
            return
        array = np.asarray(points, dtype=float)
        self.highlight_scatter.setData(pos=array[:, ::-1] + 0.5)

    def draw_brush(self):
        """Show the stroke currently being painted.

        The path is filled with the winding rule, so the dabs read as one
        translucent stroke rather than as a chain of discs darkening where they
        overlap.
        """
        if self._paint is None:
            self.brush_overlay.setPath(QtGui.QPainterPath())
            return
        colour = (255, 0, 0, 80) if self.deselect_mode else (0, 200, 255, 80)
        self.brush_overlay.setBrush(pg.mkBrush(*colour))
        self.brush_overlay.setPath(self._stroke_path)

    def draw_score(self):
        """Show the current score image as a heatmap, transparent where invalid."""
        score = self.pipeline.store.get(self.pipeline.ensure_default_score())
        finite = np.isfinite(score)
        rgba = np.zeros((*score.shape, 4), dtype=np.uint8)
        if finite.any():
            values = score[finite]
            low, high = float(values.min()), float(values.max())
            normalised = np.zeros(score.shape, dtype=float)
            if high > low:
                normalised[finite] = (values - low) / (high - low)
            colours = pg.colormap.get('viridis').map(normalised, mode='byte')
            rgba[finite] = colours[finite]
            # The NaN border keeps alpha 0, so it reads as "not scored" rather
            # than as a region that scored badly.
            rgba[..., 3] = np.where(finite, 150, 0)
        self.score_overlay.setImage(rgba, autoLevels=False)

    def _refresh_select_note(self):
        """Say what is limiting the selection, when something is.

        The point cap in particular has no other symptom: it just stops adding
        points, and the result reads as though the threshold or the separation
        did it.
        """
        if not self.pipeline.mask.any():
            self.select_note.setText('No region acts as a mask, so nothing is selected. '
                                     'Draw one in the Mask tab, or set a row to "mask".')
        elif len(getattr(self, '_points', ())) >= self.max_points_spin.value():
            self.select_note.setText(f'Stopped at the cap of {self.max_points_spin.value()} points. '
                                     'Raise the separation or the threshold to see the '
                                     'whole selection.')
        else:
            self.select_note.setText('')

    # -- results -----------------------------------------------------------

    def get_points(self):
        """Run the pipeline and return the points.

        :return: ``(n_points, 2)`` integer array in ``(row, col)`` order, ready
            for ``IDIMethod.set_points()``
        :rtype: numpy.ndarray
        """
        return self.pipeline.get_points()

    @property
    def points(self):
        """The selected points, as :meth:`get_points` returns them.

        :rtype: numpy.ndarray
        """
        return self.get_points()
