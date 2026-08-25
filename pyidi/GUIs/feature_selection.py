"""Interactive front end for the mask -> evaluate -> select pipeline.

``FeatureSelectionGUI`` drives :mod:`pyidi.selection`. The three steps of the
pipeline are the three steps of the interface, in the vocabulary settled in
issue #51: draw regions, score the image, pick the points.

The division of labour matters for how this feels to use. Only the evaluate
step is expensive, and it depends on nothing but the frame, the evaluator and
the subset size. Everything else -- painting a mask, dragging a polygon vertex,
moving the threshold slider, changing the minimum distance -- re-derives the
points from a cached score image, so it updates while the control is still
moving. Changing the subset size or the evaluator is the only thing that pays
for a recomputation.

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

import numpy as np
import pyqtgraph as pg
from PyQt6 import QtCore, QtGui, QtWidgets
from pyqtgraph import GraphicsLayoutWidget, ImageItem, ScatterPlotItem

from ..selection import (
    DEFAULT_MAX_POINTS,
    SELECTORS,
    SelectionPipeline,
    available_evaluators,
)
from ..selection_geometry import _as_size_pair

#: Grab radius, in screen pixels, within which a drag hits an existing vertex.
#: Screen rather than image pixels so hit-testing feels the same at any zoom.
VERTEX_GRAB_RADIUS_PX = 10

#: The three steps, in order. Also the toolbar button labels.
STEPS = ('Mask', 'Evaluate', 'Select')

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

    def _scene_to_rc(self, pos):
        """A scene position as ``(row, col)`` floats, or ``None`` if off-image."""
        if not self.sceneBoundingRect().contains(pos):
            return None
        point = self.mapSceneToView(pos)
        return point.y(), point.x()

    def _start_vertex_drag(self, ev):
        """Grab a vertex if the drag began near one; otherwise let the view pan."""
        gui = self.parent_gui
        if gui.step != 'Mask' or gui.tool not in VERTEX_KINDS:
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
        if not (gui.step == 'Mask' and gui.tool == 'brush'
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
        if gui.step == 'Mask' and gui.tool == 'brush':
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
        height, width = _as_size_pair(subset_size)
        self.pipeline = SelectionPipeline(self.frame, (int(height), int(width)), subset_overlap)
        self.pipeline.define_score('score', 'shi_tomasi')

        self.step = 'Mask'
        self.tool = 'polygon'
        self.deselect_mode = False
        self._paint = None
        self._syncing = False
        self.active_index = None
        self.undo_stack = []
        self.undo_limit = 50

        QtGui.QShortcut(QtGui.QKeySequence.StandardKey.Undo, self).activated.connect(self.undo)

        self._build_ui()
        self.image_item.setImage(self.frame)
        self.select_step('Mask')
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
        self.status.showMessage('Draw a region to begin.')

    def _build_step_toolbar(self):
        bar = QtWidgets.QWidget()
        row = QtWidgets.QHBoxLayout(bar)
        row.setContentsMargins(5, 4, 5, 4)
        self.step_buttons = {}
        group = QtWidgets.QButtonGroup(self)
        group.setExclusive(True)
        for index, name in enumerate(STEPS):
            button = QtWidgets.QPushButton(f'{index + 1}. {name}')
            button.setCheckable(True)
            button.setMinimumWidth(110)
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
        self.brush_overlay = ImageItem(axisOrder='row-major')

        # The subset borders live apart from the translucent fill so they can be
        # stroked with a *cosmetic* pen, whose width is in screen pixels: they stay
        # a hairline at any zoom, where a raster border cannot go below one image
        # pixel and becomes a thick band as soon as you zoom in.
        self.roi_outline = QtWidgets.QGraphicsPathItem()
        pen = pg.mkPen(0, 255, 0, 150)
        pen.setCosmetic(True)
        self.roi_outline.setPen(pen)
        self.roi_outline.setBrush(pg.mkBrush(None))

        self.geometry_line = pg.PlotDataItem(pen=pg.mkPen('y', width=2))
        self.geometry_vertices = ScatterPlotItem(pen=pg.mkPen(None), brush=pg.mkBrush(255, 255, 0, 200), size=7)
        self.point_scatter = ScatterPlotItem(pen=pg.mkPen(None), brush=pg.mkBrush(255, 100, 100, 220), size=7)
        self.highlight_scatter = ScatterPlotItem(
            pen=pg.mkPen(255, 0, 255, 230, width=2), brush=pg.mkBrush(None), size=13)

        for item, z in ((self.image_item, 0), (self.score_overlay, 0.5), (self.roi_overlay, 1),
                        (self.roi_outline, 1), (self.geometry_line, 2), (self.geometry_vertices, 2),
                        (self.point_scatter, 3), (self.highlight_scatter, 3), (self.brush_overlay, 4)):
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
        for name, builder in (('Mask', self._build_mask_page),
                              ('Evaluate', self._build_evaluate_page),
                              ('Select', self._build_select_page)):
            page = QtWidgets.QWidget()
            builder(QtWidgets.QVBoxLayout(page))
            self.step_stack.addWidget(page)
            self.step_pages[name] = page
        column.addLayout(self.step_stack, stretch=1)

        column.addWidget(self._build_selection_list())

        self.count_label = QtWidgets.QLabel('0 points')
        font = self.count_label.font()
        font.setBold(True)
        self.count_label.setFont(font)
        column.addWidget(self.count_label)

        panel.setMinimumWidth(260)
        panel.setMaximumWidth(600)
        self.splitter.addWidget(panel)
        self.splitter.setStretchFactor(0, 5)
        self.splitter.setStretchFactor(1, 0)
        self.splitter.setSizes([950, 300])

    def _build_mask_page(self, layout):
        tools = QtWidgets.QGroupBox('Region tool')
        tool_column = QtWidgets.QVBoxLayout(tools)
        self.tool_buttons = {}
        group = QtWidgets.QButtonGroup(self)
        group.setExclusive(True)
        for label, kind in TOOLS:
            button = QtWidgets.QPushButton(label)
            button.setCheckable(True)
            group.addButton(button)
            tool_column.addWidget(button)
            button.clicked.connect(lambda _, k=kind: self.select_tool(k))
            self.tool_buttons[kind] = button
        self.tool_buttons['polygon'].setChecked(True)
        layout.addWidget(tools)

        self.new_entry_button = QtWidgets.QPushButton('Start new polygon')
        self.new_entry_button.clicked.connect(self.start_new_entry)
        layout.addWidget(self.new_entry_button)

        brush = QtWidgets.QGroupBox('Brush')
        brush_column = QtWidgets.QVBoxLayout(brush)
        self.brush_radius = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.brush_radius.setRange(1, 100)
        self.brush_radius.setValue(12)
        brush_column.addWidget(QtWidgets.QLabel('Radius (hold Ctrl and drag to paint)'))
        brush_column.addWidget(self.brush_radius)
        self.deselect_button = QtWidgets.QPushButton('Deselect painted area')
        self.deselect_button.setCheckable(True)
        self.deselect_button.toggled.connect(self._set_deselect_mode)
        brush_column.addWidget(self.deselect_button)
        layout.addWidget(brush)

        layout.addWidget(self._build_subset_group())

        clear = QtWidgets.QPushButton('Clear all')
        clear.clicked.connect(self.clear_all)
        layout.addWidget(clear)
        layout.addStretch(1)

    def _build_subset_group(self):
        box = QtWidgets.QGroupBox('Subset')
        grid = QtWidgets.QGridLayout(box)
        height, width = self.pipeline.subset_size

        self.square_check = QtWidgets.QCheckBox('Square subsets')
        self.square_check.setChecked(height == width)
        self.square_check.toggled.connect(self._on_square_toggled)
        grid.addWidget(self.square_check, 0, 0, 1, 2)

        self.height_spin = QtWidgets.QSpinBox()
        self.width_spin = QtWidgets.QSpinBox()
        for spin, value in ((self.height_spin, height), (self.width_spin, width)):
            spin.setRange(3, 501)
            spin.setValue(value)
            spin.valueChanged.connect(self._on_subset_size_changed)
        self.width_spin.setEnabled(height != width)
        grid.addWidget(QtWidgets.QLabel('Height'), 1, 0)
        grid.addWidget(self.height_spin, 1, 1)
        grid.addWidget(QtWidgets.QLabel('Width'), 2, 0)
        grid.addWidget(self.width_spin, 2, 1)

        self.spacing_spin = QtWidgets.QSpinBox()
        self.spacing_spin.setRange(-500, 500)
        self.spacing_spin.setValue(self.pipeline.spacing)
        self.spacing_spin.valueChanged.connect(self._on_spacing_changed)
        self.spacing_spin.setToolTip(
            'Extra spacing between the points a "points" row lays out. Rows that '
            'act as a mask are unaffected: their spacing comes from the selector.')
        grid.addWidget(QtWidgets.QLabel('Point spacing'), 3, 0)
        grid.addWidget(self.spacing_spin, 3, 1)

        self.show_subsets = QtWidgets.QCheckBox('Show subsets')
        self.show_subsets.setChecked(True)
        self.show_subsets.toggled.connect(lambda _: self.draw_points())
        grid.addWidget(self.show_subsets, 4, 0, 1, 2)
        return box

    def _build_evaluate_page(self, layout):
        layout.addWidget(QtWidgets.QLabel('Score every subset position in the image.'))

        self.evaluator_combo = QtWidgets.QComboBox()
        self.evaluators = available_evaluators()
        for name, spec in sorted(self.evaluators.items()):
            self.evaluator_combo.addItem(spec.display_name, name)
        self.evaluator_combo.setCurrentIndex(self.evaluator_combo.findData('shi_tomasi'))
        self.evaluator_combo.currentIndexChanged.connect(self._on_evaluator_changed)
        layout.addWidget(self.evaluator_combo)

        self.param_box = QtWidgets.QGroupBox('Parameters')
        self.param_layout = QtWidgets.QFormLayout(self.param_box)
        layout.addWidget(self.param_box)
        self.param_widgets = {}
        self._rebuild_param_widgets()

        self.show_score = QtWidgets.QCheckBox('Show score overlay')
        self.show_score.toggled.connect(self._on_show_score_toggled)
        layout.addWidget(self.show_score)

        self.evaluate_note = QtWidgets.QLabel('')
        self.evaluate_note.setWordWrap(True)
        layout.addWidget(self.evaluate_note)
        layout.addStretch(1)

    def _build_select_page(self, layout):
        layout.addWidget(QtWidgets.QLabel('Pick the points from the score and the mask.'))

        self.selector_combo = QtWidgets.QComboBox()
        for name in sorted(SELECTORS):
            self.selector_combo.addItem(name, name)
        self.selector_combo.setCurrentIndex(self.selector_combo.findData('peaks'))
        self.selector_combo.currentIndexChanged.connect(self._on_selector_changed)
        layout.addWidget(self.selector_combo)

        box = QtWidgets.QGroupBox('Selection')
        form = QtWidgets.QFormLayout(box)

        self.threshold_mode = QtWidgets.QComboBox()
        self.threshold_mode.addItem('percentile of scores', 'percentile')
        self.threshold_mode.addItem('fraction of maximum', 'fraction')
        self.threshold_mode.currentIndexChanged.connect(self._on_threshold_mode_changed)
        form.addRow('Threshold by', self.threshold_mode)

        self.threshold_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.threshold_slider.setRange(0, 1000)
        self.threshold_slider.setValue(900)
        self.threshold_slider.valueChanged.connect(self._on_threshold_changed)
        self.threshold_label = QtWidgets.QLabel('90.0')
        form.addRow(self.threshold_label, self.threshold_slider)

        self.min_distance_spin = QtWidgets.QSpinBox()
        self.min_distance_spin.setRange(0, 500)
        self.min_distance_spin.setValue(self.pipeline.selector_params['min_distance'])
        self.min_distance_spin.valueChanged.connect(self._on_min_distance_changed)
        self.min_distance_row = 'Minimum distance'
        form.addRow(self.min_distance_row, self.min_distance_spin)

        self.pitch_spin = QtWidgets.QSpinBox()
        self.pitch_spin.setRange(1, 500)
        self.pitch_spin.setValue(12)
        self.pitch_spin.valueChanged.connect(self._on_pitch_changed)
        form.addRow('Grid pitch', self.pitch_spin)

        self.max_points_spin = QtWidgets.QSpinBox()
        self.max_points_spin.setRange(1, 200000)
        self.max_points_spin.setValue(DEFAULT_MAX_POINTS)
        self.max_points_spin.valueChanged.connect(self._on_max_points_changed)
        form.addRow('Maximum points', self.max_points_spin)

        layout.addWidget(box)
        self.select_note = QtWidgets.QLabel('')
        self.select_note.setWordWrap(True)
        layout.addWidget(self.select_note)
        layout.addStretch(1)
        self._update_selector_rows()

    def _build_selection_list(self):
        box = QtWidgets.QGroupBox('Selections')
        column = QtWidgets.QVBoxLayout(box)
        self.entry_list = QtWidgets.QListWidget()
        self.entry_list.currentRowChanged.connect(self._on_row_changed)
        self.entry_list.itemChanged.connect(self._on_item_changed)
        column.addWidget(self.entry_list)

        buttons = QtWidgets.QHBoxLayout()
        self.role_button = QtWidgets.QPushButton('Use as points')
        self.role_button.clicked.connect(self.toggle_role)
        buttons.addWidget(self.role_button)
        delete = QtWidgets.QPushButton('Delete')
        delete.clicked.connect(self.delete_active)
        buttons.addWidget(delete)
        column.addLayout(buttons)
        return box

    # -- steps and tools ---------------------------------------------------

    def select_step(self, name):
        """Switch to one of the three steps.

        :param name: ``'Mask'``, ``'Evaluate'`` or ``'Select'``
        :type name: str
        """
        self.step = name
        self.step_buttons[name].setChecked(True)
        self.step_stack.setCurrentWidget(self.step_pages[name])
        # The magenta ring marks the active row's points while regions are being
        # drawn; in the other steps it would ring points nobody is editing.
        self.highlight_scatter.setVisible(name == 'Mask')
        self.geometry_line.setVisible(name == 'Mask')
        self.geometry_vertices.setVisible(name == 'Mask')
        if name == 'Evaluate':
            self._refresh_evaluate_note()
        elif name == 'Select':
            self._refresh_select_note()
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
        """Drop every selection."""
        self.push_undo({'type': 'restore', 'entries': list(self.pipeline.entries)})
        self.pipeline.entries = []
        self.active_index = None
        self.refresh()

    # -- mouse -------------------------------------------------------------

    def on_mouse_click(self, event):
        """Route a click on the image to whichever tool is active."""
        if self.step != 'Mask' or event.button() != QtCore.Qt.MouseButton.LeftButton:
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
        credited = self.pipeline.points_by_entry()
        best, best_entry, best_distance = None, None, np.inf
        for entry, points in zip(self.pipeline.entries, credited):
            for point in points:
                distance = np.hypot(point[0] - position[0], point[1] - position[1])
                if distance < best_distance:
                    best, best_entry, best_distance = point, entry, distance
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
        self.draw_brush()

    def brush_end(self):
        """Commit the stroke, either as a new region or as a deselection."""
        if self._paint is None:
            return
        stroke = self._paint
        self._paint = None
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
        kind = action['type']
        if kind == 'vertex_add':
            if action['entry'].geometry:
                action['entry'].geometry.pop()
        elif kind == 'vertex_move':
            action['entry'].geometry[action['index']] = action['original']
        elif kind == 'add':
            self.pipeline.remove_entry(action['entry'])
        elif kind == 'delete':
            self.pipeline.entries.insert(action['index'], action['entry'])
        elif kind == 'restore':
            self.pipeline.entries = list(action['entries'])
            for entry, erased, geometry, removed in action.get('state', []):
                entry.erased, entry.geometry, entry.removed = erased, geometry, removed
        self.active_index = min(self.active_index or 0, len(self.pipeline.entries) - 1)
        if self.active_index < 0:
            self.active_index = None
        self.refresh()

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
        self.refresh()

    def _on_spacing_changed(self, value):
        self.pipeline.spacing = value
        self.refresh()

    def _on_evaluator_changed(self, *_):
        self._rebuild_param_widgets()
        self._redefine_score()

    def _on_show_score_toggled(self, checked):
        self.score_overlay.setVisible(checked)
        if checked:
            self.draw_score()

    def _on_selector_changed(self, *_):
        self.pipeline.selector = self.selector_combo.currentData()
        self._update_selector_rows()
        self.refresh()

    def _on_threshold_mode_changed(self, *_):
        self.pipeline.selector_params['threshold_mode'] = self.threshold_mode.currentData()
        self._on_threshold_changed(self.threshold_slider.value())

    def _on_threshold_changed(self, value):
        mode = self.threshold_mode.currentData()
        threshold = value / 10.0 if mode == 'percentile' else value / 1000.0
        self.threshold_label.setText(f'{threshold:.1f}' if mode == 'percentile' else f'{threshold:.3f}')
        self.pipeline.selector_params['threshold'] = threshold
        self.refresh()

    def _on_min_distance_changed(self, value):
        self.pipeline.selector_params['min_distance'] = value
        self.refresh()

    def _on_pitch_changed(self, value):
        self.pipeline.selector_params['pitch'] = value
        self.refresh()

    def _on_max_points_changed(self, value):
        self.pipeline.selector_params['max_points'] = value
        self.refresh()

    def _update_selector_rows(self):
        peaks = self.selector_combo.currentData() == 'peaks'
        self.min_distance_spin.setEnabled(peaks)
        self.pitch_spin.setEnabled(not peaks)

    def _rebuild_param_widgets(self):
        """Build the evaluator's parameter controls from its descriptors.

        Nothing here knows what a Shi-Tomasi or a gradient-direction parameter
        is: the registry says a parameter is a float, an int or a direction, and
        that is enough to make a widget for it. Adding an evaluator therefore
        needs no change to this module.
        """
        while self.param_layout.count():
            item = self.param_layout.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()
        self.param_widgets = {}

        spec = self.evaluators[self.evaluator_combo.currentData()]
        for parameter in spec.parameters:
            if parameter.kind == 'direction':
                widget, getter = self._direction_widget(parameter)
            else:
                widget, getter = self._number_widget(parameter)
            widget.setToolTip(parameter.description)
            self.param_layout.addRow(parameter.name.replace('_', ' ').capitalize(), widget)
            self.param_widgets[parameter.name] = getter
        self.param_box.setVisible(bool(spec.parameters))

    def _number_widget(self, parameter):
        spin = QtWidgets.QDoubleSpinBox() if parameter.kind == 'float' else QtWidgets.QSpinBox()
        spin.setRange(parameter.minimum if parameter.minimum is not None else -1e9,
                      parameter.maximum if parameter.maximum is not None else 1e9)
        spin.setValue(parameter.default)
        spin.valueChanged.connect(lambda _: self._redefine_score())
        return spin, spin.value

    def _direction_widget(self, parameter):
        widget = QtWidgets.QWidget()
        row = QtWidgets.QHBoxLayout(widget)
        row.setContentsMargins(0, 0, 0, 0)
        spins = []
        for value in parameter.default:
            spin = QtWidgets.QDoubleSpinBox()
            spin.setRange(-1e6, 1e6)
            spin.setValue(float(value))
            spin.valueChanged.connect(lambda _: self._redefine_score())
            row.addWidget(spin)
            spins.append(spin)
        for label, vector in (('X', (0.0, 1.0)), ('Y', (1.0, 0.0))):
            button = QtWidgets.QPushButton(label)
            button.setMaximumWidth(30)
            button.clicked.connect(lambda _, v=vector, s=spins: [s[i].setValue(v[i]) for i in (0, 1)])
            row.addWidget(button)
        return widget, (lambda s=spins: (s[0].value(), s[1].value()))

    def _redefine_score(self):
        """Re-declare the score from the current evaluator and parameters."""
        params = {name: getter() for name, getter in self.param_widgets.items()}
        self.pipeline.define_score('score', self.evaluator_combo.currentData(), **params)
        self.refresh()

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

    def refresh(self):
        """Re-run the pipeline and redraw everything that depends on it."""
        credited = self.pipeline.points_by_entry()
        self._points = self.pipeline.get_points()
        self._refresh_list(credited)
        self.count_label.setText(f'{len(self._points)} points')
        self.draw_points()
        self.draw_geometry()
        self.draw_highlight()
        if self.step == 'Select':
            self._refresh_select_note()
        # Part of the redraw: the overlay is drawn from the score, so a change
        # of evaluator or of subset size has to reach it too.
        if self.show_score.isChecked():
            self.draw_score()

    def draw_points(self):
        """Draw the selected points and, optionally, their subset rectangles."""
        points = getattr(self, '_points', None)
        if points is None or not len(points):
            self.point_scatter.clear()
            self.clear_subset_rectangles()
            return
        # +0.5 puts the marker at the pixel centre rather than its top-left corner.
        self.point_scatter.setData(pos=points[:, ::-1] + 0.5)
        if self.show_subsets.isChecked():
            height, width = self.pipeline.subset_size
            self.draw_subset_rectangles(points, height // 2, width // 2)
        else:
            self.clear_subset_rectangles()

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
        if self.step != 'Mask' or entry is None or entry.kind not in VERTEX_KINDS or not entry.geometry:
            self.geometry_line.clear()
            self.geometry_vertices.clear()
            return
        vertices = np.asarray(entry.geometry, dtype=float)
        closed = np.vstack([vertices, vertices[:1]]) if entry.kind == 'polygon' else vertices
        self.geometry_line.setData(closed[:, 1], closed[:, 0])
        self.geometry_vertices.setData(pos=vertices[:, ::-1])

    def draw_highlight(self):
        """Ring the points belonging to the active row."""
        if self.step != 'Mask' or self.active_index is None:
            self.highlight_scatter.clear()
            return
        credited = self.pipeline.points_by_entry()
        points = credited[self.active_index] if self.active_index < len(credited) else []
        if not points:
            self.highlight_scatter.clear()
            return
        array = np.asarray(points, dtype=float)
        self.highlight_scatter.setData(pos=array[:, ::-1] + 0.5)

    def draw_brush(self):
        """Show the stroke currently being painted."""
        if self._paint is None:
            self.brush_overlay.clear()
            return
        rgba = np.zeros((*self._paint.shape, 4), dtype=np.uint8)
        rgba[self._paint] = [255, 0, 0, 80] if self.deselect_mode else [0, 200, 255, 80]
        self.brush_overlay.setImage(rgba, autoLevels=False)

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

    def _refresh_evaluate_note(self):
        spec = self.evaluators[self.evaluator_combo.currentData()]
        self.evaluate_note.setText(
            f'{spec.description}\n\nScored over the whole image for a '
            f'{self.pipeline.subset_size[0]}x{self.pipeline.subset_size[1]} subset. '
            'Changing the subset size or a parameter recomputes it; everything else does not.')

    def _refresh_select_note(self):
        if not self.pipeline.mask.any():
            self.select_note.setText('No region acts as a mask, so nothing is selected. '
                                     'Draw one in the Mask step, or set a row to "mask".')
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
