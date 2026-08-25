import sys
import numpy as np
from PyQt6 import QtWidgets, QtCore, QtGui
from pyqtgraph import GraphicsLayoutWidget, ImageItem, ScatterPlotItem
import pyqtgraph as pg
# import pyidi  # Assuming pyidi is a custom module for video handling

from ..selection_geometry import points_along_polygon, rois_inside_polygon, rois_inside_mask, _as_size_pair

#: Grab radius (in screen pixels) within which a click/drag is considered to hit an
#: existing grid/polyline vertex. Kept constant in screen space so hit-testing feels
#: the same regardless of the current zoom level.
VERTEX_GRAB_RADIUS_PX = 10

#: Row-label prefix for each non-manual selection-entry kind, used by
#: ``SelectionGUI.add_selection`` to generate monotonic labels ("Grid 3", ...).
PRETTY = {'line': 'Line', 'grid': 'Grid', 'brush': 'Brush'}


class BrushViewBox(pg.ViewBox):
    def __init__(self, parent_gui, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setMouseMode(self.PanMode)
        self.parent_gui = parent_gui
        self._dragging_vertex = None  # set while a vertex drag is in progress

    def _vertex_drag_container(self):
        """Return the (entries, kind) vertex-list container for the active selection method.

        Only the "Grid" and "Along the line" methods support vertex dragging. The
        returned ``entries`` list is filtered from ``gui.selections`` but holds the
        same (mutable) entry dicts, so edits through it are visible in
        ``gui.selections`` too.

        :return: the list of live selection entries of the active kind and a kind
            tag ("grid" or "line"), or (None, None) if the current mode/method does
            not support vertex dragging.
        :rtype: tuple
        """
        gui = self.parent_gui
        if gui.mode != "selection":
            return None, None
        kind = gui.current_kind()
        if kind not in ('grid', 'line'):
            return None, None
        entries = [e for e in gui.selections if e['kind'] == kind and e['visible']]
        return entries, kind

    def _start_vertex_drag(self, ev):
        """Hit-test the drag start position against existing vertices; begin a drag if one is hit.

        :param ev: the pyqtgraph mouse-drag event, with ``ev.isStart()`` True
        :return: True if a vertex was grabbed and the event was accepted
        :rtype: bool
        """
        entries, kind = self._vertex_drag_container()
        if entries is None:
            return False
        pos = ev.buttonDownScenePos()
        if not self.sceneBoundingRect().contains(pos):
            return False
        point = self.parent_gui.view.mapSceneToView(pos)
        entry_idx, vertex_idx = self.parent_gui.find_vertex_to_drag(entries, point.x(), point.y())
        if entry_idx is None:
            return False
        entry = entries[entry_idx]
        self._dragging_vertex = {
            'kind': kind,
            'entry': entry,
            'vertex_index': vertex_idx,
            'original_position': entry['geometry'][vertex_idx],
        }
        ev.accept()
        return True

    def _continue_vertex_drag(self, ev):
        """Move the grabbed vertex for a mid-drag or finishing event; commit undo/recompute on finish.

        :param ev: the pyqtgraph mouse-drag event, with ``ev.isStart()`` False
        :return: True if a vertex drag was in progress and the event was consumed
        :rtype: bool
        """
        drag = self._dragging_vertex
        if drag is None:
            return False

        pos = ev.scenePos()
        if self.sceneBoundingRect().contains(pos):
            point = self.parent_gui.view.mapSceneToView(pos)
            drag['entry']['geometry'][drag['vertex_index']] = (point.x(), point.y())
            self.parent_gui.update_geometry_display()

        if ev.isFinish():
            self.parent_gui.push_undo({
                'type': 'move',
                'kind': drag['kind'],
                'entry': drag['entry'],
                'vertex_index': drag['vertex_index'],
                'original_position': drag['original_position'],
            })
            self.parent_gui.recompute_roi_points()
            self._dragging_vertex = None

        ev.accept()
        return True

    def _handle_vertex_drag(self, ev):
        """Handle a plain left-drag that starts on an existing grid/polyline vertex.

        A drag starting near a vertex moves that vertex; a drag starting elsewhere is left
        untouched so the caller falls back to panning the view.

        :param ev: the pyqtgraph mouse-drag event
        :return: True if the event was consumed as a vertex drag
        :rtype: bool
        """
        if ev.isStart():
            return self._start_vertex_drag(ev)
        return self._continue_vertex_drag(ev)

    def mouseClickEvent(self, ev):
        if self.parent_gui.mode == "selection" and self.parent_gui.current_kind() == 'brush':
            if self.parent_gui.ctrl_held:
                ev.accept()
                self.parent_gui.handle_brush_start(ev)
            else:
                ev.ignore()
        else:
            super().mouseClickEvent(ev)

    def _handle_gradient_direction_drag(self, ev):
        """Handle direction-line drag when Filter mode is setting the gradient direction.

        :param ev: the pyqtgraph mouse-drag event
        :return: True if the event was consumed
        :rtype: bool
        """
        if not (self.parent_gui.mode == "filter" and self.parent_gui.setting_direction):
            return False

        pos = ev.scenePos()
        if not self.sceneBoundingRect().contains(pos):
            return False
        point = self.mapSceneToView(pos)

        if ev.isStart():
            self.parent_gui.gradient_direction_points = [(point.x(), point.y())]
            self.parent_gui.gradient_direction_start = (point.x(), point.y())
        elif ev.isFinish():
            if hasattr(self.parent_gui, 'gradient_direction_start'):
                self.parent_gui.gradient_direction_points = [
                    self.parent_gui.gradient_direction_start,
                    (point.x(), point.y())
                ]
                self.parent_gui.compute_direction_vector()
                self.parent_gui.update_direction_line()
                # Toggle off the direction selection mode
                self.parent_gui.direction_button.setChecked(False)
                self.parent_gui.set_gradient_direction_mode()
                self.parent_gui.compute_candidate_points_gradient_direction()
        else:
            # During drag, update the line display
            if hasattr(self.parent_gui, 'gradient_direction_start'):
                temp_points = [
                    self.parent_gui.gradient_direction_start,
                    (point.x(), point.y())
                ]
                xs = [p[0] for p in temp_points]
                ys = [p[1] for p in temp_points]
                self.parent_gui.direction_line.setData(xs, ys)

        ev.accept()
        return True

    def _handle_brush_drag(self, ev):
        """Handle Ctrl+drag brush painting in Selection mode when Brush is the active method.

        :param ev: the pyqtgraph mouse-drag event
        :return: True if the event was consumed
        :rtype: bool
        """
        if not (self.parent_gui.mode == "selection" and self.parent_gui.current_kind() == 'brush'):
            return False
        if not self.parent_gui.ctrl_held:
            return False

        ev.accept()
        if ev.isStart():
            self.parent_gui._painting = True
            self.parent_gui._brush_path = []
            self.parent_gui.handle_brush_start(ev)
        elif ev.isFinish():
            self.parent_gui._painting = False
            self.parent_gui.handle_brush_end(ev)
        else:
            self.parent_gui.handle_brush_move(ev)
        return True

    def mouseDragEvent(self, ev, axis=None):
        if self._handle_gradient_direction_drag(ev):
            return
        if self._handle_brush_drag(ev):
            return
        # Plain left-drag starting on an existing grid/polyline vertex moves that vertex.
        if self._handle_vertex_drag(ev):
            return
        # fallback: pan
        super().mouseDragEvent(ev, axis)

class SelectionGUI(QtWidgets.QMainWindow):
    def __init__(self, video, subset_size=11, subset_overlap=0):
        """Initialize the selection GUI for manual subset selection.

        To extract the points, use the ``get_points`` method or the ``points`` attribute.
        
        Parameters
        ----------
        video : VideoReader or np.ndarray
            The video to be analyzed. If a VideoReader object, it should be initialized with the video file.
            If a np.ndarray, it can be either a single 2-D image (height, width) or a 3-D frame stack
            (n_frames, height, width), in which case the first frame is displayed.
        subset_size : int or (height, width) tuple, optional
            Initial side length (in pixels) of the subset drawn around each selected point. Either a
            single int for a square subset, or a ``(height, width)`` pair for an anisotropic one, where
            ``height`` is the vertical/row extent and ``width`` the horizontal/column extent -- the same
            convention as ``LucasKanade.configure(roi_size=(vertical, horizontal))``, so a value that
            works for one works for the other. Sets the starting values of the "Subset size" spinboxes/
            sliders used when computing ROI rectangles, grid/line spacing and automatic feature
            filtering. Defaults to 11. Normalized and stored as ``self.subset_size``, a ``(height,
            width)`` tuple of ints. The "Square subsets" checkbox starts checked if ``height ==
            width`` (so the width spinbox mirrors the height one) and unchecked otherwise.
        subset_overlap : int, optional
            Initial spacing (in pixels) between neighbouring subsets, used as the "Distance between
            subsets" spinbox/slider value for the Grid, Along the line and Brush selection methods.
            A positive value adds a gap between subsets, a negative value makes them overlap. This is a
            single scalar applied to both axes -- the per-axis step is ``height + subset_overlap`` and
            ``width + subset_overlap``, which is enough to get a sensible anisotropic grid spacing
            without a second overlap control. Defaults to 0.
        """
        app = QtWidgets.QApplication.instance()
        if app is None:
            app = QtWidgets.QApplication([])

        super().__init__()

        self.setWindowTitle("ROI Selection Tool")
        self.resize(1200, 800)

        h, w = _as_size_pair(subset_size)
        self.subset_size = (int(h), int(w))
        self.subset_overlap = subset_overlap

        self._paint_mask = None  # Same shape as the image
        self._paint_radius = 10  # pixels
        self.ctrl_held = False
        self.brush_deselect_mode = False
        self.installEventFilter(self)

        # Bounded undo stack: covers adding a vertex, moving a vertex, and deleting a
        # grid/polyline. Manual points, brush strokes and filter results are not undoable.
        self.undo_stack = []
        self.undo_stack_limit = 50
        self.undo_shortcut = QtGui.QShortcut(QtGui.QKeySequence.StandardKey.Undo, self)
        self.undo_shortcut.activated.connect(self.undo)

        self.gradient_direction_points = []
        self.gradient_direction = None
        self.setting_direction = False

        self.selected_points = []
        self.candidate_points = []
        # The threshold-and-show method of whichever filter produced candidate_points,
        # so a change to the selection can re-derive them; None when no filter is live.
        self._candidate_refresh = None

        # Single ordered list of selection entries, replacing the old parallel
        # manual/line/grid/brush containers -- see add_selection()/entry_points()
        # for the entry schema. Entries are created lazily on first click/stroke,
        # so this starts empty (no "always >= 1 placeholder entry" invariant).
        self.selections = []
        self.active_index = None    # int index into self.selections, or None
        self._label_counters = {'manual': 0, 'line': 0, 'grid': 0, 'brush': 0}
        # Guards against re-entrant QListWidget signal handling: programmatic
        # setCurrentRow()/setText()/setCheckState() calls set this so the
        # corresponding currentRowChanged/itemChanged handlers bail out early.
        self._syncing_list = False

        # Add status bar for instructions
        self.statusBar = self.statusBar()
        self.statusBar.showMessage("Ready. Select a method to begin.")

        # Central widget
        self.central_widget = QtWidgets.QWidget()
        self.setCentralWidget(self.central_widget)
        # Top-level layout for the central widget
        self.main_layout = QtWidgets.QVBoxLayout(self.central_widget)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)

        # Toolbar (fixed height)
        self.mode_toolbar = QtWidgets.QWidget()
        self.mode_toolbar_layout = QtWidgets.QHBoxLayout(self.mode_toolbar)
        self.mode_toolbar_layout.setContentsMargins(5, 4, 5, 4)
        self.mode_toolbar_layout.setSpacing(6)

        self.selection_mode_button = QtWidgets.QPushButton("Select") # Selection mode
        self.filter_mode_button = QtWidgets.QPushButton("Filter") # Filter mode
        for btn in [self.selection_mode_button, self.filter_mode_button]:
            btn.setCheckable(True)
            btn.setMinimumWidth(100)
            self.mode_toolbar_layout.addWidget(btn)

        self.selection_mode_button.setChecked(True)
        self.selection_mode_button.clicked.connect(lambda: self.switch_mode("selection"))
        self.filter_mode_button.clicked.connect(lambda: self.switch_mode("filter"))

        self.mode_toolbar.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)
        self.mode_toolbar.setMaximumHeight(self.selection_mode_button.sizeHint().height() + 12)

        self.main_layout.addWidget(self.mode_toolbar)

        # Add splitter directly and stretch it
        self.splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        self.main_layout.addWidget(self.splitter, stretch=1)

        # Graphics layout for image and points display
        self.ui_graphics()
        
        self.ui_right_menu()
    
        # Style
        self.setStyleSheet("""
            QTabWidget::pane { border: 0; }
            QPushButton {
                background-color: #444;
                color: white;
                padding: 6px 12px;
                border: 1px solid #555;
                border-radius: 4px;
            }
            QPushButton:checked {
                background-color: #0078d7;
                border: 1px solid #005bb5;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #555;
                border-radius: 8px;
                margin-top: 15px;
                padding-top: 15px;
                background-color: #3a3a3a;
                color: white;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 15px;
                top: 4px;
                padding: 2px 10px;
                color: #e0e0e0;
                background-color: #4a4a4a;
                border: 1px solid #666;
                border-radius: 4px;
                font-size: 11px;
                font-weight: bold;
            }
        """)

        # Connect selection change handler
        self.button_group.idClicked.connect(self.method_selected)

        # Connect mouse click
        self.pg_widget.scene().sigMouseClicked.connect(self.on_mouse_click)

        # Set the initial image
        from ..video_reader import VideoReader
        if isinstance(video, VideoReader):
            self.frame = video.get_frame(0)
        elif isinstance(video, np.ndarray) and video.ndim == 3:
            # (n_frames, height, width) - take the first frame
            self.frame = video[0]
        elif isinstance(video, np.ndarray) and video.ndim == 2:
            # (height, width) - a single image
            self.frame = video
        else:
            raise TypeError(
                f"`video` must be a VideoReader, or a 2-D (height, width) or 3-D "
                f"(n_frames, height, width) np.ndarray, got {type(video).__name__!r}."
            )

        self.image_item.setImage(self.frame.T) # axis 0 is x, while image axis 0 is y

        # Ensure method-specific widgets are visible on startup
        self.method_selected(self.button_group.checkedId())
        # Don't auto-select any filter method - let user choose when needed

        # Set the initial mode
        self.switch_mode("selection")  # Default to selection mode

        # Start the GUI
        self.show()
        # Only call sys.exit if not in IPython
        if not hasattr(sys, 'ps1'):  # Not interactive
            sys.exit(app.exec())
        else:
            app.exec()  # Don't raise SystemExit in IPythonys

    def eventFilter(self, source, event):
        if event.type() == QtCore.QEvent.Type.KeyPress:
            if event.key() == QtCore.Qt.Key.Key_Control:
                self.ctrl_held = True
        elif event.type() == QtCore.QEvent.Type.KeyRelease:
            if event.key() == QtCore.Qt.Key.Key_Control:
                self.ctrl_held = False
        return super().eventFilter(source, event)

    def create_help_button(self, tooltip_text: str) -> QtWidgets.QToolButton:
        """Create a small '?' help button with a tooltip."""
        button = QtWidgets.QToolButton()
        button.setIcon(self.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_MessageBoxQuestion))
        button.setToolTip(tooltip_text)
        button.setCursor(QtCore.Qt.CursorShape.WhatsThisCursor)
        button.setStyleSheet("""
            QToolButton {
                border: none;
                background: transparent;
                padding: 0px;
            }
            QToolButton:hover {
                color: #0078d7;
            }
        """)
        button.setFixedSize(20, 20)
        return button

    def ui_graphics(self):
        # Image viewer
        self.pg_widget = GraphicsLayoutWidget()
        self.view = BrushViewBox(parent_gui=self, lockAspect=True, invertY=True)
        self.pg_widget.addItem(self.view)

        
        self.image_item = ImageItem()
        self.polygon_line = pg.PlotDataItem(pen=pg.mkPen('y', width=2))
        self.polygon_points_scatter = ScatterPlotItem(pen=pg.mkPen(None), brush=pg.mkBrush(255, 255, 0, 200), size=6)
        self.grid_line = pg.PlotDataItem(pen=pg.mkPen('c', width=2))
        self.grid_points_scatter = ScatterPlotItem(pen=pg.mkPen(None), brush=pg.mkBrush(255, 200, 0, 200), size=6)
        self.roi_overlay = ImageItem()
        # The subset borders, kept apart from roi_overlay (which carries only the
        # translucent interior) so they can be stroked with a *cosmetic* pen: its width
        # is measured in screen pixels rather than image pixels, so the borders stay one
        # pixel thin however far you zoom in. A raster border cannot go below one image
        # pixel, which turns into a thick band at high zoom.
        self.roi_outline = QtWidgets.QGraphicsPathItem()
        outline_pen = pg.mkPen(0, 255, 0, 150)
        outline_pen.setCosmetic(True)
        self.roi_outline.setPen(outline_pen)
        self.roi_outline.setBrush(pg.mkBrush(None))
        self.scatter = ScatterPlotItem(pen=pg.mkPen(None), brush=pg.mkBrush(255, 100, 100, 200), size=8)
        # Highlights the points of the entry currently selected in selection_list,
        # drawn on top of the plain point scatter. Magenta with no fill: it is the one
        # strong hue not already taken (green = ROI fill and filter candidates, cyan =
        # grid outline, yellow = line outline and vertices, salmon = the points
        # themselves), and unlike a white ring it stays visible against both the dark
        # and the bright parts of a grayscale frame. No fill so the point underneath
        # still reads through the ring.
        self.highlight_scatter = ScatterPlotItem(
            pen=pg.mkPen(255, 0, 255, 230, width=2), brush=pg.mkBrush(None), size=13
        )

        self.candidate_scatter = ScatterPlotItem(
            pen=pg.mkPen(None),
            brush=pg.mkBrush(0, 255, 0, 200),
            size=6
        )
        self.brush_overlay = ImageItem()
        self.direction_line = pg.PlotDataItem(pen=pg.mkPen('r', width=2))

        self.view.addItem(self.image_item)
        self.view.addItem(self.polygon_line)
        self.view.addItem(self.polygon_points_scatter)
        self.view.addItem(self.grid_line)
        self.view.addItem(self.grid_points_scatter)
        self.roi_overlay.setZValue(1)
        self.view.addItem(self.roi_overlay)  # Add scatter for showing square points
        self.roi_outline.setZValue(1)
        self.view.addItem(self.roi_outline)
        self.view.addItem(self.scatter)  # Add scatter for showing points
        self.view.addItem(self.highlight_scatter)
        self.view.addItem(self.candidate_scatter)
        self.brush_overlay.setZValue(2)
        self.view.addItem(self.brush_overlay)
        self.view.addItem(self.direction_line)

        self.splitter.addWidget(self.pg_widget)

    def ui_right_menu(self):
        # The right-side menu
        self.method_widget = QtWidgets.QWidget()
        self.stack = QtWidgets.QStackedLayout(self.method_widget)

        self.manual_widget = QtWidgets.QWidget()
        self.manual_layout = QtWidgets.QVBoxLayout(self.manual_widget)
        self.stack.addWidget(self.manual_widget)

        self.automatic_widget = QtWidgets.QWidget()
        self.automatic_layout = QtWidgets.QVBoxLayout(self.automatic_widget)
        self.stack.addWidget(self.automatic_widget)

        self.ui_manual_right_menu() # The manual right menu

        self.ui_auto_right_menu() # The automatic right menu

        # Set the layout and add to splitter
        self.splitter.addWidget(self.method_widget)
        self.splitter.setStretchFactor(0, 5)  # Image area grows more
        self.splitter.setStretchFactor(1, 0)  # Menu fixed by content

        # Set initial width for right panel
        self.method_widget.setMinimumWidth(150)
        self.method_widget.setMaximumWidth(600)
        self.splitter.setSizes([1000, 300])  # Initial left/right width

        self.automatic_layout.addStretch(1)

    def _make_subset_size_spinbox(self, initial_value: int) -> QtWidgets.QSpinBox:
        """Create a subset-size QSpinBox with the styling shared by the height/width spinboxes.

        :param initial_value: starting value of the spinbox
        :type initial_value: int
        :rtype: QtWidgets.QSpinBox
        """
        spinbox = QtWidgets.QSpinBox()
        spinbox.setRange(1, 1000)
        spinbox.setValue(initial_value)
        spinbox.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        spinbox.setSingleStep(2)
        spinbox.setMinimum(1)
        spinbox.setMaximum(999)
        spinbox.setWrapping(False)
        spinbox.setSuffix("px")
        spinbox.setFixedWidth(80)
        return spinbox

    def get_subset_size(self):
        """Return the current subset size as a (height, width) tuple of ints.

        :rtype: tuple of int
        """
        return (self.subset_height_spinbox.value(), self.subset_width_spinbox.value())

    def toggle_square_subsets(self, checked: bool):
        """Handle the "Square subsets" checkbox: lock/unlock the width axis.

        When checked, the width spinbox is disabled and snapped to the current height, and the
        width slider is hidden, so the two axes are visually and functionally merged. When
        unchecked, both become independently editable.

        :param checked: new checkbox state
        :type checked: bool
        """
        self.subset_width_spinbox.setEnabled(not checked)
        self.subset_width_slider.setVisible(not checked)

        if checked:
            self._set_subset_width_value(self.subset_height_spinbox.value())
            self.recompute_roi_points()

    def _set_subset_width_value(self, value: int):
        """Set the width spinbox/slider to ``value`` without re-entering their handlers.

        :param value: new width value
        :type value: int
        """
        self.subset_width_spinbox.blockSignals(True)
        self.subset_width_spinbox.setValue(value)
        self.subset_width_spinbox.blockSignals(False)

        slider_value = min(100, max(1, value))
        self.subset_width_slider.blockSignals(True)
        self.subset_width_slider.setValue(slider_value)
        self.subset_width_slider.blockSignals(False)

    def _sync_square_width_and_recompute(self):
        """If "Square subsets" is checked, mirror the height into the width; always recompute."""
        if self.square_subsets_checkbox.isChecked():
            self._set_subset_width_value(self.subset_height_spinbox.value())
        self.recompute_roi_points()

    def update_subset_height_from_slider(self, value):
        """Update the height spinbox from the height slider value and recompute ROI points."""
        self.subset_height_spinbox.blockSignals(True)
        self.subset_height_spinbox.setValue(value)
        self.subset_height_spinbox.blockSignals(False)

        self._sync_square_width_and_recompute()

    def update_subset_height_from_spinbox(self, value):
        """Update the height slider from the height spinbox value and recompute ROI points."""
        slider_value = min(100, max(1, value))
        self.subset_height_slider.blockSignals(True)
        self.subset_height_slider.setValue(slider_value)
        self.subset_height_slider.blockSignals(False)

        self._sync_square_width_and_recompute()

    def update_subset_width_from_slider(self, value):
        """Update the width spinbox from the width slider value and recompute ROI points."""
        self.subset_width_spinbox.blockSignals(True)
        self.subset_width_spinbox.setValue(value)
        self.subset_width_spinbox.blockSignals(False)

        self.recompute_roi_points()

    def update_subset_width_from_spinbox(self, value):
        """Update the width slider from the width spinbox value and recompute ROI points."""
        slider_value = min(100, max(1, value))
        self.subset_width_slider.blockSignals(True)
        self.subset_width_slider.setValue(slider_value)
        self.subset_width_slider.blockSignals(False)

        self.recompute_roi_points()

    def ui_manual_right_menu(self):
        # Number of selected subsets
        self.points_label = QtWidgets.QLabel("Selected subsets: 0")
        font = self.points_label.font()
        font.setPointSize(10)
        font.setBold(True)
        self.points_label.setFont(font)
        
        self.manual_layout.addWidget(self.points_label)

        # Method selection group
        method_group = QtWidgets.QGroupBox("Selection Methods")
        method_layout = QtWidgets.QVBoxLayout(method_group)
        
        # Method selection buttons
        self.button_group = QtWidgets.QButtonGroup(self.method_widget)
        self.button_group.setExclusive(True)

        self.method_buttons = {}
        method_names = [
            "Grid",
            "Manual",
            "Along the line",
            "Brush",
            "Remove point",
        ]
        for i, name in enumerate(method_names):
            button = QtWidgets.QPushButton(name)
            button.setCheckable(True)
            if i == 0:
                button.setChecked(True)  # Default selection
            self.button_group.addButton(button, i)
            method_layout.addWidget(button)
            self.method_buttons[name] = button
        
        self.manual_layout.addWidget(method_group)

        # Subset configuration group
        config_group = QtWidgets.QGroupBox("Subset Configuration")
        config_layout = QtWidgets.QVBoxLayout(config_group)
        
        # Square subsets toggle: checked by default, but unchecked automatically if
        # constructed with an already-anisotropic (h, w) pair, so the checkbox state
        # matches the sizes it was started with.
        square_default = self.subset_size[0] == self.subset_size[1]
        self.square_subsets_checkbox = QtWidgets.QCheckBox("Square subsets")
        self.square_subsets_checkbox.setChecked(square_default)
        self.square_subsets_checkbox.toggled.connect(self.toggle_square_subsets)
        config_layout.addWidget(self.square_subsets_checkbox)

        # Subset size input: height x width. The label sits on its own row above the
        # spinboxes -- previously it shared a row with both spinboxes and got clipped
        # to "Subset size (h" in the panel's default width.
        config_layout.addWidget(QtWidgets.QLabel("Subset size (h x w):"))

        self.subset_size_layout = QtWidgets.QHBoxLayout()

        self.subset_height_spinbox = self._make_subset_size_spinbox(self.subset_size[0])
        self.subset_height_spinbox.valueChanged.connect(self.update_subset_height_from_spinbox)
        self.subset_size_layout.addWidget(self.subset_height_spinbox)
        # Kept as an explicit alias for the height spinbox for backward compatibility --
        # docs/source/quick_start/make_selection_animation.py and possibly user scripts
        # reach for this name.
        self.subset_size_spinbox = self.subset_height_spinbox

        self.subset_size_layout.addWidget(QtWidgets.QLabel("x"))

        self.subset_width_spinbox = self._make_subset_size_spinbox(self.subset_size[1])
        self.subset_width_spinbox.valueChanged.connect(self.update_subset_width_from_spinbox)
        self.subset_width_spinbox.setEnabled(not square_default)
        self.subset_size_layout.addWidget(self.subset_width_spinbox)

        self.subset_size_layout.addStretch()  # Push everything to the left
        config_layout.addLayout(self.subset_size_layout)

        self.subset_height_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.subset_height_slider.setRange(1, 100)
        self.subset_height_slider.setValue(self.subset_size[0])
        self.subset_height_slider.setSingleStep(1)
        self.subset_height_slider.valueChanged.connect(self.update_subset_height_from_slider)
        config_layout.addWidget(self.subset_height_slider)

        self.subset_width_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.subset_width_slider.setRange(1, 100)
        self.subset_width_slider.setValue(self.subset_size[1])
        self.subset_width_slider.setSingleStep(1)
        self.subset_width_slider.valueChanged.connect(self.update_subset_width_from_slider)
        self.subset_width_slider.setVisible(not square_default)
        config_layout.addWidget(self.subset_width_slider)

        # Show ROI rectangles
        self.show_roi_checkbox = QtWidgets.QCheckBox("Show subsets")
        self.show_roi_checkbox.setChecked(True)
        self.show_roi_checkbox.stateChanged.connect(self.update_selected_points)
        config_layout.addWidget(self.show_roi_checkbox)

        # Clear button
        self.clear_button = QtWidgets.QPushButton("Clear selections")
        self.clear_button.clicked.connect(self.clear_selection)
        config_layout.addWidget(self.clear_button)
        
        self.manual_layout.addWidget(config_group)

        # Method-specific controls group
        method_controls_group = QtWidgets.QGroupBox("Method-Specific Controls")
        method_controls_layout = QtWidgets.QVBoxLayout(method_controls_group)

        # Distance between subsets (only visible for Grid and Along the line)
        self.distance_layout = QtWidgets.QHBoxLayout()
        self.distance_layout.addWidget(QtWidgets.QLabel("Distance between subsets:"))
        
        self.distance_spinbox = QtWidgets.QSpinBox()
        self.distance_spinbox.setRange(-50, 50)
        self.distance_spinbox.setSingleStep(1)
        self.distance_spinbox.setValue(self.subset_overlap)
        self.distance_spinbox.setSuffix("px")
        self.distance_spinbox.setFixedWidth(80)
        self.distance_spinbox.valueChanged.connect(self.update_distance_from_spinbox)
        self.distance_layout.addWidget(self.distance_spinbox)
        
        self.distance_layout.addStretch()  # Push everything to the left
        
        # Create a widget to hold the distance controls
        self.distance_widget = QtWidgets.QWidget()
        self.distance_widget.setLayout(self.distance_layout)
        self.distance_widget.setVisible(False)  # Hidden by default
        method_controls_layout.addWidget(self.distance_widget)
        
        self.distance_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.distance_slider.setRange(-50, 50)
        self.distance_slider.setSingleStep(1)
        self.distance_slider.setValue(self.subset_overlap)
        self.distance_slider.setVisible(False)
        self.distance_slider.valueChanged.connect(self.update_distance_from_slider)
        method_controls_layout.addWidget(self.distance_slider)

        # Start new line (only visible in "Along the line" mode)
        self.start_new_line_button = QtWidgets.QPushButton("Start new line")
        self.start_new_line_button.clicked.connect(self.start_new_line)
        self.start_new_line_button.setVisible(False)  # Hidden by default
        method_controls_layout.addWidget(self.start_new_line_button)

        # Brush mode
        self.brush_radius_label = QtWidgets.QLabel(f"Brush radius (px): {self._paint_radius}")
        self.brush_radius_label.setVisible(False)  # shown only for Brush mode
        method_controls_layout.addWidget(self.brush_radius_label)
        
        self.brush_radius_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.brush_radius_slider.setRange(1, 50)
        self.brush_radius_slider.setSingleStep(1)
        self.brush_radius_slider.setValue(self._paint_radius)
        self.brush_radius_slider.setVisible(False)  # shown only for Brush mode
        self.brush_radius_slider.valueChanged.connect(lambda val: self.brush_radius_label.setText(f"Brush radius (px): {val}"))
        method_controls_layout.addWidget(self.brush_radius_slider)

        self.brush_deselect_button = QtWidgets.QPushButton("Deselect painted area")
        self.brush_deselect_button.setCheckable(True)
        self.brush_deselect_button.setVisible(False)  # shown only for Brush mode
        self.brush_deselect_button.clicked.connect(self.activate_brush_deselect)
        method_controls_layout.addWidget(self.brush_deselect_button)

        # Unified list of all selection entries (manual/line/grid/brush), always
        # visible regardless of the active method -- replaces the separate
        # polygon_list/grid_list. Each row's checkbox toggles that entry's
        # visibility; the delete button below removes the current row.
        self.selection_list = QtWidgets.QListWidget()
        self.selection_list.setMinimumHeight(120)
        self.selection_list.currentRowChanged.connect(self.on_entry_selected)
        self.selection_list.itemChanged.connect(self.on_entry_item_changed)
        method_controls_layout.addWidget(self.selection_list)

        self.delete_entry_button = QtWidgets.QPushButton("Delete selected")
        self.delete_entry_button.clicked.connect(self.delete_selected_entry)
        method_controls_layout.addWidget(self.delete_entry_button)

        self.manual_layout.addWidget(method_controls_group)

        self.manual_layout.addStretch(1)

    def ui_auto_right_menu(self):
        self.candidate_count_label = QtWidgets.QLabel("N candidate points: 0")
        font = self.candidate_count_label.font()
        font.setPointSize(10)
        font.setBold(True)
        self.candidate_count_label.setFont(font)
        
        self.automatic_layout.addWidget(self.candidate_count_label)

        # Filter method selection group
        filter_method_group = QtWidgets.QGroupBox("Filter Methods")
        filter_method_layout = QtWidgets.QVBoxLayout(filter_method_group)

        self.auto_method_group = QtWidgets.QButtonGroup(self.automatic_widget)
        self.auto_method_group.setExclusive(True)

        self.auto_method_buttons = {}
        method_names = [
            "Shi-Tomasi",
            "Gradient in direction",
        ]
        for i, name in enumerate(method_names):
            button = QtWidgets.QPushButton(name)
            button.setCheckable(True)
            # Don't auto-select any method - let user choose
            self.auto_method_group.addButton(button, i)
            filter_method_layout.addWidget(button)
            self.auto_method_buttons[name] = button

        self.auto_method_group.idClicked.connect(self.auto_method_selected)
        
        self.automatic_layout.addWidget(filter_method_group)

        # Display options group
        display_options_group = QtWidgets.QGroupBox("Display Options")
        display_options_layout = QtWidgets.QVBoxLayout(display_options_group)

        # Checkbox to show/hide scatter and ROI overlay
        self.show_points_checkbox = QtWidgets.QCheckBox("Show points/ROIs")
        self.show_points_checkbox.setChecked(False)
        def toggle_points_and_roi(state):
            self.roi_overlay.setVisible(state)
            self.roi_outline.setVisible(state)
            self.scatter.setVisible(state)
        self.show_points_checkbox.stateChanged.connect(toggle_points_and_roi)
        display_options_layout.addWidget(self.show_points_checkbox)
        
        # Clear the candidates button
        self.clear_candidates_button = QtWidgets.QPushButton("Clear candidates")
        self.clear_candidates_button.clicked.connect(self.clear_candidates)
        display_options_layout.addWidget(self.clear_candidates_button)
        
        self.automatic_layout.addWidget(display_options_group)

        # Method settings group
        method_settings_group = QtWidgets.QGroupBox("Method Settings")
        method_settings_layout = QtWidgets.QVBoxLayout(method_settings_group)

        # Shi-Tomasi method settings
        self.shi_tomasi_threshold = 10  # Default threshold value
        self.threshold_label = QtWidgets.QLabel(f"Threshold: {self.shi_tomasi_threshold}")
        self.threshold_label.setVisible(False)
        method_settings_layout.addWidget(self.threshold_label)
        
        self.threshold_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.threshold_slider.setRange(1, 100)
        self.threshold_slider.setSingleStep(1)
        self.threshold_slider.setValue(self.shi_tomasi_threshold)
        self.threshold_slider.setVisible(False)
        method_settings_layout.addWidget(self.threshold_slider)

        def update_label_and_recompute(val):
            self.threshold_label.setText(f"Threshold: {str(val)}")
            self.update_threshold_and_show_shi_tomsi()  # Placeholder method
        self.threshold_slider.valueChanged.connect(update_label_and_recompute)

        # Gradient in a specified direction settings
        self.direction_button = QtWidgets.QPushButton("Set direction on image")
        self.direction_button.setVisible(False)
        self.direction_button.setCheckable(True)
        self.direction_button.clicked.connect(self.set_gradient_direction_mode)
        method_settings_layout.addWidget(self.direction_button)

        # Preset direction buttons
        preset_layout = QtWidgets.QHBoxLayout()
        
        self.x_direction_button = QtWidgets.QPushButton("X Direction")
        self.x_direction_button.setVisible(False)
        self.x_direction_button.clicked.connect(self.set_x_direction_preset)
        preset_layout.addWidget(self.x_direction_button)
        
        self.y_direction_button = QtWidgets.QPushButton("Y Direction")
        self.y_direction_button.setVisible(False)
        self.y_direction_button.clicked.connect(self.set_y_direction_preset)
        preset_layout.addWidget(self.y_direction_button)
        
        # Create a widget to hold the preset buttons
        self.preset_buttons_widget = QtWidgets.QWidget()
        self.preset_buttons_widget.setLayout(preset_layout)
        self.preset_buttons_widget.setVisible(False)
        method_settings_layout.addWidget(self.preset_buttons_widget)

        self.direction_threshold = 10
        self.gradient_thresh_label = QtWidgets.QLabel(f"Threshold (grad): {self.direction_threshold}")
        self.gradient_thresh_label.setVisible(False)
        method_settings_layout.addWidget(self.gradient_thresh_label)

        self.gradient_thresh_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.gradient_thresh_slider.setRange(1, 100)
        self.gradient_thresh_slider.setSingleStep(1)
        self.gradient_thresh_slider.setValue(self.direction_threshold)
        self.gradient_thresh_slider.setVisible(False)
        method_settings_layout.addWidget(self.gradient_thresh_slider)

        def update_direction_thresh(val):
            self.gradient_thresh_label.setText(f"Threshold (grad): {val}")
            self.update_threshold_and_show_gradient_direction()
        self.gradient_thresh_slider.valueChanged.connect(update_direction_thresh)
        
        self.automatic_layout.addWidget(method_settings_group)

        self.automatic_layout.addStretch(1)

    def auto_method_selected(self, id: int):
        # Check if any button is actually checked
        if self.auto_method_group.checkedButton() is None:
            return
            
        method_name = list(self.auto_method_buttons.keys())[id]
        # print(f"Selected automatic method: {method_name}")
        # Here you can switch method behavior, show/hide widgets, etc.
        is_shi_tomasi = method_name == "Shi-Tomasi"
        is_gradient_dir = method_name == "Gradient in direction"

        # Reset gradient direction selection when switching away from gradient method
        if not is_gradient_dir and hasattr(self, 'direction_button') and self.direction_button.isChecked():
            self.direction_button.setChecked(False)
            self.set_gradient_direction_mode()
        
        # Hide direction line when not in gradient direction mode
        if not is_gradient_dir and hasattr(self, 'direction_line'):
            self.direction_line.clear()

        self.threshold_label.setVisible(is_shi_tomasi)
        self.threshold_slider.setVisible(is_shi_tomasi)

        if is_shi_tomasi:
            self.compute_candidate_points_shi_tomasi()

        self.direction_button.setVisible(is_gradient_dir)
        self.preset_buttons_widget.setVisible(is_gradient_dir)
        self.gradient_thresh_label.setVisible(is_gradient_dir)
        self.gradient_thresh_slider.setVisible(is_gradient_dir)
        self.preset_buttons_widget.setVisible(is_gradient_dir)
        self.y_direction_button.setVisible(is_gradient_dir)
        self.x_direction_button.setVisible(is_gradient_dir)

        if is_gradient_dir and self.gradient_direction is not None:
            self.compute_candidate_points_gradient_direction()
            # Show the direction line if we have gradient direction points
            if hasattr(self, 'gradient_direction_points') and len(self.gradient_direction_points) == 2:
                self.update_direction_line()

        if is_shi_tomasi:
            self.show_instruction("Use the threshold slider to filter points.")
        elif is_gradient_dir:
            self.show_instruction("Click 'Set direction on image' button and drag to define the gradient direction.")

    def show_instruction(self, message: str):
        self.statusBar.showMessage(message)

    def method_selected(self, id: int):
        method_name = list(self.method_buttons.keys())[id]
        # print(f"Selected method: {method_name}")
        kind = self.current_kind()
        self._reactivate_last_entry_of_kind(kind)
        is_along = kind == 'line'
        is_grid = kind == 'grid'
        is_brush = kind == 'brush'

        show_spacing = is_along or is_grid or is_brush

        self.start_new_line_button.setVisible(is_along or is_grid)
        self.start_new_line_button.setText("Start new grid" if is_grid else "Start new line")

        self.distance_widget.setVisible(show_spacing)
        self.distance_slider.setVisible(show_spacing)

        self.brush_deselect_button.setVisible(is_brush)
        self.brush_radius_label.setVisible(is_brush)
        self.brush_radius_slider.setVisible(is_brush)

        # Show context-sensitive instructions
        if is_brush:
            self.show_instruction("Hold Ctrl and drag to paint selection area. Use distance slider to control subset spacing.")
        elif is_along:
            self.show_instruction(
                "Click to add points along the line. Drag an existing point to move it. "
                "Click 'Start new line' to begin a new one. Ctrl+Z to undo."
            )
        elif is_grid:
            self.show_instruction(
                "Click to define grid corners. Drag an existing corner to move it. "
                "Click 'Start new grid' to begin a new grid. Ctrl+Z to undo."
            )
        elif method_name == "Manual":
            self.show_instruction("Click to add points manually.")
        elif method_name == "Remove point":
            self.show_instruction("Click on a point to remove it.")
        else:
            self.show_instruction("Ready.")

    def switch_mode(self, mode: str):
        self.mode = mode
        if mode == "selection":
            self.selection_mode_button.setChecked(True)
            self.filter_mode_button.setChecked(False)
            self.stack.setCurrentWidget(self.manual_widget)

            # Reset gradient direction selection when leaving filter mode
            if hasattr(self, 'direction_button') and self.direction_button.isChecked():
                self.direction_button.setChecked(False)
                self.set_gradient_direction_mode()
            
            # Hide direction line when leaving filter mode
            if hasattr(self, 'direction_line'):
                self.direction_line.clear()

            self.roi_overlay.setVisible(True)
            self.roi_outline.setVisible(True)
            self.scatter.setVisible(True)
            self.highlight_scatter.setVisible(True)
            self.show_instruction("Selection mode: choose a method on the left.")

        elif mode == "filter":
            self.selection_mode_button.setChecked(False)
            self.filter_mode_button.setChecked(True)
            self.stack.setCurrentWidget(self.automatic_widget)

            # Don't automatically compute anything - let user select method first
            self.show_points_checkbox.setChecked(False)
            self.roi_overlay.setVisible(False)
            self.roi_outline.setVisible(False)
            self.scatter.setVisible(False)
            # The active entry's highlight belongs to Select mode; leaving it on would
            # ring points that are no longer drawn.
            self.highlight_scatter.setVisible(False)
            self.show_instruction("Filter mode: choose a filter method and adjust settings.")

    def on_mouse_click(self, event):
        if self.mode == "filter":
            return

        kind = self.current_kind()
        if kind == 'manual':
            self.handle_manual_selection(event)
        elif kind == 'line':
            self.handle_polygon_drawing(event)
        elif kind == 'grid':
            self.handle_grid_drawing(event)
        elif kind == 'brush':
            self.handle_brush_start(event)
        elif self.method_buttons["Remove point"].isChecked():
            self.handle_remove_point(event)

    # ------------------------------------------------------------------
    # Selection-entry core helpers
    #
    # All selections (manual points, "along the line" polylines, grid
    # polygons, brush strokes) live in one ordered list, self.selections,
    # instead of four separate parallel containers. Each entry is a dict;
    # see the module-level PRETTY dict and add_selection() below for the
    # label scheme and entry schema.
    # ------------------------------------------------------------------
    def current_kind(self):
        """Return the selection-entry kind for the currently-checked method button.

        :return: ``'grid'``, ``'manual'``, ``'line'`` or ``'brush'`` for the
            correspondingly-checked method button; ``None`` if "Remove point" is
            checked (it has no entry kind of its own) or if no button is checked.
        :rtype: str or None
        """
        button_names = {'Grid': 'grid', 'Manual': 'manual', 'Along the line': 'line', 'Brush': 'brush'}
        for name, kind in button_names.items():
            if self.method_buttons[name].isChecked():
                return kind
        return None

    def add_selection(self, kind, geometry=None, label=None, make_active=True):
        """Append a new entry of `kind`, register its list row, and return the entry dict.

        This is the entry point used both by the click/stroke handlers below and by
        external callers (tests, the docs animation script) that want to build up a
        selection programmatically.

        :param kind: ``'manual'``, ``'line'``, ``'grid'`` or ``'brush'``
        :type kind: str
        :param geometry: initial geometry for the entry; defaults to ``[]`` for
            manual/line/grid. Brush entries are always created with a real mask, so
            leaving this ``None`` for ``kind='brush'`` is a programming error.
        :type geometry: list or numpy.ndarray or None
        :param label: explicit row label; if omitted, one is generated from the
            per-kind monotonic counter (see the class/module docstring)
        :type label: str or None
        :param make_active: whether to make the new entry the active one and select
            its row in ``selection_list``
        :type make_active: bool
        :return: the newly created entry dict
        :rtype: dict
        :raises ValueError: if ``kind == 'brush'`` and ``geometry`` is ``None``
        """
        if geometry is None:
            if kind == 'brush':
                raise ValueError("add_selection('brush', ...) requires an explicit mask.")
            geometry = []
        if label is None:
            self._label_counters[kind] += 1
            label = 'Manual' if kind == 'manual' else f'{PRETTY[kind]} {self._label_counters[kind]}'
        entry = {
            'kind': kind,
            'label': label,
            'geometry': geometry,
            'roi_points': [],
            'removed': set(),
            'visible': True,
        }
        self.selections.append(entry)
        row = len(self.selections) - 1
        self._insert_row(row, entry)
        if make_active:
            self.active_index = row
            self._syncing_list = True
            self.selection_list.setCurrentRow(row)
            self._syncing_list = False
        return entry

    def _reactivate_last_entry_of_kind(self, kind):
        """Make the most recent entry of `kind` active, if the active one is not already.

        There is a single ``active_index`` for all kinds, where the pre-list code kept a
        separate active index per kind. Without this, switching away from Grid (say, to
        drop a manual point) and back would leave a non-grid entry active, so the next
        click would start a *new* grid instead of continuing the one being drawn. Nothing
        happens when the active entry is already of `kind` -- that is what keeps clicking
        a specific row in ``selection_list`` from being overridden by this.

        :param kind: the entry kind the tool has just switched to, or None for
            "Remove point" (which owns no entries and leaves the active one alone)
        :type kind: str or None
        """
        if kind is None or self.active_entry(kind) is not None:
            return
        last = next((i for i in reversed(range(len(self.selections)))
                     if self.selections[i]['kind'] == kind), None)
        if last is None:
            return
        self.active_index = last
        self._syncing_list = True
        self.selection_list.setCurrentRow(last)
        self._syncing_list = False
        self.update_highlight()

    def active_entry(self, kind=None):
        """Return the active selection entry, or None.

        :param kind: if given, only return the active entry when its kind matches
        :type kind: str or None
        :return: the active entry dict, or None if there is no active entry (or its
            kind does not match `kind`)
        :rtype: dict or None
        """
        if self.active_index is None or not (0 <= self.active_index < len(self.selections)):
            return None
        entry = self.selections[self.active_index]
        if kind is not None and entry['kind'] != kind:
            return None
        return entry

    def entry_points(self, entry):
        """The entry's contributed points: roi_points minus its `removed` set, order preserved.

        :param entry: a selection entry dict
        :type entry: dict
        :return: the entry's live points, in ``roi_points`` order
        :rtype: list[tuple]
        """
        return [p for p in entry['roi_points'] if tuple(p) not in entry['removed']]

    def _brush_points(self, mask, subset_size, spacing):
        """ROI points inside a brush mask, returned as (x, y).

        ``mask`` is ``(n_x, n_y)`` indexed ``[x, y]`` (see ``handle_brush_move``),
        but ``rois_inside_mask`` documents and assumes ``mask[y, x]`` and returns
        ``(y, x)``. Transpose ``mask`` in, then flip the result back to ``(x, y)``
        out. Passing the mask untransposed happened to give correct *coordinates*
        (the two transpositions canceled) but transposed the *per-axis grid step*,
        so anisotropic subsets got the row/column spacing swapped -- fixed here.

        :param mask: boolean brush mask, ``(n_x, n_y)`` indexed ``[x, y]``
        :type mask: numpy.ndarray
        :param subset_size: ``(height, width)`` subset size
        :type subset_size: tuple
        :param spacing: extra spacing added to the subset size to get the grid step
        :type spacing: int
        :return: ``(x, y)`` ROI points inside the mask
        :rtype: list[tuple]
        """
        return [(x, y) for (y, x) in rois_inside_mask(mask.T, subset_size, spacing)]

    def recompute_entry(self, entry, subset_size=None, spacing=None):
        """Recompute `entry['roi_points']` from its geometry, in place.

        ``entry['removed']`` is deliberately NOT cleared here -- that is what lets
        removed points survive a spacing/subset-size change (see ``entry_points``).

        :param entry: the selection entry to recompute
        :type entry: dict
        :param subset_size: ``(height, width)`` subset size; defaults to
            ``self.get_subset_size()``
        :type subset_size: tuple or None
        :param spacing: spacing between subsets; defaults to
            ``self.distance_spinbox.value()``
        :type spacing: int or None
        """
        if subset_size is None:
            subset_size = self.get_subset_size()
        if spacing is None:
            spacing = self.distance_spinbox.value()

        kind, geom = entry['kind'], entry['geometry']
        if kind == 'manual':
            entry['roi_points'] = list(geom)  # no recompute; points are literal
        elif kind == 'line':
            entry['roi_points'] = points_along_polygon(geom, subset_size, spacing) if len(geom) >= 2 else []
        elif kind == 'grid':
            entry['roi_points'] = rois_inside_polygon(geom, subset_size, spacing) if len(geom) >= 3 else []
        elif kind == 'brush':
            entry['roi_points'] = self._brush_points(geom, subset_size, spacing)

    def _insert_row(self, row, entry):
        """Insert a checkable QListWidgetItem for `entry` at `row` in `selection_list`.

        :param row: row index to insert at
        :type row: int
        :param entry: the selection entry the row represents
        :type entry: dict
        """
        item = QtWidgets.QListWidgetItem(f"{entry['label']} — {len(self.entry_points(entry))} pts")
        item.setFlags(item.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
        self._syncing_list = True
        item.setCheckState(QtCore.Qt.CheckState.Checked if entry['visible'] else QtCore.Qt.CheckState.Unchecked)
        self.selection_list.insertItem(row, item)
        self._syncing_list = False

    def refresh_row_labels(self):
        """Refresh every row's text and checkbox from its entry's current state.

        Guarded by ``_syncing_list`` so the ``setText``/``setCheckState`` calls
        below do not re-trigger ``on_entry_item_changed``.
        """
        self._syncing_list = True
        for row, entry in enumerate(self.selections):
            item = self.selection_list.item(row)
            if item is None:
                continue
            item.setText(f"{entry['label']} — {len(self.entry_points(entry))} pts")
            item.setCheckState(QtCore.Qt.CheckState.Checked if entry['visible'] else QtCore.Qt.CheckState.Unchecked)
        self._syncing_list = False

    def on_entry_item_changed(self, item):
        """Sync an entry's `visible` flag from its row's checkbox state.

        :param item: the changed ``QListWidgetItem``
        :type item: QtWidgets.QListWidgetItem
        """
        if self._syncing_list:
            return
        row = self.selection_list.row(item)
        if 0 <= row < len(self.selections):
            self.selections[row]['visible'] = item.checkState() == QtCore.Qt.CheckState.Checked
            self.update_geometry_display()
            self.update_selected_points()

    def on_entry_selected(self, row):
        """Handle a row becoming the current row in `selection_list`.

        Switches the active tool to the row's kind, so its vertices become
        immediately editable (drag/undo), and refreshes the display.

        :param row: the new current row, or -1 if the selection was cleared
        :type row: int
        """
        if self._syncing_list:
            return
        if not (0 <= row < len(self.selections)):
            self.active_index = None
            self.update_geometry_display()
            self.update_selected_points()
            return

        self.active_index = row
        kind = self.selections[row]['kind']
        button_name = {'grid': 'Grid', 'line': 'Along the line', 'manual': 'Manual', 'brush': 'Brush'}[kind]
        button = self.method_buttons[button_name]
        if not button.isChecked():
            button.setChecked(True)
            self.method_selected(self.button_group.id(button))
        self.update_geometry_display()
        self.update_selected_points()

    def delete_selected_entry(self):
        """Delete the currently-selected row/entry (any kind), pushing an undo action."""
        row = self.selection_list.currentRow()
        if row < 0:
            return
        entry = self.selections[row]
        self.push_undo({
            'type': 'delete', 'kind': entry['kind'], 'entry': entry, 'row': row, 'label': entry['label'],
        })
        del self.selections[row]
        self.selection_list.takeItem(row)
        self.active_index = min(row, len(self.selections) - 1) if self.selections else None
        if self.active_index is not None:
            self._syncing_list = True
            self.selection_list.setCurrentRow(self.active_index)
            self._syncing_list = False
        self.update_geometry_display()
        self.update_selected_points()

    def update_highlight(self):
        """Highlight the active entry's points on top of the plain point scatter."""
        entry = self.active_entry()
        if entry is None or not entry['visible']:
            self.highlight_scatter.clear()
            return
        pts = self.entry_points(entry)
        if not pts:
            self.highlight_scatter.clear()
            return
        # Size/pen/brush come from the item's own defaults (see ui_graphics).
        self.highlight_scatter.setData(pos=np.array(pts) + 0.5, symbol='o')

    def _line_display_data(self):
        """Build nan-separated OPEN polyline coordinates and vertex list for 'line' entries.

        :return: ``(xs, ys, all_points)`` -- flattened, nan-separated polyline
            coordinates and the flat list of all vertices, for every visible
            ``line`` entry
        :rtype: tuple
        """
        xs, ys, all_points = [], [], []
        for entry in self.selections:
            if entry['kind'] != 'line' or not entry['visible']:
                continue
            path = entry['geometry']
            all_points.extend(path)
            if len(path) >= 2:
                xs.extend([p[0] for p in path] + [np.nan])
                ys.extend([p[1] for p in path] + [np.nan])
            elif len(path) == 1:
                xs.extend([path[0][0], path[0][0], np.nan])
                ys.extend([path[0][1], path[0][1], np.nan])
        return xs, ys, all_points

    def _grid_display_data(self):
        """Build nan-separated CLOSED polygon coordinates and vertex list for 'grid' entries.

        :return: ``(xs, ys, all_points)`` -- flattened, nan-separated closed-polygon
            coordinates and the flat list of all vertices, for every visible
            ``grid`` entry
        :rtype: tuple
        """
        xs, ys, all_points = [], [], []
        for entry in self.selections:
            if entry['kind'] != 'grid' or not entry['visible']:
                continue
            path = entry['geometry']
            all_points.extend(path)
            if len(path) >= 2:
                xs.extend([p[0] for p in path] + [path[0][0], np.nan])  # Close polygon
                ys.extend([p[1] for p in path] + [path[0][1], np.nan])
            elif len(path) == 1:
                xs.extend([path[0][0], path[0][0], np.nan])
                ys.extend([path[0][1], path[0][1], np.nan])
        return xs, ys, all_points

    def update_geometry_display(self):
        """Redraw the line/grid outlines and vertex scatters from `self.selections`.

        Walks ``self.selections`` once per kind and builds the four display
        datasets (line outline + vertices, grid outline + vertices), skipping
        entries with ``visible == False``.
        """
        line_xs, line_ys, line_points = self._line_display_data()
        self.polygon_line.setData(line_xs, line_ys)
        self.polygon_points_scatter.setData(pos=line_points)

        grid_xs, grid_ys, grid_points = self._grid_display_data()
        self.grid_line.setData(grid_xs, grid_ys)
        self.grid_points_scatter.setData(pos=grid_points)

        self.update_highlight()

    def clear_subset_rectangles(self):
        """Remove both halves of the subset-rectangle display."""
        self.roi_overlay.clear()
        self.roi_outline.setPath(QtGui.QPainterPath())

    def draw_subset_rectangles(self, points, half_h, half_w):
        """Draw a rectangle around each point, as a raster fill plus a hairline border.

        The two halves are drawn by different means because each is cheap in a
        different way. The translucent interior goes into ``roi_overlay`` as a single
        RGBA image, which costs one upload no matter how many subsets there are. The
        borders go into ``roi_outline`` as one QPainterPath stroked with a cosmetic
        pen, whose width is in *screen* pixels: that is what makes them a hairline at
        any zoom. Drawing the borders into the raster instead, as this used to, pins
        them to one *image* pixel, which is a thick band as soon as you zoom in.

        Both are built with whole-array numpy rather than a Python loop over the
        points, which is what keeps the redraw quick for tens of thousands of subsets.

        :param points: the subset centres, as an ``(n, 2)`` array of real ``(x, y)``
            = (column, row) image coordinates
        :type points: numpy.ndarray
        :param half_h: half the subset height, in pixels (``subset_h // 2``)
        :type half_h: int
        :param half_w: half the subset width, in pixels (``subset_w // 2``)
        :type half_w: int
        """
        # image_item.image / roi_overlay are column-major (pyqtgraph's default
        # axisOrder): array axis 0 is the image's x/width axis, axis 1 is its y/height
        # axis. half_w therefore pairs with axis 0 and half_h with axis 1.
        n_x, n_y = self.image_item.image.shape[:2]
        w_x, w_y = 2 * half_w + 1, 2 * half_h + 1

        ix0 = np.rint(points[:, 0]).astype(int) - half_w
        iy0 = np.rint(points[:, 1]).astype(int) - half_h
        # A subset whose rectangle would reach past the image edge is not drawn at all.
        inside = (ix0 >= 0) & (iy0 >= 0) & (ix0 + w_x < n_x) & (iy0 + w_y < n_y)
        ix0, iy0 = ix0[inside], iy0[inside]

        if not len(ix0):
            self.clear_subset_rectangles()
            return

        # Fill: mark every covered pixel at once by broadcasting the per-subset pixel
        # index ranges against each other, giving an (n, w_x, w_y) index into the mask.
        covered = np.zeros((n_x, n_y), dtype=bool)
        covered[(ix0[:, None] + np.arange(w_x))[:, :, None],
                (iy0[:, None] + np.arange(w_y))[:, None, :]] = True
        overlay = np.zeros((n_x, n_y, 4), dtype=np.uint8)  # RGBA
        overlay[..., 1] = covered * np.uint8(180)   # green
        overlay[..., 3] = covered * np.uint8(40)    # alpha
        self.roi_overlay.setImage(overlay, autoLevels=False)
        self.roi_overlay.setZValue(1)

        # Border: five corners per rectangle (the first repeated to close it) separated
        # by a nan, which is how arrayToQPath is told to start a new sub-path.
        x0, y0 = ix0.astype(float), iy0.astype(float)
        x1, y1 = x0 + w_x, y0 + w_y
        xs = np.empty((len(x0), 6))
        ys = np.empty((len(y0), 6))
        xs[:, 0] = xs[:, 3] = xs[:, 4] = x0
        xs[:, 1] = xs[:, 2] = x1
        ys[:, 0] = ys[:, 1] = ys[:, 4] = y0
        ys[:, 2] = ys[:, 3] = y1
        xs[:, 5] = ys[:, 5] = np.nan
        self.roi_outline.setPath(pg.arrayToQPath(xs.ravel(), ys.ravel(), connect='finite'))

    def refresh_candidates_for_selection(self):
        """Re-derive the filter candidates from what is currently selected.

        The Filter-mode filters score the subsets placed in Select mode, so their
        result goes stale the moment one of those subsets disappears -- painted over
        with the brush in deselect mode, clicked away with "Remove point", or removed
        by deleting or unchecking a row. A stale candidate was not merely drawn in the
        wrong place: ``get_points()`` returns the candidates whenever a filter has been
        run, so a deselected subset stayed in the returned points.

        The cached per-subset scores are deliberately kept, so this is reversible:
        re-checking a row, or undoing its deletion, brings its candidates back without
        re-running the filter.
        """
        if self._candidate_refresh is not None:
            self._candidate_refresh()

    def update_selected_points(self):
        # Order is creation order across all kinds (manual/line/grid/brush mixed
        # together as entries were added) -- a deliberate change from the old
        # manual+line+grid+brush concatenation order.
        self.selected_points = []
        for entry in self.selections:
            if entry['visible']:
                self.selected_points.extend(self.entry_points(entry))

        self.refresh_candidates_for_selection()

        if not self.selected_points:
            self.scatter.clear()
            self.clear_subset_rectangles()
            self.refresh_row_labels()
            self.update_highlight()
            return

        subset_h, subset_w = self.get_subset_size()
        half_h = subset_h // 2
        half_w = subset_w // 2

        # selected_points = np.round(np.array(self.selected_points) - 0.5)
        selected_points = np.array(self.selected_points)

        # --- Rectangles ---
        if self.show_roi_checkbox.isChecked():
            self.draw_subset_rectangles(selected_points, half_h, half_w)
        else:
            self.clear_subset_rectangles()

        # --- Center Dots ---
        self.scatter.setData(
            pos=selected_points + 0.5,
            symbol='o',
            size=6,
            brush=pg.mkBrush(255, 100, 100, 200),
            pen=pg.mkPen(None)
        )
        self.points_label.setText(f"Selected subsets: {len(self.selected_points)}")
        self.refresh_row_labels()
        self.update_highlight()

    def update_distance_from_slider(self, value):
        """Update distance spinbox from slider value and recompute ROI points."""
        # Update spinbox without triggering its signal
        self.distance_spinbox.blockSignals(True)
        self.distance_spinbox.setValue(value)
        self.distance_spinbox.blockSignals(False)
        
        # Recompute ROI points
        self.recompute_roi_points()

    def update_distance_from_spinbox(self, value):
        """Update distance slider from spinbox value and recompute ROI points."""
        # Update slider without triggering its signal
        self.distance_slider.blockSignals(True)
        self.distance_slider.setValue(value)
        self.distance_slider.blockSignals(False)
        
        # Recompute ROI points
        self.recompute_roi_points()

    def recompute_roi_points(self):
        subset_size = self.get_subset_size()
        spacing = self.distance_spinbox.value()
        for entry in self.selections:
            self.recompute_entry(entry, subset_size, spacing)
        self.update_selected_points()

    def start_new_line(self):
        # print("Starting a new line...")
        kind = self.current_kind()
        if kind in ('grid', 'line'):
            self.add_selection(kind)
            self.update_geometry_display()
            self.update_selected_points()

    def clear_selection(self):
        # print("Clearing selections...")

        # Any pending undo actions reference the entries/rows being wiped out below,
        # so they would no longer apply consistently after a full clear.
        self.undo_stack = []

        self.selections = []
        self.active_index = None
        self._label_counters = {k: 0 for k in self._label_counters}
        self.selection_list.clear()
        self.selected_points = []

        self.polygon_line.clear()
        self.polygon_points_scatter.clear()
        self.grid_line.clear()
        self.grid_points_scatter.clear()
        self.highlight_scatter.clear()
        self.scatter.clear()
        self.clear_subset_rectangles()

        # Clear candidate points from automatic filtering
        self.clear_candidates()

        self.points_label.setText("Selected subsets: 0")

        # Reset gradient direction selection and clear direction line
        if hasattr(self, 'direction_button') and self.direction_button.isChecked():
            self.direction_button.setChecked(False)
            self.set_gradient_direction_mode()
        self.direction_line.clear()

        self.update_selected_points()  # Refresh display

    def set_image(self, img: np.ndarray):
        """Display image in the manual tab."""
        self.image_item.setImage(img)

    def get_points(self):
        """Get all selected points from manual and polygons."""
        if np.array(self.candidate_points).size > 0:
            return self.get_filtered_points()
        else:
            return self.get_selected_points()
    
    @property
    def points(self):
        return self.get_points()
    
    def get_filtered_points(self):
        """Get candidate points from filtering."""
        return np.array(self.candidate_points)[:, ::-1] if hasattr(self, 'candidate_points') else []
    
    def get_selected_points(self):
        """Get all selected points from manual, polygons and grid."""
        return np.array(self.selected_points)[:, ::-1] if self.selected_points else []

    # Vertex hit-testing (shared by Grid and "Along the line" click/drag handling)
    def nearest_vertex(self, points, x, y):
        """Find the vertex nearest to (x, y) in a list of (x, y) points.

        Distance is measured in screen pixels (via the view's current pixel size) so
        that hit-testing behaves consistently regardless of the current zoom level.

        :param points: candidate vertices in view/data coordinates, native (x, y) order
        :type points: list[tuple[float, float]]
        :param x: query x coordinate in view/data units
        :type x: float
        :param y: query y coordinate in view/data units
        :type y: float
        :return: (index, distance in screen pixels) of the nearest vertex, or (None, None)
            if `points` is empty
        :rtype: tuple
        """
        if not points:
            return None, None
        px_x, px_y = self.view.viewPixelSize()
        px_x = px_x or 1e-9
        px_y = px_y or 1e-9
        arr = np.array(points, dtype=float)
        dx = (arr[:, 0] - x) / px_x
        dy = (arr[:, 1] - y) / px_y
        distances = np.hypot(dx, dy)
        idx = int(np.argmin(distances))
        return idx, float(distances[idx])

    def vertex_within_grab_radius(self, points, x, y):
        """Return the index of the vertex nearest to (x, y) if within the grab radius.

        :param points: candidate vertices in view/data coordinates, native (x, y) order
        :type points: list[tuple[float, float]]
        :param x: query x coordinate in view/data units
        :type x: float
        :param y: query y coordinate in view/data units
        :type y: float
        :return: index of the nearest vertex, or None if none is within the grab radius
        :rtype: int or None
        """
        idx, dist = self.nearest_vertex(points, x, y)
        if idx is not None and dist <= VERTEX_GRAB_RADIUS_PX:
            return idx
        return None

    def find_vertex_to_drag(self, entries, x, y):
        """Hit-test (x, y) against the vertices of every entry (grid or polyline).

        Searches across ALL entries in the container (not only the active one), so a
        vertex of any grid/polyline can be grabbed and dragged.

        :param entries: list of selection entries (dicts with a 'geometry' key),
            filtered to the active kind -- see
            ``BrushViewBox._vertex_drag_container``
        :type entries: list[dict]
        :param x: query x coordinate in view/data units
        :type x: float
        :param y: query y coordinate in view/data units
        :type y: float
        :return: (entry_index, vertex_index) of the closest vertex within the grab
            radius across all entries, or (None, None) if none is close enough
        :rtype: tuple
        """
        best_entry_idx, best_vertex_idx, best_dist = None, None, None
        for entry_idx, entry in enumerate(entries):
            idx, dist = self.nearest_vertex(entry['geometry'], x, y)
            if idx is None or dist > VERTEX_GRAB_RADIUS_PX:
                continue
            if best_dist is None or dist < best_dist:
                best_entry_idx, best_vertex_idx, best_dist = entry_idx, idx, dist
        return best_entry_idx, best_vertex_idx

    # Undo stack (add vertex / move vertex / delete a selection entry)
    def push_undo(self, action):
        """Push an action onto the bounded undo stack.

        :param action: description of the action; must include a 'type' key
            ('add', 'move' or 'delete') and a 'kind' key (the entry's kind --
            'grid', 'line', 'manual' or 'brush')
        :type action: dict
        """
        self.undo_stack.append(action)
        if len(self.undo_stack) > self.undo_stack_limit:
            self.undo_stack.pop(0)

    def undo(self):
        """Undo the most recent undoable action.

        Covers adding a vertex, moving a vertex, and deleting a selection entry --
        of any kind, since delete is generic now (manual and brush entries are
        undoable too, which they were not before). Filter results are not
        undoable. A no-op when the undo stack is empty.
        """
        if not self.undo_stack:
            return

        action = self.undo_stack.pop()

        if action['type'] == 'add':
            del action['entry']['geometry'][action['vertex_index']]
        elif action['type'] == 'move':
            action['entry']['geometry'][action['vertex_index']] = action['original_position']
        elif action['type'] == 'delete':
            row = min(action['row'], len(self.selections))
            self.selections.insert(row, action['entry'])
            self._insert_row(row, action['entry'])
            self.active_index = row
            self._syncing_list = True
            self.selection_list.setCurrentRow(row)
            self._syncing_list = False

        self.update_geometry_display()
        self.recompute_roi_points()

    # Grid selection
    def handle_grid_drawing(self, event):
        pos = event.scenePos()
        if self.view.sceneBoundingRect().contains(pos):
            mouse_point = self.view.mapSceneToView(pos)
            x, y = mouse_point.x(), mouse_point.y()

            grid = self.active_entry('grid')
            if grid is None:
                grid = self.add_selection('grid')

            # Clicking on an existing vertex is a no-op (dragging is used to move it).
            if self.vertex_within_grab_radius(grid['geometry'], x, y) is not None:
                return

            grid['geometry'].append((x, y))
            self.push_undo({
                'type': 'add', 'kind': 'grid', 'entry': grid, 'vertex_index': len(grid['geometry']) - 1
            })

            self.recompute_entry(grid)
            self.update_geometry_display()
            self.update_selected_points()  # also refreshes this row's "N pts" label

    # Manual selection
    def handle_manual_selection(self, event):
        """Handle manual selection of points."""
        pos = event.scenePos()
        if self.view.sceneBoundingRect().contains(pos):
            mouse_point = self.view.mapSceneToView(pos)
            x, y = mouse_point.x(), mouse_point.y()
            x_int, y_int = round(x - 0.5), round(y - 0.5)

            # Manual is a singleton entry -- every manual point lands in the same
            # row, never a new one.
            entry = next((e for e in self.selections if e['kind'] == 'manual'), None)
            if entry is None:
                entry = self.add_selection('manual')

            entry['geometry'].append((x_int, y_int))
            self.recompute_entry(entry)
            self.update_selected_points()  # also refreshes this row's "N pts" label

    # Along the line selection
    def handle_polygon_drawing(self, event):
        pos = event.scenePos()
        if self.view.sceneBoundingRect().contains(pos):
            mouse_point = self.view.mapSceneToView(pos)
            x, y = mouse_point.x(), mouse_point.y()

            poly = self.active_entry('line')
            if poly is None:
                poly = self.add_selection('line')

            # Clicking on an existing vertex is a no-op (dragging is used to move it).
            if self.vertex_within_grab_radius(poly['geometry'], x, y) is not None:
                return

            poly['geometry'].append((x, y))
            self.push_undo({
                'type': 'add', 'kind': 'line', 'entry': poly, 'vertex_index': len(poly['geometry']) - 1
            })

            self.recompute_entry(poly)
            self.update_geometry_display()
            self.update_selected_points()  # also refreshes this row's "N pts" label

    # Remove point selection
    def handle_remove_point(self, event):
        pos = event.scenePos()
        if self.view.sceneBoundingRect().contains(pos):
            mouse_point = self.view.mapSceneToView(pos)
            x, y = mouse_point.x(), mouse_point.y()

            # Find nearest point
            if not self.selected_points:
                return

            pts = np.array(self.selected_points)
            distances = np.linalg.norm(pts - np.array([x, y]), axis=1)
            idx = np.argmin(distances)
            closest = tuple(pts[idx])

            # Locate the entry that actually contributed this point (first visible
            # entry whose entry_points() contains it), so the removal is recorded
            # against the right owner and survives a later recompute (spacing or
            # subset-size change) instead of being silently undone by it -- see
            # entry_points()/recompute_entry().
            entry = next((e for e in self.selections if e['visible'] and closest in self.entry_points(e)), None)
            if entry is None:
                return

            if entry['kind'] == 'manual':
                # Manual points are literal -- there is nothing to regenerate, so
                # the point is removed from geometry outright instead of via
                # `removed` (which stays empty for manual entries, see the class
                # docstring / removed-point semantics notes).
                entry['geometry'] = [p for p in entry['geometry'] if tuple(p) != closest]
                self.recompute_entry(entry)
            else:
                entry['removed'].add(closest)

            self.update_selected_points()  # also refreshes the owning row's "N pts" label

    # Automatic filtering
    # Shi-Tomasi method
    def compute_candidate_points_shi_tomasi(self):
        """Compute good feature points using structure tensor analysis (Shi–Tomasi style)."""
        from scipy.ndimage import sobel

        subset_h, subset_w = self.get_subset_size()
        half_h = subset_h // 2
        half_w = subset_w // 2

        img = self.image_item.image.astype(np.float32)
        candidates = []

        # img is column-major (pyqtgraph's default axisOrder): array axis 0 is
        # the image's x/width axis, axis 1 is its y/height axis.
        # All selected points (not just manual)
        for px, py in self.selected_points:
            ix, iy = int(round(px)), int(round(py))

            if (ix - half_w < 0 or ix + half_w + 1 > img.shape[0] or
                iy - half_h < 0 or iy + half_h + 1 > img.shape[1]):
                continue

            roi = img[ix - half_w: ix + half_w + 1,
                    iy - half_h: iy + half_h + 1]

            # Compute gradients
            gx = sobel(roi, axis=1)
            gy = sobel(roi, axis=0)

            Gx2 = np.sum(gx ** 2)
            Gy2 = np.sum(gy ** 2)
            GxGy = np.sum(gx * gy)

            matrix = np.array([[Gx2, GxGy],
                            [GxGy, Gy2]])

            eigvals = np.linalg.eigvalsh(matrix)  # sorted ascending
            min_eig = eigvals[0]

            # Stored (y, x, value): update_threshold_and_show_shi_tomsi (outside this
            # function, unchanged) unpacks this as (x, y, e) and reverses it back to
            # (x, y) -- keep that round-trip intact.
            candidates.append((py + 0.0, px + 0.0, min_eig))

        if not candidates:
            self.candidate_points = []
            self._candidate_refresh = None
            self.update_candidate_display()
            return

        # Threshold by normalized eigenvalue
        eigvals = np.array([v[2] for v in candidates])
        self.max_eig_shi_tomasi = np.max(eigvals)

        self.candidates_shi_tomasi = candidates

        self._candidate_refresh = self.update_threshold_and_show_shi_tomsi
        self.update_threshold_and_show_shi_tomsi()

    def thresholded_candidates(self, cached, threshold):
        """Turn cached filter scores into the points to show as candidates.

        :param cached: per-subset ``(y, x, score)`` tuples, as stored by the
            ``compute_candidate_points_*`` methods
        :type cached: list
        :param threshold: keep only the subsets scoring strictly above this
        :type threshold: float
        :return: the surviving subsets, as rounded ``(x, y)`` tuples
        :rtype: list
        """
        selected = {(int(round(px)), int(round(py))) for px, py in self.selected_points}
        points = [(round(y), round(x)) for (x, y, score) in cached if score > threshold]
        # A subset deselected since the filter ran is dropped however well it scores.
        # `cached` itself is left alone, so re-selecting it brings the candidate back.
        return [p for p in points if p in selected]

    def update_threshold_and_show_shi_tomsi(self):
        threshold_ratio = self.threshold_slider.value() / 1000.0

        eig_threshold = self.max_eig_shi_tomasi * threshold_ratio

        self.candidate_points = self.thresholded_candidates(self.candidates_shi_tomasi, eig_threshold)
        self.update_candidate_display()
        self.update_candidate_points_count()

    def update_candidate_points_count(self):
        """Update the displayed count of candidate points."""
        if self.candidate_points:
            count_text = f"N candidate points: {len(self.candidate_points)}"
        else:
            count_text = "N candidate points: 0"

        self.candidate_count_label.setText(count_text)

    def update_candidate_display(self):
        """Show candidate points as scatter dots on the image."""
        if self.candidate_points:
            self.candidate_scatter.setData(pos=np.array(self.candidate_points) + 0.5)
        else:
            self.candidate_scatter.clear()

    def clear_candidates(self):
        """Clear candidate points."""
        # print("Clearing candidate points...")
        self.candidate_points = []
        # Otherwise update_selected_points(), below, would re-derive them from the
        # still-cached scores and undo the clear.
        self._candidate_refresh = None
        self.update_candidate_points_count()
        if hasattr(self, 'candidate_scatter'):
            self.candidate_scatter.clear()

        # Reset gradient direction selection when clearing candidates
        if hasattr(self, 'direction_button') and self.direction_button.isChecked():
            self.direction_button.setChecked(False)
            self.set_gradient_direction_mode()

        self.update_selected_points()  # Update main display to remove candidates
    
    # Gradient in a specified direction
    def set_gradient_direction_mode(self):
        """Toggle gradient direction selection mode."""
        self.setting_direction = self.direction_button.isChecked()
        
        if self.setting_direction:
            self.direction_button.setText("Cancel Direction")
            self.direction_button.setStyleSheet("background-color: #d73a00;")
            self.gradient_direction_points = []
            # Clear the direction line only when starting new selection
            self.direction_line.clear()
            self.show_instruction("Click and drag to set the gradient direction.")
        else:
            self.direction_button.setText("Set direction on image")
            self.direction_button.setStyleSheet("")
            # Don't clear the direction line when finishing selection - keep it visible
            self.show_instruction("Filter mode: choose a filter method and adjust settings.")

    def compute_direction_vector(self):
        p1, p2 = self.gradient_direction_points
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        norm = np.sqrt(dx**2 + dy**2)
        if norm == 0:
            self.gradient_direction = None
        else:
            self.gradient_direction = (dx / norm, dy / norm)

    def compute_candidate_points_gradient_direction(self):
        from scipy.ndimage import sobel

        if self.gradient_direction is None:
            return

        dy, dx = self.gradient_direction
        subset_h, subset_w = self.get_subset_size()
        half_h = subset_h // 2
        half_w = subset_w // 2

        img = self.image_item.image.astype(np.float32)
        candidates = []

        # img is column-major (pyqtgraph's default axisOrder): array axis 0 is
        # the image's x/width axis, axis 1 is its y/height axis.
        for px, py in self.selected_points:
            ix, iy = int(round(px)), int(round(py))

            if (ix - half_w < 0 or ix + half_w + 1 > img.shape[0] or
                iy - half_h < 0 or iy + half_h + 1 > img.shape[1]):
                continue

            roi = img[ix - half_w: ix + half_w + 1,
                    iy - half_h: iy + half_h + 1]

            gx = sobel(roi, axis=1)
            gy = sobel(roi, axis=0)

            gdir = np.abs(gx * dx) + np.abs(gy * dy)
            strength = np.sum(np.abs(gdir))

            # Stored (y, x, value): update_threshold_and_show_gradient_direction
            # (outside this function, unchanged) unpacks this as (x, y, v) and
            # reverses it back to (x, y) -- keep that round-trip intact.
            candidates.append((py + 0.0, px + 0.0, strength))

        if not candidates:
            self.candidate_points = []
            self._candidate_refresh = None
            self.update_candidate_display()
            return

        values = np.array([v[2] for v in candidates])
        self.max_grad_dir = np.max(values)
        self.candidates_grad_dir = candidates
        self._candidate_refresh = self.update_threshold_and_show_gradient_direction
        self.update_threshold_and_show_gradient_direction()

    def update_threshold_and_show_gradient_direction(self):
        threshold_ratio = self.gradient_thresh_slider.value() / 100.0
        threshold = self.max_grad_dir * threshold_ratio

        self.candidate_points = self.thresholded_candidates(self.candidates_grad_dir, threshold)
        self.update_candidate_display()
        self.update_candidate_points_count()

    def update_direction_line(self):
        if len(self.gradient_direction_points) == 2:
            xs = [p[0] for p in self.gradient_direction_points]
            ys = [p[1] for p in self.gradient_direction_points]
            self.direction_line.setData(xs, ys)
        else:
            self.direction_line.clear()

    # Brush
    def handle_brush_start(self, ev):
        if self.image_item.image is None:
            return
        h, w = self.image_item.image.shape[:2]
        self._paint_mask = np.zeros((h, w), dtype=bool)
        self.handle_brush_move(ev)

    def handle_brush_move(self, ev):
        if self._paint_mask is None:
            return

        pos = ev.scenePos()
        if self.view.sceneBoundingRect().contains(pos):
            mouse_point = self.view.mapSceneToView(pos)
            y, x = int(round(mouse_point.x())), int(round(mouse_point.y()))
            r = self.brush_radius_slider.value()

            h, w = self._paint_mask.shape
            yy, xx = np.ogrid[max(0, y - r):min(h, y + r + 1),
                            max(0, x - r):min(w, x + r + 1)]
            mask = (yy - y) ** 2 + (xx - x) ** 2 <= r ** 2
            self._paint_mask[max(0, y - r):min(h, y + r + 1),
                            max(0, x - r):min(w, x + r + 1)][mask] = True

            self.update_brush_overlay()

    def handle_brush_end(self, ev):
        if self._paint_mask is None:
            return

        if self.brush_deselect_mode:
            self._apply_brush_deselect()
            self.brush_deselect_mode = False
            self.brush_deselect_button.setChecked(False)
        else:
            # One entry per stroke.
            entry = self.add_selection('brush', geometry=self._paint_mask.copy())
            self.recompute_entry(entry)

        self._paint_mask = None
        self.update_geometry_display()
        self.update_selected_points()
        self.update_brush_overlay()

    def _apply_brush_deselect(self):
        """Remove every point covered by the current deselect-mode brush stroke.

        For a ``brush`` entry the stroke is subtracted from the painted mask itself,
        so only the overlapping area is lost and the rest of the stroke survives; the
        entry is dropped (and its row removed from ``selection_list``) only once
        nothing is left painted. Editing the mask rather than the derived points is
        what makes the deselection outlast a recompute.

        For every other kind the covered points are recorded in the entry's
        ``removed`` set (``manual`` excepted -- see below), which ``entry_points()``
        applies on read, so those deselections survive a spacing/subset-size change
        too.
        """
        def point_inside_mask(pt, mask):
            y, x = int(round(pt[0])), int(round(pt[1]))
            h, w = mask.shape
            return 0 <= y < h and 0 <= x < w and mask[y, x]

        active = self.active_entry()
        rows_to_delete = []
        for row, entry in enumerate(self.selections):
            if entry['kind'] == 'brush':
                entry['geometry'] = entry['geometry'] & ~self._paint_mask
                if not entry['geometry'].any():
                    rows_to_delete.append(row)
                else:
                    self.recompute_entry(entry)
                continue
            covered = [tuple(pt) for pt in self.entry_points(entry) if point_inside_mask(pt, self._paint_mask)]
            if entry['kind'] == 'manual':
                # Manual points are literal: drop them from geometry outright, the same
                # way handle_remove_point does. Recording them in `removed` instead would
                # make a later click on the very same pixel silently do nothing.
                covered_set = set(covered)
                entry['geometry'] = [p for p in entry['geometry'] if tuple(p) not in covered_set]
                self.recompute_entry(entry)
            else:
                entry['removed'].update(covered)

        for row in reversed(rows_to_delete):
            del self.selections[row]
            self.selection_list.takeItem(row)

        # Deleting rows shifts every later index, so re-derive the active one from the
        # entry object rather than leaving a stale (possibly out-of-range) index behind.
        # Matched by identity: `==` on entry dicts compares their values, which raises
        # on a brush entry's numpy mask ("truth value of an array is ambiguous").
        self.active_index = next((i for i, e in enumerate(self.selections) if e is active), None)
        if self.active_index is not None:
            self._syncing_list = True
            self.selection_list.setCurrentRow(self.active_index)
            self._syncing_list = False

    def update_brush_overlay(self):
        if self._paint_mask is not None:
            rgba = np.zeros((*self._paint_mask.shape, 4), dtype=np.uint8)
            if self.brush_deselect_mode:
                rgba[self._paint_mask] = [255, 0, 0, 80] # Red with transparency
            else:
                rgba[self._paint_mask] = [0, 200, 255, 80]  # Cyan with transparency
            self.brush_overlay.setImage(rgba, autoLevels=False)
            self.brush_overlay.setZValue(2)
        else:
            self.brush_overlay.clear()

    def activate_brush_deselect(self):
        if self.brush_deselect_button.isChecked():
            self.brush_deselect_mode = True

    def set_x_direction_preset(self):
        """Set horizontal (X) direction preset."""
        if self.image_item.image is None:
            return
        
        # Get image dimensions
        w, h = self.image_item.image.shape[:2]
        
        # Set horizontal line in the center of the image
        center_y = h // 2
        margin = min(w // 4, 50)  # Use 1/4 of width or 50 pixels, whichever is smaller
        
        # Create horizontal line points
        start_x = margin
        end_x = w - margin
        
        self.gradient_direction_points = [
            (start_x, center_y),
            (end_x, center_y)
        ]
        
        # Compute and set the direction vector
        self.compute_direction_vector()
        self.update_direction_line()
        
        # Ensure direction selection is off
        if self.direction_button.isChecked():
            self.direction_button.setChecked(False)
            self.set_gradient_direction_mode()
        
        # Compute candidate points
        self.compute_candidate_points_gradient_direction()
        
        self.show_instruction("X (horizontal) direction preset applied.")

    def set_y_direction_preset(self):
        """Set vertical (Y) direction preset."""
        if self.image_item.image is None:
            return
        
        # Get image dimensions
        w, h = self.image_item.image.shape[:2]
        
        # Set vertical line in the center of the image
        center_x = w // 2
        margin = min(h // 4, 50)  # Use 1/4 of height or 50 pixels, whichever is smaller
        
        # Create vertical line points
        start_y = margin
        end_y = h - margin
        
        self.gradient_direction_points = [
            (center_x, start_y),
            (center_x, end_y)
        ]
        
        # Compute and set the direction vector
        self.compute_direction_vector()
        self.update_direction_line()
        
        # Ensure direction selection is off
        if self.direction_button.isChecked():
            self.direction_button.setChecked(False)
            self.set_gradient_direction_mode()
        
        # Compute candidate points
        self.compute_candidate_points_gradient_direction()
        
        self.show_instruction("Y (vertical) direction preset applied.")

if __name__ == "__main__":
    # import pyidi
    # filename = "data/data_showcase.cih"
    # video = pyidi.VideoReader(filename)
    # example_image = (video.get_frame(0).T)[:, ::-1]


    import requests
    from PIL import Image
    import io
    import numpy as np
    # Example black and white image (public domain)
    url = "https://raw.githubusercontent.com/scikit-image/scikit-image/main/skimage/data/camera.png"
    # Fetch the image
    response = requests.get(url)
    img = Image.open(io.BytesIO(response.content)).convert("L")  # Convert to grayscale
    # Convert to numpy array
    example_image = (np.array(img).T)[:, ::-1]


    Points = SelectionGUI(example_image.astype(np.uint8))

    print(Points.get_points())  # # print selected points for testing
