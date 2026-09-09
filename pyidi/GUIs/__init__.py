"""Graphical interfaces, each available only if its own toolkit is installed.

Every class here needs the ``[qt]`` extra, but not the same part of it. The
selection windows and ``ResultViewer`` are PyQt6 and pyqtgraph; the napari
``GUI`` needs napari and magicgui instead. The requirement is therefore
checked per class rather than once for the package -- checking once means a
partial install, PyQt6 without napari being the likely one, turns ``import
pyidi`` into an ``ImportError`` from deep inside a submodule rather than a
message saying what to install.

A class whose dependencies are missing is replaced by a stub that imports
cleanly and raises ``RuntimeError`` when constructed, naming what is absent.
"""

import importlib.util
import typing

#: What each name needs beyond the base dependencies. Checked, not imported.
_REQUIREMENTS = {
    'SelectionGUI': ('PyQt6', 'pyqtgraph'),
    'SelectionGUIOld': ('PyQt6', 'pyqtgraph'),
    'ResultViewer': ('PyQt6', 'pyqtgraph'),
    'Viewer': ('PyQt6', 'pyqtgraph'),
    'GUI': ('napari', 'magicgui'),
}


def _installed(*modules):
    """Whether every named top-level module can be found, without importing it."""
    return all(importlib.util.find_spec(name) is not None for name in modules)


def _unavailable(name):
    """A stand-in for ``name`` that raises only when someone constructs it."""
    missing = ', '.join(m for m in _REQUIREMENTS[name] if not _installed(m))

    class Unavailable:
        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                f"{name} requires the qt extras: pip install pyidi[qt] "
                f"(missing: {missing})."
            )

    Unavailable.__name__ = name
    Unavailable.__qualname__ = name
    return Unavailable


HAS_PYQT6 = _installed('PyQt6', 'pyqtgraph')
HAS_NAPARI = _installed('napari', 'magicgui')

if HAS_PYQT6 or typing.TYPE_CHECKING:
    from .feature_selection import SelectionGUI
    from .subset_selection import SelectionGUIOld
    from .result_viewer import ResultViewer
    from .result_viewer import Viewer
else:
    SelectionGUI = _unavailable('SelectionGUI')
    SelectionGUIOld = _unavailable('SelectionGUIOld')
    ResultViewer = _unavailable('ResultViewer')
    Viewer = _unavailable('Viewer')

if HAS_NAPARI or typing.TYPE_CHECKING:
    from .gui import GUI
else:
    GUI = _unavailable('GUI')


class FeatureSelectionGUI:
    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "FeatureSelectionGUI was a working name that never shipped. "
            "It is now called SelectionGUI."
        )


class SubsetSelection:
    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "SubsetSelection was removed in favour of SelectionGUI. "
            "Replace SubsetSelection(...) with "
            "SelectionGUI(video, subset_size=..., subset_overlap=...)."
        )
