"""Each GUI class checks its own dependencies, so a partial install still imports.

``pyidi.GUIs`` used to gate every class on PyQt6 alone, then import the napari
``GUI`` unconditionally. With PyQt6 present and napari absent -- ``pip install
pyqt6 pyqtgraph`` without the extra -- ``import pyidi`` therefore died with a
``ModuleNotFoundError`` from inside a submodule, taking the whole package with
it rather than just the one class that was unusable.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/../')

import pyidi  # noqa: E402
from pyidi import GUIs  # noqa: E402

EXPORTED = ('SelectionGUI', 'SelectionGUIOld', 'ResultViewer', 'Viewer', 'GUI')


@pytest.mark.parametrize('name', EXPORTED)
def test_every_gui_name_is_bound(name):
    """Present as the real class or as a stub, but never missing."""
    assert getattr(pyidi, name, None) is not None


@pytest.mark.parametrize('name', EXPORTED)
def test_every_gui_name_declares_what_it_needs(name):
    """A stub cannot say what is missing unless the requirement is recorded."""
    assert name in GUIs._REQUIREMENTS


def test_the_selection_windows_do_not_depend_on_napari():
    """The reason the package survives a PyQt6-without-napari install."""
    assert 'napari' not in GUIs._REQUIREMENTS['SelectionGUI']
    assert 'napari' not in GUIs._REQUIREMENTS['SelectionGUIOld']
    assert 'napari' in GUIs._REQUIREMENTS['GUI']


def test_a_stub_names_the_extra_and_the_missing_package():
    """Constructing an unavailable class says what to install, and what is absent."""
    stub = GUIs._unavailable('GUI')
    assert stub.__name__ == 'GUI'

    with pytest.raises(RuntimeError) as excinfo:
        stub()
    message = str(excinfo.value)
    assert 'pip install pyidi[qt]' in message
    assert 'GUI requires' in message


def test_a_stub_is_importable_rather_than_raising_on_definition():
    """The whole point: the failure waits for construction."""
    stub = GUIs._unavailable('ResultViewer')
    assert isinstance(stub, type)
