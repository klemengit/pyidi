"""
Tests for the hardened ``IDIMethod.set_points()`` in
``pyidi/methods/idi_method.py``.

set_points() now:
  * accepts a plain array-like of (row, col) points, or any object exposing
    a ``.points`` attribute (duck-typed, so both selection GUIs work without
    idi_method importing either of them);
  * raises ValueError for empty input, non-2D input, wrong column count, and
    (when the method's video reports its size) out-of-bounds points;
  * rounds non-integer points to the nearest int (not truncating) and warns
    with UserWarning when it actually changes a value;
  * skips the bounds check entirely when ``self.video`` doesn't expose both
    ``image_width`` and ``image_height`` (see the ``getattr``/``hasattr``
    guard in the implementation).

These tests avoid running any full displacement analysis; they only build
method instances and call ``set_points`` / ``configure``.
"""

import types
import warnings

import numpy as np
import pytest
import sys
import os

my_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, my_path + '/../')

import pyidi
from pyidi.methods.idi_method import IDIMethod

DATA = os.path.join(my_path, '..', 'data', 'data_synthetic.cih')


@pytest.fixture
def video():
    return pyidi.VideoReader(input_file=DATA)


@pytest.fixture
def method(video):
    # LucasKanade is a thin IDIMethod subclass; set_points() itself is
    # implemented on IDIMethod and not overridden, so this exercises the
    # real code path used by every method.
    return pyidi.LucasKanade(video)


class DummySelectionGUI:
    """Stand-in for a selection GUI: exposes only the `.points` contract
    that set_points() duck-types against (both the tkinter and Qt subset
    selection GUIs satisfy this)."""

    def __init__(self, points):
        self.points = points


def _bare_method_without_video_info():
    """An object with a `set_points`-compatible `self.video` that lacks
    `image_width`/`image_height`, so the bounds check must be skipped.

    Confirmed from the implementation: it does
    ``video = getattr(self, 'video', None)`` then checks
    ``hasattr(video, 'image_width') and hasattr(video, 'image_height')``
    before doing any bounds checking at all.
    """
    obj = types.SimpleNamespace()
    obj.video = types.SimpleNamespace()  # no image_width / image_height
    return obj


# ---------------------------------------------------------------------------
# ValueError cases
# ---------------------------------------------------------------------------

def test_empty_points_raises_value_error(method):
    with pytest.raises(ValueError, match=r"empty"):
        method.set_points(np.array([]))


def test_non_2d_points_raises_value_error(method):
    with pytest.raises(ValueError, match=r"2-dimensional"):
        method.set_points(np.array([1, 2, 3]))


def test_wrong_column_count_raises_value_error(method):
    with pytest.raises(ValueError, match=r"two columns"):
        method.set_points(np.array([[1, 2, 3], [4, 5, 6]]))


def test_out_of_bounds_points_raise_value_error(method, video):
    # image is 128 (height/rows) x 256 (width/cols); row 200 is out of bounds
    with pytest.raises(ValueError, match=r"bounds"):
        method.set_points(np.array([[10, 10], [200, 10]]))


def test_negative_points_are_out_of_bounds(method):
    with pytest.raises(ValueError, match=r"bounds"):
        method.set_points(np.array([[-1, 10]]))


# ---------------------------------------------------------------------------
# Rounding: nearest, not truncation
# ---------------------------------------------------------------------------

def test_float_points_round_to_nearest_not_truncate(method):
    """The specific regression being guarded: 1.7 must become 2, not 1.

    The old LucasKanade code truncated toward zero (int(x)); the old
    SimplifiedOpticalFlow code hard-crashed on non-integer input.
    """
    with pytest.warns(UserWarning):
        method.set_points(np.array([[1.7, 2.2], [50.4, 60.6]]))

    assert np.issubdtype(method.points.dtype, np.integer)
    np.testing.assert_array_equal(method.points, np.array([[2, 2], [50, 61]]))


def test_rounding_warning_fires_only_when_values_actually_change(method):
    with pytest.warns(UserWarning, match=r"rounded"):
        method.set_points(np.array([[1.5, 2.0], [3.0, 4.0]]))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        method.set_points(np.array([[1, 2], [3, 4]]))
    assert not [w for w in caught if issubclass(w.category, UserWarning)]


def test_integer_points_round_trip_unchanged(method):
    points = np.array([[10, 20], [50, 100], [0, 0], [127, 255]])
    method.set_points(points)
    np.testing.assert_array_equal(method.points, points)
    assert np.issubdtype(method.points.dtype, np.integer)


# ---------------------------------------------------------------------------
# Duck typing
# ---------------------------------------------------------------------------

def test_accepts_object_with_points_attribute(method):
    """This is what makes ``method.set_points(selection_gui)`` work."""
    gui = DummySelectionGUI(points=[[10, 20], [30, 40]])
    method.set_points(gui)
    np.testing.assert_array_equal(method.points, np.array([[10, 20], [30, 40]]))


def test_duck_typed_points_are_still_validated(method):
    """Duck-typed input goes through the same validation as a plain array."""
    gui = DummySelectionGUI(points=[[1, 2, 3]])
    with pytest.raises(ValueError, match=r"two columns"):
        method.set_points(gui)


# ---------------------------------------------------------------------------
# Bounds check is skipped without usable video info
# ---------------------------------------------------------------------------

def test_bounds_check_skipped_when_video_lacks_image_size():
    obj = _bare_method_without_video_info()
    wild_points = np.array([[-999, 99999], [5, 5]])

    # must not raise, even though these coordinates would be out of bounds
    # for any real image
    IDIMethod.set_points(obj, wild_points)
    np.testing.assert_array_equal(obj.points, wild_points)
