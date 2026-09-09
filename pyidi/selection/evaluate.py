"""Whole-image evaluation of subset quality.

An *evaluator* answers one question for every pixel of the image at once: how
well would a subset centred here track? The answer comes back as a *score
image* -- a ``float32`` array the shape of the input, with ``NaN`` wherever the
subset window would reach past the image edge.

Everything here is vectorised over the whole image. The point-by-point
alternative (one Sobel and one 2x2 eigendecomposition per subset, as in
``pyidi/GUIs/subset_selection.py``) costs roughly 100 us per subset, which is
fine for a few hundred grid points and takes minutes at one megapixel. The
box-filter formulation below is a handful of separable O(1)-per-pixel passes,
so scoring every pixel of a megapixel frame is a matter of tens of
milliseconds. That is the whole reason a mask can go back to meaning "where I
want points" rather than "how much scoring I can afford".

Coordinate convention: ``(row, col)`` throughout, i.e. plain numpy indexing.
The transpose that pyqtgraph's column-major image display needs belongs at the
GUI boundary, not here.

Why ``NaN`` for the invalid border rather than a companion boolean array: every
comparison against ``NaN`` is already ``False``, so an invalid pixel can never
be picked without a single explicit check anywhere downstream, and
``np.nanmax``/``np.nanpercentile`` normalise correctly by construction.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Tuple

import numpy as np
from scipy.ndimage import sobel, uniform_filter

from ..selection_geometry import _as_size_pair

#: Radius, in pixels, of the gradient operator every evaluator here uses. The
#: score at a pixel therefore depends on the image up to ``half + this`` away,
#: which is what the bounding-box crop below has to pad by.
GRADIENT_RADIUS = 1

#: Evaluate the bounding box rather than the whole frame once the box is at
#: most this fraction of the frame. Below it the crop saves real time; above it
#: the bookkeeping costs more than the convolutions it avoids.
CROP_AREA_FRACTION = 0.25


@dataclass(frozen=True)
class Parameter:
    """A single evaluator parameter, described well enough to build a widget from.

    :param name: keyword name, as the evaluator function takes it
    :type name: str
    :param kind: ``'float'``, ``'int'`` or ``'direction'`` (a ``(row, col)``
        pair). A GUI switches on this to decide which control to create.
    :type kind: str
    :param default: value used when the caller does not supply one
    :type default: object
    :param minimum: lower bound, or ``None`` when unbounded
    :type minimum: float or None
    :param maximum: upper bound, or ``None`` when unbounded
    :type maximum: float or None
    :param description: one-line explanation, suitable for a tooltip
    :type description: str
    """

    name: str
    kind: str
    default: Any
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    description: str = ''


@dataclass(frozen=True)
class Evaluator:
    """A registered evaluator: the function plus everything needed to drive it.

    :param name: registry key, e.g. ``'shi_tomasi'``
    :type name: str
    :param display_name: human-readable name for a menu
    :type display_name: str
    :param function: ``f(image, window, **params) -> ndarray``, where ``window``
        is the ``(rows, cols)`` scoring window. It scores the whole array and
        need not care about the border -- :func:`evaluate` masks that off.
    :type function: callable
    :param parameters: descriptors for every keyword the function takes beyond
        ``image`` and ``window``
    :type parameters: tuple[Parameter, ...]
    :param description: one-line explanation of what the score means
    :type description: str
    """

    name: str
    display_name: str
    function: Callable
    parameters: Tuple[Parameter, ...] = field(default_factory=tuple)
    description: str = ''


_REGISTRY = {}


def register_evaluator(evaluator):
    """Add an evaluator to the registry, replacing any existing one of that name.

    :param evaluator: the evaluator to register
    :type evaluator: Evaluator
    :return: the evaluator, so this can be used as a decorator-ish one-liner
    :rtype: Evaluator
    """
    _REGISTRY[evaluator.name] = evaluator
    return evaluator


def available_evaluators():
    """Every registered evaluator, keyed by name.

    :return: a copy of the registry, safe to iterate while registering
    :rtype: dict[str, Evaluator]
    """
    return dict(_REGISTRY)


def get_evaluator(name):
    """Look up a registered evaluator by name.

    :param name: registry key
    :type name: str
    :return: the evaluator
    :rtype: Evaluator
    :raises ValueError: if no evaluator of that name is registered
    """
    if name not in _REGISTRY:
        known = ', '.join(sorted(_REGISTRY)) or '(none registered)'
        raise ValueError(f"Unknown evaluator {name!r}. Registered evaluators: {known}.")
    return _REGISTRY[name]


def resolve_parameters(evaluator, params):
    """Fill in an evaluator's defaults and reject anything it does not accept.

    :param evaluator: the evaluator whose descriptors define the accepted keys
    :type evaluator: Evaluator
    :param params: caller-supplied parameters, possibly partial
    :type params: dict
    :return: every parameter the evaluator takes, defaults filled in
    :rtype: dict
    :raises ValueError: if ``params`` contains a key the evaluator does not take
    """
    accepted = {p.name: p.default for p in evaluator.parameters}
    unknown = set(params) - set(accepted)
    if unknown:
        known = ', '.join(sorted(accepted)) or '(none)'
        raise ValueError(
            f"Evaluator {evaluator.name!r} does not take {sorted(unknown)}. Parameters: {known}."
        )
    accepted.update(params)
    return accepted


def window_size(subset_size):
    """The scoring window, per axis, for a subset size.

    Always odd, so the window is symmetric about the pixel it scores: an even
    ``subset_size`` of 10 gives an 11-pixel window, exactly reproducing the
    ``img[c - half : c + half + 1]`` slicing used for subsets elsewhere in the
    package. An even-width box filter would instead sit half a pixel off centre.

    :param subset_size: scalar, or a ``(height, width)`` pair
    :type subset_size: int or tuple
    :return: ``(rows, cols)`` window extent, both odd
    :rtype: tuple[int, int]
    """
    h, w = _as_size_pair(subset_size)
    return 2 * (int(h) // 2) + 1, 2 * (int(w) // 2) + 1


def half_window(subset_size):
    """Half the scoring window, per axis -- the depth of the invalid border.

    :param subset_size: scalar, or a ``(height, width)`` pair
    :type subset_size: int or tuple
    :return: ``(rows, cols)`` half-extent
    :rtype: tuple[int, int]
    """
    win_r, win_c = window_size(subset_size)
    return win_r // 2, win_c // 2


def _gradients(image):
    """Sobel gradients of the whole image, as ``(d/drow, d/dcol)`` in float64.

    float64 rather than float32 because the box sums below reach ~1e13 for a
    16-bit image and an 11x11 window, which float32 cannot hold to the precision
    the equivalence test against the per-subset reference asks for. The returned
    score is narrowed back to float32.
    """
    img = np.asarray(image, dtype=np.float64)
    return sobel(img, axis=0), sobel(img, axis=1)


def _box_sum(values, window):
    """Sum of ``values`` over ``window``, via the O(1)-per-pixel mean."""
    return uniform_filter(values, size=window, mode='constant') * (window[0] * window[1])


def shi_tomasi(image, window):
    """Smaller eigenvalue of the gradient structure tensor summed over the window.

    The Shi-Tomasi corner criterion: high where the image content inside the
    subset constrains motion in *both* directions, so a subset on a plain edge
    scores low (it can slide along the edge) and one on a corner scores high.

    Computed in closed form rather than through ``eigvalsh``, which needs a
    Python-level loop::

        lambda_min = (a + c)/2 - sqrt(((a - c)/2)**2 + b**2)

    with ``a = sum(gx**2)``, ``c = sum(gy**2)``, ``b = sum(gx*gy)``. The sums are
    true sums over the window, not means, so the values are directly comparable
    with the per-subset implementation in ``SelectionGUIOld``.

    :param image: 2-D image, indexed ``[row, col]``
    :type image: numpy.ndarray
    :param window: ``(rows, cols)`` scoring window
    :type window: tuple[int, int]
    :return: score over the whole array, border included and meaningless
    :rtype: numpy.ndarray
    """
    g_row, g_col = _gradients(image)
    a = _box_sum(g_col * g_col, window)
    c = _box_sum(g_row * g_row, window)
    b = _box_sum(g_col * g_row, window)

    half_trace = 0.5 * (a + c)
    spread = np.sqrt(np.square(0.5 * (a - c)) + np.square(b))
    # The structure tensor is positive semi-definite, so the smaller eigenvalue
    # is non-negative; clip away the rounding noise that would otherwise leave a
    # flat region at -1e-20 instead of exactly zero.
    return np.maximum(half_trace - spread, 0.0)


def gradient_direction(image, window, direction=(0.0, 1.0)):
    """Summed squared image gradient projected onto one direction.

    Answers a narrower question than :func:`shi_tomasi`: not "can this subset be
    tracked at all" but "can it be tracked *along this axis*". Useful when only
    one component of the motion matters, e.g. a beam bending in one plane.

    :param image: 2-D image, indexed ``[row, col]``
    :type image: numpy.ndarray
    :param window: ``(rows, cols)`` scoring window
    :type window: tuple[int, int]
    :param direction: ``(row, col)`` direction to project onto; normalised
        internally, so only its orientation matters
    :type direction: tuple[float, float]
    :return: score over the whole array, border included and meaningless
    :rtype: numpy.ndarray
    :raises ValueError: if ``direction`` has zero length
    """
    d = np.asarray(direction, dtype=np.float64).ravel()
    if d.size != 2:
        raise ValueError(f'direction must be a (row, col) pair, got {len(d)} values.')
    norm = np.hypot(d[0], d[1])
    if norm == 0:
        raise ValueError('direction must be non-zero.')
    d = d / norm

    g_row, g_col = _gradients(image)
    projected = d[0] * g_row + d[1] * g_col
    return _box_sum(projected * projected, window)


register_evaluator(Evaluator(
    name='shi_tomasi',
    display_name='Shi-Tomasi',
    function=shi_tomasi,
    parameters=(),
    description='Corner strength: high where the subset is constrained in both directions.',
))

register_evaluator(Evaluator(
    name='gradient_direction',
    display_name='Gradient in direction',
    function=gradient_direction,
    parameters=(
        Parameter(
            name='direction',
            kind='direction',
            default=(0.0, 1.0),
            description='(row, col) direction the gradient is projected onto.',
        ),
    ),
    description='Gradient strength along one chosen direction.',
))


def _bounding_box(mask):
    """Inclusive-exclusive ``(r0, r1, c0, c1)`` bounding box of a boolean mask.

    :return: the box, or ``None`` if the mask is empty
    :rtype: tuple or None
    """
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    if not rows.size or not cols.size:
        return None
    return int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1


def _evaluation_box(shape, mask, crop, half):
    """Decide which slice of the image to run the evaluator on.

    Returns the *padded* box to evaluate together with the sub-box whose scores
    that padding makes trustworthy. The padding is ``half + GRADIENT_RADIUS``:
    the score at a pixel reads the gradient up to ``half`` away, and each
    gradient reads the image one further.

    The crop follows the mask's bounding *box*, never its shape. Handing the
    evaluator a mask-shaped image would let the mask boundary act as an image
    edge and manufacture a strong gradient all along it, and it would force a
    recompute on every brush stroke.

    :return: ``((r0, r1, c0, c1), (vr0, vr1, vc0, vc1))`` -- the box to evaluate
        and the globally-valid box within it -- or ``None`` when nothing is
        worth evaluating
    :rtype: tuple or None
    """
    h, w = shape
    half_r, half_c = half

    box = None
    if crop is not False and mask is not None:
        bbox = _bounding_box(mask)
        if bbox is None:
            return None
        auto = (bbox[1] - bbox[0]) * (bbox[3] - bbox[2]) <= CROP_AREA_FRACTION * h * w
        if crop is True or auto:
            box = bbox

    if box is None:
        r0, r1, c0, c1 = 0, h, 0, w
    else:
        pad_r, pad_c = half_r + GRADIENT_RADIUS, half_c + GRADIENT_RADIUS
        r0, r1 = max(0, box[0] - pad_r), min(h, box[1] + pad_r)
        c0, c1 = max(0, box[2] - pad_c), min(w, box[3] + pad_c)

    # A score is trustworthy where the evaluated slice supplied every pixel it
    # depends on. Where the slice edge *is* the image edge the slice saw exactly
    # what a full-image evaluation would have, so only the subset-window rule
    # applies there.
    valid_r0 = max(half_r, r0 + half_r + GRADIENT_RADIUS if r0 > 0 else half_r)
    valid_r1 = min(h - half_r, r1 - half_r - GRADIENT_RADIUS if r1 < h else h - half_r)
    valid_c0 = max(half_c, c0 + half_c + GRADIENT_RADIUS if c0 > 0 else half_c)
    valid_c1 = min(w - half_c, c1 - half_c - GRADIENT_RADIUS if c1 < w else w - half_c)

    if valid_r0 >= valid_r1 or valid_c0 >= valid_c1:
        return None
    return (r0, r1, c0, c1), (valid_r0, valid_r1, valid_c0, valid_c1)


def evaluate(image, evaluator='shi_tomasi', subset_size=11, mask=None, crop=None, **params):
    """Score every subset position in the image.

    :param image: 2-D image, indexed ``[row, col]``
    :type image: numpy.ndarray
    :param evaluator: registry name of the evaluator to run
    :type evaluator: str
    :param subset_size: scalar, or a ``(height, width)`` pair
    :type subset_size: int or tuple
    :param mask: boolean array the shape of ``image``; only used to decide the
        bounding-box crop, never to restrict which pixels are scored inside it
    :type mask: numpy.ndarray or None
    :param crop: ``True`` to force the bounding-box crop, ``False`` to forbid it,
        ``None`` (default) to crop when the box is at most
        :data:`CROP_AREA_FRACTION` of the frame
    :type crop: bool or None
    :param params: evaluator-specific parameters; see its descriptors
    :return: ``float32`` score image the shape of ``image``, ``NaN`` wherever the
        subset window would leave the image or the crop leaves the score unknown
    :rtype: numpy.ndarray
    :raises ValueError: if ``image`` is not 2-D, the evaluator is unknown, or a
        parameter is not one the evaluator takes
    """
    image = np.asarray(image)
    if image.ndim != 2:
        raise ValueError(f'image must be 2-D (row, col), got shape {image.shape}.')

    spec = get_evaluator(evaluator)
    resolved = resolve_parameters(spec, params)
    window = window_size(subset_size)
    half = (window[0] // 2, window[1] // 2)

    score = np.full(image.shape, np.nan, dtype=np.float32)

    boxes = _evaluation_box(image.shape, mask, crop, half)
    if boxes is None:
        return score

    (r0, r1, c0, c1), (vr0, vr1, vc0, vc1) = boxes
    raw = spec.function(image[r0:r1, c0:c1], window, **resolved)
    score[vr0:vr1, vc0:vc1] = raw[vr0 - r0:vr1 - r0, vc0 - c0:vc1 - c0]
    return score
