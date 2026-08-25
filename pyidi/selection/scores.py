"""Named, cached score images.

A :class:`ScoreStore` holds one image and one subset size, and hands out score
images by name. The point of the name is that several scores coexist: a
Shi-Tomasi score for a speckled plate and a directional-gradient score for a
beam can both be live, each referenced by whichever selection entry wants it,
without either recomputing the other.

Caching is keyed on the ``(evaluator, parameters, subset size)`` triple rather
than on the name, so two names describing the same computation share one array.

What invalidates what is deliberate and coarse. Changing the image or the
subset size drops everything, because every cached array is wrong. Changing a
threshold, a mask or a selection parameter drops nothing, because none of them
enter a score. That asymmetry is the whole reason the interface can re-derive
points on every slider tick.

The store deliberately evaluates the *full frame* and never the bounding-box
crop that :func:`~pyidi.selection.evaluate.evaluate` also offers. A cropped
score is only valid inside its box, so it would have to be invalidated whenever
the mask grew -- exactly the "a mask edit never costs an evaluation" guarantee
this class exists to provide. Callers scoring a small region of a very large
frame once, headlessly, can pass ``crop=True`` to ``evaluate`` directly.
"""

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from ..selection_geometry import _as_size_pair
from .evaluate import evaluate, get_evaluator, resolve_parameters

#: How many score arrays a store keeps before dropping the least recently used.
#:
#: Each one is a ``float32`` the size of the frame -- 16 MB at 2560x1600 -- and
#: every distinct set of evaluator parameters is a different array. A spin box
#: dragged through sixty values therefore asks for sixty of them, and an
#: unbounded cache would hold the lot. Eight is comfortably more than the two or
#: three scores a session actually switches between, and recomputing one is tens
#: of milliseconds per megapixel.
DEFAULT_CACHE_SIZE = 8


def _freeze(value):
    """Turn a parameter value into something hashable and comparable.

    Sequences become tuples (recursively) so that ``[0, 1]`` and ``(0, 1)``
    produce the same cache key; a numpy scalar becomes a plain Python number so
    that ``np.float64(0.5)`` and ``0.5`` do too.
    """
    if isinstance(value, np.ndarray):
        return tuple(_freeze(v) for v in value.tolist())
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(v) for v in value)
    if isinstance(value, np.generic):
        return value.item()
    return value


@dataclass(frozen=True)
class ScoreSpec:
    """What a score image is: an evaluator, its parameters, and a subset size.

    Two specs that compare equal describe the same array, which is what makes
    this usable as a cache key.

    :param evaluator: registry name of the evaluator
    :type evaluator: str
    :param parameters: every parameter the evaluator takes, defaults filled in,
        as a sorted tuple of ``(name, frozen value)`` pairs
    :type parameters: tuple
    :param subset_size: ``(height, width)``
    :type subset_size: tuple[int, int]
    """

    evaluator: str
    parameters: Tuple[Tuple[str, object], ...]
    subset_size: Tuple[int, int]

    @classmethod
    def build(cls, evaluator, subset_size, **params):
        """Normalise loose arguments into a spec.

        :param evaluator: registry name of the evaluator
        :type evaluator: str
        :param subset_size: scalar or ``(height, width)`` pair
        :type subset_size: int or tuple
        :param params: evaluator-specific parameters, possibly partial
        :return: the normalised spec
        :rtype: ScoreSpec
        :raises ValueError: if the evaluator or a parameter name is unknown
        """
        spec = get_evaluator(evaluator)
        resolved = resolve_parameters(spec, params)
        frozen = tuple(sorted((k, _freeze(v)) for k, v in resolved.items()))
        h, w = _as_size_pair(subset_size)
        return cls(evaluator=evaluator, parameters=frozen, subset_size=(int(h), int(w)))

    def as_kwargs(self):
        """The parameters as a keyword dict, ready to pass to ``evaluate``.

        :rtype: dict
        """
        return dict(self.parameters)


class ScoreStore:
    """One image, one subset size, and every score computed from them.

    :param image: 2-D reference frame, indexed ``[row, col]``
    :type image: numpy.ndarray
    :param subset_size: scalar or ``(height, width)`` pair
    :type subset_size: int or tuple
    :param max_cached: how many score arrays to keep; see
        :data:`DEFAULT_CACHE_SIZE`
    :type max_cached: int
    """

    def __init__(self, image, subset_size=11, max_cached=DEFAULT_CACHE_SIZE):
        self._image = None
        self._subset_size = None
        self._definitions = {}
        self._cache = {}
        self.max_cached = max(1, int(max_cached))
        #: How many times an evaluator has actually run. Only ever increases,
        #: including across an invalidation, so a test or a GUI assertion can
        #: check "this interaction did not re-evaluate" by comparing before and
        #: after.
        self.n_evaluations = 0
        self.set_image(image)
        self.set_subset_size(subset_size)

    @property
    def image(self):
        """The reference frame every score is computed from.

        :rtype: numpy.ndarray
        """
        return self._image

    @property
    def subset_size(self):
        """The ``(height, width)`` subset size every score is computed for.

        :rtype: tuple[int, int]
        """
        return self._subset_size

    @property
    def names(self):
        """The names of the scores currently defined, in definition order.

        :rtype: list[str]
        """
        return list(self._definitions)

    def set_image(self, image):
        """Replace the reference frame, discarding every cached score.

        :param image: 2-D frame, indexed ``[row, col]``
        :type image: numpy.ndarray
        :raises ValueError: if ``image`` is not 2-D
        """
        image = np.asarray(image)
        if image.ndim != 2:
            raise ValueError(f'image must be 2-D (row, col), got shape {image.shape}.')
        self._image = image
        self.invalidate()

    def set_subset_size(self, subset_size):
        """Replace the subset size, discarding every cached score.

        The score definitions survive: a name still refers to the same evaluator
        and parameters, and its array is recomputed at the new size on the next
        request.

        :param subset_size: scalar or ``(height, width)`` pair
        :type subset_size: int or tuple
        """
        h, w = _as_size_pair(subset_size)
        new_size = (int(h), int(w))
        if new_size == self._subset_size:
            return
        self._subset_size = new_size
        self._definitions = {
            name: ScoreSpec(spec.evaluator, spec.parameters, new_size)
            for name, spec in self._definitions.items()
        }
        self.invalidate()

    def invalidate(self):
        """Drop every cached array, keeping the definitions."""
        self._cache = {}

    def define(self, name, evaluator, **params):
        """Declare a named score. Nothing is computed until it is requested.

        Redefining an existing name replaces its spec.

        :param name: the name to address this score by
        :type name: str
        :param evaluator: registry name of the evaluator
        :type evaluator: str
        :param params: evaluator-specific parameters
        :return: the spec now bound to ``name``
        :rtype: ScoreSpec
        """
        spec = ScoreSpec.build(evaluator, self._subset_size, **params)
        self._definitions[name] = spec
        return spec

    def remove(self, name):
        """Forget a named score. The cached array survives if another name shares it.

        :param name: the name to drop
        :type name: str
        """
        self._definitions.pop(name, None)

    def spec(self, name):
        """The spec bound to a name.

        :param name: a defined score name
        :type name: str
        :rtype: ScoreSpec
        :raises KeyError: if no score of that name is defined
        """
        if name not in self._definitions:
            known = ', '.join(self._definitions) or '(none defined)'
            raise KeyError(f"No score named {name!r}. Defined scores: {known}.")
        return self._definitions[name]

    def get(self, name):
        """The score image for a name, computing it only if it is not cached.

        :param name: a defined score name
        :type name: str
        :return: ``float32`` score image, ``NaN`` on the invalid border
        :rtype: numpy.ndarray
        :raises KeyError: if no score of that name is defined
        """
        return self.get_for_spec(self.spec(name))

    def get_for_spec(self, spec):
        """The score image for a spec, computing it only if it is not cached.

        :param spec: what to compute
        :type spec: ScoreSpec
        :return: ``float32`` score image, ``NaN`` on the invalid border
        :rtype: numpy.ndarray
        """
        # Popped and reinserted, so that the dict's insertion order is
        # least-recently-used first and the eviction below is its first key.
        score = self._cache.pop(spec, None)
        if score is None:
            score = evaluate(
                self._image,
                evaluator=spec.evaluator,
                subset_size=spec.subset_size,
                crop=False,
                **spec.as_kwargs(),
            )
            self.n_evaluations += 1
        self._cache[spec] = score
        while len(self._cache) > self.max_cached:
            del self._cache[next(iter(self._cache))]
        return score

    def is_cached(self, name):
        """Whether a name's array is already computed.

        :param name: a defined score name
        :type name: str
        :rtype: bool
        """
        return self.spec(name) in self._cache
