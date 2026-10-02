"""Lazy random array creation."""

from ..autoray import do, random_array, register_backend
from .core import LazyArray, parse_creation_shape


def _random_array(shape, like, **kwargs):
    # infer backend, dtype and device from ``like`` at execution
    return do("random.array", shape, like=like, **kwargs)


_random_array.__name__ = "random_array"
# draw new samples on each compiled call
_random_array._autoray_nonfoldable = True


def _default_rng(seed, like, **kwargs):
    return do("random.default_rng", seed, like=like, **kwargs)


_default_rng.__name__ = "default_rng"
# make a new generator on each compiled call
_default_rng._autoray_nonfoldable = True


class LazyGenerator:
    """A lazy random generator, whose draws are lazy arrays that depend on a
    single generator node. The concrete generator is created when that node
    is computed.

    Parameters
    ----------
    node : LazyArray
        The node that creates the concrete generator.
    like : str or LazyArray
        The backend name or lazy array that draws take ``like`` from.
    """

    __slots__ = ("_node", "_like")

    def __init__(self, node, like):
        self._node = node
        self._like = like

    @property
    def backend(self):
        return self._node.backend

    def _draw(self, size, **kwargs):
        if size is None:
            size = ()
        return array(size, rng=self, **kwargs)

    def normal(self, loc=0.0, scale=1.0, size=None, **kwargs):
        return self._draw(size, dist="normal", loc=loc, scale=scale, **kwargs)

    def standard_normal(self, size=None, **kwargs):
        return self._draw(size, dist="normal", **kwargs)

    def uniform(self, low=0.0, high=1.0, size=None, **kwargs):
        return self._draw(
            size, dist="uniform", loc=low, scale=high - low, **kwargs
        )

    def random(self, size=None, **kwargs):
        return self._draw(size, dist="uniform", **kwargs)

    def __repr__(self):
        return f"<LazyGenerator(backend={self.backend!r})>"


register_backend(LazyGenerator, "autoray.lazy")


def default_rng(seed=None, backend="numpy", _like=None, **kwargs):
    """Create a lazy random generator.

    Parameters
    ----------
    seed : None, int, LazyGenerator or backend seed, optional
        Seed passed to the concrete ``random.default_rng`` when computed. A
        ``LazyGenerator`` is returned unchanged.
    backend : str, optional
        Backend used when ``_like`` is ``None``. Defaults to ``"numpy"``.
    _like : LazyArray or None, optional
        Lazy array that supplies the backend, dtype and device at execution.
    **kwargs
        Options passed to ``random.default_rng`` when computed.

    Returns
    -------
    LazyGenerator
    """
    if isinstance(seed, LazyGenerator):
        return seed

    if _like is None:
        like = backend
    else:
        like = _like
        backend = _like.backend

    node = LazyArray(
        backend=backend,
        fn=_default_rng,
        args=(seed, like),
        kwargs=kwargs,
        shape=(),
    )
    return LazyGenerator(node, like)


def array(shape, rng=None, backend="numpy", _like=None, **kwargs):
    """Create lazy random samples.

    Extra options, such as ``dist``, ``dtype`` and ``device``, are passed to
    :func:`~autoray.autoray.random_array` when the array is computed.

    Parameters
    ----------
    shape : tuple[int]
        Shape of the output array.
    rng : None, int, LazyGenerator or backend generator, optional
        Passed to ``random.array`` when computed. A ``LazyGenerator`` also
        supplies the backend when ``_like`` is ``None``.
    backend : str, optional
        Backend used when ``_like`` and ``rng`` give none. Defaults to
        ``"numpy"``.
    _like : LazyArray or None, optional
        Lazy array that supplies the backend, dtype and device at execution.
    **kwargs
        Options passed to ``random.array`` when computed.

    Returns
    -------
    LazyArray
        Random samples as a lazy array.
    """
    shape = parse_creation_shape(shape)

    if isinstance(rng, LazyGenerator):
        if _like is None:
            _like = rng._like
        rng = rng._node

    if _like is None:
        like = backend
    elif isinstance(_like, str):
        like = backend = _like
    else:
        # the concrete backend, dtype and device are taken from this later
        like = _like
        backend = _like.backend

    if rng is not None:
        kwargs["rng"] = rng

    return LazyArray(
        backend=backend,
        fn=_random_array,
        args=(shape, like),
        kwargs=kwargs,
        shape=shape,
    )


random_array.register("autoray.lazy", array)


__all__ = (
    "LazyGenerator",
    "array",
    "default_rng",
)
