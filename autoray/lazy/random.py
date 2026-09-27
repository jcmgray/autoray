"""Lazy random array creation."""

from ..autoray import do, random_array
from .core import LazyArray, parse_creation_shape


def _random_array(shape, like, **kwargs):
    # infer backend, dtype and device from ``like`` at execution
    return do("random.array", shape, like=like, **kwargs)


_random_array.__name__ = "random_array"
# draw new samples on each compiled call
_random_array._autoray_nonfoldable = True


def array(shape, rng=None, backend="numpy", _like=None, **kwargs):
    """Create lazy random samples using the backend's shared state.

    Extra options, such as ``dist``, ``dtype`` and ``device``, are passed to
    :func:`~autoray.autoray.random_array` when the array is computed.

    Parameters
    ----------
    shape : tuple[int]
        Shape of the output array.
    rng : None, optional
        Must be ``None``. Seeds and generators are not supported.
    backend : str, optional
        Backend used when ``_like`` is ``None``. Defaults to ``"numpy"``.
    _like : LazyArray or None, optional
        Lazy array that supplies the backend, dtype and device at execution.
    **kwargs
        Options passed to ``random.array`` when computed.

    Returns
    -------
    LazyArray
        Random samples as a lazy array.
    """
    if rng is not None:
        raise TypeError("lazy random.array supports only rng=None")

    shape = parse_creation_shape(shape)
    if _like is None:
        like = backend
    else:
        # the concrete backend, dtype and device are taken from this later
        like = _like
        backend = _like.backend

    return LazyArray(
        backend=backend,
        fn=_random_array,
        args=(shape, like),
        kwargs=kwargs,
        shape=shape,
    )


random_array.register("autoray.lazy", array)


__all__ = ("array",)
