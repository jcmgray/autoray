import functools
import importlib.util
import os

import pytest

_ALL_BACKENDS = [
    "cupy",
    "dask",
    "jax",
    "mlx",
    "numpy",
    "paddle",
    "sparse",
    "tensorflow",
    "torch",
]

_AVAILABLE_BACKENDS = set()
for _lib in _ALL_BACKENDS:
    _lib_available = importlib.util.find_spec(_lib) is not None
    if _lib_available:
        _AVAILABLE_BACKENDS.add(_lib)
        if _lib == "jax":
            import jax

            jax.config.update("jax_enable_x64", True)
            jax.config.update("jax_platform_name", "cpu")
            os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

        elif _lib == "mlx":
            # default to cpu, since many linear algebra decompositions are
            # only supported on cpu, and we are only testing the interface
            import mlx.core as mx

            mx.set_default_device(mx.cpu)


FLOAT_DTYPES = ("float32", "float64")
COMPLEX_DTYPES = ("complex64", "complex128")
ALL_DTYPES = FLOAT_DTYPES + COMPLEX_DTYPES


# xfail registry
# keys:
#   (backend, dtype)     → global dtype xfail (any fn should xfail)
#   (backend, fn)        → function xfail (any dtype should xfail)
#   (backend, fn, dtype) → only specific function+dtype xfail
#
# values:
#   str                        → always xfail (parametrize + test time)
#   callable(args, kwargs)     → returns reason str or None (test time only)

XFAILS = {
    # cupy
    ("cupy", "choice"): "cupy: no choice interface yet",
    ("cupy", "gumbel"): "cupy: no gumbel interface yet",
    ("cupy", "linalg.eig"): "cupy doesn't support linalg.eig",
    ("cupy", "normal"): "cupy: no normal interface yet",
    ("cupy", "permutation"): "cupy: no permutation interface yet",
    (
        "cupy",
        "scipy.linalg.solve_triangular",
    ): "cupy doesn't support scipy.linalg.solve_triangular",
    # dask
    # https://github.com/dask/dask/issues/12335
    ("dask", "linalg.cholesky", "complex64"): "dask complex cholesky broken",
    ("dask", "linalg.cholesky", "complex128"): "dask complex cholesky broken",
    ("dask", "linalg.eig"): "dask doesn't support linalg.eig",
    ("dask", "linalg.eigh"): "dask doesn't support linalg.eigh",
    ("dask", "linalg.norm"): lambda a, kw: (
        "dask doesn't support linalg.norm with ord=2 and ndim>2"
        if kw.get("ord") == 2 and a[0] == 3
        else None
    ),
    ("dask", "scipy.linalg.expm"): "dask doesn't support scipy",
    (
        "dask",
        "scipy.linalg.solve_triangular",
    ): "dask doesn't support scipy.linalg.solve_triangular",
    ("dask", "split"): "dask doesn't support split",
    ("dask", "linalg.svd"): lambda a, kw: (
        "dask svd doesn't support batched" if a and a[0] == "batched" else None
    ),
    # mlx
    ("mlx", "binomial"): "mlx: no binomial interface yet",
    ("mlx", "choice"): "mlx: no choice interface yet",
    ("mlx", "complex", "float64"): "mlx doesn't support complex128",
    ("mlx", "complex128"): "mlx doesn't support complex128 dtype",
    ("mlx", "einsum"): lambda a, kw: (
        "mlx einsum doesn't support interleaved format"
        if a[0] == "interleaved"
        else None
    ),
    ("mlx", "empty"): "mlx: no `empty` function yet",
    ("mlx", "exponential"): "mlx: no exponential interface yet",
    ("mlx", "indices"): "mlx doesn't support indices",
    (
        "mlx",
        "linalg.cholesky",
        "complex64",
    ): "mlx doesn't support complex64 cholesky",
    (
        "mlx",
        "linalg.inv",
        "complex64",
    ): "mlx doesn't support complex64 inv",
    (
        "mlx",
        "linalg.qr",
        "complex64",
    ): "mlx doesn't support complex64 qr",
    (
        "mlx",
        "linalg.solve",
        "complex64",
    ): "mlx doesn't support complex64 solve",
    ("mlx", "moveaxis"): lambda a, kw: (
        "mlx doesn't support multi axis move" if a[0] == "multiple" else None
    ),
    ("mlx", "nonzero"): "mlx doesn't support nonzero",
    ("mlx", "poisson"): "mlx: no poisson interface yet",
    ("mlx", "random.uniform", "float64"): "mlx doesn't support float64 random",
    ("mlx", "scipy.linalg.expm"): "mlx doesn't support scipy",
    (
        "mlx",
        "scipy.linalg.solve_triangular",
    ): "mlx doesn't support scipy.linalg.solve_triangular",
    # paddle
    ("paddle", "eye", "complex128"): "paddle doesn't support complex eye",
    ("paddle", "eye", "complex64"): "paddle doesn't support complex eye",
    (
        "paddle",
        "identity",
        "complex128",
    ): "paddle doesn't support complex identity",
    (
        "paddle",
        "identity",
        "complex64",
    ): "paddle doesn't support complex identity",
    (
        "paddle",
        "linalg.cholesky",
        "complex128",
    ): "paddle doesn't support complex cholesky",
    ("paddle", "linalg.eig"): "paddle doesn't support linalg.eig",
    (
        "paddle",
        "linalg.solve",
        "complex128",
    ): "paddle doesn't support complex linalg.solve",
    (
        "paddle",
        "linalg.solve",
        "complex64",
    ): "paddle doesn't support complex linalg.solve",
    (
        "paddle",
        "linalg.svd",
        "complex128",
    ): "paddle doesn't support complex SVD",
    ("paddle", "scipy.linalg.expm"): "paddle doesn't support scipy",
    (
        "paddle",
        "scipy.linalg.solve_triangular",
    ): "paddle doesn't support scipy.linalg.solve_triangular",
    ("paddle", "take"): "paddle doesn't support take",
    ("paddle", "where"): "paddle doesn't support where",
    # sparse
    ("sparse", "allclose"): "sparse doesn't support allclose",
    ("sparse", "arange"): "sparse doesn't support arange",
    ("sparse", "array"): "sparse needs explicit constructor",
    ("sparse", "asarray"): "sparse needs explicit constructor",
    ("sparse", "cumsum"): "sparse doesn't support cumsum",
    ("sparse", "diag"): "sparse doesn't support diag",
    ("sparse", "indices"): "sparse doesn't support indices",
    ("sparse", "linalg.cholesky"): "sparse doesn't support linalg",
    ("sparse", "linalg.eig"): "sparse doesn't support linalg",
    ("sparse", "linalg.eigh"): "sparse doesn't support linalg",
    ("sparse", "linalg.inv"): "sparse doesn't support linalg",
    ("sparse", "linalg.norm"): "sparse doesn't support linalg",
    ("sparse", "linalg.qr"): "sparse doesn't support linalg",
    ("sparse", "linalg.solve"): "sparse doesn't support linalg",
    ("sparse", "linalg.svd"): "sparse doesn't support linalg",
    ("sparse", "max"): lambda a, kw: (
        "sparse doesn't support max with axis" if a else None
    ),
    ("sparse", "mean"): lambda a, kw: (
        "sparse doesn't support mean with axis" if a else None
    ),
    ("sparse", "power"): "sparse doesn't support power",
    ("sparse", "ravel"): "sparse doesn't support ravel",
    ("sparse", "scipy.linalg.expm"): "sparse doesn't support scipy",
    (
        "sparse",
        "scipy.linalg.solve_triangular",
    ): "sparse doesn't support scipy.linalg.solve_triangular",
    ("sparse", "split"): "sparse doesn't support split",
    ("sparse", "swapaxes"): "sparse doesn't support swapaxes",
    ("sparse", "take"): "sparse doesn't support take",
    ("sparse", "trace"): "sparse doesn't support trace",
    ("sparse", "where"): "sparse doesn't support where",
    # tensorflow
    ("tensorflow", "binomial"): "tensorflow: no binomial interface yet",
    ("tensorflow", "choice"): "tensorflow: no choice interface yet",
    ("tensorflow", "empty"): "tensorflow doesn't support empty",
    ("tensorflow", "exponential"): "tensorflow: no exponential interface yet",
    ("tensorflow", "full"): "tensorflow doesn't support full",
    ("tensorflow", "gumbel"): "tensorflow: no gumbel interface yet",
    ("tensorflow", "permutation"): "tensorflow: no permutation interface yet",
    ("tensorflow", "poisson"): "tensorflow: no poisson interface yet",
    ("tensorflow", "scipy.linalg.expm"): "tensorflow doesn't support scipy",
    # torch
    ("torch", "binomial"): "torch: no binomial interface yet",
    ("torch", "exponential"): "torch: no exponential interface yet",
    ("torch", "gumbel"): "torch: no gumbel interface yet",
    ("torch", "poisson"): "torch: no poisson interface yet",
    ("torch", "prod"): lambda a, kw: (
        "torch doesn't support prod with tuple axis"
        if isinstance(kw.get("axis"), tuple)
        else None
    ),
    ("torch", "scipy.linalg.expm"): "torch doesn't support scipy",
}


def _get_xfail(backend, fn=None, dtype=None, args=(), kwargs=None):
    """Return xfail reason if this combo is registered, else None. String
    entries are always checked. Callable entries are only evaluated when args
    or kwargs are provided.
    """
    # we need to check multiply keys so that more specific entries (fn+dtype)
    # can be caught be more global registered xfails (fn or dtype only)
    keys = []
    if fn and dtype:
        keys.append((backend, fn, dtype))
    if fn:
        keys.append((backend, fn))
    if dtype:
        keys.append((backend, dtype))

    if kwargs is None:
        kwargs = {}
    check_callables = bool(args or kwargs)

    for key in keys:
        reason_or_sig_checker = XFAILS.get(key)
        if reason_or_sig_checker is None:
            # not registered as xfail combo
            continue

        if isinstance(reason_or_sig_checker, str):
            # blanket xfail
            return reason_or_sig_checker

        if check_callables and callable(reason_or_sig_checker):
            # xfail depends on args/kwargs
            reason = reason_or_sig_checker(args, kwargs)
            if reason:
                return reason

    # all valid
    return None


@functools.cache
def _get_marks(
    backend,
    fn=None,
    dtype=None,
    fn_args=(),
    fn_kwargs=(),
    requires=None,
):
    """Compute skip/xfail marks for a parameter combination. Parameters are all
    hashable for caching. ``fn_kwargs`` is a tuple of ``(key, value)`` pairs
    (converted to dict internally).
    """
    marks = []

    if backend not in _AVAILABLE_BACKENDS:
        marks.append(pytest.mark.skipif(True, reason=f"No {backend}."))

    kwargs = dict(fn_kwargs) if fn_kwargs else {}
    reason = _get_xfail(
        backend, fn=fn, dtype=dtype, args=fn_args, kwargs=kwargs
    )
    if reason:
        marks.append(pytest.mark.xfail(reason=reason, strict=True))

    if requires is not None:
        if isinstance(requires, str):
            requires = [requires]

        for required_fn in requires:
            reason = _get_xfail(backend, dtype=dtype, fn=required_fn)
            if reason:
                marks.append(pytest.mark.skip(reason=reason))
                break

    return tuple(marks)


def xfail_if(backend, fn=None, dtype=None, args=(), kwargs=None):
    """Call at top of test body to xfail registered combos. Checks both string
    and callable entries. Use this when the xfail depends on args/kwargs not
    known at parametrize time.
    """
    reason = _get_xfail(backend, fn=fn, dtype=dtype, args=args, kwargs=kwargs)
    if reason:
        pytest.xfail(reason)


def _normalize_fns(fns):
    """Normalize fn specs into ``(name, args, kwargs)`` triples.

    Each entry in ``fns`` can be:

    - ``"fn_name"`` — just a function name
    - ``("fn_name", args_tuple)`` — name + positional args
    - ``("fn_name", args_tuple, kwargs_dict)`` — name + args + kwargs

    Returns ``(normalized, use_args, use_kwargs)`` where ``use_args`` /
    ``use_kwargs`` indicate whether any entry carried args or kwargs
    (controls what values appear in the yielded ``pytest.param``).
    """
    normalized = []
    use_args = False
    use_kwargs = False

    for f in fns:
        if isinstance(f, str):
            normalized.append((f, (), {}))
        else:
            name = f[0]
            args = tuple(f[1]) if len(f) > 1 else ()
            use_args = True
            kwargs = f[2] if len(f) > 2 else {}
            if kwargs:
                use_kwargs = True
            normalized.append((name, args, kwargs))

    return normalized, use_args, use_kwargs


def gen_params(backends=None, dtypes=None, fns=None, requires=None):
    """Generate pytest.param tuples for the cartesian product of backends x
    dtypes x fns, with skip/xfail marks attached.

    Parameters
    ----------
    backends : list, Ellipsis, or None
        ``...`` for all backends, list for subset, None to omit.
    dtypes : list, Ellipsis, or None
        ``...`` for ALL_DTYPES (float32, float64, complex64, complex128),
        list for subset, None to omit.
    fns : list or None
        Function specs. Each entry can be:

        - ``"fn_name"`` — just a name
        - ``("fn_name", args)`` — name + positional args tuple
        - ``("fn_name", args, kwargs)`` — name + args + kwargs dict

        When any entry carries args (or kwargs), *every* yielded
        param includes an ``args`` (and ``kwargs``) value so the
        test signature stays consistent.
    requires : str, list[str] or None
        Optional list of functions required for this test, but which are not
        being explicitly parametrized over. If any of these are not supported
        by a backend, the test will be skipped for that backend with an
        appropriate reason.

    Returns
    -------
    list[pytest.param]
    """
    if backends is None:
        raise ValueError("backends is required")
    elif backends is ...:
        backends = _ALL_BACKENDS
    else:
        backends = list(backends)

    if dtypes is None:
        dtypes = (None,)
    elif dtypes is ...:
        dtypes = ALL_DTYPES
    else:
        dtypes = list(dtypes)

    # normalize fns
    if fns is None:
        fn_specs = [(None, (), {})]
        use_args = False
        use_kwargs = False
    else:
        fn_specs, use_args, use_kwargs = _normalize_fns(fns)

    result = []
    for backend in backends:
        for dtype in dtypes:
            for fn_name, fn_args, fn_kwargs in fn_specs:
                marks = _get_marks(
                    backend,
                    fn=fn_name,
                    dtype=dtype,
                    fn_args=fn_args,
                    fn_kwargs=tuple(sorted(fn_kwargs.items())),
                    requires=requires,
                )
                values = [backend]
                if dtype is not None:
                    values.append(dtype)
                if fn_name is not None:
                    values.append(fn_name)
                if use_args:
                    values.append(fn_args)
                if use_kwargs:
                    values.append(fn_kwargs)
                result.append(pytest.param(*values, marks=marks))

    return result


# backward compat
BACKENDS = gen_params(backends=...)
