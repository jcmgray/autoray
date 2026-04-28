from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
from importlib.metadata import version as _version

try:
    __version__ = _version("autoray")
except _PackageNotFoundError:
    try:
        # fallback for source trees where hatch-vcs has generated _version.py.
        from ._version import version as __version__
    except ImportError:
        __version__ = "0.0.0+unknown"


# useful constants
from math import (
    e,
    inf,
    nan,
    pi,
)

from . import lazy
from .autoray import (
    DoFunc,
    astype,
    backend_like,
    compose,
    conj,
    dag,
    do,
    get_backend,
    get_common_dtype,
    get_dtype_name,
    get_lib_fn,
    get_namespace,
    imag,
    infer_backend,
    infer_backend_multi,
    is_array,
    is_scalar,
    ndim,
    numpy,
    real,
    register_backend,
    register_function,
    reshape,
    set_backend,
    shape,
    size,
    to_backend_dtype,
    to_numpy,
    transpose,
    tree_apply,
    tree_flatten,
    tree_iter,
    tree_map,
    tree_unflatten,
)
from .compiler import autojit
from .grad import stop_gradient

__all__ = (
    "astype",
    "autojit",
    "backend_like",
    "compose",
    "conj",
    "dag",
    "do",
    "DoFunc",
    "e",
    "get_backend",
    "get_common_dtype",
    "get_dtype_name",
    "get_lib_fn",
    "get_namespace",
    "imag",
    "inf",
    "infer_backend_multi",
    "infer_backend",
    "is_array",
    "is_scalar",
    "lazy",
    "nan",
    "ndim",
    "numpy",
    "pi",
    "real",
    "register_backend",
    "register_function",
    "reshape",
    "set_backend",
    "shape",
    "size",
    "stop_gradient",
    "to_backend_dtype",
    "to_numpy",
    "transpose",
    "tree_apply",
    "tree_flatten",
    "tree_iter",
    "tree_map",
    "tree_unflatten",
)
