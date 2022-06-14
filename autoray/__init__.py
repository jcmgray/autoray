from .autoray import (
    do,
    get_backend,
    set_backend,
    backend_like,
    infer_backend,
    get_lib_fn,
    conj,
    transpose,
    dag,
    real,
    imag,
    reshape,
    to_backend_dtype,
    astype,
    get_dtype_name,
    get_common_dtype,
    to_numpy,
    register_backend,
    register_function,
    # the numpy mimic submodule
    numpy,
)
from .compiler import autojit
from . import lazy


__all__ = (
    "do",
    "get_backend",
    "set_backend",
    "backend_like",
    "infer_backend",
    "get_lib_fn",
    "conj",
    "transpose",
    "dag",
    "real",
    "imag",
    "reshape",
    "to_backend_dtype",
    "get_dtype_name",
    "get_common_dtype",
    "astype",
    "to_numpy",
    "register_backend",
    "register_function",
    # the numpy mimic submodule
    "numpy",
    # abstract function compilation
    "autojit",
    # lazy array library
    "lazy",
)

from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions
