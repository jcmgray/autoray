from .autoray import (
    infer_backend,
    get_lib_fn,
    do,
    conj,
    transpose,
    dag,
    real,
    imag,
    reshape,
    to_backend_dtype,
    astype,
    get_dtype_name,
    to_numpy,
    # the numpy mimic submodule
    numpy,
)


__all__ = (
    'do',
    'infer_backend',
    'get_lib_fn',
    'conj',
    'transpose',
    'dag',
    'real',
    'imag',
    'reshape',
    'to_backend_dtype',
    'get_dtype_name',
    'astype',
    'to_numpy',
    # the numpy mimic submodule
    'numpy',
)

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
