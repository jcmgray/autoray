from . import linalg

from .core import (
    LazyArray,
    shared_intermediates,
    array,
    transpose,
    reshape,
    tensordot,
    einsum,
    trace,
    matmul,
    clip,
    flip,
    sort,
    argsort,
    stack,
    # binary
    multiply,
    add,
    floordivide,
    truedivide,
    # unary
    sin,
    cos,
    tan,
    arcsin,
    arccos,
    arctan,
    sinh,
    cosh,
    tanh,
    arcsinh,
    arccosh,
    arctanh,
    exp,
    log,
    log2,
    log10,
    conj,
    sign,
    real,
    imag,
    # reductions
    prod,
)
from .core import abs_ as abs
from .core import sum_ as sum
from .core import min_ as min
from .core import max_ as max

__all__ = (
    "LazyArray",
    "shared_intermediates",
    "linalg",
    "array",
    "transpose",
    "reshape",
    "tensordot",
    "einsum",
    "conj",
    "trace",
    "matmul",
    "clip",
    "flip",
    "sort",
    "argsort",
    "stack",
    # binary
    "multiply",
    "add",
    "floordivide",
    "truedivide",
    # unary
    "sin",
    "cos",
    "tan",
    "arcsin",
    "arccos",
    "arctan",
    "sinh",
    "cosh",
    "tanh",
    "arcsinh",
    "arccosh",
    "arctanh",
    "exp",
    "log",
    "log2",
    "log10",
    "conj",
    "sign",
    "abs",
    "real",
    "imag",
    # reductions
    "sum",
    "prod",
    "min",
    "max",
)


try:
    from opt_einsum.backends.dispatch import _aliases

    _aliases["autoray"] = "autoray.lazy"
except ImportError:  # pragma: no cover
    pass
