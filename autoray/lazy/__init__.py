from . import linalg

from .core import (
    LazyArray,
    Variable,
    shared_intermediates,
    array,
    transpose,
    reshape,
    tensordot,
    einsum,
    trace,
    matmul,
    kron,
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
    angle,
    real,
    imag,
    # reductions
    prod,
)
from .core import abs_ as abs
from .core import sum_ as sum
from .core import min_ as min
from .core import max_ as max
from .core import complex_ as complex

__all__ = (
    "LazyArray",
    "Variable",
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
    "kron",
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
    "angle",
    "real",
    "imag",
    # reductions
    "sum",
    "prod",
    "min",
    "max",
    "complex",
)


try:
    from opt_einsum.backends.dispatch import _aliases

    _aliases["autoray"] = "autoray.lazy"
except ImportError:  # pragma: no cover
    pass
