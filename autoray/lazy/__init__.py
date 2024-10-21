from . import linalg

from .core import (
    add,
    angle,
    arccos,
    arccosh,
    arcsin,
    arcsinh,
    arctan,
    arctanh,
    argsort,
    array,
    ascend,
    clip,
    compute,
    concatenate,
    conj,
    cos,
    cosh,
    descend,
    diag,
    einsum,
    empty,
    exp,
    eye,
    flip,
    floordivide,
    Function,
    get_source,
    identity,
    imag,
    kron,
    LazyArray,
    log,
    log10,
    log2,
    matmul,
    multiply,
    ones,
    prod,
    real,
    reshape,
    shared_intermediates,
    sign,
    sin,
    sinh,
    sort,
    split,
    sqrt,
    stack,
    take,
    tan,
    tanh,
    tensordot,
    trace,
    transpose,
    truedivide,
    Variable,
    where,
    zeros,
)
from .core import abs_ as abs
from .core import sum_ as sum
from .core import min_ as min
from .core import max_ as max
from .core import complex_ as complex

__all__ = (
    "abs",
    "add",
    "angle",
    "arccos",
    "arccosh",
    "arcsin",
    "arcsinh",
    "arctan",
    "arctanh",
    "argsort",
    "array",
    "ascend",
    "clip",
    "complex",
    "compute",
    "concatenate",
    "conj",
    "conj",
    "cos",
    "cosh",
    "descend",
    "diag",
    "einsum",
    "empty",
    "exp",
    "eye",
    "flip",
    "floordivide",
    "Function",
    "get_source",
    "identity",
    "imag",
    "kron",
    "LazyArray",
    "linalg",
    "log",
    "log10",
    "log2",
    "matmul",
    "max",
    "min",
    "multiply",
    "ones",
    "prod",
    "real",
    "reshape",
    "shared_intermediates",
    "sign",
    "sin",
    "sinh",
    "sort",
    "split",
    "sqrt",
    "stack",
    "sum",
    "take",
    "tan",
    "tanh",
    "tensordot",
    "trace",
    "transpose",
    "truedivide",
    "Variable",
    "where",
    "zeros",
)


try:
    from opt_einsum.backends.dispatch import _aliases

    _aliases["autoray"] = "autoray.lazy"
except ImportError:  # pragma: no cover
    pass
