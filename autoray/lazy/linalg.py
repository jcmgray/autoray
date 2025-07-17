"""
TODO: lstsq, pinv, eigvals, eigvalsh
"""

import operator

from ..autoray import get_lib_fn

from .core import (
    shape,
    ensure_lazy,
    lazy_cache,
    find_common_backend,
)


@lazy_cache("linalg.svd")
def svd(a):
    a = ensure_lazy(a)
    fn_svd = get_lib_fn(a.backend, "linalg.svd")
    lsvd = a.to(fn_svd, (a,), shape=(3,))
    *b, m, n = shape(a)
    k = min(m, n)
    lU = lsvd.to(operator.getitem, (lsvd, 0), shape=(*b, m, k))
    ls = lsvd.to(operator.getitem, (lsvd, 1), shape=(*b, k))
    lV = lsvd.to(operator.getitem, (lsvd, 2), shape=(*b, k, n))
    return lU, ls, lV


@lazy_cache("linalg.qr")
def qr(a):
    a = ensure_lazy(a)
    lQR = a.to(get_lib_fn(a.backend, "linalg.qr"), (a,), shape=(2,))
    *b, m, n = shape(a)
    k = min(m, n)
    lQ = lQR.to(operator.getitem, (lQR, 0), shape=(*b, m, k))
    lR = lQR.to(operator.getitem, (lQR, 1), shape=(*b, k, n))
    return lQ, lR


@lazy_cache("linalg.eig")
def eig(a):
    a = ensure_lazy(a)
    fn_eig = get_lib_fn(a.backend, "linalg.eig")
    leig = a.to(fn_eig, (a,), shape=(2,))
    *b, m, _ = shape(a)
    el = leig.to(
        operator.getitem,
        (leig, 0),
        shape=(
            *b,
            m,
        ),
    )
    ev = leig.to(operator.getitem, (leig, 1), shape=(*b, m, m))
    return el, ev


@lazy_cache("linalg.eigh")
def eigh(a):
    a = ensure_lazy(a)
    fn_eigh = get_lib_fn(a.backend, "linalg.eigh")
    leigh = a.to(fn_eigh, (a,), shape=(2,))
    *b, m, _ = shape(a)
    el = leigh.to(
        operator.getitem,
        (leigh, 0),
        shape=(
            *b,
            m,
        ),
    )
    ev = leigh.to(operator.getitem, (leigh, 1), shape=(*b, m, m))
    return el, ev


@lazy_cache("linalg.inv")
def inv(a):
    a = ensure_lazy(a)
    fn_inv = get_lib_fn(a.backend, "linalg.inv")
    return a.to(fn_inv, (a,))


@lazy_cache("linalg.cholesky")
def cholesky(a):
    a = ensure_lazy(a)
    fn_inv = get_lib_fn(a.backend, "linalg.cholesky")
    return a.to(fn_inv, (a,))


@lazy_cache("linalg.solve")
def solve(a, b):
    # XXX: support non square a?
    a = ensure_lazy(a)
    b = ensure_lazy(b)
    backend = find_common_backend(a, b)
    fn_solve = get_lib_fn(backend, "linalg.solve")
    return b.to(
        backend=backend,
        fn=fn_solve,
        args=(a, b),
        deps=(a, b),
    )


@lazy_cache("linalg.norm")
def norm(x, order=None):
    x = ensure_lazy(x)
    fn_inv = get_lib_fn(x.backend, "linalg.norm")
    newshape = ()
    return x.to(fn_inv, (x, order), shape=newshape)
