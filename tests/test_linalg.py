"""Group tests for linear algebra decompositions etc."""

import numpy as np
import pytest

import autoray as ar

from .conftest import gen_params, gen_rand


@pytest.mark.parametrize(
    "backend,dtype,fn,args",
    gen_params(
        backends=...,
        dtypes=...,
        fns=[
            ("linalg.svd", ("matrix",)),
            ("linalg.svd", ("batched",)),
        ],
    ),
)
def test_svd(backend, dtype, fn, args):
    svdtype = args[0]
    if svdtype == "matrix":
        x = gen_rand((5, 4), backend, dtype)
        U, s, V = ar.do("linalg.svd", x)
        assert (
            ar.infer_backend(x)
            == ar.infer_backend(U)
            == ar.infer_backend(s)
            == ar.infer_backend(V)
            == backend
        )
        # XXX: tensorflow can't multiply complex * real
        s = ar.do("astype", s, U.dtype)
        y = U @ ar.do("diag", s, like=x) @ V
        yn = ar.to_numpy(y)
        xn = ar.to_numpy(x)
        np.testing.assert_allclose(yn, xn, rtol=1e-2, atol=1e-3)

    elif svdtype == "batched":
        x = gen_rand((2, 5, 4), backend, dtype)
        U, s, VH = ar.do("linalg.svd", x)
        # XXX: tensorflow can't multiply complex * real
        s = ar.do("astype", s, U.dtype)
        y = U @ (ar.do("reshape", s, (2, 4, 1)) * VH)
        assert ar.shape(U) == (2, 5, 4)
        assert ar.shape(s) == (2, 4)
        assert ar.shape(VH) == (2, 4, 4)
        yn = ar.to_numpy(y)
        xn = ar.to_numpy(x)
        np.testing.assert_allclose(yn, xn, rtol=1e-2, atol=1e-3)


@pytest.mark.parametrize(
    "backend", gen_params(backends=..., requires="linalg.qr")
)
@pytest.mark.parametrize("shape", [(4, 3), (4, 4), (3, 4)])
def test_qr_thin_square_fat(backend, shape):
    x = gen_rand(shape, backend)
    Q, R = ar.do("linalg.qr", x)
    xn, Qn, Rn = map(ar.to_numpy, (x, Q, R))
    assert ar.do("allclose", xn, Qn @ Rn)


@pytest.mark.parametrize(
    "backend,dtype",
    gen_params(backends=..., dtypes=..., requires="linalg.solve"),
)
def test_solve(backend, dtype):

    A = gen_rand((4, 4), backend, dtype)
    b = gen_rand((4, 1), backend, dtype)
    x = ar.do("linalg.solve", A, b)

    Ax = ar.to_numpy(A @ x)
    b = ar.to_numpy(b)
    np.testing.assert_allclose(Ax, b, rtol=1e-2, atol=1e-3)


@pytest.mark.parametrize(
    "backend,dtype",
    gen_params(backends=..., dtypes=..., requires="linalg.inv"),
)
def test_inv(backend, dtype):
    A = gen_rand((4, 4), backend, dtype)
    A_inv = ar.do("linalg.inv", A)
    I = ar.to_numpy(A @ A_inv)
    np.testing.assert_allclose(I, np.eye(4), rtol=1e-3, atol=1e-4)


@pytest.mark.parametrize(
    "backend,dtype",
    gen_params(backends=..., dtypes=..., requires="linalg.eigh"),
)
def test_eigh(backend, dtype):

    A = gen_rand((4, 4), backend, dtype)
    A = A + ar.dag(A)
    el, ev = ar.do("linalg.eigh", A)
    B = (ev * ar.reshape(el, (1, -1))) @ ar.dag(ev)
    assert ar.do("allclose", ar.to_numpy(A), ar.to_numpy(B), rtol=1e-3)


@pytest.mark.parametrize(
    "backend,dtype",
    gen_params(backends=..., dtypes=..., requires="linalg.cholesky"),
)
def test_cholesky_upper(backend, dtype):
    x = gen_rand((4, 4), backend, dtype)
    xp = ar.get_namespace(x)
    A = x @ ar.dag(x) + 1e-3 * xp.eye(4)

    U = xp.linalg.cholesky(A, upper=True)
    reconstructed = ar.dag(U) @ U

    assert xp.shape(U) == (4, 4)
    ar.do(
        "testing.assert_allclose",
        ar.to_numpy(reconstructed),
        ar.to_numpy(A),
        rtol=1e-3,
        atol=1e-6,
    )


@pytest.mark.parametrize(
    "backend,dtype",
    gen_params(
        backends=...,
        dtypes=...,
        requires="scipy.linalg.solve_triangular",
    ),
)
def test_scipy_linalg_solve_triangular(backend, dtype):

    A = gen_rand((4, 4), backend, dtype)
    xp = ar.get_namespace(A)
    # make A a well-conditioned triangular matrix
    Au = xp.triu(A)
    Au = Au + 2 * xp.eye(4)
    b = gen_rand((4, 1), backend, dtype)

    # solve with upper triangular (default)
    x = xp.scipy.linalg.solve_triangular(Au, b)
    assert ar.do(
        "allclose",
        ar.to_numpy(Au @ x),
        ar.to_numpy(b),
        rtol=1e-3,
        atol=1e-6,
    )

    # solve with lower triangular
    Al = xp.tril(A)
    Al = Al + 2 * xp.eye(4)
    x = xp.scipy.linalg.solve_triangular(Al, b, lower=True)
    assert ar.do(
        "allclose",
        ar.to_numpy(Al @ x),
        ar.to_numpy(b),
        rtol=1e-3,
        atol=1e-6,
    )
