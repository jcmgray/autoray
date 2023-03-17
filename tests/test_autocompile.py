import pytest

from autoray import do, autojit, infer_backend, to_numpy, shape
from .test_autoray import BACKENDS, gen_rand

from numpy.testing import assert_allclose


BACKENDS = [
    p for p in BACKENDS if p.values[0] in ("jax", "torch", "tensorflow")
]


def modified_gram_schmidt(X):
    Q = []
    for j in range(0, shape(X)[0]):
        q = X[j, :]
        for i in range(0, j):
            rij = do("tensordot", do("conj", Q[i]), q, axes=1)
            q = q - rij * Q[i]
        rjj = do("linalg.norm", q, 2)
        Q.append(q / rjj)
    return do("stack", tuple(Q), axis=0)


@pytest.fixture
def mgs_case():
    x = gen_rand((10, 10), "numpy")
    y = modified_gram_schmidt(x)
    return x, y


@pytest.mark.parametrize("share_intermediates", [False, True])
@pytest.mark.parametrize("nested", [False, True])
def test_compile_python(mgs_case, share_intermediates, nested):
    x, y = mgs_case
    compiler_opts = {"python": {"share_intermediates": share_intermediates}}
    mgs = autojit(modified_gram_schmidt, compiler_opts=compiler_opts)
    if nested:
        mgs = autojit(mgs, compiler_opts=compiler_opts)
    y2 = mgs(x)
    assert_allclose(y, y2)


@pytest.mark.parametrize("backend", BACKENDS)
def test_others_numpy(backend, mgs_case):
    x, y = mgs_case
    mgs = autojit(modified_gram_schmidt)
    y2 = mgs(x, backend=backend)
    assert infer_backend(y2) == "numpy"
    assert_allclose(y, y2)


@pytest.mark.parametrize("backend", BACKENDS)
def test_autodispatch(backend, mgs_case):
    x, y = mgs_case
    x = do("array", x, like=backend)
    mgs = autojit(modified_gram_schmidt)
    y2 = mgs(x, backend=backend)
    assert infer_backend(y2) == backend
    assert_allclose(y, to_numpy(y2))


def test_complicated_signature():
    @autojit
    def foo(a, b, c):
        a1, a2 = a
        b1 = b["1"]
        c1, c2 = c["sub"]
        return do("sum", do("stack", (a1, a2, b1, c1, c2)), axis=0)

    x = do("random.uniform", size=(5, 7), like="numpy")
    y = foo((x[0, :], x[1, :]), {"1": x[2, :]}, c={"sub": (x[3, :], x[4, :])})
    assert_allclose(y, x.sum(0))


def test_multi_output():
    @autojit
    def foo(a, b, c):
        a = a - do("sum", b)
        b = b - do("sum", a)
        return a + c, b - c

    a = gen_rand((2, 3), "numpy")
    b = gen_rand((4, 5), "numpy")
    x, y = foo(a, b, 1)

    assert_allclose(x, a - b.sum() + 1)
    assert_allclose(y, b - (a - b.sum()).sum() - 1)


def test_static_kwargs_change():
    @autojit
    def foo(a, b, c):
        if c == "sum":
            return a + b
        elif c == "sub":
            return a - b

    assert (
        foo(
            do("array", 100, like="numpy"), do("array", 1, like="numpy"), "sum"
        )
        == 101
    )
    assert (
        foo(
            do("array", 100, like="numpy"), do("array", 1, like="numpy"), "sub"
        )
        == 99
    )
