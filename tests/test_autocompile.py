import pytest

from autoray import do, autocompile, infer_backend, to_numpy
from .test_autoray import BACKENDS, gen_rand

from numpy.testing import assert_allclose


BACKENDS = [
    p for p in BACKENDS if p.values[0] in ("jax", "torch", "tensorflow")
]


def modified_gram_schmidt(X):
    Q = []
    for j in range(0, X.shape[0]):
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
    mgs = autocompile(modified_gram_schmidt, compiler_opts=compiler_opts)
    if nested:
        mgs = autocompile(mgs, compiler_opts=compiler_opts)
    y2 = mgs(x)
    assert_allclose(y, y2)


@pytest.mark.parametrize("backend", BACKENDS)
def test_others_numpy(backend, mgs_case):
    x, y = mgs_case
    mgs = autocompile(modified_gram_schmidt)
    y2 = mgs(x, backend=backend)
    assert infer_backend(y2) == "numpy"
    assert_allclose(y, y2)


@pytest.mark.parametrize("backend", BACKENDS)
def test_autodispatch(backend, mgs_case):
    x, y = mgs_case
    x = do("array", x, like=backend)
    mgs = autocompile(modified_gram_schmidt)
    y2 = mgs(x, backend=backend)
    assert infer_backend(y2) == backend
    assert_allclose(y, to_numpy(y2))


def test_complicated_signature():

    @autocompile
    def foo(a, b, c):
        a1, a2 = a
        b1 = b['1']
        c1, c2 = c['sub']
        return do('sum', do('stack', (a1, a2, b1, c1, c2)), axis=0)

    x = do('random.uniform', size=(5, 7), like='numpy')
    y = foo((x[0, :], x[1, :]), {'1': x[2, :]}, c={'sub': (x[3, :], x[4, :])})
    assert_allclose(y, x.sum(0))
