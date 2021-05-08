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
def test_compile_python(mgs_case, share_intermediates):
    x, y = mgs_case
    mgs = autocompile(
        modified_gram_schmidt,
        compiler_opts={"python": {"share_intermediates": share_intermediates}},
    )
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
