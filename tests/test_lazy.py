import functools
import re

import pytest

from autoray import do, lazy, to_numpy, infer_backend, get_dtype_name, astype
from numpy.testing import assert_allclose

from .test_autoray import BACKENDS, gen_rand


def test_manual_construct():

    def foo(a, b, c):
        a1, a2 = a
        b1 = b['1']
        c1, c2 = c['sub']
        return do('sum', do('stack', (a1, a2, b1, c1, c2)), axis=0)

    x = do('random.uniform', size=(5, 7), like='numpy')
    x0 = lazy.array(x[0, :])
    x1 = lazy.array(x[1, :])
    x2 = lazy.array(x[2, :])
    x3 = lazy.array(x[3, :])
    x4 = lazy.array(x[4, :])

    y = lazy.LazyArray(
        backend=infer_backend(x),
        fn=foo,
        args=((x0, x1), {'1': x2}),
        kwargs=dict(c={'sub': (x3, x4)}),
        shape=(7,),
        dtype='float64',
    )

    assert y.deps == (x0, x1, x2, x3, x4)
    assert re.match(
        r'x\d+ = foo\d+\(\(x\d+, x\d+,\), '
        r'{1: x\d+}, c: {sub: \(x\d+, x\d+,\)}\)',
        y.get_source()
    )
    assert_allclose(y.compute(), x.sum(0))


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


def wrap_strict_check(larray):

    fn_orig = larray._fn

    @functools.wraps(fn_orig)
    def checked(*args, **kwargs):
        data = fn_orig(*args, **kwargs)
        assert tuple(data.shape) == larray.shape
        assert get_dtype_name(data) == larray.dtype
        assert infer_backend(data) == larray.backend
        return data

    return checked


def make_strict(larray):
    for node in larray:
        larray._fn = wrap_strict_check(larray)


@pytest.mark.parametrize("backend", BACKENDS)
def test_lazy_mgs(backend):
    if backend == "sparse":
        pytest.xfail("Sparse doesn't support 'linalg.norm' yet...")
    x = gen_rand((5, 5), backend)
    lx = lazy.array(x)
    ly = modified_gram_schmidt(lx)
    make_strict(ly)
    assert str(ly) == (
        f"<LazyArray(fn=stack, shape=(5, 5), "
        f"dtype=float64, backend='{backend}')>"
    )
    assert isinstance(ly, lazy.LazyArray)
    assert ly.history_max_size() == 25
    assert ly.history_total_size() > 25
    assert len(tuple(ly)) == 57
    assert len({node.fn_name for node in ly}) == 9
    assert_allclose(to_numpy(ly.compute()), to_numpy(modified_gram_schmidt(x)))
    with lazy.shared_intermediates():
        ly = modified_gram_schmidt(lx)
        make_strict(ly)
    assert len(tuple(ly)) == 51
    assert len({node.fn_name for node in ly}) == 9
    assert_allclose(to_numpy(ly.compute()), to_numpy(modified_gram_schmidt(x)))


def test_partial_evaluation():
    la = lazy.array(gen_rand((10, 10), "numpy"))
    lb = lazy.array(gen_rand((10, 10), "numpy"))
    lc = lazy.array(gen_rand((10, 10), "numpy"))
    ld = lazy.array(gen_rand((10, 10), "numpy"))
    lab = do("tanh", la @ lb)
    lcd = lc @ ld
    ls = lab + lcd
    ld = do("abs", lab / lcd)
    le = do("einsum", "ab,ba->a", ls, ld)
    lf = do("sum", le)
    make_strict(lf)
    assert len(tuple(lf)) == 12
    lf.compute_constants(variables=[lc, ld])  # constants = [la, lb]
    assert len(tuple(lf)) == 9
    assert "tanh" not in {node.fn_name for node in lf}
    lf.compute()


def test_plot():
    import matplotlib

    matplotlib.use("Template")
    la = lazy.array(gen_rand((10, 10), "numpy"))
    lb = lazy.array(gen_rand((10, 10), "numpy"))
    lc = lazy.array(gen_rand((10, 10), "numpy"))
    ld = lazy.array(gen_rand((10, 10), "numpy"))
    lab = do("tanh", la @ lb)
    lcd = lc @ ld
    ls = lab + lcd
    ld = do("abs", lab / lcd)
    le = do("einsum", "ab,ba->a", ls, ld)
    lf = do("sum", le)
    lf.plot()
    lf.plot(variables=[lc, ld])


def test_share_intermediates():
    la = lazy.array(gen_rand((10, 10), "numpy"))
    lb = lazy.array(gen_rand((10, 10), "numpy"))
    l1 = do("tanh", la @ lb)
    l2 = do("tanh", la @ lb)
    ly = l1 + l2
    assert len(tuple(ly)) == 7
    y1 = ly.compute()
    with lazy.shared_intermediates():
        l1 = do("tanh", la @ lb)
        l2 = do("tanh", la @ lb)
        ly = l1 + l2
    assert len(tuple(ly)) == 5
    y2 = ly.compute()
    assert_allclose(y1, y2)


@pytest.mark.parametrize("backend", BACKENDS)
def test_transpose_chain(backend):
    lx = lazy.array(gen_rand((2, 3, 4, 5, 6), backend))
    l1 = do("transpose", lx, (1, 0, 3, 2, 4))
    l2 = do("transpose", l1, (1, 0, 3, 2, 4))
    assert l2.args[0] is lx
    assert l2.deps == (lx,)
    assert len(tuple(l1)) == 2
    assert len(tuple(l2)) == 2
    assert_allclose(
        to_numpy(lx.compute()), to_numpy(l2.compute()),
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_reshape_chain(backend):
    lx = lazy.array(gen_rand((2, 3, 4, 5, 6), backend))
    l1 = do("reshape", lx, (6, 4, 30))
    l2 = do("reshape", l1, (-1,))
    assert len(tuple(l1)) == 2
    assert len(tuple(l2)) == 2
    assert l2.args[0] is lx
    assert l2.deps == (lx,)
    assert_allclose(
        to_numpy(lx.compute()).flatten(), to_numpy(l2.compute()),
    )


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("dtype", ["float64", "complex128"])
def test_svd(backend, dtype):
    if backend == "sparse":
        pytest.xfail("Sparse doesn't support 'linalg.svd' yet...")
    x = lazy.array(gen_rand((4, 5), backend, dtype))
    U, s, VH = do("linalg.svd", x)
    assert U.shape == (4, 4)
    assert s.shape == (4,)
    assert VH.shape == (4, 5)
    s = astype(s, dtype)
    ly = U @ (do("reshape", s, (-1, 1)) * VH)
    make_strict(ly)
    assert_allclose(
        to_numpy(x.compute()), to_numpy(ly.compute()),
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_qr(backend):
    if backend == "sparse":
        pytest.xfail("Sparse doesn't support 'linalg.qr' yet...")
    x = lazy.array(gen_rand((4, 5), backend))
    Q, R = do("linalg.qr", x)
    assert Q.shape == (4, 4)
    assert R.shape == (4, 5)
    ly = Q @ R
    make_strict(ly)
    assert_allclose(
        to_numpy(x.compute()), to_numpy(ly.compute()),
    )


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("dtype", ["float64", "complex128"])
def test_eig_inv(backend, dtype):
    if backend in ("cupy", "dask", "torch", "mars", "sparse"):
        pytest.xfail(f"{backend} doesn't support 'linalg.eig' yet...")

    # N.B. the prob that a real gaussian matrix has all real eigenvalues is
    # ``2**(-d * (d - 1) / 4)`` - see Edelman 1997 - so need ``d >> 5``
    d = 20
    x = lazy.array(gen_rand((d, d), backend, dtype))
    el, ev = do("linalg.eig", x)
    assert el.shape == (d,)
    assert ev.shape == (d, d)
    ly = ev @ (do("reshape", el, (-1, 1)) * do("linalg.inv", ev))
    make_strict(ly)
    assert_allclose(
        to_numpy(x.compute()), to_numpy(ly.compute()),
    )


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("dtype", ["float64", "complex128"])
def test_eigh(backend, dtype):
    if backend in ("dask", "mars", "sparse",):
        pytest.xfail(f"{backend} doesn't support 'linalg.eig' yet...")
    x = lazy.array(gen_rand((5, 5), backend, dtype))
    x = x + x.H
    el, ev = do("linalg.eigh", x)
    assert get_dtype_name(ev) == dtype
    assert el.shape == (5,)
    assert ev.shape == (5, 5)
    ly = ev @ (do("reshape", el, (-1, 1)) * ev.H)
    make_strict(ly)
    assert_allclose(
        to_numpy(x.compute()), to_numpy(ly.compute()),
    )


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("dtype", ["float64", "complex128"])
def test_cholesky(backend, dtype):
    if backend in ("sparse",):
        pytest.xfail(f"{backend} doesn't support 'linalg.cholesky' yet...")
    x = lazy.array(gen_rand((5, 5), backend, dtype))
    x = x @ x.H
    C = do("linalg.cholesky", x)
    assert C.shape == (5, 5)
    ly = C @ C.H
    make_strict(ly)
    assert_allclose(
        to_numpy(x.compute()), to_numpy(ly.compute()),
    )


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("dtype", ["float64", "complex128"])
def test_solve(backend, dtype):
    if backend in ("sparse",):
        pytest.xfail(f"{backend} doesn't support 'linalg.solve' yet...")
    A = lazy.array(gen_rand((5, 5), backend, dtype))
    y = lazy.array(gen_rand((5,), backend, dtype))

    x = do("linalg.solve", A, y)
    assert x.shape == (5,)
    # tensorflow e.g. doesn't allow ``A @ x`` for vector x ...
    ly = do("tensordot", A, x, axes=1)
    make_strict(ly)
    assert_allclose(
        to_numpy(y.compute()), to_numpy(ly.compute()),
    )


def test_dunder_magic():
    a = do('random.uniform', size=(), like='numpy')
    b = lazy.array(a)
    x, y, z = do('random.uniform', size=(3), like='numpy')
    a = x * a
    b = x * b
    a = a * y
    b = b * y
    a *= z
    b *= z
    assert_allclose(a, b.compute())

    a = do('random.uniform', size=(), like='numpy')
    b = lazy.array(a)
    x, y, z = do('random.uniform', size=(3), like='numpy')
    a = x + a
    b = x + b
    a = a + y
    b = b + y
    a += z
    b += z
    assert_allclose(a, b.compute())

    a = do('random.uniform', size=(), like='numpy')
    b = lazy.array(a)
    x, y, z = do('random.uniform', size=(3), like='numpy')
    a = x - a
    b = x - b
    a = a - y
    b = b - y
    a -= z
    b -= z
    assert_allclose(a, b.compute())

    a = do('random.uniform', size=(), like='numpy')
    b = lazy.array(a)
    x, y, z = do('random.uniform', size=(3), like='numpy')
    a = x / a
    b = x / b
    a = a / y
    b = b / y
    a /= z
    b /= z
    assert_allclose(a, b.compute())

    a = do('random.uniform', size=(), like='numpy')
    b = lazy.array(a)
    x, y, z = do('random.uniform', size=(3), like='numpy')
    a = x // a
    b = x // b
    a = a // y
    b = b // y
    a //= z
    b //= z
    assert_allclose(a, b.compute())

    a = do('random.uniform', size=(), like='numpy')
    b = lazy.array(a)
    x, y, z = do('random.uniform', size=(3), like='numpy')
    a = x ** a
    b = x ** b
    a = a ** y
    b = b ** y
    a **= z
    b **= z
    assert_allclose(a, b.compute())

    a = do('random.uniform', size=(3, 3), like='numpy')
    b = lazy.array(a)
    x, y, z = do('random.uniform', size=(3, 3, 3), like='numpy')
    a = x @ a
    b = x @ b
    a = a @ y
    b = b @ y
    a = a @ z
    b @= z
    assert_allclose(a, b.compute())


def test_indexing():
    a = do('random.uniform', size=(2, 3, 4, 5), like='numpy')
    b = lazy.array(a)

    for key in [
        0,
        (1, ..., -1),
        (0, 1, slice(None), -2)
    ]:
        assert_allclose(a[key], b[key].compute())


def test_einsum():
    a = do('random.uniform', size=(2, 3, 4, 5), like='numpy')
    b = do('random.uniform', size=(4, 5), like='numpy')
    c = do('random.uniform', size=(6, 2, 3), like='numpy')
    eq = 'abcd,cd,fab->fd'
    x1 = do('einsum', eq, a, b, c)
    la, lb, lc = map(lazy.array, (a, b, c))
    x2 = do('einsum', eq, la, lb, lc)
    assert_allclose(x1, x2.compute())


def test_tensordot():
    a = do('random.uniform', size=(7, 3, 4, 5), like='numpy')
    b = do('random.uniform', size=(5, 6, 3, 2), like='numpy')
    x1 = do('tensordot', a, b, axes=[(1, 3), (2, 0)])
    la, lb = map(lazy.array, (a, b))
    x2 = do('tensordot', la, lb, axes=[(1, 3), (2, 0)])
    assert_allclose(x1, x2.compute())
