import functools
import re

import pytest
from numpy.testing import assert_allclose, assert_raises

import autoray as ar
from autoray import lazy

from .test_autoray import BACKENDS, gen_rand


def assert_unary_fn_with_kwargs(fn, shapes, seed, kwargs, backend="numpy"):
    xp = ar.get_namespace(backend)
    rng = xp.random.default_rng(seed)
    args = [rng.uniform(size=shape) for shape in shapes]
    y = ar.do(fn, *args, **kwargs)
    largs = [lazy.array(arg) for arg in args]
    ly = ar.do(fn, *largs, **kwargs)
    assert ly.shape == y.shape
    xp.testing.assert_allclose(y, ly.compute())


def test_manual_construct():
    def foo(a, b, c):
        a1, a2 = a
        b1 = b["1"]
        c1, c2 = c["sub"]
        return ar.do("sum", ar.do("stack", (a1, a2, b1, c1, c2)), axis=0)

    x = ar.do("random.uniform", size=(5, 7), like="numpy")
    x0 = lazy.array(x[0, :])
    x1 = lazy.array(x[1, :])
    x2 = lazy.array(x[2, :])
    x3 = lazy.array(x[3, :])
    x4 = lazy.array(x[4, :])

    y = lazy.LazyArray(
        backend=ar.infer_backend(x),
        fn=foo,
        args=((x0, x1), {"1": x2}),
        kwargs=dict(c={"sub": (x3, x4)}),
        shape=(7,),
    )

    assert y.deps == (x0, x1, x2, x3, x4)
    assert re.match(
        r"x\d+ = foo\d+\(\(x\d+, x\d+,\), "
        r"{1: x\d+}, c={sub: \(x\d+, x\d+,\)}\)",
        y.get_source(),
    )
    assert_allclose(y.compute(), x.sum(0))


def modified_gram_schmidt(X):
    Q = []
    for j in range(0, ar.shape(X)[0]):
        q = X[j, :]
        for i in range(0, j):
            rij = ar.do("tensordot", ar.do("conj", Q[i]), q, axes=1)
            q = q - rij * Q[i]
        rjj = ar.do("linalg.norm", q, 2)
        Q.append(q / rjj)
    return ar.do("stack", tuple(Q), axis=0)


def wrap_strict_check(larray):
    fn_orig = larray._fn

    @functools.wraps(fn_orig)
    def checked(*args, **kwargs):
        data = fn_orig(*args, **kwargs)
        assert ar.shape(data) == ar.shape(larray)
        assert ar.infer_backend(data) == larray.backend
        return data

    return checked


def make_strict(larray):
    for node in larray.descend():
        larray._fn = wrap_strict_check(larray)


@pytest.mark.parametrize("backend", BACKENDS)
def test_lazy_mgs(backend):
    if backend == "sparse":
        pytest.xfail("Sparse doesn't support 'linalg.norm' yet...")
    x = gen_rand((5, 5), backend)
    lx = lazy.array(x)
    ly = modified_gram_schmidt(lx)
    ly.show()
    make_strict(ly)
    assert str(ly) == (
        f"<LazyArray(fn=stack, shape=(5, 5), backend='{backend}')>"
    )
    assert isinstance(ly, lazy.LazyArray)
    hmax = ly.history_max_size()
    hpeak = ly.history_peak_size()
    htot = ly.history_total_size()
    assert hmax == 25
    assert 25 < hpeak < htot
    assert ly.history_num_nodes() == 57
    assert len(ly.history_fn_frequencies()) == 9
    assert_allclose(
        ar.to_numpy(ly.compute()), ar.to_numpy(modified_gram_schmidt(x))
    )
    with lazy.shared_intermediates():
        ly = modified_gram_schmidt(lx)
        make_strict(ly)
    assert ly.history_num_nodes() == 51
    assert len(ly.history_fn_frequencies()) == 9
    assert_allclose(
        ar.to_numpy(ly.compute()), ar.to_numpy(modified_gram_schmidt(x))
    )


def test_partial_evaluation():
    la = lazy.array(gen_rand((10, 10), "numpy"))
    lb = lazy.array(gen_rand((10, 10), "numpy"))
    lc = lazy.array(gen_rand((10, 10), "numpy"))
    ld = lazy.array(gen_rand((10, 10), "numpy"))
    lab = ar.do("tanh", la @ lb)
    lcd = lc @ ld
    ls = lab + lcd
    ld = ar.do("abs", lab / lcd)
    le = ar.do("einsum", "ab,ba->a", ls, ld)
    lf = ar.do("sum", le)
    make_strict(lf)
    assert lf.history_num_nodes() == 12
    lf.compute_constants(variables=[lc, ld])  # constants = [la, lb]
    assert lf.history_num_nodes() == 9
    assert "tanh" not in {node.fn_name for node in lf.descend()}
    lf.compute()


def test_history_fn_frequencies():
    la = lazy.array(gen_rand((10, 10), "numpy"))
    lb = lazy.array(gen_rand((10, 10), "numpy"))
    lc = lazy.array(gen_rand((10, 10), "numpy"))
    ld = lazy.array(gen_rand((10, 10), "numpy"))
    lab = ar.do("tanh", la @ lb)
    lcd = lc @ ld
    ls = lab + lcd
    ld = ar.do("abs", lab / lcd)
    le = ar.do("einsum", "ab,ba->a", ls, ld)
    lf = ar.do("sum", le)
    assert lf.history_fn_frequencies() == {
        "None": 4,  # the inputs
        "tanh": 1,
        "matmul": 2,
        "add": 1,
        "absolute": 1,
        "truediv": 1,
        "einsum": 1,
        "sum": 1,
    }


def test_plot():
    pytest.importorskip("networkx")
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Template")
    la = lazy.array(gen_rand((10, 10), "numpy"))
    lb = lazy.array(gen_rand((10, 10), "numpy"))
    lc = lazy.array(gen_rand((10, 10), "numpy"))
    ld = lazy.array(gen_rand((10, 10), "numpy"))
    lab = ar.do("tanh", la @ lb)
    lcd = lc @ ld
    ls = lab + lcd
    ld = ar.do("abs", lab / lcd)
    le = ar.do("einsum", "ab,ba->a", ls, ld)
    lf = ar.do("sum", le)
    lf.plot_graph()
    lf.plot_graph(initial_layout="layers")
    lf.plot_graph(variables=[lc, ld], color_by="variables")
    lf.plot_circuit()
    lf.plot_circuit(color_by="id")
    lf.plot_history_size_footprint()
    lf.plot_history_functions_scatter()
    lf.plot_history_functions_lines(log=2)
    lf.plot_history_functions_image(rasterize=True)
    lf.plot_history_stats()


def test_share_intermediates():
    la = lazy.array(gen_rand((10, 10), "numpy"))
    lb = lazy.array(gen_rand((10, 10), "numpy"))
    l1 = ar.do("tanh", la @ lb)
    l2 = ar.do("tanh", la @ lb)
    ly = l1 + l2
    assert ly.history_num_nodes() == 7
    y1 = ly.compute()
    with lazy.shared_intermediates():
        l1 = ar.do("tanh", la @ lb)
        l2 = ar.do("tanh", la @ lb)
        ly = l1 + l2
    assert ly.history_num_nodes() == 5
    y2 = ly.compute()
    assert_allclose(y1, y2)


@pytest.mark.parametrize("backend", BACKENDS)
def test_transpose_chain(backend):
    lx = lazy.array(gen_rand((2, 3, 4, 5, 6), backend))
    l1 = ar.do("transpose", lx, (1, 0, 3, 2, 4))
    l2 = ar.do("transpose", l1, (1, 0, 3, 2, 4))
    assert l2.args[0] is lx
    assert l2.deps == (lx,)
    assert l1.history_num_nodes() == 2
    assert l2.history_num_nodes() == 2
    assert_allclose(
        ar.to_numpy(lx.compute()),
        ar.to_numpy(l2.compute()),
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_reshape_chain(backend):
    lx = lazy.array(gen_rand((2, 3, 4, 5, 6), backend))
    l1 = ar.do("reshape", lx, (6, 4, 30))
    l2 = ar.do("reshape", l1, (-1,))
    assert l1.history_num_nodes() == 2
    assert l2.history_num_nodes() == 2
    assert l2.args[0] is lx
    assert l2.deps == (lx,)
    assert_allclose(
        ar.to_numpy(lx.compute()).flatten(),
        ar.to_numpy(l2.compute()),
        atol=1e-6,
    )


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("dtype", ["float64", "complex128"])
def test_svd(backend, dtype):
    if backend == "sparse":
        pytest.xfail("Sparse doesn't support 'linalg.svd' yet...")

    if backend in ("paddle",) and "complex" in dtype:
        pytest.xfail(
            f"{backend} `linalg.solve` doesn't support complex dtype..."
        )

    x = lazy.array(gen_rand((4, 5), backend, dtype))
    U, s, VH = ar.do("linalg.svd", x)
    assert ar.shape(U) == (4, 4)
    assert ar.shape(s) == (4,)
    assert ar.shape(VH) == (4, 5)
    s = ar.astype(s, dtype)
    ly = U @ (ar.do("reshape", s, (-1, 1)) * VH)
    make_strict(ly)
    assert_allclose(
        ar.to_numpy(x.compute()),
        ar.to_numpy(ly.compute()),
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_qr(backend):
    if backend == "sparse":
        pytest.xfail("Sparse doesn't support 'linalg.qr' yet...")
    x = lazy.array(gen_rand((4, 5), backend))
    Q, R = ar.do("linalg.qr", x)
    assert ar.shape(Q) == (4, 4)
    assert ar.shape(R) == (4, 5)
    ly = Q @ R
    make_strict(ly)
    assert_allclose(
        ar.to_numpy(x.compute()),
        ar.to_numpy(ly.compute()),
    )


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("dtype", ["float64", "complex128"])
def test_eig_inv(backend, dtype):
    if backend in ("cupy", "dask", "torch", "mars", "sparse", "paddle"):
        pytest.xfail(f"{backend} doesn't support 'linalg.eig' yet...")

    # N.B. the prob that a real gaussian matrix has all real eigenvalues is
    # ``2**(-d * (d - 1) / 4)`` - see Edelman 1997 - so need ``d >> 5``
    d = 20
    x = lazy.array(gen_rand((d, d), backend, dtype))
    el, ev = ar.do("linalg.eig", x)
    assert ar.shape(el) == (d,)
    assert ar.shape(ev) == (d, d)
    ly = ev @ (ar.do("reshape", el, (-1, 1)) * ar.do("linalg.inv", ev))
    make_strict(ly)
    assert_allclose(
        ar.to_numpy(x.compute()),
        ar.to_numpy(ly.compute()),
    )


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("dtype", ["float64", "complex128"])
def test_eigh(backend, dtype):
    if backend in (
        "dask",
        "mars",
        "sparse",
    ):
        pytest.xfail(f"{backend} doesn't support 'linalg.eig' yet...")
    x = lazy.array(gen_rand((5, 5), backend, dtype))
    x = x + x.H
    el, ev = ar.do("linalg.eigh", x)
    assert ar.shape(el) == (5,)
    assert ar.shape(ev) == (5, 5)
    ly = ev @ (ar.do("reshape", el, (-1, 1)) * ev.H)
    make_strict(ly)
    assert_allclose(
        ar.to_numpy(x.compute()),
        ar.to_numpy(ly.compute()),
    )


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("dtype", ["float64", "complex128"])
def test_cholesky(backend, dtype):
    if backend in ("sparse",):
        pytest.xfail(f"{backend} doesn't support 'linalg.cholesky' yet...")

    if backend in ("paddle",) and "complex" in dtype:
        pytest.xfail(f"{backend} doesn't support complex dtype...")

    x = lazy.array(gen_rand((5, 5), backend, dtype))
    x = x @ x.H
    C = ar.do("linalg.cholesky", x)
    assert ar.shape(C) == (5, 5)
    ly = C @ C.H
    make_strict(ly)
    assert_allclose(
        ar.to_numpy(x.compute()),
        ar.to_numpy(ly.compute()),
    )


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("dtype", ["float64", "complex128"])
def test_solve(backend, dtype):
    if backend in ("sparse",):
        pytest.xfail(f"{backend} doesn't support 'linalg.solve' yet...")

    if backend in ("paddle",) and "complex" in dtype:
        pytest.xfail(
            f"{backend} `linalg.solve` doesn't support complex dtype..."
        )

    A = lazy.array(gen_rand((5, 5), backend, dtype))
    y = lazy.array(gen_rand((5,), backend, dtype))

    x = ar.do("linalg.solve", A, y)
    assert ar.shape(x) == (5,)
    # tensorflow e.g. doesn't allow ``A @ x`` for vector x ...
    ly = ar.do("tensordot", A, x, axes=1)
    make_strict(ly)
    assert_allclose(
        ar.to_numpy(y.compute()),
        ar.to_numpy(ly.compute()),
    )


def test_dunder_magic():
    a = ar.do("random.uniform", size=(), like="numpy")
    b = lazy.array(a)
    x, y, z = ar.do("random.uniform", size=(3), like="numpy")
    a = x * a
    b = x * b
    a = a * y
    b = b * y
    a *= z
    b *= z
    assert_allclose(a, b.compute())

    a = ar.do("random.uniform", size=(), like="numpy")
    b = lazy.array(a)
    x, y, z = ar.do("random.uniform", size=(3), like="numpy")
    a = x + a
    b = x + b
    a = a + y
    b = b + y
    a += z
    b += z
    assert_allclose(a, b.compute())

    a = ar.do("random.uniform", size=(), like="numpy")
    b = lazy.array(a)
    x, y, z = ar.do("random.uniform", size=(3), like="numpy")
    a = x - a
    b = x - b
    a = a - y
    b = b - y
    a -= z
    b -= z
    assert_allclose(a, b.compute())

    a = ar.do("random.uniform", size=(), like="numpy")
    b = lazy.array(a)
    x, y, z = ar.do("random.uniform", size=(3), like="numpy")
    a = x / a
    b = x / b
    a = a / y
    b = b / y
    a /= z
    b /= z
    assert_allclose(a, b.compute())

    a = ar.do("random.uniform", size=(), like="numpy")
    b = lazy.array(a)
    x, y, z = ar.do("random.uniform", size=(3), like="numpy")
    a = x // a
    b = x // b
    a = a // y
    b = b // y
    a //= z
    b //= z
    assert_allclose(a, b.compute())

    a = ar.do("random.uniform", size=(), like="numpy")
    b = lazy.array(a)
    x, y, z = ar.do("random.uniform", size=(3), like="numpy")
    a = x**a
    b = x**b
    a = a**y
    b = b**y
    a **= z
    b **= z
    assert_allclose(a, b.compute())

    a = ar.do("random.uniform", size=(3, 3), like="numpy")
    b = lazy.array(a)
    x, y, z = ar.do("random.uniform", size=(3, 3, 3), like="numpy")
    a = x @ a
    b = x @ b
    a = a @ y
    b = b @ y
    a = a @ z
    b @= z
    assert_allclose(a, b.compute())


def test_indexing():
    a = ar.do("random.uniform", size=(2, 3, 4, 5), like="numpy")
    b = lazy.array(a)

    for key in [0, (1, ..., -1), (0, 1, slice(None), -2)]:
        assert_allclose(a[key], b[key].compute())


@pytest.mark.parametrize("k", [-3, -1, 0, 2, 4])
@pytest.mark.parametrize(
    "shape",
    [
        (3,),
        (2, 2),
        (3, 4),
        (4, 3),
    ],
)
def test_diag(shape, k):
    a = ar.do("random.uniform", size=shape, like="numpy")
    b = lazy.array(a)
    ad = ar.do("diag", a, k)
    bd = ar.do("diag", b, k)
    assert_allclose(ad, bd.compute())


def test_einsum():
    a = ar.do("random.uniform", size=(2, 3, 4, 5), like="numpy")
    b = ar.do("random.uniform", size=(4, 5), like="numpy")
    c = ar.do("random.uniform", size=(6, 2, 3), like="numpy")
    eq = "abcd,cd,fab->fd"
    x1 = ar.do("einsum", eq, a, b, c)
    la, lb, lc = map(lazy.array, (a, b, c))
    x2 = ar.do("einsum", eq, la, lb, lc)
    assert_allclose(x1, x2.compute())


def test_tensordot():
    a = ar.do("random.uniform", size=(7, 3, 4, 5), like="numpy")
    b = ar.do("random.uniform", size=(5, 6, 3, 2), like="numpy")
    x1 = ar.do("tensordot", a, b, axes=[(1, 3), (2, 0)])
    la, lb = map(lazy.array, (a, b))
    x2 = ar.do("tensordot", la, lb, axes=[(1, 3), (2, 0)])
    assert_allclose(x1, x2.compute())


def test_use_variable_to_trace_function():
    a = lazy.Variable(shape=(2, 3), backend="numpy")
    b = lazy.Variable(shape=(3, 4), backend="numpy")
    c = ar.do("tanh", a @ b)
    f = c.get_function([a, b])
    x = ar.do("random.uniform", size=(2, 3), like="numpy")
    y = ar.do("random.uniform", size=(3, 4), like="numpy")
    z = f([x, y])
    assert ar.shape(z) == (2, 4)


def test_can_pickle_traced_function():
    import pickle

    a = lazy.Variable(shape=(2, 3), backend="numpy")
    b = lazy.Variable(shape=(3, 4), backend="numpy")
    c = ar.do("tanh", a @ b)
    f = c.get_function([a, b])
    x = ar.do("random.uniform", size=(2, 3), like="numpy")
    y = ar.do("random.uniform", size=(3, 4), like="numpy")
    z = f([x, y])
    assert ar.shape(z) == (2, 4)

    s = pickle.dumps(f)
    g = pickle.loads(s)
    z = g([x, y])
    assert ar.shape(z) == (2, 4)


def test_where():
    a = lazy.Variable(shape=(4,), backend="numpy")
    b = lazy.Variable(shape=(4,), backend="numpy")
    c = ar.do("where", *(a > 0, b, 1))
    f = c.get_function([a, b])
    x = ar.do("asarray", [-0.5, -0.5, 1, 2], like="numpy")
    y = ar.do("asarray", [1, 2, 3, 4], like="numpy")
    z = f(x, y)
    assert_allclose(z, [1, 1, 3, 4])


def test_lazy_function_pytree_input_and_output():
    inputs = {
        "a": lazy.Variable(shape=(2, 3), backend="numpy"),
        "b": lazy.Variable(shape=(3, 4), backend="numpy"),
    }
    outputs = {
        "outa": ar.do("tanh", inputs["a"] @ inputs["b"]),
        "outb": [inputs["a"] - 1, inputs["b"] - 1],
    }
    f = lazy.Function(inputs, outputs)

    a = ar.do("random.uniform", size=(2, 3), like="numpy")
    b = ar.do("random.uniform", size=(3, 4), like="numpy")

    outs = f({"a": a, "b": b})

    assert_allclose(outs["outa"], ar.do("tanh", a @ b))
    assert_allclose(outs["outb"][0], a - 1)
    assert_allclose(outs["outb"][1], b - 1)


@pytest.mark.parametrize(
    "indices",
    [
        [0, 1],
        [[0, 1], [1, 2]],
        [[[0, 1], [1, 2]], [[1, 1], [2, 2]]],
        [[[[0, 1, 2, 3]]]],
        [[[[0], [1]]], [[[2], [3]]]],
    ],
)
@pytest.mark.parametrize(
    "shape",
    [
        (4,),
        (4, 5),
        (4, 5, 6),
        (4, 5, 6, 7),
    ],
)
def test_take(indices, shape):
    a = ar.do("random.uniform", size=shape, like="numpy")
    b = lazy.Variable(shape=shape, backend="numpy")
    np_shape = ar.do("take", a, indices).shape
    lazy_shape = ar.do("take", b, indices).shape

    fn = ar.do("take", b, indices).get_function([b])
    lazy_func_shape = fn([a]).shape
    assert_allclose(np_shape, lazy_shape)
    assert_allclose(np_shape, lazy_func_shape)


@pytest.mark.parametrize(
    "indices",
    [
        [0, 1],
        [[0, 1], [1, 2]],
        [[[0, 1], [1, 2]], [[1, 1], [2, 2]]],
        [[[[0, 1, 2, 3]]]],
        [[[[0], [1]]], [[[2], [3]]]],
    ],
)
@pytest.mark.parametrize(
    "shape",
    [
        (4,),
        (4, 5),
        (4, 5, 6),
        (4, 5, 6, 7),
    ],
)
def test_getitem(indices, shape):
    a = ar.do("random.uniform", size=shape, like="numpy")
    b = lazy.Variable(shape=shape, backend="numpy")
    np_shape = a[indices].shape
    lazy_shape = b[indices].shape

    fn = b[indices].get_function([b])
    lazy_func_shape = fn([a]).shape
    assert_allclose(np_shape, lazy_shape)
    assert_allclose(np_shape, lazy_func_shape)


def random_indexer(ndim_min=0, ndim_max=10, d_min=1, d_max=5, seed=None):
    """Generate a random shape and valid indexing object into that shape."""
    import numpy as np

    rng = np.random.default_rng(seed=seed)

    ndim = rng.integers(ndim_min, ndim_max + 1)

    # if we have a advanced indexing arrays, the shape of the array
    adv_ix_ndim = rng.integers(1, 4)
    adv_ix_shape = tuple(rng.integers(d_min, d_max + 1, size=adv_ix_ndim))

    def rand_adv_ix_broadcastable_shape():
        # get a random shape that broadcast matches adv_ix_shape
        ndim = rng.integers(1, adv_ix_ndim + 1)
        matching_shape = adv_ix_shape[-ndim:]
        return tuple(rng.choice([d, 1]) for d in matching_shape)

    shape = []
    indexer = []
    choices = ["index", "slice", "ellipsis", "array", "list", "newaxis"]

    i = 0
    while i < ndim:
        kind = rng.choice(choices)

        if kind == "newaxis":
            indexer.append(None)
            continue

        d = rng.integers(d_min, d_max + 1)
        shape.append(d)

        if kind == "index":
            ix = rng.integers(-d, d)
            if rng.random() > 0.5:
                # randomly supply integers and numpy ints
                ix = int(ix)

        elif kind == "ellipsis":
            # only one ellipsis allowed
            ix = ...
            choices.remove("ellipsis")
            # how many dims ellipsis should expand to
            i += rng.integers(0, 4)

        elif kind == "slice":
            start = rng.integers(-d - 2, d + 2)
            stop = rng.integers(-d - 2, d - 2)
            step = rng.choice([-3, -2, -1, 1, 2, 3])
            ix = slice(start, stop, step)

        elif kind == "array":
            ai_shape = rand_adv_ix_broadcastable_shape()
            ix = rng.integers(-d, d, size=ai_shape)

        elif kind == "list":
            ai_shape = rand_adv_ix_broadcastable_shape()
            ix = rng.integers(-d, d, size=ai_shape).tolist()

        indexer.append(ix)
        i += 1

    if (len(indexer) == 1) and (rng.random() > 0.5):
        # return the raw object
        (indexer,) = indexer
    else:
        indexer = tuple(indexer)

    return tuple(shape), indexer


@pytest.mark.parametrize("seed", range(1000))
def test_lazy_getitem_random(seed):
    shape, indexer = random_indexer()
    a = ar.do("random.uniform", size=shape, like="numpy")
    ai = a[indexer]
    b = lazy.array(a)
    bi = b[indexer]
    assert bi.shape == ai.shape
    assert_allclose(bi.compute(), ai)


@pytest.mark.parametrize(
    "shape1, shape2",
    [
        ((3,), (3,)),
        ((3,), (3, 2)),
        ((6, 5, 4, 3), (3,)),
        ((7, 6, 5, 4), (7, 6, 4, 3)),
    ],
)
def test_matmul_shape(shape1, shape2):
    a = lazy.Variable(shape=shape1)
    b = lazy.Variable(shape=shape2)
    np_a = ar.do("random.uniform", size=shape1, like="numpy")
    np_b = ar.do("random.uniform", size=shape2, like="numpy")

    lazy_shape = (a @ b).shape
    np_shape = (np_a @ np_b).shape
    assert_allclose(lazy_shape, np_shape)


@pytest.mark.parametrize(
    "shape1, shape2",
    [
        ((3,), (1,)),
        ((3,), (4, 3)),
        ((3,), (3, 2, 1)),
        (
            (2, 2, 3, 4),
            (
                1,
                2,
                4,
                5,
            ),
        ),
        ((6, 5, 4), (6, 3, 3)),
    ],
)
def test_matmul_shape_error(shape1, shape2):
    a = lazy.Variable(shape=shape1)
    b = lazy.Variable(shape=shape2)

    def f(x, y):
        return x @ y

    assert_raises(ValueError, f, a, b)


def test_pytree_compute():
    x = ar.do("random.uniform", size=(5, 6), like="numpy")
    lx = lazy.array(x)
    lu, ls, lv = ar.do("linalg.svd", lx)
    lresults = {"u": lu, "s": ls, "v": lv}
    results = lazy.compute(lresults)
    assert isinstance(results, dict)
    assert ar.infer_backend(results["s"]) == ar.infer_backend(x)


def test_kron():
    x = ar.do("random.uniform", size=(2, 3), like="numpy")
    y = ar.do("random.uniform", size=(2, 3), like="numpy")
    xy = ar.do("kron", x, y)

    lx = lazy.array(x)
    ly = lazy.array(y)
    lxy = ar.do("kron", lx, ly)
    assert lxy.shape == xy.shape
    assert_allclose(lxy.compute(), xy)

    x = ar.do("random.uniform", size=(3,), like="numpy")
    y = ar.do("random.uniform", size=(3, 4, 5), like="numpy")
    xy = ar.do("kron", x, y)

    lx = lazy.array(x)
    ly = lazy.array(y)
    lxy = ar.do("kron", lx, ly)
    assert lxy.shape == xy.shape
    assert_allclose(lxy.compute(), xy)

    x = ar.do("random.uniform", size=(3, 4, 5), like="numpy")
    y = ar.do("random.uniform", size=(3,), like="numpy")
    xy = ar.do("kron", x, y)

    lx = lazy.array(x)
    ly = lazy.array(y)
    lxy = ar.do("kron", lx, ly)
    assert lxy.shape == xy.shape
    assert_allclose(lxy.compute(), xy)


def test_concatenate():
    x = ar.do("random.uniform", size=(3, 4, 5), like="numpy")
    y = ar.do("random.uniform", size=(3, 1, 5), like="numpy")
    z = ar.do("random.uniform", size=(3, 7, 5), like="numpy")
    xyz = ar.do("concatenate", (x, y, z), axis=1)

    lx = lazy.array(x)
    ly = lazy.array(y)
    lz = lazy.array(z)
    lxyz = ar.do("concatenate", (lx, ly, lz), axis=1)

    assert lxyz.shape == xyz.shape
    assert_allclose(lxyz.compute(), xyz)


@pytest.mark.parametrize("shape_in", [(2, 3, 4)])
@pytest.mark.parametrize("keepdims", [False, True])
@pytest.mark.parametrize(
    "kwargs",
    [
        {"axis": 0},
        {"axis": 2},
        {"ord": None, "axis": (0, 2)},
        {"ord": 1, "axis": (0, 2)},
        {"ord": 2, "axis": (2, 1)},
    ],
)
@pytest.mark.parametrize("backend", BACKENDS)
def test_norm(shape_in, keepdims, kwargs, backend):
    if backend in ("dask", "sparse",):
        pytest.xfail(f"{backend} doesn't support all 'linalg.norm' options...")

    fn = "linalg.norm"
    kwargs["keepdims"] = keepdims
    assert_unary_fn_with_kwargs(
        fn, [shape_in], seed=1234, kwargs=kwargs, backend=backend
    )
