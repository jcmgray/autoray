import numpy as np
import pytest

import autoray as ar
from autoray import shape

from .conftest import gen_params, gen_rand


@pytest.mark.parametrize(
    "backend,fn,args",
    gen_params(
        backends=...,
        fns=[
            ("all", ()),
            ("clip", (0.2, 0.7)),
            ("conj", ()),
            ("cos", ()),
            ("cosh", ()),
            ("count_nonzero", ()),
            ("cumsum", (0,)),
            ("diag", ()),
            ("diag", (1,)),
            ("diag", (-1,)),
            ("exp", ()),
            ("imag", ()),
            ("log", ()),
            ("log10", ()),
            ("max", (-1,)),
            ("max", ()),
            ("mean", ()),
            ("mean", (0,)),
            ("min", ()),
            ("power", (2,)),
            ("prod", ()),
            ("ravel", ()),
            ("real", ()),
            ("sin", ()),
            ("sinh", ()),
            ("sqrt", ()),
            ("sum", ()),
            ("sum", (1,)),
            ("tan", ()),
            ("tanh", ()),
            ("trace", ()),
            ("tril", ()),
            ("tril", (1,)),
            ("tril", (-1,)),
            ("triu", ()),
            ("triu", (1,)),
            ("triu", (-1,)),
        ],
    ),
)
def test_unary_functions(backend, fn, args):

    xn = ar.do("random.uniform", size=(4, 5), like="numpy")
    yn = ar.do(fn, xn, *args)
    x = ar.do("asarray", xn, like=backend)
    y = ar.do(fn, x, *args)
    yt = ar.do("to_numpy", y)
    assert ar.do("allclose", yt, yn)


@pytest.mark.parametrize(
    "backend,fn",
    gen_params(
        backends=...,
        fns=["add", "allclose", "divide", "matmul", "multiply", "subtract"],
    ),
)
def test_binary_functions(backend, fn):
    xan = ar.do("random.uniform", size=(3, 3), like="numpy")
    xbn = ar.do("random.uniform", size=(3, 3), like="numpy")
    yn = ar.do(fn, xan, xbn)
    xa = ar.do("asarray", xan, like=backend)
    xb = ar.do("asarray", xbn, like=backend)
    y = ar.do(fn, xa, xb)
    yt = ar.do("to_numpy", y)
    assert ar.do("allclose", yt, yn)


@pytest.mark.parametrize(
    "backend,f,args,kwargs",
    gen_params(
        backends=...,
        fns=[
            (f, (), kw)
            for f in ("sum", "prod", "max", "min", "mean")
            for kw in (
                {},
                {"axis": 1},
                {"axis": 1, "keepdims": True},
                {"axis": (0, 2)},
            )
        ],
    ),
)
def test_reduce_functions(backend, f, args, kwargs):
    x = ar.do("random.normal", size=(2, 3, 4), like="numpy")
    y = ar.do(f, x, **kwargs)
    xb = ar.do("asarray", x, like=backend)
    yb = ar.do(f, xb, **kwargs)
    yt = ar.do("to_numpy", yb)
    assert ar.do("allclose", yt, y)


@pytest.mark.parametrize(
    "backend,fn", gen_params(backends=..., fns=["sqrt", "exp", "sum"])
)
def test_basic(backend, fn):
    x = gen_rand((2, 3, 4), backend)
    y = ar.do(fn, x)
    if (backend == "sparse") and (fn == "sum"):
        pytest.xfail("Sparse 'sum' outputs dense.")
    assert ar.infer_backend(x) == ar.infer_backend(y) == backend


def test_infer_backend_multi():
    x = 1.0
    y = gen_rand((2, 3), "numpy")
    z = ar.lazy.Variable((4, 5))
    assert ar.infer_backend_multi(x) == "builtins"
    assert ar.infer_backend_multi(x, y) == "numpy"
    assert ar.infer_backend_multi(x, y, z) == "autoray.lazy"


def test_raises_import_error_when_missing():
    with pytest.raises(ImportError):
        ar.do("anonexistantfunction", 1, like="numpy")
    with pytest.raises(ImportError):
        ar.do("ones", 1, like="anonexistantbackend")


@pytest.mark.parametrize("backend", gen_params(backends=...))
@pytest.mark.parametrize(
    "fn,args",
    [
        (ar.conj, []),
        (ar.transpose, []),
        (ar.real, []),
        (ar.imag, []),
        (ar.reshape, [(5, 3)]),
    ],
)
def test_attribute_prefs(backend, fn, args):
    x = gen_rand((3, 5), backend)
    y = fn(x, *args)
    assert ar.infer_backend(x) == ar.infer_backend(y) == backend


def modified_gram_schmidt(X):
    Q = []
    for j in range(0, shape(X)[0]):
        q = X[j, :]
        for i in range(0, j):
            rij = ar.do("tensordot", ar.do("conj", Q[i]), q, 1)
            q = q - rij * Q[i]

        rjj = ar.do("linalg.norm", q, 2)
        Q.append(q / rjj)

    return ar.do("stack", Q, axis=0)


@pytest.mark.parametrize(
    "backend", gen_params(backends=..., requires="linalg.norm")
)
def test_full_fn_mgs(backend):
    x = gen_rand((3, 5), backend)
    Ux = modified_gram_schmidt(x)
    y = ar.do("sum", Ux @ ar.dag(Ux))
    assert ar.to_numpy(y) == pytest.approx(3)


def modified_gram_schmidt_np_mimic(X, explicit_namespace=False):
    if explicit_namespace:
        xp = ar.get_namespace(X)
    else:
        from autoray import numpy as xp

        print(xp)

    Q = []
    for j in range(0, shape(X)[0]):
        q = X[j, :]
        for i in range(0, j):
            rij = xp.tensordot(xp.conj(Q[i]), q, 1)
            q = q - rij * Q[i]

        rjj = xp.linalg.norm(q, 2)
        Q.append(q / rjj)

    return xp.stack(Q, axis=0)


def test_numpy_mimic_dunder_methods():
    from abc import ABC

    from autoray import numpy as np

    class Base(ABC):
        pass

    assert isinstance(np, object)
    assert not isinstance(np, Base)
    print(np)
    dir(np)


@pytest.mark.parametrize(
    "backend", gen_params(backends=..., requires="linalg.norm")
)
@pytest.mark.parametrize("explicit_namespace", [True, False])
def test_mgs_np_mimic(backend, explicit_namespace):
    x = gen_rand((3, 5), backend)
    Ux = modified_gram_schmidt_np_mimic(x, explicit_namespace)
    y = ar.do("sum", Ux @ ar.dag(Ux))
    assert ar.to_numpy(y) == pytest.approx(3)


@pytest.mark.parametrize("backend", gen_params(backends=...))
def test_translator_random_uniform(backend):
    from autoray import numpy as anp

    if backend == "sparse":
        pytest.xfail("Sparse will have zeros")

    x = anp.random.uniform(low=-10, size=(4, 5), like=backend)
    assert (ar.to_numpy(x) > -10).all()
    assert (ar.to_numpy(x) < 1.0).all()

    # test default single scalar
    x = anp.random.uniform(low=1000, high=2000, like=backend)
    assert 1000 <= ar.to_numpy(x) < 2000


@pytest.mark.parametrize("backend", gen_params(backends=...))
def test_translator_random_normal(backend):
    from autoray import numpy as anp

    x = anp.random.normal(100.0, 0.1, size=(4, 5), like=backend)

    if backend == "sparse":
        assert (x.data > 90.0).all()
        assert (x.data < 110.0).all()
        return

    assert (ar.to_numpy(x) > 90.0).all()
    assert (ar.to_numpy(x) < 110.0).all()

    if backend == "tensorflow":
        x32 = ar.do(
            "random.normal",
            100.0,
            0.1,
            dtype="float32",
            size=(4, 5),
            like=backend,
        )
        assert x32.dtype == "float32"
        assert (ar.to_numpy(x32) > 90.0).all()
        assert (ar.to_numpy(x32) < 110.0).all()

    # test default single scalar
    x = anp.random.normal(loc=1500, scale=10, like=backend)
    assert 1000 <= ar.to_numpy(x) < 2000


@pytest.mark.parametrize("backend", gen_params(backends=..., requires="tril"))
def test_tril(backend):
    x = gen_rand((4, 4), backend)
    xl = ar.do("tril", x)
    xln = ar.to_numpy(xl)
    assert xln[0, 1] == 0.0
    if backend != "sparse":
        # this won't work for sparse because density < 1
        assert (xln > 0.0).sum() == 10
    xl = ar.do("tril", x, k=1)
    xln = ar.to_numpy(xl)
    if backend != "sparse":
        # this won't work for sparse because density < 1
        assert xln[0, 1] != 0.0
    assert xln[0, 2] == 0.0
    if backend != "sparse":
        # this won't work for sparse because density < 1
        assert (xln > 0.0).sum() == 13


@pytest.mark.parametrize("backend", gen_params(backends=..., requires="triu"))
def test_triu(backend):
    x = gen_rand((4, 4), backend)
    xl = ar.do("triu", x)
    xln = ar.to_numpy(xl)
    assert xln[1, 0] == 0.0
    if backend != "sparse":
        # this won't work for sparse because density < 1
        assert (xln > 0.0).sum() == 10
    xl = ar.do("triu", x, k=-1)
    xln = ar.to_numpy(xl)
    if backend != "sparse":
        # this won't work for sparse because density < 1
        assert xln[1, 0] != 0.0
    assert xln[2, 0] == 0.0
    if backend != "sparse":
        # this won't work for sparse because density < 1
        assert (xln > 0.0).sum() == 13


@pytest.mark.parametrize(
    "backend", gen_params(backends=..., requires="count_nonzero")
)
@pytest.mark.parametrize("array_dtype", ["int", "float", "bool"])
def test_count_nonzero(backend, array_dtype):
    if array_dtype == "int":
        x = ar.do("asarray", [0, 1, 2, 0, 3], like=backend)
    elif array_dtype == "float":
        x = ar.do("asarray", [0.0, 1.0, 2.0, 0.0, 3.0], like=backend)
    elif array_dtype == "bool":
        x = ar.do("asarray", [False, True, True, False, True], like=backend)
    nz = ar.do("count_nonzero", x)
    assert ar.to_numpy(nz) == 3


def test_pseudo_submodules():
    x = gen_rand((2, 3), "numpy")
    xT = ar.do("numpy.transpose", x, like="autoray")
    assert shape(xT) == (3, 2)


@pytest.mark.parametrize(
    "backend,dtype,fn",
    gen_params(backends=..., fns=["ones", "zeros"], dtypes=...),
)
def test_dtype_specials(backend, dtype, fn):
    import numpy as np

    x = ar.do(fn, shape=(2, 3), like=backend)

    x = ar.astype(x, dtype)
    assert ar.get_dtype_name(x) == dtype
    x = ar.to_numpy(x)
    assert isinstance(x, np.ndarray)
    assert ar.get_dtype_name(x) == dtype


@pytest.mark.parametrize(
    "backend,real_dtype",
    gen_params(
        backends=..., dtypes=["float32", "float64"], requires="complex"
    ),
)
def test_complex_creation(backend, real_dtype):
    x = ar.do(
        "complex",
        ar.astype(
            ar.do("random.uniform", size=(3, 4), like=backend), real_dtype
        ),
        ar.astype(
            ar.do("random.uniform", size=(3, 4), like=backend), real_dtype
        ),
    )
    assert (
        ar.get_dtype_name(x)
        == {"float32": "complex64", "float64": "complex128"}[real_dtype]
    )


@pytest.mark.parametrize(
    "backend,dtype_in", gen_params(backends=..., dtypes=...)
)
def test_real_imag(backend, dtype_in):

    dtype_out = {
        "float32": "float32",
        "float64": "float64",
        "complex64": "float32",
        "complex128": "float64",
    }[dtype_in]

    x = gen_rand((3, 4), backend, dtype_in)

    re = ar.do("real", x)
    im = ar.do("imag", x)

    assert ar.infer_backend(re) == backend
    assert ar.infer_backend(im) == backend

    assert ar.get_dtype_name(re) == dtype_out
    assert ar.get_dtype_name(im) == dtype_out

    assert ar.do("allclose", ar.to_numpy(x).real, ar.to_numpy(re))
    assert ar.do("allclose", ar.to_numpy(x).imag, ar.to_numpy(im))


@pytest.mark.parametrize("backend", gen_params(backends=..., requires="pad"))
def test_pad(backend):

    A = gen_rand((3, 4, 5), backend)

    for pad_width, new_shape in [
        # same pad before and after for every axis
        (2, (7, 8, 9)),
        # same pad for every axis
        (((1, 2),), (6, 7, 8)),
        # different pad for every axis
        (((4, 3), (2, 4), (3, 2)), (10, 10, 10)),
    ]:
        B = ar.do("pad", A, pad_width)
        assert shape(B) == new_shape
        assert ar.to_numpy(ar.do("sum", A)) == pytest.approx(
            ar.to_numpy(ar.do("sum", B))
        )


@pytest.mark.parametrize("backend", gen_params(backends=...))
def test_register_function(backend):
    x = ar.do("ones", shape=(2, 3), like=backend)

    def direct_fn(x):
        return 1

    # first test we can provide the function directly
    ar.register_function(backend, "test_register", direct_fn)
    assert ar.do("test_register", x) == 1

    def wrap_fn(fn):
        def new_fn(*args, **kwargs):
            res = fn(*args, **kwargs)
            return res + 1

        return new_fn

    # then check we can wrap the old (previous) function
    ar.register_function(backend, "test_register", wrap_fn, wrap=True)
    assert ar.do("test_register", x) == 2


@pytest.mark.parametrize("backend", gen_params(backends=...))
def test_register_function_decorator(backend):
    x = ar.do("ones", shape=(2, 3), like=backend)

    @ar.register_function(backend, "test_register_decorator")
    def test_register_decorator(x):
        return 1

    assert ar.do("test_register_decorator", x) == 1

    @ar.register_function(backend, "test_register_decorator", wrap=True)
    def wrap_fn(fn):
        def new_fn(*args, **kwargs):
            res = fn(*args, **kwargs)
            return res + 1

        return new_fn

    assert ar.do("test_register_decorator", x) == 2


@pytest.mark.parametrize("backend", gen_params(backends=..., requires="take"))
def test_take(backend):
    num_inds = 4
    A = gen_rand((2, 3, 4), backend)
    if backend == "jax":  # gen_rand doesn't work with ints for JAX
        ind = gen_rand((num_inds,), "numpy", dtype="int64")
    else:
        ind = gen_rand((num_inds,), backend, dtype="int64")

    # Take along axis 1, and check if result makes sense
    B = ar.do("take", A, ind, axis=1)
    assert shape(B) == (2, 4, 4)
    for i in range(num_inds):
        assert ar.do(
            "allclose", ar.to_numpy(A[:, ind[0], :]), ar.to_numpy(B[:, 0, :])
        )
    assert ar.infer_backend(A) == ar.infer_backend(B)


@pytest.mark.parametrize(
    "backend", gen_params(backends=..., requires="concatenate")
)
def test_concatenate(backend):
    mats = [gen_rand((2, 3, 4), backend) for _ in range(3)]

    # Concatenate along axis 1, check if shape is correct
    # also check if automatically inferring backend works
    mats_concat1 = ar.do("concatenate", mats, axis=1)
    mats_concat2 = ar.do("concatenate", mats, axis=1, like=backend)
    assert shape(mats_concat1) == shape(mats_concat2) == (2, 9, 4)
    assert (
        backend
        == ar.infer_backend(mats_concat1)
        == ar.infer_backend(mats_concat2)
    )


@pytest.mark.parametrize("backend", gen_params(backends=..., requires="stack"))
def test_stack(backend):
    mats = [gen_rand((2, 3, 4), backend) for _ in range(3)]

    # stack, creating a new axis (at position 0)
    # also check if automatically inferring backend works
    mats_stack1 = ar.do("stack", mats)
    mats_stack2 = ar.do("stack", mats, like=backend)
    assert shape(mats_stack1) == shape(mats_stack2) == (3, 2, 3, 4)
    assert (
        backend
        == ar.infer_backend(mats_stack1)
        == ar.infer_backend(mats_stack2)
    )


@pytest.mark.parametrize(
    "backend,fn,args",
    gen_params(
        backends=...,
        fns=[
            ("einsum", ("eq",)),
            ("einsum", ("interleaved",)),
        ],
    ),
)
def test_einsum(backend, fn, args):
    A = gen_rand((2, 3, 4), backend)
    B = gen_rand((3, 4, 2), backend)

    (einsum_type,) = args
    if einsum_type == "eq":
        C1 = ar.do(fn, "ijk,jkl->il", A, B, like=backend)
        C2 = ar.do(fn, "ijk,jkl->il", A, B)
    elif einsum_type == "interleaved":
        C1 = ar.do("einsum", A, [0, 1, 2], B, [1, 2, 3], [0, 3], like=backend)
        C2 = ar.do("einsum", A, [0, 1, 2], B, [1, 2, 3], [0, 3])

    C3 = ar.do("reshape", A, (2, 12)) @ ar.do("reshape", B, (12, 2))

    assert shape(C1) == shape(C2) == (2, 2)
    assert ar.do("allclose", ar.to_numpy(C1), ar.to_numpy(C3))
    assert ar.do("allclose", ar.to_numpy(C2), ar.to_numpy(C3))
    assert (
        ar.infer_backend(C1)
        == ar.infer_backend(C2)
        == ar.infer_backend(C3)
        == backend
    )


@pytest.mark.parametrize("backend", gen_params(backends=..., requires="trace"))
def test_trace(backend):

    x = gen_rand((4, 4), backend)
    tr = ar.do("trace", x)
    assert shape(tr) == ()
    assert ar.infer_backend(tr) == backend

    x = gen_rand((3, 5, 5), backend)
    tr = ar.do("trace", x, axis1=1, axis2=2)
    assert shape(tr) == (3,)
    assert ar.infer_backend(tr) == backend


@pytest.mark.parametrize(
    "backend,fn,args",
    gen_params(
        backends=...,
        fns=[
            ("moveaxis", ("single",)),
            ("moveaxis", ("multiple",)),
        ],
    ),
)
def test_moveaxis(backend, fn, args):
    axis_type = args[0]

    if axis_type == "single":
        x = gen_rand((2, 3, 4), backend)
        xn = ar.to_numpy(x)

        # single axis, positive and negative index
        y = ar.do("moveaxis", x, 0, -1)
        assert shape(y) == (3, 4, 2)
        assert ar.infer_backend(y) == backend
        assert ar.do("allclose", ar.to_numpy(y), ar.do("moveaxis", xn, 0, -1))

        y = ar.do("moveaxis", x, 2, 0)
        assert shape(y) == (4, 2, 3)
        assert ar.infer_backend(y) == backend
        assert ar.do("allclose", ar.to_numpy(y), ar.do("moveaxis", xn, 2, 0))

    elif axis_type == "multiple":
        x = gen_rand((2, 3, 4, 5), backend)
        xn = ar.to_numpy(x)

        y = ar.do("moveaxis", x, [0, 1], [-1, -2])
        assert shape(y) == (4, 5, 3, 2)
        assert ar.infer_backend(y) == backend
        assert ar.do(
            "allclose", ar.to_numpy(y), ar.do("moveaxis", xn, [0, 1], [-1, -2])
        )


@pytest.mark.parametrize(
    "backend", gen_params(backends=..., requires="swapaxes")
)
def test_swapaxes(backend):

    x = gen_rand((2, 3, 4), backend)
    xn = ar.to_numpy(x)

    y = ar.do("swapaxes", x, 0, 2)
    assert shape(y) == (4, 3, 2)
    assert ar.infer_backend(y) == backend
    assert ar.do("allclose", ar.to_numpy(y), ar.do("swapaxes", xn, 0, 2))

    # negative indices
    y = ar.do("swapaxes", x, -1, -3)
    assert shape(y) == (4, 3, 2)
    assert ar.infer_backend(y) == backend
    assert ar.do("allclose", ar.to_numpy(y), ar.do("swapaxes", xn, -1, -3))


@pytest.mark.parametrize("backend", gen_params(backends=..., requires="split"))
@pytest.mark.parametrize("int_or_section", ["int", "section", "empty"])
def test_split(backend, int_or_section):
    A = ar.do("ones", (10, 20, 10), like=backend)
    if int_or_section == "section":
        sections = [2, 4, 14]
        splits = ar.do("split", A, sections, axis=1)
        assert len(splits) == 4
        assert ar.shape(splits[3]) == (10, 6, 10)
    elif int_or_section == "empty":
        splits = ar.do("split", A, [], axis=1)
        assert len(splits) == 1
        assert ar.shape(splits[0]) == (10, 20, 10)
    else:
        splits = ar.do("split", A, 5, axis=2)
        assert len(splits) == 5
        assert ar.shape(splits[2]) == (10, 20, 2)


@pytest.mark.parametrize(
    "backend", gen_params(backends=..., requires=("nonzero", "arange"))
)
def test_nonzero(backend):
    A = ar.do("arange", 10, like=backend)
    B = ar.do("arange", 10, like=backend) + 1
    C = ar.do("stack", [A, B])
    D = ar.do("nonzero", C < 5)
    if backend == "dask":
        for x in D:
            x.compute_chunk_sizes()
    for x in D:
        assert ar.to_numpy(x).shape == (9,)


@pytest.mark.parametrize("backend", gen_params(backends=..., requires="where"))
def test_where(backend):
    x = ar.do("asarray", [-1.0, 0.0, 1.0], like=backend)
    y = ar.do("asarray", [10.0, 10.0, 10.0], like=backend)
    out = ar.do("where", x > 0.0, x, y)
    assert ar.do("allclose", ar.to_numpy(out), [10.0, 10.0, 1.0])


@pytest.mark.parametrize(
    "backend,dtype_str,fn",
    gen_params(
        backends=...,
        dtypes=["float32", "float64"],
        fns=["random.normal", "random.uniform", "zeros", "ones", "eye"],
    ),
)
@pytest.mark.parametrize("str_or_backend", ("str", "backend"))
def test_dtype_kwarg(backend, dtype_str, fn, str_or_backend):
    if str_or_backend == "str":
        dtype = dtype_str
    else:
        dtype = ar.to_backend_dtype(dtype_str, like=backend)

    if fn in ("random.normal", "random.uniform"):
        A = ar.do(fn, size=(10, 5), dtype=dtype, like=backend)
    elif fn in ("zeros", "ones"):
        A = ar.do(fn, shape=(10, 5), dtype=dtype, like=backend)
    else:  # fn = 'eye'
        A = ar.do(fn, 10, dtype=dtype, like=backend)
        assert shape(A) == (10, 10)
        A = ar.do(fn, 10, 5, dtype=dtype, like=backend)
    assert shape(A) == (10, 5)
    assert ar.get_dtype_name(A) == dtype_str


@pytest.mark.parametrize("backend", gen_params(backends=..., requires="ones"))
def test_get_common_dtype(backend):
    x = ar.do("ones", (1,), like=backend, dtype="complex64")
    y = ar.do("ones", (1,), like=backend, dtype="float64")
    assert ar.get_common_dtype(x, y) == "complex128"


@pytest.mark.parametrize("backend", gen_params(backends=..., requires="ones"))
def test_backend_like(backend):
    assert ar.get_backend() is None
    ar.set_backend("test")
    assert ar.get_backend() == "test"
    ar.set_backend(None)
    assert ar.get_backend() is None
    with ar.backend_like(backend):
        assert ar.get_backend() == backend
        x = ar.do("ones", (2,), like=backend)
        assert ar.infer_backend(x) == backend
    assert ar.get_backend() is None


def test_nested_multihreaded_backend_like():
    from concurrent.futures import ThreadPoolExecutor

    from autoray.autoray import choose_backend

    def foo(backend1, backend2):
        bs = []
        bs.append((ar.get_backend(), choose_backend("test", 1)))
        with ar.backend_like(backend1):
            bs.append((ar.get_backend(), choose_backend("test", 1)))
            with ar.backend_like(backend2):
                bs.append((ar.get_backend(), choose_backend("test", 1)))
            bs.append((ar.get_backend(), choose_backend("test", 1)))
        bs.append((ar.get_backend(), choose_backend("test", 1)))
        return bs

    b_exp = [("A", "A"), ("B", "B"), ("C", "C"), ("B", "B"), ("A", "A")]
    with ar.backend_like("A"):
        b = foo("B", "C")
    assert b == b_exp

    b_exp = [
        ("A", "A"),
        ("B", "B"),
        (None, "builtins"),
        ("B", "B"),
        ("A", "A"),
    ]
    with ar.backend_like("A"):
        b = foo("B", None)
    assert b == b_exp

    with ThreadPoolExecutor(3) as pool:
        b_exp = [(None, "A"), ("B", "B"), ("C", "C"), ("B", "B"), (None, "A")]
        with ar.backend_like("A"):
            bs = [pool.submit(foo, "B", "C") for _ in range(3)]
            for b in bs:
                assert b.result() == b_exp

        b_exp = [(None, "A"), ("B", "B"), (None, "A"), ("B", "B"), (None, "A")]
        with ar.backend_like("A"):
            bs = [pool.submit(foo, "B", None) for _ in range(3)]
            for b in bs:
                assert b.result() == b_exp


def test_compose():
    @ar.compose
    def mycomposedfn(x, backend):
        x = ar.do("exp", x, like=backend)
        x = ar.do("log", x, like=backend)
        return x

    x = ar.do("ones", (2,), like="numpy")
    y = ar.do("mycomposedfn", x)
    assert ar.do("allclose", x, y)
    y = mycomposedfn(x)
    assert ar.do("allclose", x, y)
    mycomposedfn.register("numpy", lambda x: 1)
    y = ar.do("mycomposedfn", x)
    assert y == 1
    y = mycomposedfn(x)
    assert y == 1

    @mycomposedfn.register("numpy")
    def f(x):
        return 2

    y = ar.do("mycomposedfn", x)
    assert y == 2
    y = mycomposedfn(x)
    assert y == 2


def test_builtins_complex():
    re = 1.0
    im = 2.0
    z = ar.do("complex", re, im)
    assert z == 1.0 + 2.0j
    assert ar.infer_backend(z) == "builtins"


def test_shape_ndim_builtins():
    import numpy as np

    xs = [
        1,
        4.0,
        7j,
        (),
        [],
        [[]],
        [np.ones(3), np.ones(3)],
        np.ones((5, 4, 3)),
    ]
    for x in xs:
        assert ar.shape(x) == np.shape(x)
        assert ar.ndim(x) == np.ndim(x)


@pytest.mark.parametrize(
    "backend",
    gen_params(backends=..., requires="scipy.linalg.expm"),
)
def test_scipy_expm_dispatching(backend):
    x = gen_rand((3, 3), backend=backend)
    ar.do("scipy.linalg.expm", x)


def check_array_dtypes(x, y):
    assert x.dtype == y.dtype
    if hasattr(x, "device"):
        assert x.device == y.device


_CREATION_CALLS = {
    "empty": ((2, 3),),
    "eye": (3,),
    "full": ((2, 3), 7),
    "identity": (4,),
    "ones": ((2, 3),),
    "zeros": ((2, 3),),
}


@pytest.mark.parametrize(
    "backend,dtype,fn",
    gen_params(
        backends=...,
        dtypes=...,
        fns=list(_CREATION_CALLS),
    ),
)
@pytest.mark.parametrize("use_namespace", [False, True])
def test_creation_passes_dtype_device(backend, dtype, fn, use_namespace):
    x = gen_rand((1,), backend, dtype)
    call_args = _CREATION_CALLS[fn]
    if use_namespace:
        xp = ar.get_namespace(x)
        y = getattr(xp, fn)(*call_args)
    else:
        y = ar.do(fn, *call_args, like=x)
    check_array_dtypes(x, y)


creation_funcs_with_args = [
    ("empty", ((2, 3),)),
    ("eye", (4,)),
    ("full", ((2, 3), 7)),
    ("identity", (4,)),
    ("ones", ((2, 3),)),
    ("zeros", ((2, 3),)),
]

creation_builtins = [
    (float, [np.float64]),
    (int, [np.int32, np.int64]),  # np.int32 on Windows and np.int64 else
    (complex, [np.complex128]),
]


@pytest.mark.parametrize("fn, args", creation_funcs_with_args)
@pytest.mark.parametrize("dtype, expected", creation_builtins)
def test_creation_with_builtins(fn, args, dtype, expected):
    x = dtype(4)
    y = ar.do(fn, *args, like=x)
    assert y.dtype in expected


@pytest.mark.parametrize(
    "backend", gen_params(backends=..., requires="indices")
)
def test_indices(backend):
    from numpy.testing import assert_array_equal

    x = ar.do("indices", (3, 4), like=backend)
    xn = ar.to_numpy(x)
    xe = ar.do("indices", (3, 4), like="numpy")
    assert_array_equal(xn, xe)


@pytest.mark.parametrize("backend", gen_params(backends=...))
def test_is_array(backend):
    x = gen_rand((2, 3), backend)
    assert ar.is_array(x)
    x = gen_rand((), backend)
    assert ar.is_array(x)
    y = 5
    assert not ar.is_array(y)
    y = [5]
    assert not ar.is_array(y)


@pytest.mark.parametrize("backend", gen_params(backends=...))
def test_is_scalar(backend):
    x = gen_rand((2, 3), backend)
    assert not ar.is_scalar(x)
    x = gen_rand((), backend)
    assert ar.is_scalar(x)
    y = 5
    assert ar.is_scalar(y)
    y = [5]
    assert not ar.is_scalar(y)


@pytest.mark.parametrize("backend", gen_params(backends=..., requires="array"))
def test_function_array(backend):

    x = 2.0
    z1 = ar.do("array", [x], like=backend)
    assert ar.do("shape", z1) == (1,)
    assert ar.infer_backend(z1) == backend

    z2 = ar.do("array", (x,), like=backend)
    assert ar.do("shape", z2) == (1,)
    assert ar.infer_backend(z2) == backend

    y = ar.do("array", x, like=backend)
    z3 = ar.do("array", [y], like=y)
    assert ar.do("shape", z3) == (1,)
    assert ar.infer_backend(z3) == backend

    z4 = ar.do("array", (y,), like=y)
    assert ar.do("shape", z4) == (1,)
    assert ar.infer_backend(z4) == backend

    z5 = ar.do("array", z4, like=z4)
    assert ar.do("shape", z5) == (1,)
    assert ar.infer_backend(z5) == backend


@pytest.mark.parametrize(
    "backend", gen_params(backends=..., requires="asarray")
)
def test_function_asarray(backend):

    x = 2.0
    z1 = ar.do("asarray", [x], like=backend)
    assert ar.do("shape", z1) == (1,)
    assert ar.infer_backend(z1) == backend

    z2 = ar.do("asarray", (x,), like=backend)
    assert ar.do("shape", z2) == (1,)
    assert ar.infer_backend(z2) == backend

    y = ar.do("asarray", x, like=backend)
    z3 = ar.do("asarray", [y], like=y)
    assert ar.do("shape", z3) == (1,)
    assert ar.infer_backend(z3) == backend

    z4 = ar.do("asarray", (y,), like=y)
    assert ar.do("shape", z4) == (1,)
    assert ar.infer_backend(z4) == backend

    z5 = ar.do("asarray", z4, like=z4)
    assert ar.do("shape", z5) == (1,)
    assert ar.infer_backend(z5) == backend
