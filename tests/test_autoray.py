import importlib.util

import pytest

import autoray as ar
from autoray import shape


# find backends to tests
BACKENDS = [pytest.param("numpy")]
for lib in ["cupy", "dask", "tensorflow", "torch", "mars", "jax", "sparse"]:
    if importlib.util.find_spec(lib):
        BACKENDS.append(pytest.param(lib))

        if lib == "jax":
            import os
            from jax.config import config

            config.update("jax_enable_x64", True)
            config.update("jax_platform_name", "cpu")
            os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
    else:
        BACKENDS.append(
            pytest.param(
                lib, marks=pytest.mark.skipif(True, reason=f"No {lib}.")
            )
        )


JAX_RANDOM_KEY = None


def gen_rand(shape, backend, dtype="float64"):
    if "complex" in dtype:
        re = gen_rand(shape, backend)
        im = gen_rand(shape, backend)
        return ar.astype(ar.do("complex", re, im), dtype)

    if backend == "jax":
        from jax import random as jrandom

        global JAX_RANDOM_KEY

        if JAX_RANDOM_KEY is None:
            JAX_RANDOM_KEY = jrandom.PRNGKey(42)
        JAX_RANDOM_KEY, subkey = jrandom.split(JAX_RANDOM_KEY)

        return jrandom.uniform(subkey, shape=shape, dtype=dtype)

    elif backend == "sparse":
        x = ar.do(
            "random.uniform",
            size=shape,
            like=backend,
            density=0.5,
            format="coo",
            fill_value=0,
        )

    else:
        x = ar.do("random.uniform", size=shape, like=backend)

    x = ar.astype(x, ar.to_backend_dtype(dtype, backend))
    assert ar.get_dtype_name(x) == dtype
    return x


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("fn", ["sqrt", "exp", "sum"])
def test_basic(backend, fn):
    if (backend == "ctf") and fn in ("sqrt", "sum"):
        pytest.xfail("ctf doesn't have sqrt, and converts sum output to numpy")

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


@pytest.mark.parametrize("backend", BACKENDS)
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
    if (backend == "torch") and fn in (ar.real, ar.imag):
        pytest.xfail("Pytorch doesn't support complex numbers yet...")

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


@pytest.mark.parametrize("backend", BACKENDS)
def test_mgs(backend):
    if backend == "sparse":
        pytest.xfail("Sparse doesn't support linear algebra yet...")
    if backend == "ctf":
        pytest.xfail("ctf does not have 'stack' function.")
    x = gen_rand((3, 5), backend)
    Ux = modified_gram_schmidt(x)
    y = ar.do("sum", Ux @ ar.dag(Ux))
    assert ar.to_numpy(y) == pytest.approx(3)


def modified_gram_schmidt_np_mimic(X):
    from autoray import numpy as np

    print(np)

    Q = []
    for j in range(0, shape(X)[0]):
        q = X[j, :]
        for i in range(0, j):
            rij = np.tensordot(np.conj(Q[i]), q, 1)
            q = q - rij * Q[i]

        rjj = np.linalg.norm(q, 2)
        Q.append(q / rjj)

    return np.stack(Q, axis=0)


@pytest.mark.parametrize("backend", BACKENDS)
def test_mgs_np_mimic(backend):
    if backend == "sparse":
        pytest.xfail("Sparse doesn't support linear algebra yet...")
    if backend == "ctf":
        pytest.xfail("ctf does not have 'stack' function.")
    x = gen_rand((3, 5), backend)
    Ux = modified_gram_schmidt_np_mimic(x)
    y = ar.do("sum", Ux @ ar.dag(Ux))
    assert ar.to_numpy(y) == pytest.approx(3)


@pytest.mark.parametrize("backend", BACKENDS)
def test_linalg_svd_square(backend):
    if backend == "sparse":
        pytest.xfail("Sparse doesn't support linear algebra yet...")
    x = gen_rand((5, 4), backend)
    U, s, V = ar.do("linalg.svd", x)
    assert (
        ar.infer_backend(x)
        == ar.infer_backend(U)
        == ar.infer_backend(s)
        == ar.infer_backend(V)
        == backend
    )
    y = U @ ar.do("diag", s, like=x) @ V
    diff = ar.do("sum", abs(y - x))
    assert ar.to_numpy(diff) < 1e-8


@pytest.mark.parametrize("backend", BACKENDS)
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


@pytest.mark.parametrize("backend", BACKENDS)
def test_translator_random_normal(backend):
    if backend == "ctf":
        pytest.xfail()

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


@pytest.mark.parametrize("backend", BACKENDS)
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

    if backend == "tensorflow":
        with pytest.raises(ValueError):
            ar.do("tril", x, -1)


@pytest.mark.parametrize("backend", BACKENDS)
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

    if backend == "tensorflow":
        with pytest.raises(ValueError):
            ar.do("triu", x, 1)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("shape", [(4, 3), (4, 4), (3, 4)])
def test_qr_thin_square_fat(backend, shape):
    if backend == "sparse":
        pytest.xfail("Sparse doesn't support linear algebra yet...")
    x = gen_rand(shape, backend)
    Q, R = ar.do("linalg.qr", x)
    xn, Qn, Rn = map(ar.to_numpy, (x, Q, R))
    assert ar.do("allclose", xn, Qn @ Rn)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("array_dtype", ["int", "float", "bool"])
def test_count_nonzero(backend, array_dtype):
    if backend == "mars":
        import mars

        if tuple(map(int, mars.__version__.split("."))) < (0, 4, 0):
            pytest.xfail("mars count_nonzero bug fixed in version 0.4.")
    if backend == "ctf" and array_dtype == "bool":
        pytest.xfail("ctf doesn't support bool array dtype")

    if array_dtype == "int":
        x = ar.do("array", [0, 1, 2, 0, 3], like=backend)
    elif array_dtype == "float":
        x = ar.do("array", [0.0, 1.0, 2.0, 0.0, 3.0], like=backend)
    elif array_dtype == "bool":
        x = ar.do("array", [False, True, True, False, True], like=backend)
    nz = ar.do("count_nonzero", x)
    assert ar.to_numpy(nz) == 3


def test_pseudo_submodules():
    x = gen_rand((2, 3), "numpy")
    xT = ar.do("numpy.transpose", x, like="autoray")
    assert shape(xT) == (3, 2)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("creation", ["ones", "zeros"])
@pytest.mark.parametrize(
    "dtype", ["float32", "float64", "complex64", "complex128"]
)
def test_dtype_specials(backend, creation, dtype):
    import numpy as np

    x = ar.do(creation, shape=(2, 3), like=backend)

    if backend == "torch" and "complex" in dtype:
        pytest.xfail("Pytorch doesn't support complex numbers yet...")

    x = ar.astype(x, dtype)
    assert ar.get_dtype_name(x) == dtype
    x = ar.to_numpy(x)
    assert isinstance(x, np.ndarray)
    assert ar.get_dtype_name(x) == dtype


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("real_dtype", ["float32", "float64"])
def test_complex_creation(backend, real_dtype):
    if backend == "torch":
        pytest.xfail("Pytorch doesn't support complex numbers yet...")
    if (backend == "sparse") and (real_dtype == "float32"):
        pytest.xfail(
            "Bug in sparse where single precision isn't maintained "
            "after scalar multiplication."
        )

    if (backend == "ctf") and (real_dtype == "float32"):
        pytest.xfail(
            "ctf currently doesn't preserve single precision when "
            "multiplying by python scalars."
        )

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


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    "dtype_in,dtype_out",
    [
        ("float32", "float32"),
        ("float64", "float64"),
        ("complex64", "float32"),
        ("complex128", "float64"),
    ],
)
def test_real_imag(backend, dtype_in, dtype_out):
    x = gen_rand((3, 4), backend, dtype_in)

    re = ar.do("real", x)
    im = ar.do("imag", x)

    assert ar.infer_backend(re) == backend
    assert ar.infer_backend(im) == backend

    assert ar.get_dtype_name(re) == dtype_out
    assert ar.get_dtype_name(im) == dtype_out

    assert ar.do("allclose", ar.to_numpy(x).real, ar.to_numpy(re))
    assert ar.do("allclose", ar.to_numpy(x).imag, ar.to_numpy(im))


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    "dtype",
    ["float32", "float64", "complex64", "complex128"],
)
def test_linalg_solve(backend, dtype):
    if backend == "sparse":
        pytest.xfail("Sparse doesn't support linear algebra yet...")

    A = gen_rand((4, 4), backend, dtype)
    b = gen_rand((4, 1), backend, dtype)
    x = ar.do("linalg.solve", A, b)
    assert ar.do(
        "allclose", ar.to_numpy(A @ x), ar.to_numpy(b), rtol=1e-3, atol=1e-6
    )


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    "dtype",
    ["float32", "float64", "complex64", "complex128"],
)
def test_linalg_eigh(backend, dtype):
    if backend == "sparse":
        pytest.xfail("sparse doesn't support linalg.eigh yet.")
    if backend == "dask":
        pytest.xfail("dask doesn't support linalg.eigh yet.")
    if backend == "mars":
        pytest.xfail("mars doesn't support linalg.eigh yet.")
    if (backend == "torch") and ("complex" in dtype):
        pytest.xfail("Pytorch doesn't fully support complex yet.")

    A = gen_rand((4, 4), backend, dtype)
    A = A + ar.dag(A)
    el, ev = ar.do("linalg.eigh", A)
    B = (ev * ar.reshape(el, (1, -1))) @ ar.dag(ev)
    assert ar.do("allclose", ar.to_numpy(A), ar.to_numpy(B), rtol=1e-3)


@pytest.mark.parametrize("backend", BACKENDS)
def test_pad(backend):
    if backend == "sparse":
        pytest.xfail("sparse doesn't support linalg.eigh yet.")
    if backend == "mars":
        pytest.xfail("mars doesn't support linalg.eigh yet.")

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


@pytest.mark.parametrize("backend", BACKENDS)
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


@pytest.mark.parametrize("backend", BACKENDS)
def test_take(backend):
    if backend == "sparse":
        pytest.xfail("sparse doesn't support take yet")
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


@pytest.mark.parametrize("backend", BACKENDS)
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


@pytest.mark.parametrize("backend", BACKENDS)
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


@pytest.mark.parametrize("backend", BACKENDS)
def test_einsum(backend):
    if backend == "sparse":
        pytest.xfail("sparse doesn't support einsum yet")
    A = gen_rand((2, 3, 4), backend)
    B = gen_rand((3, 4, 2), backend)
    C1 = ar.do("einsum", "ijk,jkl->il", A, B, like=backend)
    C2 = ar.do("einsum", "ijk,jkl->il", A, B)
    if backend not in ("torch", "tensorflow"):  # this syntax is not supported
        C3 = ar.do("einsum", A, [0, 1, 2], B, [1, 2, 3], [0, 3])
    else:
        C3 = C1
    C4 = ar.do("reshape", A, (2, 12)) @ ar.do("reshape", B, (12, 2))

    assert shape(C1) == shape(C2) == shape(C3) == (2, 2)
    assert ar.do("allclose", ar.to_numpy(C1), ar.to_numpy(C4))
    assert ar.do("allclose", ar.to_numpy(C2), ar.to_numpy(C4))
    assert ar.do("allclose", ar.to_numpy(C3), ar.to_numpy(C4))
    assert (
        ar.infer_backend(C1)
        == ar.infer_backend(C2)
        == ar.infer_backend(C3)
        == ar.infer_backend(C4)
        == backend
    )


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("int_or_section", ["int", "section"])
def test_split(backend, int_or_section):
    if backend == "sparse":
        pytest.xfail("sparse doesn't support split yet")
    if backend == "dask":
        pytest.xfail("dask doesn't support split yet")
    A = ar.do("ones", (10, 20, 10), like=backend)
    if int_or_section == "section":
        sections = [2, 4, 14]
        splits = ar.do("split", A, sections, axis=1)
        assert len(splits) == 4
        assert splits[3].shape == (10, 6, 10)
    else:
        splits = ar.do("split", A, 5, axis=2)
        assert len(splits) == 5
        assert splits[2].shape == (10, 20, 2)


@pytest.mark.parametrize("backend", BACKENDS)
def test_where(backend):
    if backend == "sparse":
        pytest.xfail("sparse doesn't support where yet")
    A = ar.do("arange", 10, like=backend)
    B = ar.do("arange", 10, like=backend) + 1
    C = ar.do("stack", [A, B])
    D = ar.do("where", C < 5)
    if backend == "dask":
        for x in D:
            x.compute_chunk_sizes()
    for x in D:
        assert ar.to_numpy(x).shape == (9,)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("dtype_str", ["float32", "float64"])
@pytest.mark.parametrize(
    "fn", ["random.normal", "random.uniform", "zeros", "ones", "eye"]
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


@pytest.mark.parametrize("backend", BACKENDS)
def test_get_common_dtype(backend):
    x = ar.do("ones", (1,), like=backend, dtype="complex64")
    y = ar.do("ones", (1,), like=backend, dtype="float64")
    assert ar.get_common_dtype(x, y) == "complex128"


@pytest.mark.parametrize("backend", BACKENDS)
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
    from autoray.autoray import choose_backend
    from concurrent.futures import ThreadPoolExecutor

    def foo(backend1, backend2):
        bs = []
        bs.append(
            (
                ar.get_backend(),
                choose_backend("test", 1),
            )
        )
        with ar.backend_like(backend1):
            bs.append(
                (
                    ar.get_backend(),
                    choose_backend("test", 1),
                )
            )
            with ar.backend_like(backend2):
                bs.append(
                    (
                        ar.get_backend(),
                        choose_backend("test", 1),
                    )
                )
            bs.append(
                (
                    ar.get_backend(),
                    choose_backend("test", 1),
                )
            )
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
