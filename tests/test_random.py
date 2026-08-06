from collections import Counter

import pytest

import autoray as ar

from .conftest import gen_params, gen_rand

_RANDOM_BACKENDS = [
    "cupy",
    "dask",
    "jax",
    "mlx",
    "numpy",
    "tensorflow",
    "torch",
]


@pytest.mark.parametrize(
    "backend,fn,args,kwargs",
    gen_params(
        backends=_RANDOM_BACKENDS,
        fns=[
            ("binomial", (7, 0.424), {"size": (3, 4)}),
            (
                "choice",
                (tuple(11.1 * i for i in range(100)),),
                {"size": (3, 4)},
            ),
            ("choice", (tuple(11.1 * i for i in range(1000)),), {}),
            ("exponential", (), {"size": (3, 4)}),
            ("exponential", (), {}),
            ("gumbel", (), {"size": (3, 4)}),
            ("gumbel", (), {}),
            ("integers", (100, 1000), {"size": (3, 4)}),
            ("integers", (100, 1000), {}),
            ("normal", (), {"size": (3, 4)}),
            ("normal", (), {}),
            ("permutation", (tuple(11.1 * i for i in range(100)),), {}),
            ("poisson", (100,), {"size": (3, 4, 5)}),
            ("random", (), {"size": (3, 4)}),
            ("random", (), {}),
            ("uniform", (), {"size": (3, 4)}),
            ("uniform", (), {}),
        ],
    ),
)
def test_random_default_rng(backend, fn, args, kwargs):
    if fn in ("choice", "permutation"):
        args = (ar.do("array", args[0], like=backend), *args[1:])

    seed = 42
    seed2 = 43

    rng = ar.do("random.default_rng", seed, like=backend)
    x = ar.do("to_numpy", getattr(rng, fn)(*args, **kwargs))
    if "size" in kwargs:
        assert ar.do("shape", x) == kwargs["size"]
    y = ar.do("to_numpy", getattr(rng, fn)(*args, **kwargs))
    assert not ar.do("allclose", x, y)
    rng = ar.do("random.default_rng", seed2, like=backend)
    z = ar.do("to_numpy", getattr(rng, fn)(*args, **kwargs))
    assert not ar.do("allclose", x, z)
    rng = ar.do("random.default_rng", seed, like=backend)
    x2 = ar.do("to_numpy", getattr(rng, fn)(*args, **kwargs))
    assert ar.do("allclose", x, x2)


@pytest.mark.parametrize("fn", ["normal", "random", "uniform"])
@pytest.mark.parametrize(
    "dtype_name", ["float16", "bfloat16", "float32", "float64"]
)
def test_torch_default_rng_inherits_float_dtype(fn, dtype_name):
    torch = pytest.importorskip("torch")
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")

    for device in devices:
        like = torch.empty((), dtype=getattr(torch, dtype_name), device=device)
        rng = ar.do("random.default_rng", 42, like=like)
        x = getattr(rng, fn)(size=(2, 3))
        assert x.dtype == like.dtype
        assert x.device == like.device


@pytest.mark.parametrize("fn", ["normal", "random", "uniform"])
def test_torch_default_rng_explicit_dtype_overrides_like(fn):
    torch = pytest.importorskip("torch")
    like = torch.empty((), dtype=torch.float64)
    rng = ar.do("random.default_rng", 42, like=like)
    x = getattr(rng, fn)(size=(2, 3), dtype=torch.float32)
    assert x.dtype == torch.float32


@pytest.mark.parametrize("dtype_name", ["bool", "int64", "complex64"])
def test_torch_default_rng_ignores_non_float_dtype(dtype_name):
    torch = pytest.importorskip("torch")
    like = torch.empty((), dtype=getattr(torch, dtype_name))
    rng = ar.do("random.default_rng", 42, like=like)
    x = rng.normal(size=(2, 3))
    assert x.dtype == torch.get_default_dtype()


def test_torch_default_rng_integer_distribution_ignores_float_dtype():
    torch = pytest.importorskip("torch")
    like = torch.empty((), dtype=torch.float64)
    rng = ar.do("random.default_rng", 42, like=like)
    x = rng.integers(0, 10, size=(2, 3))
    assert x.dtype == torch.int64


@pytest.mark.parametrize(
    "fn,args,kwargs",
    [
        ("choice", (10,), {"size": (2, 3)}),
        ("integers", (10,), {"size": (2, 3)}),
        ("normal", (), {"size": (2, 3)}),
        ("permutation", (10,), {}),
        ("random", (), {"size": (2, 3)}),
        ("uniform", (), {"size": (2, 3)}),
    ],
)
def test_torch_default_rng_generates_on_cuda(fn, args, kwargs):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    like = torch.empty((), device="cuda")
    rng = ar.do("random.default_rng", 42, like=like)
    x = getattr(rng, fn)(*args, **kwargs)
    assert x.device == like.device


def test_torch_default_rng_uses_generator_device_by_default():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    default_device = torch.get_default_device()
    try:
        torch.set_default_device("cuda")
        rng = ar.do("random.default_rng", 42, like="torch")
        x = rng.normal(size=(2, 3))
        assert x.device == rng._generator.device
    finally:
        torch.set_default_device(default_device)


def test_jax_jit_random():
    pytest.importorskip("jax")

    @ar.autojit(backend="jax")
    def f(seed):
        rng = ar.do("random.default_rng", seed)
        return rng.normal(size=(3, 4))

    x1 = ar.do("to_numpy", f(ar.do("array", 42)))
    x2 = ar.do("to_numpy", f(ar.do("array", 42)))
    assert ar.do("allclose", x1, x2)
    x3 = ar.do("to_numpy", f(ar.do("array", 43)))
    assert not ar.do("allclose", x1, x3)


class TestRandomArray:
    @pytest.mark.parametrize("device", ["inherited", "cpu", "cuda"])
    @pytest.mark.parametrize("dist", ["normal", "uniform", "rademacher"])
    @pytest.mark.parametrize(
        "backend,dtype,fn",
        gen_params(
            backends=_RANDOM_BACKENDS,
            dtypes=...,
            fns=["random.array"],
        ),
    )
    def test_backend_dtype_device(self, backend, dtype, fn, dist, device):
        if device == "cuda":
            if backend == "torch":
                import torch

                if not torch.cuda.is_available():
                    pytest.skip("CUDA is not available")
            elif backend != "cupy":
                pytest.skip(f"{backend} CUDA is not tested")
        elif (device == "cpu") and backend in ("cupy", "dask", "mlx"):
            pytest.skip(f"{backend} does not report a cpu device")

        like = gen_rand((), backend, dtype)
        _, like_device, _ = ar.infer_backend_device_dtype(like)
        kwargs = {} if device == "inherited" else {"device": device}

        x, y = (
            ar.do(fn, (3, 4), dist=dist, rng=42, like=like, **kwargs)
            for _ in range(2)
        )

        assert ar.do("shape", x) == (3, 4)
        assert ar.get_dtype_name(x) == dtype
        assert ar.do("allclose", x, y)
        _, actual_device, _ = ar.infer_backend_device_dtype(x)

        if device == "inherited":
            assert actual_device == like_device
        elif backend == "torch":
            assert actual_device.type == device
        elif backend == "jax":
            assert actual_device.platform == device
        elif backend == "tensorflow":
            # a full device string like '/job:localhost/.../device:CPU:0'
            assert actual_device.lower().endswith(f"{device}:0")
        elif backend == "cupy":
            assert actual_device == like_device
        else:
            assert actual_device == device

        if "complex" in dtype:
            assert ar.do("std", ar.real(x)) > 0.0
            assert ar.do("std", ar.imag(x)) > 0.0

    @pytest.mark.parametrize(
        "backend,dtype,fn",
        gen_params(
            backends=_RANDOM_BACKENDS,
            dtypes=["complex64", "complex128"],
            fns=["random.array"],
        ),
    )
    def test_complex_normal_has_unit_total_variance(self, backend, dtype, fn):
        x = ar.do(
            fn,
            (100_000,),
            dist="normal",
            dtype=dtype,
            rng=42,
            like=backend,
        )
        assert ar.get_dtype_name(x) == dtype
        x = ar.do("to_numpy", x)
        assert (abs(x) ** 2).mean() == pytest.approx(1.0, abs=0.03)

    @pytest.mark.parametrize("dist", ["normal", "uniform", "rademacher"])
    @pytest.mark.parametrize(
        "backend,fn",
        gen_params(
            backends=_RANDOM_BACKENDS,
            fns=["random.array"],
        ),
    )
    def test_loc_scale(self, backend, fn, dist):
        x = ar.do(
            fn,
            (3, 4),
            dist=dist,
            dtype="float32",
            rng=42,
            like=backend,
        )
        y = ar.do(
            fn,
            (3, 4),
            dist=dist,
            loc=2.0,
            scale=3.0,
            dtype="float32",
            rng=42,
            like=backend,
        )
        assert ar.do("to_numpy", y) == pytest.approx(
            2.0 + 3.0 * ar.do("to_numpy", x)
        )

    # rng=None takes the generic path, an rng the per backend fast one
    @pytest.mark.parametrize("rng", [42, None])
    @pytest.mark.parametrize(
        "backend,dtype,fn",
        gen_params(
            backends=_RANDOM_BACKENDS,
            dtypes=...,
            fns=["random.array"],
        ),
    )
    def test_rademacher_values(self, backend, dtype, fn, rng):
        n = 20_000
        x = ar.do(
            fn, (n,), dist="rademacher", dtype=dtype, rng=rng, like=backend
        )
        assert ar.get_dtype_name(x) == dtype
        x = ar.do("to_numpy", x)
        assert abs(x) == pytest.approx(1.0)

        # both signs appear, and all four roots of unity for a complex dtype
        counts = Counter(x.tolist())
        assert len(counts) == (4 if "complex" in dtype else 2)
        # each is equally likely, well within 5 sigma at this size
        expected = n / len(counts)
        assert min(counts.values()) > expected * 0.9
        assert max(counts.values()) < expected * 1.1

    def test_unknown_distribution(self):
        with pytest.raises(ValueError, match="Unknown distribution 'cauchy'"):
            ar.do("random.array", (2, 3), dist="cauchy", like="numpy")

    def test_numpy_generator_selects_backend(self):
        numpy = pytest.importorskip("numpy")
        rng = numpy.random.default_rng(42)
        x = ar.do("random.array", (2, 3), rng=rng)
        assert isinstance(x, numpy.ndarray)
        assert x.dtype == numpy.float64

        x = ar.do("random.array", (2, 3), dtype="complex64")
        assert isinstance(x, numpy.ndarray)
        assert x.dtype == numpy.complex64

    @pytest.mark.parametrize(
        "dist,method",
        [("normal", "standard_normal"), ("uniform", "random")],
    )
    def test_numpy_uses_dtype_aware_primitive(self, dist, method):
        numpy = pytest.importorskip("numpy")
        rng = numpy.random.default_rng(42)
        x = ar.do(
            "random.array",
            (3, 4),
            dist=dist,
            dtype="float32",
            rng=rng,
        )

        rng = numpy.random.default_rng(42)
        y = getattr(rng, method)((3, 4), dtype=numpy.float32)
        assert x == pytest.approx(y)

    @pytest.mark.parametrize("dist", ["normal", "uniform", "rademacher"])
    @pytest.mark.parametrize(
        "backend,dtype,fn",
        gen_params(
            backends=_RANDOM_BACKENDS,
            dtypes=...,
            fns=["random.array"],
            requires="random.seed",
        ),
    )
    def test_none_uses_global_rng(self, backend, dtype, fn, dist):
        ar.do("random.seed", 42, like=backend)
        x = ar.do(fn, (3, 4), dist=dist, dtype=dtype, like=backend)
        ar.do("random.seed", 42, like=backend)
        y = ar.do(fn, (3, 4), dist=dist, dtype=dtype, like=backend)
        assert ar.get_dtype_name(x) == dtype
        assert ar.do("allclose", x, y)
        if "complex" in dtype:
            assert ar.do("std", ar.real(x)) > 0.0
            assert ar.do("std", ar.imag(x)) > 0.0

    def test_namespace_inherits_dtype(self):
        numpy = pytest.importorskip("numpy")
        xp = ar.get_namespace(numpy.empty((), dtype=numpy.float32))
        x = xp.random.array((2, 3), rng=42)
        assert x.dtype == numpy.float32

    def test_jax_key_selects_backend(self):
        jax = pytest.importorskip("jax")
        key = jax.random.key(42)
        x = ar.do("random.array", (2, 3), dtype="float32", rng=key)
        assert ar.infer_backend(x) == "jax"

    def test_torch_generator_selects_backend(self):
        torch = pytest.importorskip("torch")
        rng = torch.Generator().manual_seed(42)
        x = ar.do("random.array", (2, 3), dtype="float32", rng=rng)
        assert ar.infer_backend(x) == "torch"

    def test_jax_seedless_default_rng_warns(self):
        pytest.importorskip("jax")
        # the warning is emitted one time only, so reset it first
        ar.autoray._warn_jax_generated_seed.cache_clear()
        with pytest.warns(RuntimeWarning, match="reuses the same seed"):
            ar.do("random.default_rng", None, like="jax")

    def test_jax_jit_none_rng_is_frozen(self):
        jax = pytest.importorskip("jax")

        @jax.jit
        def f():
            return ar.do(
                "random.array",
                (2, 3),
                dtype="float32",
                rng=None,
                like="jax",
            )

        try:
            x1 = f()
            x2 = f()
            assert ar.do("allclose", x1, x2)

            # tracing leaves a tracer in the shared generator, so this fails
            with pytest.raises(jax.errors.UnexpectedTracerError):
                ar.do("random.array", (2, 3), dtype="float32", like="jax")
        finally:
            ar.do("random.seed", 42, like="jax")

    def test_jax_jit_captured_generator_is_frozen(self):
        jax = pytest.importorskip("jax")
        rng = ar.do("random.default_rng", 42, like="jax")

        @jax.jit
        def f():
            return ar.do("random.array", (2, 3), dtype="float32", rng=rng)

        # each call hits the trace cache, so no error, just the same values
        x1 = f()
        x2 = f()
        assert ar.do("allclose", x1, x2)

        # tracing leaves a tracer in the captured generator
        with pytest.raises(jax.errors.UnexpectedTracerError):
            rng.normal(size=(2, 3))

    def test_jax_jit(self):
        jax = pytest.importorskip("jax")
        jnp = pytest.importorskip("jax.numpy")

        @jax.jit
        def f(seed, loc, scale):
            rng = ar.do("random.default_rng", seed, like="jax")
            return (
                ar.do(
                    "random.array",
                    (3, 4),
                    loc=loc,
                    scale=scale,
                    dtype="float32",
                    rng=rng,
                    like="jax",
                ),
                ar.do(
                    "random.array",
                    (3, 4),
                    loc=loc,
                    scale=scale,
                    dtype="float32",
                    rng=rng,
                    like="jax",
                ),
            )

        x1, y1 = f(jnp.asarray(42), jnp.asarray(2.0), jnp.asarray(3.0))
        x2, y2 = f(jnp.asarray(42), jnp.asarray(2.0), jnp.asarray(3.0))
        x3, _ = f(jnp.asarray(43), jnp.asarray(2.0), jnp.asarray(3.0))
        assert ar.do("allclose", x1, x2)
        assert ar.do("allclose", y1, y2)
        assert not ar.do("allclose", x1, y1)
        assert not ar.do("allclose", x1, x3)

    def test_torch_compile(self):
        torch = pytest.importorskip("torch")
        if not hasattr(torch, "compile"):
            pytest.skip("torch.compile is unavailable")

        # warm autoray's function cache, dynamo cannot trace the lazy import
        ar.do("random.array", (1,), dtype="float32", rng=None, like="torch")

        @torch.compile(backend="eager", fullgraph=True)
        def f():
            return ar.do(
                "random.array",
                (3, 4),
                dtype="float32",
                rng=None,
                like="torch",
            )

        x = f()
        y = f()
        assert not torch.allclose(x, y)

    def test_torch_none_uses_global_rng(self):
        torch = pytest.importorskip("torch")
        torch.manual_seed(42)
        x = ar.do("random.array", (3, 4), dtype="float32", like="torch")
        torch.manual_seed(42)
        y = ar.do("random.array", (3, 4), dtype="float32", like="torch")
        assert x.dtype == torch.float32
        assert torch.allclose(x, y)

    def test_mlx_global_rng_is_the_native_one(self):
        mx = pytest.importorskip("mlx.core")

        # mlx's own seed reaches autoray
        mx.random.seed(42)
        x = ar.do("random.array", (3, 4), dtype="float32", like="mlx")
        mx.random.seed(42)
        y = ar.do("random.array", (3, 4), dtype="float32", like="mlx")
        assert ar.do("allclose", x, y)

        # and autoray's seed reaches mlx
        ar.do("random.seed", 42, like="mlx")
        a = mx.random.normal((3, 4))
        ar.do("random.seed", 42, like="mlx")
        b = mx.random.normal((3, 4))
        assert ar.do("allclose", a, b)
