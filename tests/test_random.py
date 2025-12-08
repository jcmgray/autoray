import pytest

import autoray as ar

from .test_autoray import BACKENDS


@pytest.mark.parametrize(
    "backend",
    [
        b
        for b in BACKENDS
        if any(
            b.values[0] == other
            for other in (
                "cupy",
                "dask",
                "jax",
                "numpy",
                "tensorflow",
                "torch",
            )
        )
    ],
)
@pytest.mark.parametrize(
    "dist,args,kwargs",
    [
        ("binomial", (7, 0.424), {"size": (3, 4)}),
        ("choice", ([11.1 * i for i in range(100)],), {"size": (3, 4)}),
        ("choice", ([11.1 * i for i in range(1000)],), {}),
        ("exponential", (), {"size": (3, 4)}),
        ("exponential", (), {}),
        ("gumbel", (), {"size": (3, 4)}),
        ("gumbel", (), {}),
        ("integers", (100, 1000), {"size": (3, 4)}),
        ("integers", (100, 1000), {}),
        ("normal", (), {"size": (3, 4)}),
        ("normal", (), {}),
        ("permutation", ([11.1 * i for i in range(100)],), {}),
        ("poisson", (100,), {"size": (3, 4, 5)}),
        ("random", (), {"size": (3, 4)}),
        ("random", (), {}),
        ("uniform", (), {"size": (3, 4)}),
        ("uniform", (), {}),
    ],
)
def test_random_default_rng(backend, dist, args, kwargs):
    if dist in ("choice", "permutation"):
        args = (ar.do("array", args[0], like=backend), *args[1:])

    if dist == "permutation" and backend == "dask":
        pytest.xfail("bug: https://github.com/dask/dask/issues/12029")

    if backend == "torch" and dist in (
        "binomial",
        "exponential",
        "gumbel",
        "poisson",
    ):
        pytest.xfail(f"torch: no {dist} interface yet.")

    if backend == "cupy" and dist in (
        "choice",
        "gumbel",
        "normal",
        "permutation",
    ):
        pytest.xfail(f"torch: no {dist} interface yet.")

    if backend == "tensorflow" and dist in (
        "binomial",
        "choice",
        "exponential",
        "gumbel",
        "permutation",
        "poisson",
    ):
        pytest.xfail(f"tensorflow: no {dist} interface yet.")

    seed = 42
    seed2 = 43

    rng = ar.do("random.default_rng", seed, like=backend)
    x = ar.do("to_numpy", getattr(rng, dist)(*args, **kwargs))
    if "size" in kwargs:
        assert ar.do("shape", x) == kwargs["size"]
    y = ar.do("to_numpy", getattr(rng, dist)(*args, **kwargs))
    assert not ar.do("allclose", x, y)
    rng = ar.do("random.default_rng", seed2, like=backend)
    z = ar.do("to_numpy", getattr(rng, dist)(*args, **kwargs))
    assert not ar.do("allclose", x, z)
    rng = ar.do("random.default_rng", seed, like=backend)
    x2 = ar.do("to_numpy", getattr(rng, dist)(*args, **kwargs))
    assert ar.do("allclose", x, x2)


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
