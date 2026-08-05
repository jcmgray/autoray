# Random numbers

Random number generation is one of the areas where array backends diverge most:
the function names differ, the state handling differs, and not every library
has a shared global state at all. [`autoray`](autoray) provides a unified
interface at three different levels:

* [`do("random.array", ...)`](autoray.autoray.random_array) - a single
  *unified* backend agnostic creation routine, recommended for most use cases.
  It supports `dtype` and `device` matching, distribution as a parameter,
  consistent handling of complex dtypes, `loc` and `scale`, and both implicit
  and explicit random state.

* `do("random.default_rng", seed, like=...)` - an *explicit* generator object
  with a `numpy.random.Generator`-like API, supporting many distributions.

* `do("random.normal", ...)` and friends - the backend's *shared* random
  state, seeded with `do("random.seed", seed, like=...)`.


## Generating an array

[`"random.array"`](autoray.autoray.random_array) is the recommended entry
point. It is a creation routine, so it takes the shape first and picks up
`dtype` and `device` from `like`:

```python
import autoray as ar

x = ar.do("random.array", (2, 3), like="torch")

# match the backend, dtype and device of an existing array
y = ar.do("random.array", ar.shape(x), like=x)
```

The full signature is:

```python
ar.do(
    "random.array",
    shape,
    dist="normal",
    loc=0.0,
    scale=1.0,
    dtype=None,
    device=None,
    rng=None,
    like=None,
)
```

The sampled value `z` is set by `dist`:

* `dist="normal"` draws a normal sample with mean zero and variance one.
* `dist="uniform"` draws a sample from `[0, 1)`.

If `loc` and `scale` are provided, the final values are transformed as
`scale * z + loc`. Either can be an array, which broadcasts against the
generated samples.

```{note}
`loc` and `scale` shift and scale the *whole* sample, so for `"uniform"` they
describe an interval of width `scale` starting at `loc`, rather than the
`low` and `high` pair that
[`numpy.random.Generator.uniform`](numpy.random.Generator.uniform) takes.
```

Unlike a raw `do("random.normal", ...)` call, `"random.array"` aims to mean the
same thing across backends: the requested `dtype` is the dtype sampled at,
complex dtypes are handled explicitly, and the result is generated on the
requested device.


### Complex dtypes

A complex `dtype` is supported directly, and follows the standard convention
for a *complex normal* `z`, that is,

$$
\mathbb{E}[z] = 0, \quad \mathbb{E}[|z|^2] = 1,
$$

with the real and imaginary parts each having variance $1/2$. This makes
`scale` mean the same thing for real and complex dtypes, and makes
$\mathbb{E}[z z^\dagger] = I$, as required for e.g. Hutchinson trace
estimation:

```python
z = ar.do("random.array", (10000,), dtype="complex128", rng=42, like="numpy")

ar.do("mean", ar.do("abs", z) ** 2)
# 1.00945...
```

It is the same convention that `jax` and `torch` use for their own complex
normals, which `"random.array"` defers to directly. Elsewhere it draws the two
real parts and scales them by $1/\sqrt{2}$.

````{warning}
The raw `do("random.normal", ...)` call gives no such guarantee. On a backend
whose generator has no complex support, a complex `dtype` is applied by casting
a *real* sample, silently leaving the imaginary part zero:

```python
ar.do("random.normal", size=(3,), dtype="complex128", like="numpy")
# array([-0.15591091+0.j, -0.36737385+0.j,  0.55108074+0.j])
```
````

A complex `"uniform"` sample fills the complex unit square, i.e. the real and
imaginary parts are each drawn from `[0, 1)`.


## Controlling the state with `rng`

The `rng` argument selects between shared and explicit state, and accepts the
following forms:

| `rng` | state | behaviour |
| :--- | :--- | :--- |
| `None` (default) | shared | use the backend's global random state |
| `int` | neither | make a fresh generator seeded with this, for this call |
| a generator | explicit | use and advance that generator's own state |
| a backend seed object | explicit | whatever that backend's `default_rng` takes |

An integer gives a reproducible array without touching any global state:

```python
xa = ar.do("random.array", (2, 3), rng=42, like="numpy")
xb = ar.do("random.array", (2, 3), rng=42, like="numpy")
ar.do("allclose", xa, xb)
# True
```

A generator carries state between calls, so successive draws differ, and it
also supplies the backend, meaning `like` is not needed:

```python
rng = ar.do("random.default_rng", 42, like="torch")

x = ar.do("random.array", (2, 3), rng=rng)
y = ar.do("random.array", (2, 3), rng=rng)  # different values to x
```

The last row of the table means each backend additionally accepts its own
native seed and state objects, since `rng` is passed through to that backend's
`"random.default_rng"`. For instance `numpy` accepts a `SeedSequence` or a
`BitGenerator`, `jax` accepts a key, and `torch` accepts a `torch.Generator`:

```python
key = jax.random.key(42)
x = ar.do("random.array", (2, 3), rng=key, like="jax")
```

```{note}
A `torch.Generator` is bound to the device it was made on, and torch requires
the generator and the output to share a device. Requesting a `device` that the
supplied generator cannot serve raises a descriptive `ValueError` rather than
a torch internal error.
```


## Generator objects

`do("random.default_rng", seed, like=...)` returns an object holding explicit
state, with the [`numpy.random.Generator`](numpy.random.Generator) API, for
`numpy`, `cupy`, `dask`, `jax`, `mlx`, `tensorflow` and `torch`. For the
libraries that do not natively provide one, `autoray` supplies a shim class
implementing the same methods on top of that backend's primitives, so a `jax`
generator for instance holds and splits a key behind the same interface.

```python
rng = ar.do("random.default_rng", 42, like="jax")

rng.normal(size=(2, 3))
rng.integers(0, 10, size=(2, 3))
rng.choice(ar.do("arange", 10, like=rng), size=5)
```

In each case `seed` defaults to `None`, meaning non-deterministic
initialization. The generator itself is a registered `autoray` backend object,
so it is also accepted anywhere a `like` argument is.

Method coverage depends on what the underlying library exposes:

| method | numpy | cupy | dask | jax | mlx | tensorflow | torch |
| :--- | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| `uniform` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `random` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `integers` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `normal` | ✓ | | ✓ | ✓ | ✓ | ✓ | ✓ |
| `binomial` | ✓ | ✓ | ✓ | ✓ | | | |
| `choice` | ✓ | | ✓ | ✓ | | | ✓ |
| `exponential` | ✓ | ✓ | ✓ | ✓ | | | |
| `gumbel` | ✓ | | ✓ | ✓ | ✓ | | |
| `permutation` | ✓ | | ✓ | ✓ | ✓ | | ✓ |
| `poisson` | ✓ | ✓ | ✓ | ✓ | | | |

A missing entry raises rather than silently doing something else, and the gaps
are tracked in `tests/conftest.py` so that they close automatically as the
backends grow the relevant interfaces.

```{hint}
The gaps do not carry over to `"random.array"`, which works for all the
backends listed. `cupy` for example has no `normal` method, but does have the
equivalent `standard_normal`, which `"random.array"` uses instead.
```


## Shared random state

With `rng=None`, `"random.array"` uses the backend's shared state, which is
seeded through `do("random.seed", seed, like=...)`:

```python
ar.do("random.seed", 42, like="torch")
x = ar.do("random.array", (2, 3), like="torch")
```

This is registered function for the backends above, e.g. for `torch` it
forwards to `torch.manual_seed`, since `torch.random.seed` takes no seed at
all, and for `tensorflow` to `tf.random.set_seed`.

`jax` is the exception, having no shared state of its own, since every
`jax.random` call requires a key. For it `autoray` keeps a module level
generator and uses that for `rng=None`, so seedless code still works and
`do("random.seed", ..., like="jax")` still makes it reproducible.


## Compiled functions

Random state and tracing interact awkwardly, in opposite ways for the two main
compilers.

For `jax.jit`, pass a seed or key *into* the function and build the generator
inside it:

```python
@jax.jit
def sample(seed):
    rng = ar.do("random.default_rng", seed, like="jax")
    return ar.do("random.array", (2, 3), rng=rng, like="jax")
```

```{warning}
Capturing a generator made outside the function, or using `rng=None`, fails
quietly rather than raising: the key is frozen at trace time, so each call hits
the trace cache and returns the *same* values. The error comes later, since the
generator is left holding a tracer, and both eager use of it and any retrace of
the compiled function then raise `UnexpectedTracerError`. For `rng=None` the
affected generator is autoray's module level one, which
`ar.do("random.seed", seed, like="jax")` restores.
```

For `torch.compile` it is the other way round: use `rng=None`, since the
compiler handles torch's shared random state itself, whereas an explicit
`torch.Generator` causes a graph break and so is incompatible with
`fullgraph=True`.

```python
@torch.compile(fullgraph=True)
def sample():
    return ar.do("random.array", (2, 3), like="torch")


# call once eagerly first, so that dynamo does not have to trace the import
ar.do("random.array", (2, 3), like="torch")

sample()
```

```{note}
The warm up call matters: `autoray` imports and caches a backend function the
first time it is used, and dynamo cannot trace that import machinery. Without
it, the first compiled call fails with a `torch._dynamo.exc.Unsupported` error.
This is not specific to the random functions, but applies to the first use of
an `autoray` function inside a `fullgraph=True` region.
```
