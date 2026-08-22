# Changelog

Release notes for `autoray`.

## v0.11.0 (unreleased)

**Enhancements:**

- `compose` now supplies an [`AutoNamespace`](autoray.autoray.AutoNamespace) to any *default* implementation taking a `namespace` parameter: the namespace itself when called as `xp.my_func(...)`, so its dtype and device defaults apply, otherwise one from the first argument if it matches the dispatched backend. Implementations registered for a specific backend keep their own signatures.
- Namespaces from `get_namespace` are now stable objects: registering a function drops their cached lookups rather than discarding the namespaces, so one already held picks up the new function.
- `autoray.lazy` `"sum"`, `"prod"`, `"min"` and `"max"` now accept `keepdims`, matching the eager and array API signatures. The reduced axes are kept as size 1 in the inferred lazy shape.
- Added the lazy reductions `"mean"`, `"std"`, `"var"`, `"all"`, `"any"`, `"count_nonzero"`, `"argmin"` and `"argmax"`, which also take `axis` and `keepdims`, and `"cumsum"`, which takes `axis` and accumulates over the flattened array when it is `None`. `"argmin"` and `"argmax"` follow the eager convention of only accepting a scalar `axis`. Backend support for `keepdims` varies: see `XFAILS` in `tests/conftest.py`.
- The lazy reductions and `"cumsum"` pass any further keyword arguments, such as `ddof` or `dtype`, straight through to the backend function. A `LazyArray` supplied this way, for example as `where`, is tracked as a dependency of the result.

**Bug Fixes:**

- `autoray.lazy` reductions now pass `axis` on as supplied, rather than always normalizing it to a tuple. `do("prod", x, axis=0)` on a lazy `torch` array therefore works, since `torch.prod` only accepts a scalar `dim`.


## v0.10.1 (2026-08-07)

This is a re-release of v0.10.1, see below.


## v0.10.0 (2026-08-06)

**Enhancements:**

- Added [`random_array`](autoray.autoray.random_array), i.e. `do("random.array", ...)`, for backend-agnostic normal, uniform and rademacher random arrays. It accepts backend-specific generators or seeds, inherits dtype and device from `like`, and generates complex normal samples with unit total variance. A complex rademacher sample is a choice from the four roots of unity, so it also has modulus one. `jax` `"random.default_rng"` with `seed=None` warns one time, because tracing fixes its generated seed.
- `jax`, `mlx` and `tensorflow` `"random.default_rng"` now default to `seed=None`, and `jax` also accepts a key. `torch` `"random.default_rng"` accepts a `torch.Generator`, and raises a descriptive error if a device is also requested that the generator cannot supply.
- Added a [random numbers](random.md) documentation guide, covering [`random_array`](autoray.autoray.random_array), the generator objects and their per backend method coverage, the shared random state, the complex normal convention, and the rules for `jax.jit` and `torch.compile`.

**Bug Fixes:**

- `torch` `"random.normal"` and `"random.uniform"` now generate at the requested `dtype` rather than casting afterwards. A complex `dtype` therefore gets a complex sample rather than a real one cast up, which previously left the imaginary part zero, and `float64` now carries full double precision entropy.
- `torch` `"random.seed"` is now registered, and seeds the shared random state. Previously it resolved to `torch.random.seed`, which takes no seed and thus raised `TypeError`.
- `tensorflow` `"random.seed"` is now registered, as an alias of `tf.random.set_seed`. Previously it raised `ImportError`.
- `mlx` now uses mlx's own shared random state, rather than a separate module level generator of autoray's. `do("random.seed", seed, like="mlx")` therefore reaches `mx.random.seed`, and mixing autoray and raw `mlx.core.random` calls is now reproducible in both directions.


## v0.9.1 (2026-08-03)

**Bug Fixes:**

- Registration now removes the cache entries that depend on it. A registration applies immediately, also to functions and classes that autoray used before. Previously, [`register_function`](autoray.register_function) with `module=`, `alias=` or `wrapper=` had no effect if the code called the function first, because only the first import of a function reads these three arguments.
- [`register_backend`](autoray.register_backend) and [`register_backend_alias`](autoray.autoray.register_backend_alias) now clear the caches that hold the backend of each class. Autoray then finds the new backend, also for a class that it used before.
- [`tree_register_container`](autoray.autoray.tree_register_container) now clears the pytree dispatch caches. Autoray then uses the new functions for a container class, also if it used that class as a leaf before, or if it used the functions of a parent class.
- Registration also clears the namespaces from [`get_namespace`](autoray.get_namespace) and the dtype cache of [`to_backend_dtype`](autoray.to_backend_dtype). A namespace object that you already have keeps the functions that it found. This is the correct behavior: get a new namespace to use a later registration.
- `autoray` no longer shows the internal exception chain in the `ImportError` that it raises when it cannot find a function.
- `torch` `"random.default_rng"` now defaults to `seed=None`


## v0.9.0 (2026-07-15)

**Enhancements:**

- Added [`to`](autoray.to) for converting arrays, or nested collections ("pytrees") of arrays, to a target backend, dtype and/or device, all specifiable in a single string such as `"torch-float32-cuda:0"`, via explicit kwargs, or an example array. Repeated references to the same input array are converted once and share the same output array. Matching `torch.nn.Module.to` semantics, only floating point and complex arrays are cast when a dtype is given, so e.g. integer index arrays are preserved.
- Added [`to_device`](autoray.to_device) composed function for moving arrays between devices, with `"gpu"` accepted as an alias for `"cuda"` where relevant, and a bare device type such as `"gpu"` meaning 'ensure on this type of device', without migrating arrays between device indices.
- Added [`from_numpy`](autoray.from_numpy) composed creation routine for converting a numpy array into a backend array, directly with a given dtype and on a given device where possible, e.g. via a single `torch.as_tensor` call. An example array supplied as `like` supplies its backend, dtype and device as defaults.
- Made [`to_numpy`](autoray.to_numpy) a composed function with default implementation `np.asarray`, so that unknown backends are handled automatically.
- Reworked [`register_function`](autoray.register_function) into the single entry point for all function-level registration, taking `module=`, `alias=`, `wrapper=`, `inject_dtype=` and `inject_device=`, deprecating `register_creation_routine`.
- Dtype name resolution now handles scalar types such as `np.float32` and builtins such as `float` and `complex`, anywhere a dtype is specified.
- MLX: `array` and `asarray` now preserve the input array's dtype rather than applying mlx defaults, e.g. no longer silently downcasting float64 to float32.
- MLX: `count_nonzero` now uses the native mlx implementation where available (v0.32+), falling back to a manual version which also supports the `axis` and `keepdims` kwargs.
- The `"backend[alt]"` fallback, used when a function is missing from an older version of a library, now also finds directly registered implementations, and caches its result against the original backend so the lookup only happens once.
- MLX: requesting a device warns that mlx arrays live in unified memory with per-op computation placement, rather than failing generically.
- Lazy: [`to_numpy`](autoray.to_numpy) and [`to_device`](autoray.to_device) on `LazyArray`s now raise explicit errors, while [`from_numpy`](autoray.from_numpy) creates a lazy leaf node.
- Added the `testcupy` pixi environment, and refreshed the contributing and developer guides, including a code of conduct and AI policy.

**Bug Fixes:**

- Fixed [`get_namespace`](autoray.get_namespace) caching for backends with unhashable device objects such as `cupy.cuda.Device`.
- Added cupy `linalg.cholesky` support for the `upper` kwarg via the new generic [`cholesky_manual_upper`](autoray.autoray.cholesky_manual_upper) wrapper.
- Torch: `random.default_rng` now only inherits floating point dtypes from the `like` argument and creates generated arrays on the same device as its generator.


## v0.8.11 (2026-06-08)

**Enhancements:**

- Added initial MLX backend support including `random.default_rng`, dtype-aware creation wrappers, [`to_numpy`](autoray.to_numpy), `count_nonzero`, `ravel`, and `linalg.svd` support.
- Added [`infer_backend_device_dtype`](autoray.infer_backend_device_dtype) for shared backend, device, and dtype inference from `like` values.
- Exported [`DoFunc`](autoray.DoFunc) for reusable call-time auto-dispatch, with faster [`do`](autoray.do) and [`get_namespace`](autoray.get_namespace) dispatch paths.
- Refreshed package metadata, pixi environments, docs infrastructure, and CI, including the `testmlx` environment and MLX CI job.
- Reworked backend tests around the unified `XFAILS` registry and expanded linear algebra coverage.

**Bug Fixes:**

- Cached failed `.device` and `.dtype` probes in [`get_namespace`](autoray.get_namespace) to avoid repeated slow exceptions for array classes such as JAX tracers ({pull}`30`).
- Fixed `autojit` with lazy [`lazy_astype`](autoray.lazy.core.lazy_astype) dtype evaluation.
- Added TensorFlow [`astype`](autoray.autoray.tensorflow_astype) handling and a NumPy-like Torch [`nonzero`](autoray.autoray.torch_nonzero_wrap) wrapper.
- Fixed reduced and batched SVD shape handling in [`svd_manual_full_matrices_kwarg`](autoray.autoray.svd_manual_full_matrices_kwarg).
- Fixed [`prime_factors`](autoray.experimental.complexity_tracing.prime_factors) to preserve integer factors.

---

Previous release notes can be found on the project releases page.
