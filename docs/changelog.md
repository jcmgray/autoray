# Changelog

Release notes for `autoray`.

## v0.9.0 (unreleased)

**Enhancements:**

- Added [`to`](autoray.to) for converting arrays, or nested collections ("pytrees") of arrays, to a target backend, dtype and/or device, all specifiable in a single string such as `"torch-float32-cuda:0"`, via explicit kwargs, or an example array. Matching `torch.nn.Module.to` semantics, only floating point and complex arrays are cast when a dtype is given, so e.g. integer index arrays are preserved.
- Added [`to_device`](autoray.to_device) composed function for moving arrays between devices, with `"gpu"` accepted as an alias for `"cuda"` where relevant, and a bare device type such as `"gpu"` meaning 'ensure on this type of device', without migrating arrays between device indices.
- Added [`from_numpy`](autoray.from_numpy) composed creation routine for converting a numpy array into a backend array, directly with a given dtype and on a given device where possible, e.g. via a single `torch.as_tensor` call. An example array supplied as `like` supplies its backend, dtype and device as defaults.
- Made [`to_numpy`](autoray.to_numpy) a composed function with default implementation `np.asarray`, so that unknown backends are handled automatically.
- Reworked [`register_function`](autoray.register_function) into the single entry point for all function-level registration, taking `module=`, `alias=`, `wrapper=`, `inject_dtype=` and `inject_device=`, deprecating `register_creation_routine`.
- Dtype name resolution now handles scalar types such as `np.float32` and builtins such as `float` and `complex`, anywhere a dtype is specified.
- MLX: `array` and `asarray` now preserve the input array's dtype rather than applying mlx defaults, e.g. no longer silently downcasting float64 to float32.
- MLX: requesting a device warns that mlx arrays live in unified memory with per-op computation placement, rather than failing generically.
- Lazy: [`to_numpy`](autoray.to_numpy) and [`to_device`](autoray.to_device) on `LazyArray`s now raise explicit errors, while [`from_numpy`](autoray.from_numpy) creates a lazy leaf node.
- Added the `testcupy` pixi environment, and refreshed the contributing and developer guides, including a code of conduct and AI policy.

**Bug Fixes:**

- Fixed [`get_namespace`](autoray.get_namespace) caching for backends with unhashable device objects such as `cupy.cuda.Device`.
- Added cupy `linalg.cholesky` support for the `upper` kwarg via the new generic [`cholesky_manual_upper`](autoray.autoray.cholesky_manual_upper) wrapper.
- Torch: `random.default_rng` now only injects floating point dtypes from the `like` argument.

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
