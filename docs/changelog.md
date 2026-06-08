# Changelog

Release notes for `autoray`.

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
