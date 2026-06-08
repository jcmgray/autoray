# Contributing

Contributions to `autoray` in the form of
[pull requests](https://github.com/jcmgray/autoray/pulls) are very welcome.
Opening an [issue](https://github.com/jcmgray/autoray/issues) first can be
useful for larger changes, design questions, or work that might affect public
APIs.

If this is your first time contributing on GitHub, the following guide may be
useful:

- [GitHub - Creating a pull request](https://help.github.com/articles/creating-a-pull-request/)


## Development Setup

`autoray` uses [pixi](https://pixi.sh) for development environments and
predefined project tasks. The environments and commands are defined in
[`pyproject.toml`](../pyproject.toml).

From a fresh clone:

```bash
git clone https://github.com/jcmgray/autoray.git
cd autoray
pixi install
```


## Common Commands

Run the full test suite with coverage (matches CI):

```bash
pixi run -e testpynew test
```

Run a focused test via the `pytest` task (arguments after `--` are forwarded
to pytest):

```bash
pixi run pytest -- tests/test_autoray.py
pixi run pytest -- tests/test_autoray.py::test_basic -v
pixi run pytest -- tests/test_autoray.py::test_basic[numpy-sum] -v
pixi run pytest -- -k "test_mgs" -v
```

Run the matrix-style backend checks in a specific environment:

```bash
pixi run -e testpyold test
pixi run -e testpymid test
pixi run -e testpynew test
pixi run -e testjax test
pixi run -e testtorch test
pixi run -e testtensorflow test
pixi run -e testmlx test
```

Format and lint:

```bash
pixi run lint
pixi run format
pixi run format-all   # also runs `squeaky` on notebooks
```

Build and serve the docs:

```bash
pixi run docs
pixi run docs-clean
pixi run docs-serve
```

More developer details are in the
[development guide](https://autoray.readthedocs.io/en/latest/develop.html).


## Contribution Checklist

- [ ] Tests have been added for new functionality. If a backend cannot
      support the new behaviour, register the limitation in the `XFAILS`
      dict in `tests/conftest.py` rather than scattering `pytest.xfail()`
      calls across test files.
- [ ] Tests use the `gen_params()` helper from `tests/conftest.py` to
      parametrize over backends, dtypes, and function variants.
- [ ] Public functions have
      [NumPy-style docstrings](http://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html).
- [ ] New public API is exported from `autoray/__init__.py` and added to
      `"__all__"` if appropriate.
- [ ] New functionality is documented in `docs/` or demonstrated with an
      example notebook when appropriate.
- [ ] User-facing changes are noted in `docs/changelog.md`.
- [ ] Backend libraries (`numpy`, `torch`, `jax`, ...) are **not** imported
      at module level. Use lazy imports (and `@functools.cache` where
      appropriate) so the zero-dependency runtime contract is preserved.
- [ ] Formatting and lint checks pass with `pixi run lint`.
