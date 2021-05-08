import functools

from .autoray import infer_backend, do
from . import lazy


class CompilePython:
    """A simple compiler that unravels all autoray calls, optionally sharing
    intermediates and folding constants, converts this to a code object using
    ``compile``, then executes this using ``exec``.

    Parameters
    ----------
    fn : callable
        Function to compile - should have signature ``fn(*arrays)``, and
        perform array operations on these using ``autoray.do`` syntax.
    fold_constants : bool, optional
        Whether to fold all constant operations into the graph, which might
        increase memory usage.
    share_intermediates : bool, optional
        Whether to cache all computational nodes during the trace, so that any
        shared intermediate results can be identified.
    """

    def __init__(self, fn, fold_constants=True, share_intermediates=True):
        self._fn = fn
        self._fold_constants = fold_constants
        self._share_intermediates = share_intermediates
        self._fn_compiled = None

    def _trace(self, arrays):
        """Convert the example arrays to lazy variables and trace them through
        the function.
        """
        if self._share_intermediates:
            with lazy.shared_intermediates():
                variables = tuple(map(lazy.array, arrays))
                out = self._fn(*variables)
        else:
            variables = tuple(map(lazy.array, arrays))
            out = self._fn(*variables)

        return out, variables

    def _setup(self, arrays):
        """Based on example ``arrays``, compile the function.
        """
        out, variables = self._trace(arrays)
        self._fn_compiled = out.get_function(
            variables, fold_constants=self._fold_constants
        )

    def __call__(self, *arrays):
        """If necessary, build, then call the compiled function.
        """
        if self._fn_compiled is None:
            self._setup(arrays)
        return self._fn_compiled(arrays)


class CompileJax:
    """
    """

    def __init__(self, fn, enable_x64=None, platform_name=None, **kwargs):
        self.fn = fn
        self._enable_x64 = enable_x64
        self._platform_name = platform_name
        self._jit_fn = None
        self._jit_kwargs = kwargs

    def setup(self):
        import jax

        if self._enable_x64 is not None:
            from jax.config import config

            config.update("jax_enable_x64", self._enable_x64)

        if self._platform_name is not None:
            from jax.config import config

            config.update("jax_platform_name", self._platform_name)

        self._jit_fn = jax.jit(self.fn, **self._jit_kwargs)

    def __call__(self, *arrays):
        array_backend = infer_backend(arrays[0])
        if self._jit_fn is None:
            self.setup()
        out = self._jit_fn(*arrays)
        if array_backend != "jax":
            out = do("asarray", out, like=array_backend)
        return out


class CompileTensorFlow:
    """
    """

    def __init__(self, fn, **kwargs):
        self.fn = fn
        kwargs.setdefault("autograph", False)
        kwargs.setdefault("experimental_compile", False)
        self._jit_fn = None
        self._jit_kwargs = kwargs

    def setup(self):
        import tensorflow as tf

        self._jit_fn = tf.function(**self._jit_kwargs)(self.fn)

    def __call__(self, *arrays):
        array_backend = infer_backend(arrays[0])
        if self._jit_fn is None:
            self.setup()
        out = self._jit_fn(*arrays)
        if array_backend != "tensorflow":
            out = do("asarray", out, like=array_backend)
        return out


class CompileTorch:
    """
    """

    def __init__(self, fn, script=True, **kwargs):
        import torch

        self.torch = torch
        self.fn = fn
        self.script = script
        self._jit_fn = None
        self._jit_kwargs = kwargs

    def setup(self, arrays):
        self._jit_fn = self.torch.jit.trace(
            self.fn, arrays, **self._jit_kwargs
        )
        if self.script:
            self._jit_fn = self.torch.jit.script(self._jit_fn)

    def __call__(self, *arrays):
        array_backend = infer_backend(arrays[0])
        if array_backend != "torch":
            arrays = tuple(map(self.torch.as_tensor, arrays))
        if self._jit_fn is None:
            self.setup(arrays)
        out = self._jit_fn(*arrays)
        if array_backend != "torch":
            out = do("asarray", out, like=array_backend)
        return out


_backend_lookup = {
    "jax": "jax",
    "tensorflow": "tensorflow",
    "torch": "torch",
}


_compiler_lookup = {
    "python": CompilePython,
    "jax": CompileJax,
    "tensorflow": CompileTensorFlow,
    "torch": CompileTorch,
}


class AutoCompiled:
    """Just in time compile a ``autoray.do`` using function. See the main
    wrapper ``autocompile``.
    """

    def __init__(self, fn, backend=None, compiler_opts=None):
        self._fn = fn
        self._backend = backend
        self._compiled_fns = {}
        if compiler_opts is None:
            self._compiler_kwargs = {}
        else:
            self._compiler_kwargs = compiler_opts

    def __call__(self, *arrays, backend=None):
        if backend is None:
            if self._backend is None:
                backend = infer_backend(arrays[0])
            else:
                backend = self._backend
        backend = _backend_lookup.get(backend, 'python')

        try:
            fn_compiled = self._compiled_fns[backend]
        except KeyError:
            backend_compiler = _compiler_lookup.get(backend, CompilePython)
            kwargs = self._compiler_kwargs.get(backend, {})
            fn_compiled = backend_compiler(self._fn, **kwargs)
            self._compiled_fns[backend] = fn_compiled

        return fn_compiled(*arrays)


def autocompile(fn, *, backend=None, compiler_opts=None):
    """Just-in-time compile an ``autoray`` function, which should have
    signature:

        fn(*arrays) -> array

    The backend used to do the compilation can be set in three ways:

        1. Automatically based on the arrays the function is called with,
           i.e. ``cfn(*torch_arrays)`` will use ``torch.jit.trace``.
        2. In this wrapper, ``@autocompile(backend='jax')``, to provide a
           specific default instead.
        3. When you call the function ``cfn(*arrays, backend='torch')`` to
           override on a per-call basis.

    If the arrays supplied are of a different backend type to the compiler,
    then the returned array will also be converted back, i.e.
    ``cfn(*numpy_arrays, backend='tensorflow')`` will return a ``numpy`` array.

    The ``'python'`` backend simply extracts and unravels all the ``do`` calls
    into a code object using ``compile`` which is then run with ``exec``.
    This makes use of shared intermediates and constant folding, strips
    away any python scaffoliding, and is compatible with any library, but the
    resulting function is not 'low-level' in the same way as the other
    backends.

    Parameters
    ----------
    fn : callable
        The autoray function to compile.
    backend : {None, 'python', 'jax', 'torch', 'tensorflow'}, optional
        If set, use this as the default backend.
    compiler_opts : dict[dict], optional
        Dict of dicts when you can supply options for each compiler backend
        separately, e.g.:
        ``@autocompile(compiler_opts={'torch': {'script': False}})``.

    Returns
    -------
    cfn : callable
        The function with auto compilation.
    """
    kws = dict(backend=backend, compiler_opts=compiler_opts)
    if fn is None:
        return functools.partial(autocompile, **kws)
    return functools.wraps(fn)(AutoCompiled(fn, **kws))
