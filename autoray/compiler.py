import functools

from .autoray import infer_backend, do, backend_like
from . import lazy


def find_array(x, variables):
    lx = lazy.array(x)
    variables.append(lx)
    return lx


def find_tuple(x, variables):
    return tuple(find_and_inject_variables(subx, variables) for subx in x)


def find_list(x, variables):
    return [find_and_inject_variables(subx, variables) for subx in x]


def find_dict(x, variables):
    return {k: find_and_inject_variables(v, variables) for k, v in x.items()}


def find_identity(x, variables):
    return x


_find_dispatch = {
    tuple: find_tuple,
    list: find_list,
    dict: find_dict,
}


def find_and_inject_variables(x, variables):
    cls = x.__class__
    try:
        find_fn = _find_dispatch[cls]
    except KeyError:
        # cache array types based on whether they have a shape attribute
        if hasattr(x, "shape"):
            find_fn = _find_dispatch[cls] = find_array
        else:
            find_fn = _find_dispatch[cls] = find_identity
    return find_fn(x, variables)


def extract_arrays(x, constants=None):
    if hasattr(x, "shape"):
        yield x
    elif isinstance(x, (tuple, list)):
        for subx in x:
            yield from extract_arrays(subx, constants)
    elif isinstance(x, dict):
        for subx in x.values():
            yield from extract_arrays(subx, constants)
    elif constants is not None:
        constants.append(x)


class CompilePython:
    """A simple compiler that unravels all autoray calls, optionally sharing
    intermediates and folding constants, converts this to a code object using
    ``compile``, then executes this using ``exec``. Non-array function
    arguments are treated as static, with a new function compiled for each
    unique hash of their values.

    Parameters
    ----------
    fn : callable
        Function to compile - should have signature
        ``fn(*args, **kwargs) -> array``, with ``args`` and ``kwargs`` any
        nested combination of ``tuple``, ``list`` and ``dict`` objects
        containing arrays (or other constant arguments), and perform array
        operations on these using ``autoray.do``.
    fold_constants : bool, optional
        Whether to fold all constant array operations into the graph, which
        might increase memory usage.
    share_intermediates : bool, optional
        Whether to cache all computational nodes during the trace, so that any
        shared intermediate results can be identified.
    """

    def __init__(self, fn, fold_constants=True, share_intermediates=True):
        self._fn = fn
        self._fold_constants = fold_constants
        self._share_intermediates = share_intermediates
        self._fns = {}

    def _trace(self, *args, **kwargs):
        """Convert the example arrays to lazy variables and trace them through
        the function.
        """
        variables = []

        def _run_lazy():
            lz_args = find_tuple(args, variables)
            lz_kwargs = find_dict(kwargs, variables)
            return self._fn(*lz_args, **lz_kwargs)

        if self._share_intermediates:
            with backend_like("autoray.lazy"), lazy.shared_intermediates():
                out = _run_lazy()
        else:
            with backend_like("autoray.lazy"):
                out = _run_lazy()

        return out, variables

    def _setup(self, *args, **kwargs):
        """Based on example ``arrays``, compile the function.
        """
        out, variables = self._trace(*args, **kwargs)
        return out.get_function(variables, fold_constants=self._fold_constants)

    def __call__(self, *args, **kwargs):
        """If necessary, build, then call the compiled function.
        """
        # separate variable arrays from constant kwargs
        constants = []
        arrays = (
            *extract_arrays(args, constants),
            *extract_arrays(kwargs, constants),
        )
        key = hash(tuple(constants))

        try:
            return self._fns[key](arrays)
        except KeyError:
            fn = self._fns[key] = self._setup(*args, **kwargs)
            return fn(arrays)


class CompileJax:
    """
    """

    def __init__(self, fn, enable_x64=None, platform_name=None, **kwargs):
        self._fn = fn
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

        self._jit_fn = jax.jit(self._fn, **self._jit_kwargs)
        self._fn = None

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
        self._fn = fn
        kwargs.setdefault("autograph", False)
        kwargs.setdefault("experimental_compile", False)
        self._jit_fn = None
        self._jit_kwargs = kwargs

    def setup(self):
        import tensorflow as tf
        self._jit_fn = tf.function(**self._jit_kwargs)(self._fn)
        self._fn = None

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
        self._fn = fn
        self.script = script
        self._jit_fn = None
        self._jit_kwargs = kwargs

    def setup(self, arrays):
        self._jit_fn = self.torch.jit.trace(
            self._fn, arrays, **self._jit_kwargs
        )
        if self.script:
            self._jit_fn = self.torch.jit.script(self._jit_fn)
        self._fn = None

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


_backend_lookup = {}

_compiler_lookup = {
    "jax": CompileJax,
    "tensorflow": CompileTensorFlow,
    "torch": CompileTorch,
}


class AutoCompiled:
    """Just in time compile a ``autoray.do`` using function. See the main
    wrapper ``autojit``.
    """

    def __init__(self, fn, backend=None, compiler_opts=None):
        self._fn = fn
        self._backend = backend
        self._compiled_fns = {}
        if compiler_opts is None:
            self._compiler_kwargs = {}
        else:
            self._compiler_kwargs = compiler_opts

    def __call__(self, *args, backend=None, **kwargs):
        array_backend = infer_backend(next(extract_arrays((args, kwargs))))
        if backend is None:
            if self._backend is None:
                backend = array_backend
            else:
                backend = self._backend

        try:
            key = _backend_lookup[backend, array_backend]
        except KeyError:
            if backend in _compiler_lookup:
                key = backend
            else:
                key = f"python-{array_backend}"
            _backend_lookup[backend, array_backend] = key

        try:
            fn_compiled = self._compiled_fns[key]
        except KeyError:
            if "python" in key:
                backend = "python"
            backend_compiler = _compiler_lookup.get(backend, CompilePython)
            compiler_kwargs = self._compiler_kwargs.get(backend, {})
            fn_compiled = backend_compiler(self._fn, **compiler_kwargs)
            self._compiled_fns[key] = fn_compiled

        return fn_compiled(*args, **kwargs)


def autojit(fn=None, *, backend=None, compiler_opts=None):
    """Just-in-time compile an ``autoray`` function, which should have
    signature:

        fn(*arrays) -> array

    The backend used to do the compilation can be set in three ways:

        1. Automatically based on the arrays the function is called with,
           i.e. ``cfn(*torch_arrays)`` will use ``torch.jit.trace``.
        2. In this wrapper, ``@autojit(backend='jax')``, to provide a
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
        ``@autojit(compiler_opts={'torch': {'script': False}})``.

    Returns
    -------
    cfn : callable
        The function with auto compilation.
    """
    kws = dict(backend=backend, compiler_opts=compiler_opts)
    if fn is None:
        return functools.partial(autojit, **kws)
    return functools.wraps(fn)(AutoCompiled(fn, **kws))
