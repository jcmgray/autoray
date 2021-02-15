"""
AUTORAY - backend agnostic array operations.


Copyright 2019 Johnnie Gray

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import importlib
import functools
import itertools
from collections import OrderedDict, defaultdict

import numpy as _numpy


def do(fn, *args, like=None, **kwargs):
    """Do function named ``fn`` on ``(*args, **kwargs)``, peforming single
    dispatch to retrieve ``fn`` based on whichever library defines the class of
    the ``args[0]``, or the ``like`` keyword argument if specified.

    Examples
    --------

    Works on numpy arrays:

        >>> import numpy as np
        >>> x_np = np.random.uniform(size=[5])
        >>> y_np = do('sqrt', x_np)
        >>> y_np
        array([0.32464973, 0.90379787, 0.85037325, 0.88729814, 0.46768083])

        >>> type(y_np)
        numpy.ndarray

    Works on cupy arrays:

        >>> import cupy as cp
        >>> x_cp = cp.random.uniform(size=[5])
        >>> y_cp = do('sqrt', x_cp)
        >>> y_cp
        array([0.44541656, 0.88713113, 0.92626237, 0.64080557, 0.69620767])

        >>> type(y_cp)
        cupy.core.core.ndarray

    Works on tensorflow arrays:

        >>> import tensorflow as tf
        >>> x_tf = tf.random.uniform(shape=[5])
        >>> y_tf = do('sqrt', x_tf)
        >>> y_tf
        <tf.Tensor 'Sqrt_1:0' shape=(5,) dtype=float32>

        >>> type(y_tf)
        tensorflow.python.framework.ops.Tensor

    You get the idea.

    For functions that don't dispatch on the first argument you can use the
    ``like`` keyword:

        >>> do('eye', 3, like=x_tf)
        <tf.Tensor: id=91, shape=(3, 3), dtype=float32>
    """
    if like is None:
        dispatch_arg = _DISPATCHERS[fn](*args, **kwargs)
        backend = infer_backend(dispatch_arg)
    elif isinstance(like, str):
        backend = like
    else:
        backend = infer_backend(like)

    return get_lib_fn(backend, fn)(*args, **kwargs)


@functools.lru_cache(128)
def _infer_class_backend_cached(T):
    if issubclass(T, _numpy.ndarray):
        return "numpy"

    lib = T.__module__.split(".")[0]

    # check if lib should mapped entirely to another lib
    backend = _BACKEND_ALIASES.get(lib, lib)

    return backend


def infer_backend(array):
    """Get the name of the library that defined the class of ``array`` - unless
    ``array`` is directly a subclass of ``numpy.ndarray``, in which case assume
    ``numpy`` is the desired backend.
    """
    return _infer_class_backend_cached(array.__class__)


def get_lib_fn(backend, fn):
    """Cached retrieval of correct function for backend

    Parameters
    ----------
    backend : str
        The module defining the array class to dispatch on.
    fn : str
        The function to retrieve.

    Returns
    -------
    callable
    """

    try:
        lib_fn = _FUNCS[backend, fn]
    except KeyError:
        # alias for global module,
        #     e.g. 'decimal' -> 'math'
        module = _MODULE_ALIASES.get(backend, backend)

        # submodule where function is found for backend,
        #     e.g. ['tensorflow', trace'] -> 'tensorflow.linalg'
        try:
            full_location = _SUBMODULE_ALIASES[backend, fn]

            # if explicit submodule alias given, don't use prepended location
            #     for example, ('torch', 'linalg.svd') -> torch.svd
            only_fn = fn.split(".")[-1]

        except KeyError:
            full_location = module

            # move any prepended location into the full module path
            #     e.g. 'fn=linalg.eigh' -> ['linalg', 'eigh']
            split_fn = fn.split(".")
            full_location = ".".join([full_location] + split_fn[:-1])
            only_fn = split_fn[-1]

        # cached lookup of custom name function might take
        #     e.g. ['tensorflow', 'sum'] -> 'reduce_sum'
        fn_name = _FUNC_ALIASES.get((backend, fn), only_fn)

        # import the function into the cache
        try:
            lib = importlib.import_module(full_location)
        except ImportError:
            # sometimes libraries hack an attribute to look like submodule
            mod, submod = full_location.split(".")
            lib = getattr(importlib.import_module(mod), submod)

        # check for a custom wrapper but default to identity
        wrapper = _CUSTOM_WRAPPERS.get((backend, fn), lambda fn: fn)

        # store the function!
        lib_fn = _FUNCS[backend, fn] = wrapper(getattr(lib, fn_name))

    return lib_fn


# ---------------------- special top level functions ------------------------ #


def conj(x):
    """Array conjugate."""
    return do("conj", x)


def transpose(x, *args):
    """Array transpose."""
    return do("transpose", x, *args)


def dag(x):
    """Array Hermitian transpose."""
    try:
        return x.H
    except AttributeError:
        return do("conj", do("transpose", x))


def real(x):
    """Array real part."""
    return do("real", x)


def imag(x):
    """Array imaginary part."""
    return do("imag", x)


def reshape(x, shape):
    """Array reshaped."""
    try:
        return x.reshape(shape)
    except AttributeError:
        return do("reshape", x, shape)


def to_backend_dtype(dtype_name, like):
    """Turn string specifier ``dtype_name`` into dtype of backend ``like``."""
    if not isinstance(like, str):
        like = infer_backend(like)

    try:
        return get_lib_fn(like, dtype_name)
    except AttributeError:
        return dtype_name


def get_dtype_name(x):
    """Find string specifier ``dtype_name`` of array ``x``."""
    try:
        return x.dtype.name
    except AttributeError:
        # let modules provide their own
        return do("get_dtype_name", x)


def astype(x, dtype_name, **kwargs):
    """Cast array as type ``dtype_name`` - tries ``x.astype`` first."""
    dtype = to_backend_dtype(dtype_name, like=x)
    try:
        return x.astype(dtype, **kwargs)
    except AttributeError:
        return do("astype", x, dtype, **kwargs)


def to_numpy(x):
    """Get a numpy version of array ``x``."""
    return do("to_numpy", x)


# -------------------------- some common wrappers --------------------------- #


def svd_not_full_matrices_wrapper(fn):
    @functools.wraps(fn)
    def default_not_full_matrices(*args, **kwargs):
        kwargs.setdefault("full_matrices", False)
        return fn(*args, **kwargs)

    return default_not_full_matrices


def svd_sUV_to_UsVH_wrapper(fn):
    @functools.wraps(fn)
    def numpy_like(*args, **kwargs):
        s, U, V = fn(*args, **kwargs)
        return U, s, dag(V)

    return numpy_like


def svd_UsV_to_UsVH_wrapper(fn):
    @functools.wraps(fn)
    def numpy_like(*args, **kwargs):
        U, s, V = fn(*args, **kwargs)
        return U, s, dag(V)

    return numpy_like


def svd_manual_full_matrices_kwarg(fn):
    @functools.wraps(fn)
    def numpy_like(*args, full_matrices=False, **kwargs):
        U, s, VH = fn(*args, **kwargs)

        if not full_matrices:
            U, VH = U[:, : s.size], VH[: s.size, :]

        return U, s, VH

    return numpy_like


def qr_allow_fat(fn):
    @functools.wraps(fn)
    def numpy_like(a, **kwargs):
        m, n = a.shape

        if m >= n:
            # square or thin
            return fn(a, **kwargs)

        Q, R_sq = fn(a[:, :m])
        R_r = dag(Q) @ a[:, m:]
        R = do("concatenate", (R_sq, R_r), axis=1, like=a)

        return Q, R

    return numpy_like


def tril_to_band_part(fn):
    @functools.wraps(fn)
    def numpy_like(x, k=0):

        if k < 0:
            raise ValueError(
                "'k' must be positive to recreate 'numpy.tril' "
                "behaviour with 'tensorflow.matrix_band_part'."
            )

        return fn(x, -1, k)

    return numpy_like


def triu_to_band_part(fn):
    @functools.wraps(fn)
    def numpy_like(x, k=0):

        if k > 0:
            raise ValueError(
                "'k' must be negative to recreate 'numpy.triu' "
                "behaviour with 'tensorflow.matrix_band_part'."
            )

        return fn(x, -k, -1)

    return numpy_like


def scale_random_uniform_manually(fn):
    @functools.wraps(fn)
    def numpy_like(low=0.0, high=1.0, size=None, **kwargs):
        if size is None:
            size = ()

        x = fn(size=size, **kwargs)

        if (low != 0.0) or (high != 1.0):
            x = (high - low) * x + low

        return x

    return numpy_like


def scale_random_normal_manually(fn):
    @functools.wraps(fn)
    def numpy_like(loc=0.0, scale=1.0, size=None, **kwargs):
        if size is None:
            size = ()

        x = fn(size=size, **kwargs)

        if (loc != 0.0) or (scale != 1.0):
            x = scale * x + loc

        return x

    return numpy_like


def translate_wrapper(fn, translator):
    """Wrap a function to match the api of another according to a translation.
    The ``translator`` entries in the form of an ordered dict should have
    entries like:

        (desired_kwarg: (backend_kwarg, default_value))

    with the order defining the args of the function.
    """

    @functools.wraps(fn)
    def translated_function(*args, **kwargs):
        new_kwargs = {}
        translation = translator.copy()

        # convert args
        for arg_value in args:
            new_arg_name = translation.popitem(last=False)[1][0]
            new_kwargs[new_arg_name] = arg_value

        # convert kwargs -  but only those in the translation
        for key, value in kwargs.items():
            try:
                new_kwargs[translation.pop(key)[0]] = value
            except KeyError:
                new_kwargs[key] = value

        # set remaining default kwargs
        for key, value in translation.items():
            new_kwargs[value[0]] = value[1]

        return fn(**new_kwargs)

    return translated_function


def make_translator(t):
    return functools.partial(translate_wrapper, translator=OrderedDict(t))


def complex_add_re_im(re, im):
    return re + 1j * im


# ----------------------------- Custom dispatchers -------------------------- #


def register_dispatch(fun, dispatcher):
    """Register a new dispatcher.

    This is useful in case the backend to be used by a function cannot be
    inferred from the first argument.
    """
    _DISPATCHERS[fun] = dispatcher


def default_dispatcher(*args, **kwargs):
    """Try to infer backend from first argument passed to function."""
    return args[0]


# lookup of custom dispatcher methods, for cases when backend cannot be
#     inferred accurately from first argument.
_DISPATCHERS = defaultdict(lambda: default_dispatcher)


def join_array_dispatcher(*args, **kwargs):
    """Dispatcher for functions where first argument is a sequence."""
    try:
        return args[0][0]
    except (TypeError, ValueError):
        # user passed an empty sequence, or something non-iterable
        # try to infer backend from first argument as fallback
        return args[0]


# List of functions listed in numpy API as array joining operations
register_dispatch("concatenate", join_array_dispatcher)
register_dispatch("stack", join_array_dispatcher)
register_dispatch("block", join_array_dispatcher)
register_dispatch("vstack", join_array_dispatcher)
register_dispatch("hstack", join_array_dispatcher)
register_dispatch("dstack", join_array_dispatcher)
register_dispatch("column_stack", join_array_dispatcher)
register_dispatch("row_stack", join_array_dispatcher)


def einsum_dispatcher(*args, **kwargs):
    """Dispatcher for handling einsum.

    einsum can either take string as first argument, in which case backend
    should be inferred from second argument. Or it can take an array as first
    argument, which should be used to infer backend.
    """
    if isinstance(args[0], str):
        return args[1]
    return args[0]


register_dispatch("einsum", einsum_dispatcher)

# --------------- object to act as drop-in replace for numpy ---------------- #

_partial_functions = {}


class NumpyMimic:
    """A class to mimic the syntax of using `numpy` directly."""

    def __init__(self, submodule=None):
        self.submodule = submodule

    def __getattribute__(self, fn):

        # look out for certain submodules which are not functions
        if fn == "linalg":
            return numpy_linalg
        if fn == "random":
            return numpy_random

        # if this is the e.g. linalg mimic, preprend 'linalg.'
        submod = object.__getattribute__(self, "submodule")
        if submod is not None:
            fn = ".".join((submod, fn))

        # cache the correct partial function
        try:
            pfn = _partial_functions[fn]
        except KeyError:
            pfn = _partial_functions[fn] = functools.partial(do, fn)

        return pfn

    @staticmethod
    def __repr__():
        return "<autoray.numpy>"


numpy = NumpyMimic()
numpy_linalg = NumpyMimic("linalg")
numpy_random = NumpyMimic("random")


# --------------------------------------------------------------------------- #
#                    storage & backend specific functions                     #
# --------------------------------------------------------------------------- #

# lookup for mapping entire lib to another
_BACKEND_ALIASES = {}

# global (non function specific) aliases
_MODULE_ALIASES = {}

# lookup for when functions are elsewhere than the expected location
_SUBMODULE_ALIASES = {}

# lookup for when functions are simply called something else
_FUNC_ALIASES = {}

# custom wrappers for when functions don't just have different location or
#     name. For example, when kwargs need to be translated or results modified
_CUSTOM_WRAPPERS = {}

# actual cache of funtions to use - this is populated lazily and can be used
#     to directly set an implementation of a function for a specific backend
_FUNCS = {}

# ------------------------------ standard-lib ------------------------------- #

_MODULE_ALIASES["decimal"] = "math"
_MODULE_ALIASES["builtins"] = "numpy"


# ---------------------------------- numpy ---------------------------------- #


def numpy_to_numpy(x):
    return do("array", x, like="numpy")


_FUNCS["numpy", "to_numpy"] = numpy_to_numpy
_FUNCS["numpy", "complex"] = complex_add_re_im
_FUNCS["builtins", "to_numpy"] = numpy_to_numpy
_SUBMODULE_ALIASES["numpy", "linalg.expm"] = "scipy.linalg"
_CUSTOM_WRAPPERS["numpy", "linalg.svd"] = svd_not_full_matrices_wrapper


# ---------------------------------- cupy ----------------------------------- #


def cupy_to_numpy(x):  # pragma: no cover
    return x.get()


_FUNCS["cupy", "to_numpy"] = cupy_to_numpy
_FUNCS["cupy", "complex"] = complex_add_re_im
_CUSTOM_WRAPPERS["cupy", "linalg.svd"] = svd_not_full_matrices_wrapper


# ----------------------------------- jax ----------------------------------- #


_JAX_RANDOM_KEY = None


def jax_random_seed(seed=None):
    from jax.random import PRNGKey

    global _JAX_RANDOM_KEY
    if seed is None:
        from random import SystemRandom

        seed = SystemRandom().randint(
            -(2 ** 63), 2 ** 63 - 1
        )  # inclusive high
    _JAX_RANDOM_KEY = PRNGKey(seed)


def jax_random_get_key():
    from jax.random import split

    global _JAX_RANDOM_KEY
    if _JAX_RANDOM_KEY is None:
        jax_random_seed()
    _JAX_RANDOM_KEY, subkey = split(_JAX_RANDOM_KEY)
    return subkey


def jax_random_uniform(low=0.0, high=1.0, size=None, **kwargs):
    from jax.random import uniform

    if size is None:
        size = ()
    return uniform(
        jax_random_get_key(), shape=size, minval=low, maxval=high, **kwargs
    )


def jax_random_normal(loc=0.0, scale=1.0, size=None, **kwargs):
    from jax.random import normal

    if size is None:
        size = ()
    x = normal(jax_random_get_key(), shape=size, **kwargs)
    if scale != 1.0:
        x *= scale
    if loc != 0.0:
        x += loc
    return x


def jax_to_numpy(x):
    return x.__array__()


_BACKEND_ALIASES["jaxlib"] = "jax"
_MODULE_ALIASES["jax"] = "jax.numpy"
_SUBMODULE_ALIASES["jax", "complex"] = "jax.lax"
_SUBMODULE_ALIASES["jax", "linalg.expm"] = "jax.scipy.linalg"
_CUSTOM_WRAPPERS["jax", "linalg.qr"] = qr_allow_fat
_CUSTOM_WRAPPERS["jax", "linalg.svd"] = svd_not_full_matrices_wrapper
_FUNCS["jax", "to_numpy"] = jax_to_numpy
_FUNCS["jax", "random.seed"] = jax_random_seed
_FUNCS["jax", "random.uniform"] = jax_random_uniform
_FUNCS["jax", "random.normal"] = jax_random_normal


# -------------------------------- autograd --------------------------------- #

_MODULE_ALIASES["autograd"] = "autograd.numpy"
_CUSTOM_WRAPPERS["autograd", "linalg.svd"] = svd_not_full_matrices_wrapper
_FUNCS["autograd", "complex"] = complex_add_re_im


# ---------------------------------- dask ----------------------------------- #


def dask_to_numpy(x):
    return x.compute()


_FUNCS["dask", "to_numpy"] = dask_to_numpy
_FUNCS["dask", "complex"] = complex_add_re_im
_FUNC_ALIASES["dask", "abs"] = "absolute"
_MODULE_ALIASES["dask"] = "dask.array"
_CUSTOM_WRAPPERS["dask", "linalg.svd"] = svd_manual_full_matrices_kwarg


# ---------------------------------- mars ----------------------------------- #


def mars_to_numpy(x):
    return x.to_numpy()


_FUNCS["mars", "to_numpy"] = mars_to_numpy
_FUNCS["mars", "complex"] = complex_add_re_im
_MODULE_ALIASES["mars"] = "mars.tensor"


# ----------------------------------- ctf ----------------------------------- #


def ctf_array(x):
    return do("astensor", x, like="ctf")


def ctf_to_numpy(x):
    return x.to_nparray()


def ctf_get_dtype_name(x):
    return x.dtype.__name__


_FUNCS["ctf", "array"] = ctf_array
_FUNCS["ctf", "to_numpy"] = ctf_to_numpy
_FUNCS["ctf", "get_dtype_name"] = ctf_get_dtype_name
_SUBMODULE_ALIASES["ctf", "linalg.svd"] = "ctf"
_SUBMODULE_ALIASES["ctf", "linalg.eigh"] = "ctf"
_SUBMODULE_ALIASES["ctf", "linalg.qr"] = "ctf"


# ------------------------------- sparse------------------------------------- #


def sparse_array(x):
    return do("COO.from_numpy", x, like="sparse")


def sparse_to_numpy(x):
    return x.todense()


def sparse_complex(x, y):
    return x + 1j * y


def sparse_transpose(x, axes=None):
    return x.transpose(axes)


def sparse_sum(x, axis=None, keepdims=False, dtype=None, out=None):
    return x.sum(axis=axis, keepdims=keepdims, dtype=dtype, out=out)


def sparse_prod(x, axis=None, keepdims=False, dtype=None, out=None):
    return x.prod(axis=axis, keepdims=keepdims, dtype=dtype, out=out)


def sparse_conj(x):
    return x.conj()


def sparse_real(x):
    return x.real


def sparse_imag(x):
    return x.imag


def sparse_count_nonzero(x):
    return x.nnz


def sparse_random_uniform(low=0.0, high=1.0, size=None, **kwargs):
    def rvs(nnz):
        return do("random.uniform", low, high, (nnz,), like="numpy")

    return do("random", size, data_rvs=rvs, **kwargs, like="sparse")


def sparse_random_normal(loc=0.0, scale=1.0, size=None, **kwargs):
    def rvs(nnz):
        return do("random.normal", loc, scale, (nnz,), like="numpy")

    return do("random", size, data_rvs=rvs, **kwargs, like="sparse")


_FUNCS["sparse", "array"] = sparse_array
_FUNCS["sparse", "to_numpy"] = sparse_to_numpy
_FUNCS["sparse", "transpose"] = sparse_transpose
_FUNCS["sparse", "sum"] = sparse_sum
_FUNCS["sparse", "prod"] = sparse_prod
_FUNCS["sparse", "conj"] = sparse_conj
_FUNCS["sparse", "real"] = sparse_real
_FUNCS["sparse", "imag"] = sparse_imag
_FUNCS["sparse", "complex"] = sparse_complex
_FUNCS["sparse", "count_nonzero"] = sparse_count_nonzero
_FUNCS["sparse", "random.uniform"] = sparse_random_uniform
_FUNCS["sparse", "random.normal"] = sparse_random_normal

# sparse uses numpys __array_func__ interface
for f in (
    "log",
    "log2",
    "log10",
    "exp",
    "sqrt",
    "sign",
    "sin",
    "cos",
    "tan",
    "arcsin",
    "arccos",
    "arctan",
    "sinh",
    "cosh",
    "tanh",
    "arcsinh",
    "arccosh",
    "arctanh",
):
    _SUBMODULE_ALIASES["sparse", f] = "numpy"


# ------------------------------- tensorflow -------------------------------- #


def tensorflow_to_numpy(x):
    return x.numpy()


def tensorflow_pad_wrap(tf_pad):
    def numpy_like(array, pad_width, mode="constant", constant_values=0):
        if mode != "constant":
            raise NotImplementedError

        try:
            if len(pad_width) == 1:
                pad_width = pad_width * len(array.shape)
        except TypeError:
            pad_width = ((pad_width, pad_width),) * len(array.shape)

        return tf_pad(
            array, pad_width, mode="CONSTANT", constant_values=constant_values
        )

    return numpy_like


_FUNCS["tensorflow", "to_numpy"] = tensorflow_to_numpy

_SUBMODULE_ALIASES["tensorflow", "log"] = "tensorflow.math"
_SUBMODULE_ALIASES["tensorflow", "conj"] = "tensorflow.math"
_SUBMODULE_ALIASES["tensorflow", "real"] = "tensorflow.math"
_SUBMODULE_ALIASES["tensorflow", "imag"] = "tensorflow.math"
_SUBMODULE_ALIASES["tensorflow", "power"] = "tensorflow.math"
_SUBMODULE_ALIASES["tensorflow", "count_nonzero"] = "tensorflow.math"
_SUBMODULE_ALIASES["tensorflow", "diag"] = "tensorflow.linalg"
_SUBMODULE_ALIASES["tensorflow", "trace"] = "tensorflow.linalg"
_SUBMODULE_ALIASES["tensorflow", "tril"] = "tensorflow.linalg"
_SUBMODULE_ALIASES["tensorflow", "triu"] = "tensorflow.linalg"

_FUNC_ALIASES["tensorflow", "sum"] = "reduce_sum"
_FUNC_ALIASES["tensorflow", "min"] = "reduce_min"
_FUNC_ALIASES["tensorflow", "max"] = "reduce_max"
_FUNC_ALIASES["tensorflow", "mean"] = "reduce_mean"
_FUNC_ALIASES["tensorflow", "prod"] = "reduce_prod"
_FUNC_ALIASES["tensorflow", "concatenate"] = "concat"
_FUNC_ALIASES["tensorflow", "clip"] = "clip_by_value"
_FUNC_ALIASES["tensorflow", "arange"] = "range"
_FUNC_ALIASES["tensorflow", "tril"] = "band_part"
_FUNC_ALIASES["tensorflow", "triu"] = "band_part"
_FUNC_ALIASES["tensorflow", "diag"] = "tensor_diag"
_FUNC_ALIASES["tensorflow", "array"] = "convert_to_tensor"
_FUNC_ALIASES["tensorflow", "astype"] = "cast"
_FUNC_ALIASES["tensorflow", "power"] = "pow"
_FUNC_ALIASES["tensorflow", "take"] = "gather"

_CUSTOM_WRAPPERS["tensorflow", "linalg.svd"] = svd_sUV_to_UsVH_wrapper
_CUSTOM_WRAPPERS["tensorflow", "linalg.qr"] = qr_allow_fat
_CUSTOM_WRAPPERS["tensorflow", "tril"] = tril_to_band_part
_CUSTOM_WRAPPERS["tensorflow", "triu"] = triu_to_band_part
_CUSTOM_WRAPPERS["tensorflow", "pad"] = tensorflow_pad_wrap
_CUSTOM_WRAPPERS["tensorflow", "random.uniform"] = make_translator(
    [
        ("low", ("minval", 0.0)),
        ("high", ("maxval", 1.0)),
        ("size", ("shape", ())),
    ]
)
_CUSTOM_WRAPPERS["tensorflow", "random.normal"] = make_translator(
    [
        ("loc", ("mean", 0.0)),
        ("scale", ("stddev", 1.0)),
        ("size", ("shape", ())),
    ]
)
_CUSTOM_WRAPPERS["tensorflow", "clip"] = make_translator(
    [
        ("a", ("t", 0.0)),
        ("a_min", ("clip_value_min",)),
        ("a_max", ("clip_value_max",)),
    ]
)


# ---------------------------------- torch ---------------------------------- #


def torch_to_numpy(x):
    return x.detach().cpu().numpy()


def torch_transpose(x, axes=None):
    if axes is None:
        axes = reversed(range(0, x.ndimension()))
    return x.permute(*axes)


def torch_count_nonzero(x):
    return do("sum", x != 0, like="torch")


def torch_astype(x, dtype):
    return x.to(dtype=to_backend_dtype(dtype, like=x))


@functools.lru_cache(32)
def _torch_get_dtype_name(dtype):
    return str(dtype).split(".")[-1]


def torch_get_dtype_name(x):
    return _torch_get_dtype_name(x.dtype)


def torch_real(x):
    # torch doesn't support calling real on real arrays
    try:
        if x.is_complex():
            return x.real
    except AttributeError:
        pass
    return x


def torch_imag(x):
    # torch doesn't support calling imag on real arrays
    try:
        if x.is_complex():
            return x.imag
    except AttributeError:
        pass
    return do("zeros_like", x, like="torch")


def torch_linalg_solve(a, b):
    return do("solve", b, a, like="torch")[0]


def torch_linalg_eigh(x):
    return tuple(do("symeig", x, eigenvectors=True, like="torch"))


def torch_linalg_eigvalsh(x):
    return do("symeig", x, eigenvectors=False, like="torch")[0]


def torch_pad(array, pad_width, mode="constant", constant_values=0):
    if mode != "constant":
        raise NotImplementedError

    try:
        # numpy takes pads like ((0, 0), (1, 1), ... (n-1, n-1))
        # torch takes pads like (n-1, n-1, n-2, n-2, n-3, n-3, ...)
        pad = tuple(itertools.chain.from_iterable(pad_width))[::-1]

        # a single tuple was specified ((a, b),) - use for all axes
        if len(pad) == 2:
            pad = pad * array.ndimension()

    except TypeError:
        # assume int
        pad = (pad_width,) * 2 * array.ndimension()

    return do(
        "nn.functional.pad",
        array,
        pad=pad,
        mode=mode,
        value=constant_values,
        like="torch",
    )


_FUNCS["torch", "pad"] = torch_pad
_FUNCS["torch", "real"] = torch_real
_FUNCS["torch", "imag"] = torch_imag
_FUNCS["torch", "astype"] = torch_astype
_FUNCS["torch", "to_numpy"] = torch_to_numpy
_FUNCS["torch", "complex"] = complex_add_re_im
_FUNCS["torch", "transpose"] = torch_transpose
_FUNCS["torch", "count_nonzero"] = torch_count_nonzero
_FUNCS["torch", "get_dtype_name"] = torch_get_dtype_name
_FUNCS["torch", "linalg.solve"] = torch_linalg_solve
_FUNCS["torch", "linalg.eigh"] = torch_linalg_eigh
_FUNCS["torch", "linalg.eigvalsh"] = torch_linalg_eigvalsh

_FUNC_ALIASES["torch", "clip"] = "clamp"
_FUNC_ALIASES["torch", "power"] = "pow"
_FUNC_ALIASES["torch", "array"] = "tensor"
_FUNC_ALIASES["torch", "concatenate"] = "cat"
_FUNC_ALIASES["torch", "random.normal"] = "randn"
_FUNC_ALIASES["torch", "random.uniform"] = "rand"
_FUNC_ALIASES["torch", "linalg.expm"] = "matrix_exp"
_FUNC_ALIASES["torch", "take"] = "index_select"

_SUBMODULE_ALIASES["torch", "linalg.qr"] = "torch"
_SUBMODULE_ALIASES["torch", "linalg.svd"] = "torch"
_SUBMODULE_ALIASES["torch", "linalg.expm"] = "torch"
_SUBMODULE_ALIASES["torch", "random.normal"] = "torch"
_SUBMODULE_ALIASES["torch", "random.uniform"] = "torch"

_CUSTOM_WRAPPERS["torch", "linalg.svd"] = svd_UsV_to_UsVH_wrapper
_CUSTOM_WRAPPERS["torch", "linalg.qr"] = qr_allow_fat
_CUSTOM_WRAPPERS["torch", "random.normal"] = scale_random_normal_manually
_CUSTOM_WRAPPERS["torch", "random.uniform"] = scale_random_uniform_manually
_CUSTOM_WRAPPERS["torch", "stack"] = make_translator(
    [
        ("arrays", ("tensors",)),
        ("axis", ("dim", 0)),
    ]
)
_CUSTOM_WRAPPERS["torch", "concatenate"] = make_translator(
    [("arrays", ("tensors",)), ("axis", ("dim", 0))]
)
_CUSTOM_WRAPPERS["torch", "tril"] = make_translator(
    [
        ("m", ("input",)),
        ("k", ("diagonal", 0)),
    ]
)
_CUSTOM_WRAPPERS["torch", "triu"] = make_translator(
    [
        ("m", ("input",)),
        ("k", ("diagonal", 0)),
    ]
)
_CUSTOM_WRAPPERS["torch", "clip"] = make_translator(
    [
        ("a", ("input",)),
        ("a_min", ("min",)),
        ("a_max", ("max",)),
    ]
)
_CUSTOM_WRAPPERS["torch", "ones"] = make_translator(
    [
        ("shape", ("size",)),
    ]
)
_CUSTOM_WRAPPERS["torch", "zeros"] = make_translator(
    [
        ("shape", ("size",)),
    ]
)
_CUSTOM_WRAPPERS["torch", "empty"] = make_translator(
    [
        ("shape", ("size",)),
    ]
)
_CUSTOM_WRAPPERS["torch", "take"] = make_translator(
    [
        ("a", ("input",)),
        ("indices", ("index",)),
        ("axis", ("dim",)),
    ]
)


# --------------------------- register your own! ---------------------------- #


def register_function(backend, name, fn, wrap=False):
    """Directly provide your own function.

    Parameters
    ----------
    backend : str
        The name of the backend to register the function for.
    name : str
        Name of the function, e.g. `'sum'` or `'linalg.svd'`.
    fn : callable
        The function to register.
    wrap : bool, optional
        Whether to wrap the old function like ``fn(old_fn)`` rather than
        directly supply the entire new function.
    """
    if wrap:
        old = get_lib_fn(backend, name)
        _FUNCS[backend, name] = fn(old)
    else:
        _FUNCS[backend, name] = fn
