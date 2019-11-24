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
from collections import OrderedDict

import numpy as _numpy


def infer_backend(array):
    """Get the name of the library that defined the class of ``array`` - unless
    ``array`` is directly a subclass of ``numpy.ndarray``, in which case assume
    ``numpy`` is the desired backend.
    """
    if isinstance(array, _numpy.ndarray):
        return 'numpy'
    return array.__class__.__module__.split('.')[0]


# global (non function specific) aliases
_module_aliases = {
    'decimal': 'math',
    'jax': 'jax.numpy',
    'builtins': 'numpy',
    'dask': 'dask.array',
    'mars': 'mars.tensor',
    'autograd': 'autograd.numpy',
}


# lookup for when functions are elsewhere than the expected location
_submodule_aliases = {
    ('numpy', 'linalg.expm'): 'scipy.linalg',
    ('tensorflow', 'log'): 'tensorflow.math',
    ('tensorflow', 'conj'): 'tensorflow.math',
    ('tensorflow', 'real'): 'tensorflow.math',
    ('tensorflow', 'imag'): 'tensorflow.math',
    ('tensorflow', 'diag'): 'tensorflow.linalg',
    ('tensorflow', 'trace'): 'tensorflow.linalg',
    ('tensorflow', 'tril'): 'tensorflow.linalg',
    ('tensorflow', 'triu'): 'tensorflow.linalg',
    ('torch', 'linalg.svd'): 'torch',
    ('torch', 'linalg.norm'): 'torch',
    ('torch', 'random.normal'): 'torch',
    ('torch', 'random.uniform'): 'torch',
    ('ctf', 'linalg.svd'): 'ctf',
    ('ctf', 'linalg.eigh'): 'ctf',
    ('ctf', 'linalg.qr'): 'ctf',
}


# lookup for when functions don't have the same name
_func_aliases = {
    ('tensorflow', 'sum'): 'reduce_sum',
    ('tensorflow', 'min'): 'reduce_min',
    ('tensorflow', 'max'): 'reduce_max',
    ('tensorflow', 'mean'): 'reduce_mean',
    ('tensorflow', 'prod'): 'reduce_prod',
    ('tensorflow', 'concatenate'): 'concat',
    ('tensorflow', 'clip'): 'clip_by_value',
    ('tensorflow', 'arange'): 'range',
    ('tensorflow', 'tril'): 'band_part',
    ('tensorflow', 'triu'): 'band_part',
    ('tensorflow', 'diag'): 'tensor_diag',
    ('tensorflow', 'array'): 'convert_to_tensor',
    ('torch', 'array'): 'tensor',
    ('torch', 'arange'): 'range',
    ('torch', 'random.normal'): 'randn',
    ('torch', 'random.uniform'): 'rand',
}


def svd_not_full_matrices_wrapper(fn):

    @functools.wraps(fn)
    def default_not_full_matrices(*args, **kwargs):
        kwargs.setdefault('full_matrices', False)
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
            U, VH = U[:, :s.size], VH[:s.size, :]

        return U, s, VH

    return numpy_like


def tril_to_band_part(fn):

    @functools.wraps(fn)
    def numpy_like(x, k=0):

        if k < 0:
            raise ValueError("'k' must be positive to recreate 'numpy.tril' "
                             "behaviour with 'tensorflow.matrix_band_part'.")

        return fn(x, -1, k)

    return numpy_like


def triu_to_band_part(fn):

    @functools.wraps(fn)
    def numpy_like(x, k=0):

        if k > 0:
            raise ValueError("'k' must be negative to recreate 'numpy.triu' "
                             "behaviour with 'tensorflow.matrix_band_part'.")

        return fn(x, -k, -1)

    return numpy_like


def scale_random_uniform_manually(fn):

    @functools.wraps(fn)
    def numpy_like(low=0.0, high=1.0, size=None):
        if size is None:
            size = ()

        x = fn(size=size)

        if (low != 0.0) or (high != 1.0):
            x = (high - low) * x + low

        return x

    return numpy_like


def scale_random_normal_manually(fn):

    @functools.wraps(fn)
    def numpy_like(loc=0.0, scale=1.0, size=None):
        if size is None:
            size = ()

        x = fn(size=size)

        if (loc != 0.0) or (scale != 1.0):
            x = scale * x + loc

        return x

    return numpy_like


# custom wrapper for when functions don't just have different location or name
_custom_wrappers = {
    ('numpy', 'linalg.svd'): svd_not_full_matrices_wrapper,
    ('cupy', 'linalg.svd'): svd_not_full_matrices_wrapper,
    ('jax', 'linalg.svd'): svd_not_full_matrices_wrapper,
    ('dask', 'linalg.svd'): svd_manual_full_matrices_kwarg,
    ('tensorflow', 'linalg.svd'): svd_sUV_to_UsVH_wrapper,
    ('tensorflow', 'tril'): tril_to_band_part,
    ('tensorflow', 'triu'): triu_to_band_part,
    ('torch', 'linalg.svd'): svd_UsV_to_UsVH_wrapper,
    ('torch', 'random.normal'): scale_random_normal_manually,
    ('torch', 'random.uniform'): scale_random_uniform_manually,
}


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


_custom_wrappers['tensorflow', 'random.uniform'] = make_translator([
    ('low', ('minval', 0.0)),
    ('high', ('maxval', 1.0)),
    ('size', ('shape', ())),
])
_custom_wrappers['tensorflow', 'random.normal'] = make_translator([
    ('loc', ('mean', 0.0)),
    ('scale', ('stddev', 1.0)),
    ('size', ('shape', ())),
])
_custom_wrappers['torch', 'stack'] = make_translator([
    ('arrays', ('tensors',)),
    ('axis', ('dim', 0)),
])
_custom_wrappers['torch', 'tril'] = make_translator([
    ('m', ('input',)),
    ('k', ('diagonal', 0)),
])
_custom_wrappers['torch', 'triu'] = make_translator([
    ('m', ('input',)),
    ('k', ('diagonal', 0)),
])


# actual cache of funtions to use - this is populated lazily
_funcs = {}


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
        lib_fn = _funcs[backend, fn]
    except KeyError:
        # alias for global module,
        #     e.g. 'decimal' -> 'math'
        module = _module_aliases.get(backend, backend)

        # submodule where function is found for backend,
        #     e.g. ['tensorflow', trace'] -> 'tensorflow.linalg'
        try:
            full_location = _submodule_aliases[backend, fn]

            # if explicit submodule alias given, don't use prepended location
            #     for example, ('torch', 'linalg.svd') -> torch.svd
            only_fn = fn.split('.')[-1]

        except KeyError:
            full_location = module

            # move any prepended location into the full module path
            #     e.g. 'fn=linalg.eigh' -> ['linalg', 'eigh']
            split_fn = fn.split('.')
            full_location = '.'.join([full_location] + split_fn[:-1])
            only_fn = split_fn[-1]

        # cached lookup of custom name function might take
        #     e.g. ['tensorflow', 'sum'] -> 'reduce_sum'
        fn_name = _func_aliases.get((backend, fn), only_fn)

        # import the function into the cache
        try:
            lib = importlib.import_module(full_location)
        except ImportError:
            # sometimes libraries hack an attribute to look like submodule
            mod, submod = full_location.split('.')
            lib = getattr(importlib.import_module(mod), submod)

        # check for a custom wrapper but default to identity
        wrapper = _custom_wrappers.get((backend, fn), lambda fn: fn)

        # store the function!
        lib_fn = _funcs[backend, fn] = wrapper(getattr(lib, fn_name))

    return lib_fn


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
        backend = infer_backend(args[0])
    elif isinstance(like, str):
        backend = like
    else:
        backend = infer_backend(like)

    return get_lib_fn(backend, fn)(*args, **kwargs)


# --------------------- attribute preferring functions ---------------------- #

def conj(x):
    try:
        return x.conj()
    except AttributeError:
        return do('conj', x)


def transpose(x, *args):
    return do('transpose', x, *args)


def dag(x):
    try:
        return x.H
    except AttributeError:
        return do('conj', do('transpose', x))


def real(x):
    try:
        return x.real
    except AttributeError:
        return do('real', x)


def imag(x):
    try:
        return x.imag
    except AttributeError:
        return do('imag', x)


def reshape(x, shape):
    try:
        return x.reshape(shape)
    except AttributeError:
        return do('reshape', x, shape)


# --------------------- manually specify some functions --------------------- #

def torch_conj(x):
    import warnings
    msg = ("Torch does not currently support complex data types, and has no "
           "'conj' function, using the identity instead for now.")
    warnings.warn(msg, FutureWarning)
    return x


def torch_real(x):
    import warnings
    msg = ("Torch does not currently support complex data types, and has no "
           "'real' function, using the identity instead for now.")
    warnings.warn(msg, FutureWarning)
    return x


def torch_imag(x):
    import warnings
    msg = ("Torch does not currently support complex data types, and has no "
           "'imag' function, using zeros_like instead for now.")
    warnings.warn(msg, FutureWarning)
    return do('zeros_like', x)


def torch_transpose(x, axes=None):
    if axes is None:
        axes = reversed(range(0, x.ndimension()))
    return x.permute(*axes)


_funcs['torch', 'conj'] = torch_conj
_funcs['torch', 'real'] = torch_real
_funcs['torch', 'imag'] = torch_imag
_funcs['torch', 'transpose'] = torch_transpose


# --------------- object to act as drop-in replace for numpy ---------------- #

_partial_functions = {}


class NumpyMimic:
    """A class to mimic the syntax of using `numpy` directly.
    """

    def __init__(self, submodule=None):
        self.submodule = submodule

    def __getattribute__(self, fn):

        # look out for certain submodules which are not functions
        if fn == 'linalg':
            return numpy_linalg
        if fn == 'random':
            return numpy_random

        # if this is the e.g. linalg mimic, preprend 'linalg.'
        submod = object.__getattribute__(self, 'submodule')
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
numpy_linalg = NumpyMimic('linalg')
numpy_random = NumpyMimic('random')
