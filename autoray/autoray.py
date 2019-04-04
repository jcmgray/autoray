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
    ('numpy', 'linalg.expm'): 'scipy',
    ('tensorflow', 'trace'): 'tensorflow.linalg',
}


# lookup for when functions don't have the same name
_func_aliases = {
    ('tensorflow', 'sum'): 'reduce_sum',
    ('tensorflow', 'min'): 'reduce_min',
    ('tensorflow', 'max'): 'reduce_max',
    ('tensorflow', 'mean'): 'reduce_mean',
    ('tensorflow', 'prod'): 'reduce_prod',
    ('tensorflow', 'concatenate'): 'concat',
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


# custom wrapper for when functions don't just have different location or name
_custom_wrappers = {
    ('numpy', 'linalg.svd'): svd_not_full_matrices_wrapper,
    ('cupy', 'linalg.svd'): svd_not_full_matrices_wrapper,
    ('tensorflow', 'linalg.svd'): svd_sUV_to_UsVH_wrapper,
}


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
        submodule_name = _submodule_aliases.get((backend, fn), module)

        # parse out extra submodules
        #     e.g. 'fn=linalg.eigh' -> ['linalg', 'eigh']
        split_fn = fn.split('.')
        submodule_name = '.'.join([submodule_name] + split_fn[:-1])
        only_fn = split_fn[-1]

        # cached lookup of custom name function might take
        #     e.g. ['tensorflow', 'sum'] -> 'reduce_sum'
        fn_name = _func_aliases.get((backend, fn), only_fn)

        # import the function into the cache
        lib = importlib.import_module(submodule_name)

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
    try:
        return x.transpose(*args)
    except AttributeError:
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


# --------------- object to act as drop-in replace for numpy ---------------- #

_partial_functions = {}


class NumpyMimic:
    """A class to mimic the syntax of using `numpy` directly.
    """

    def __init__(self, submodule=None):
        self.submodule = submodule

    def __getattribute__(self, fn):

        # know that linalg is a submodule rather than a function
        if fn == 'linalg':
            return numpy_linalg

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
