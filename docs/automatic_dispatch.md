# Automatic dispatch

The primary function of [`autoray`](autoray) is to enable writing high level
array / tensor code that is agnostic to the backend arrays being supplied.
It does this via ***'automatic dispatch'***, which has a few notable
differences to other approaches:

* It is automatic - generally neither you or the backend array library needs
  to implement any dispatch logic, instead [`autoray`](autoray) finds, if
  neccesary 'translates', and then caches the relevant functions when they are
  first called.

* It is specialized for array functions and treats [`numpy`](numpy) as the
  reference interface for call signatures of 'equivalent' functions, although
  it doesn't rely or numpy or require it to be installed.

* Despite this, there is no fixed API as such - if a backend can be
  inferred, and the relevant function imported, a [`do`](autoray.do) call is
  valid.


## Basics

The main function of [`autoray`](autoray) is [`do`](autoray.do), which takes a
function name followed by `*args` and `**kwargs`, and automatically looks up
(and caches) the correct backend function. There are four main ways that the
backend is inferred:

***1. Automatic backend:***

```python
do('sqrt', x)
```

Here the backend is inferred from ``x``. By default dispatch happens on the
first argument, but various functions (such as ``'stack'`` and ``'einsum'``)
know to dispatch on other arguments.

***2. Backend 'like' another array:***

```python
do('random.normal', size=(2, 3, 4), like=x)
```

Here the backend is inferred from another array and can thus be implicitly
propagated, even when functions take no array arguments.

***3. Explicit backend:***

```python
do('einsum', eq, x, y, like='customlib')
```

Here one simply supplies the desired function backend explicitly.

***4. Context manager***

```python
with backend_like('autoray.lazy'):
    xy = do('tensordot', x, y, 1)
    z = do('trace', xy)
```

Here you set a default backend for a whole block of code. This default
overrides method 1. above but 2. and 3. still take precedence. The argument to
[`backend_like`](autoray.backend_like) can be a backend string or an example
array.


````{hint}
In all the above cases `do(fn_name, *args, like=like, **kwargs)` could be
replaced with:
```python
from autoray import numpy as np

np.fn_name(*args, like=like, **kwargs)
```
````

### Manual dispatch functions

You can manually break the process into two steps with the following functions:

* [`autoray.infer_backend`](autoray.infer_backend) - return the backend name
  for a single array.
* [`autoray.infer_backend_multi`](autoray.infer_backend_multi) - return the
  backend name based on multiple arrays.
* [`autoray.get_lib_fn`](autoray.get_lib_fn) - return the actual function for a
  given backend and function name.

If you know you are going to use a function repeatedly, you can thus avoid the
(albeit minor) overhead of dispatching each call separately, for instance:

```python
def matmul_chain(*arrays):
    # if the arrays might be a mix of backends, use infer_backend_multi,
    # but here we just dispatch on the first array
    backend = infer_backend(arrays[0])
    fn = get_lib_fn(backend, 'matmul')
    return functools.reduce(fn, arrays)
```

### Other special functions

There are a few high level functions that might be preferred to attribute
access, for reasons of consitency:

* [`autoray.shape`](autoray.shape) - return the shape of an array. In most
  cases `x.shape` is fine, but this ensures the output is `tuple[int]`
  and also works for builtins without calling `numpy`.
* [`autoray.ndim`](autoray.ndim) - return the number of dimensions of an array.
* [`autoray.size`](autoray.size) - return the total number of elements in an
  array
* [`autoray.dag`](autoray.dag) - return the adjoint of an array, i.e. the
  transpose with complex conjugation.

Functions for dealing with dtypes:

* [`autoray.get_dtype_name`](autoray.get_dtype_name) - return the name of the
  dtype of an array as a string
* [`autoray.to_backend_dtype`](autoray.to_backend_dtype) - turn a string
  specified dtype into the equivalent dtype for a given backend
* [`autoray.astype`](autoray.astype) - cast an array to a given dtype,
  specified as a string.

And for converting any array to a numpy array:

* [`autoray.to_numpy`](autoray.to_numpy)

```{hint}
All of these can be called via [`do`](autoray.do) as well, e.g.
`do('shape', x)`.
```


## Backends

In [`autoray`](autoray) a backend internally is simply specified by a string.
By default, the `backend` of an array is name of the library that the class is
defined in, and the relevant functions are assumed to be in the namespace of
`backend`. If that is the case (e.g. `cupy`), then that library is already
compatible with `autoray`. Note all backend lookups are cached on
`obj.__class__` for speed.

`autoray` also handles common cases where the functions are in a different
library or sub-module (such as `jax -> jax.numpy`). This requires a simple
mapping to be specified, which `autoray` does for various libraries.

You can explicitly register a backend name (and thus default location) for a
specific class with the function
[`register_backend`](autoray.register_backend):

```python
register_backend(mylib.myobjs.MyClass, 'mylib.myfuncs')
```
Now when `autoray` encounters an instance of `MyClass` it will look for
functions in `mylib.myfuncs` instead of `mylib`. You could also use an
arbitrary name for the backend, and then alias it to the correct location
separately.


````{note}
`autoray` is aware of the `scipy` namespace and relevant submodules for
`numpy`, `cupy`, `jax`, for example:

```python
do('scipy.linalg.exp', x)
```
````

## Functions

Once a `backend` is inferred and the location of the relevant functions is
known, `autoray` tries to import and cache the relevant function from that
namespace. Many libraries (e.g. `cupy`, `dask`, `jax`, `autograd`, `sparse`,
...) actively mirror the `numpy` API, so there is little else to be done.

Some other libraries (e.g. `tensorflow`, `pytorch`, ...) diverge from the
`numpy` API more, and yet have largely equivalent functions, simply defined in
slight different places with different names and / or signatures. `autoray`
has a simple translation mechanism for:

* when functions are in a different module (e.g.
  `'trace' -> tensorflow.linalg.trace`)
* when functions have a different name (e.g. `'sum' -> tensorflow.reduce_sum`)
* when functions have a different signature (e.g.
  `tensordot(a, b, axes) -> torch.tensordot(a, b, dims)`)

If you want to directly provide a missing or *alternative* implementation of
some function for a particular backend you can swap one in with
[`register_function`](autoray.register_function):

```python
def my_custom_torch_svd(x):
    import torch

    print('Hello SVD!')
    u, s, v = torch.svd(x)

    return u, s, v.T

ar.register_function('torch', 'linalg.svd', my_custom_torch_svd)

x = ar.do('random.uniform', size=(3, 4), like='torch')

ar.do('linalg.svd', x)
# Hello SVD!
# (tensor([[-0.5832,  0.6188, -0.5262],
#          [-0.5787, -0.7711, -0.2655],
#          [-0.5701,  0.1497,  0.8078]]),
#  tensor([2.0336, 0.8518, 0.4572]),
#  tensor([[-0.4568, -0.3166, -0.6835, -0.4732],
#          [-0.5477,  0.2825, -0.2756,  0.7377],
#          [ 0.2468, -0.8423, -0.0993,  0.4687]]))
```

If you want to make use of the existing function you can supply ``wrap=True``
in which case the custom function supplied should act like a decorator:

```python
def my_custom_sum_wrapper(old_fn):

    def new_fn(*args, **kwargs):
        print('Hello sum!')
        return old_fn(*args **kwargs)

    return new_fn

ar.register_function('torch', 'sum', my_custom_sum_wrapper, wrap=True)

ar.do('sum', x)
# Hello sum!
# tensor(5.4099)
```

Though be careful, if you call [`register_function`](autoray.register_function)
again it will now wrap the
*new* function! Note you can combine
[`register_backend`](autoray.register_backend) and
[`register_function`](autoray.register_function) to dynamically define array
types and functions from anywhere. See also
[`register_dispatch`](autoray.register_dispatch) for
controlling which arguments are used to infer the backend for any function.


### Composing new functions

Sometimes you want to define a function that is composed of many array
functions, but you want to dispatch at the level of the whole block, not each
individual call, or indeed use a completely different implementation. For
instance, you might want to use a [`numba`](https://numba.pydata.org/) or
[`pythran`](https://pythran.readthedocs.io/en/latest/) compiled version for
`numpy`.

The [`autoray.compose`](autoray.compose) function allows you to do this. You
decorate a function, that forms the default implementation, then you can
register alternative implementations for specific backends. For instance:

```python
from autoray import compose
from numba import njit

@compose
def my_func(x):
    # get how many elements are needed to sum to 20
    return ar.do('sum', ar.do('cumsum', x, 0) < 20)

# register a numba implementation
@my_func.register('numpy')
@njit
def my_func_numba(x):
    s = 0.0
    i = 0
    while s < 20:
        s += x[i]
        i += 1
    return i - 1

# any calls like this now dispatch to my_func_numba
do('my_func', x_numpy)
```


### Deviations from `numpy`

As stated above, `autoray` does not have an explicit API, but where there exist
equivalent functions, `autoray` uses the call signature of `numpy` as a
reference. The following are deviations from this:

* `do('linalg.svd', x)` - `autoray` defaults to `full_matrices=False`, since
  this is generally always desired, and many libraries do not even support
  `full_matrices=True`.


-------------------------------------------------------------------------------

## Comparison to alternatives

* The ``__array_function__`` protocol has been
  [suggested](https://www.numpy.org/neps/nep-0018-array-function-protocol.html)
  and now implemented in ``numpy``. This will hopefully eventually be a nice
  solution for array dispatch. However, it requires the backend library to
  implement the protocol, which has not been done for common libraries yet.

* The [uarray](https://github.com/Quansight-Labs/uarray) project appears to
  have similar goals but is still being developed.

* [`functools.singledispatch`](https://docs.python.org/3/library/functools.html#functools.singledispatch) is a general *single* dispatch mechanism, but it is slower
  and requires the user to explicitly register each function they want to
  dispatch on.

* [`plum`](https://github.com/beartype/plum) is a general *multiple* dispatch
  mechanism, but again it would require registering every function for every
  backend explicitly.
