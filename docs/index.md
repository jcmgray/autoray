# Welcome to autoray's documentation!
[![tests](https://github.com/jcmgray/autoray/actions/workflows/tests.yml/badge.svg)](https://github.com/jcmgray/autoray/actions/workflows/tests.yml) [![codecov](https://codecov.io/gh/jcmgray/autoray/branch/master/graph/badge.svg?token=Q5evNiuT9S)](https://codecov.io/gh/jcmgray/autoray) [![Codacy Badge](https://app.codacy.com/project/badge/Grade/ba896d74c4954dd58da01df30c7bf326)](https://www.codacy.com/gh/jcmgray/autoray/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=jcmgray/autoray&amp;utm_campaign=Badge_Grade) [![PyPI](https://img.shields.io/pypi/v/autoray?color=teal)](https://pypi.org/project/autoray/) [![Anaconda-Server Badge](https://anaconda.org/conda-forge/autoray/badges/version.svg)](https://anaconda.org/conda-forge/autoray)

[`autoray`](autoray) is a lightweight python AUTOmatic-arRAY library for
abstracting your tensor operations. Primarily it provides an
[*automatic* dispatch mechanism](automatic_dispatch) that means you can
write backend agnostic code that works for:

* [numpy](https://github.com/numpy/numpy)
* [pytorch](https://pytorch.org/)
* [jax](https://github.com/google/jax)
* [cupy](https://github.com/cupy/cupy)
* [dask](https://github.com/dask/dask)
* [autograd](https://github.com/HIPS/autograd)
* [tensorflow](https://github.com/tensorflow/tensorflow)
* [sparse](https://sparse.pydata.org/)
* [mars](https://github.com/mars-project/mars)
* ... and indeed **any** library that provides a numpy-*ish* api, even if it
  knows nothing about `autoray`.

Beyond that, abstracting the array interface allows you to:

* *swap [custom versions of functions](automatic_dispatch.md#functions)
  for specific backends*
* *trace through computations [lazily](lazy_computation) without actually
  running them*
* *automatically [share intermediates and fold constants](lazy_computation)
  in computations*
* *compile functions with a [unified interface](compilation) for different
  backends*


## Basic usage

The main function of `autoray` is [`do`](autoray.do), which takes a function
name followed by `*args` and `**kwargs`, and automatically looks up (and
caches) the correct function to match the equivalent numpy call:

```python
from autoray as ar

def noised_svd(x):
    # automatic dispatch based on supplied array
    U, s, VH = ar.do('linalg.svd', x)

    # automatic dispatch based on different array
    sn = s + 0.1 * ar.do('random.normal', size=ar.shape(s), like=s)

    # automatic dispatch for multiple arrays for certain functions
    return ar.do('einsum', 'ij,j,jk->ik', U, sn, VH)

# explicit backend given by string
x = ar.do('random.uniform', size=(100, 100), like="torch")

# this function now works for any backend
y = noised_svd(x)

# explicit inference of backend from array
ar.infer_backend(y)
# 'torch'
```

If you don't like the explicit [`do`](autoray.do) syntax, or simply want a
drop-in replacement for existing code, you can also import the `autoray.numpy`
module:

```{code-block} python
from autoray import numpy as np

# set a temporary default backend
with ar.backend_like('cupy'):
    z = np.ones((3, 4), dtype='float32')

np.exp(z)
# array([[2.7182817, 2.7182817, 2.7182817, 2.7182817],
#        [2.7182817, 2.7182817, 2.7182817, 2.7182817],
#        [2.7182817, 2.7182817, 2.7182817, 2.7182817]], dtype=float32)
```

Custom backends and functions can be dynamically registered with:

* [`register_backend`](autoray.register_backend)
* [`register_function`](autoray.register_function)

---

## Advanced details

```{toctree}
:caption: Guides
:maxdepth: 2

installation.md
automatic_dispatch.md
lazy_computation.ipynb
compilation.ipynb
development.md
```

```{toctree}
:caption: Links
:hidden:

GitHub Repository <https://github.com/jcmgray/autoray>
```
