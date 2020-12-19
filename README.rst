.. raw:: html

    <img src="https://github.com/jcmgray/autoray/blob/master/docs/images/autoray-header.png" width="500px">


A lightweight python AUTOmatic-arRAY library. Write numeric code that works for:

* `numpy <https://github.com/numpy/numpy>`_
* `cupy <https://github.com/cupy/cupy>`_
* `dask <https://github.com/dask/dask>`_
* `autograd <https://github.com/HIPS/autograd>`_
* `jax <https://github.com/google/jax>`_
* `mars <https://github.com/mars-project/mars>`_
* `tensorflow <https://github.com/tensorflow/tensorflow>`_
* `pytorch <https://pytorch.org/>`_
* ... and indeed **any** library that provides a numpy-*ish* api.

.. image:: https://dev.azure.com/autoray-org/autoray/_apis/build/status/jcmgray.autoray?branchName=master
  :target: https://dev.azure.com/autoray-org/autoray/_build/latest?definitionId=1&branchName=master
  :alt: Azure Pipelines
.. image:: https://codecov.io/gh/jcmgray/autoray/branch/master/graph/badge.svg
  :target: https://codecov.io/gh/jcmgray/autoray
  :alt: Code Coverage
.. image:: https://img.shields.io/lgtm/grade/python/g/jcmgray/autoray.svg
  :target: https://lgtm.com/projects/g/jcmgray/autoray/
  :alt: Code Quality

As an example consider this function that orthogonalizes a matrix using the modified Gram-Schmidt algorithm:

.. code:: python3

    from autoray import do

    def modified_gram_schmidt(X):

        Q = []
        for j in range(0, X.shape[0]):

            q = X[j, :]
            for i in range(0, j):
                rij = do('tensordot', do('conj', Q[i]), q, 1)
                q = q - rij * Q[i]

            rjj = do('linalg.norm', q, 2)
            Q.append(q / rjj)

        return do('stack', Q, axis=0, like=X)

Which is now compatible with **all** of the above mentioned libraries! (N.B. this particular example is also probably slow). If you don't like the explicit ``do`` syntax, then you can import the fake ``numpy`` object as a **drop-in replacement** instead:

.. code:: python3

    from autoray import numpy as np

    x = np.random.uniform(size=(2, 3, 4), like='tensorflow')
    np.tensordot(x, x, [(2, 1), (2, 1)])
    # <tf.Tensor 'Tensordot:0' shape=(2, 2) dtype=float32>

    np.eye(3, like=x)  # many functions obviously can't dispatch without the `like` keyword
    # <tf.Tensor 'eye/MatrixDiag:0' shape=(3, 3) dtype=float32>

Of course complete compatibility is not going to be possible for all functions, operations and libraries, but ``autoray`` hopefully makes the job much easier. Of the above, ``tensorflow`` has *quite* a different interface and ``pytorch`` probably the *most* different. Whilst for example not every function will work out-of-the-box for these two, ``autoray`` is also designed with the easy addition of new functions in mind (for example adding new translations is often a one-liner).

**How does it work?**

``autoray`` works using essentially a single dispatch mechanism on the first  argument for ``do``, or the ``like`` keyword argument if specified, fetching functions from the whichever module defined that supplied array. Additionally, it caches a few custom translations and lookups so as to handle libraries like ``tensorflow`` that don't exactly replicate the ``numpy`` api (for example ``sum`` gets translated to ``tensorflow.reduce_sum``).

Special Functions
-----------------

The main function is ``do``, but the following special (i.e. not in ``numpy``) functions are also implemented that may be useful:

* ``autoray.infer_backend`` - check what library is being inferred for a given array
* ``autoray.to_backend_dtype`` - convert a string specified dtype like ``'float32'`` to ``torch.float32`` for example
* ``autoray.get_dtype_name`` - convert a backend dtype back into the equivalent string specifier like ``'complex64'``
* ``autoray.astype`` - backend agnostic dtype conversion of arrays
* ``autoray.to_numpy`` - convert any array to a ``numpy.ndarray``

Here are all of those in action:

.. code:: python3

    import autoray as ar

    backend = 'torch'
    dtype = ar.to_backend_dtype('float64', like=backend)
    dtype
    # torch.float64

    x = ar.do('random.normal', size=(4,), dtype=dtype, like=backend)
    x
    # tensor([ 0.0461,  0.3028,  0.1790, -0.1494], dtype=torch.float64)

    ar.infer_backend(x)
    # 'torch'

    ar.get_dtype_name(x)
    # 'float64'

    x32 = ar.astype(x, 'float32')
    ar.to_numpy(x32)
    # array([ 0.04605161,  0.30280888,  0.17903718, -0.14936243], dtype=float32)

Registering Your Own functions
------------------------------

If you want to directly provide a missing or alternative implementation of some function for a particular backend you can do so with ``autoray.register_function``:

.. code:: python3

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

If you want to make use of the existing function you can supply ``wrap=True`` in which case the custom function supplied should act like a decorator:

.. code:: python3

    def my_custom_sum_wrapper(old_fn):

        def new_fn(*args, **kwargs):
            print('Hello sum!')
            return old_fn(*args **kwargs)

        return new_fn

    ar.register_function('torch', 'sum', my_custom_sum_wrapper, wrap=True)

    ar.do('sum', x)
    # Hello sum!
    # tensor(5.4099)

Though be careful, if you call ``register_function`` again it will now wrap the *new* function!

Deviations from `numpy`
=======================

`autoray` doesn't have an API as such, since it is essentially just a fancy single dispatch mechanism.
On the other hand, where translations *are* in place, they generally use the numpy API. So
``autoray.do('stack', arrays=pytorch_tensors, axis=0)``
gets automatically translated into
``torch.stack(tensors=pytorch_tensors, dims=0)``
and so forth.

Currently the one place this isn't true is ``autoray.do('linalg.svd', x)`` where instead ``full_matrices=False``
is used as the default since this generally makes more sense and many libraries don't even implement the other case.
Autoray also dispatches ``'linalg.expm'`` for ``numpy`` arrays to ``scipy``, and may well do with other scipy-only functions at some point.

Installation
------------

You can install ``autoray`` via `conda-forge <https://conda-forge.org/>`_ as well as with ``pip``. Alternatively, simply copy the monolithic ``autoray.py`` into your project internally (if dependencies aren't your thing).

**Alternatives**

* The ``__array_function__`` protocol has been `suggested <https://www.numpy.org/neps/nep-0018-array-function-protocol.html>`_ and now implemented in ``numpy``. Hopefully this will eventually negate the need for ``autoray``. On the other hand, third party libraries themselves need to implement the interface, which has not been done, for example, in ``tensorflow`` yet.
* The `uarray <https://github.com/Quansight-Labs/uarray>`_ project aims to develop a generic array interface but comes with the warning *"This is experimental and very early research code. Don't use this."*.

Contributing
------------

Pull requests such as extra translations are very welcome!
