.. raw:: html

    <img src="https://github.com/jcmgray/autoray/blob/master/docs/images/autoray-header.png" width="500px">


A lightweight python AUTOmatic-arRAY library. Write numeric code that works for:

* `numpy <https://github.com/numpy/numpy>`_
* `cupy <https://github.com/cupy/cupy>`_
* `dask <https://github.com/dask/dask>`_
* `tensorflow <https://github.com/tensorflow/tensorflow>`_
* `autograd <https://github.com/HIPS/autograd>`_
* `jax <https://github.com/google/jax>`_
* `mars <https://github.com/mars-project/mars>`_
* ... and indeed **any** library that provides a numpy-*ish* api.


.. image:: https://travis-ci.org/jcmgray/autoray.svg?branch=master
  :target: https://travis-ci.org/jcmgray/autoray
  :alt: Travis-CI
.. image:: https://codecov.io/gh/jcmgray/autoray/branch/master/graph/badge.svg
  :target: https://codecov.io/gh/jcmgray/autoray
  :alt: Code Coverage
.. image:: https://img.shields.io/lgtm/grade/python/g/jcmgray/autoray.svg
  :target: https://lgtm.com/projects/g/jcmgray/autoray/
  :alt: Code Quality

For example consider this function that orthogonalizes a matrix using the modified Gram-Schmidt algorithm:

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

    >>> from autoray import numpy as np

    >>> import tensorflow as tf
    >>> x = tf.random.uniform(shape=(2, 3, 4))

    >>> np.tensordot(x, x, [(2, 1), (2, 1)])
    <tf.Tensor 'Tensordot:0' shape=(2, 2) dtype=float32>

    >>> np.eye(3, like=x)  # many functions obviously can't dispatch without the `like` keyword
    <tf.Tensor 'eye/MatrixDiag:0' shape=(3, 3) dtype=float32>

Of course complete compatibility is not going to be possible for all functions, operations and libraries, but ``autoray`` hopefully makes the job much easier.


**How does it work?**

``autoray`` works using essentially a single dispatch mechanism on the first  argument for ``do``, or the ``like`` keyword argument if specified, fetching functions from the whichever module defined that supplied array. Additionally, it caches a few custom translations and lookups so as to handle libraries like ``tensorflow`` that don't exactly replicate the ``numpy`` api (for example ``sum`` gets translated to ``tensorflow.reduce_sum``).

**Alternatives**

* The ``__array_function__`` protocol has been `suggested <https://www.numpy.org/neps/nep-0018-array-function-protocol.html>`_ and now implemented in ``numpy``. Hopefully this will eventually negate the need for ``autoray``. On the other hand, third party libraries themselves need to implement the interface, which has not been done, for example, in ``tensorflow`` yet.
* The `uarray <https://github.com/Quansight-Labs/uarray>`_ project aims to develop a generic array interface but comes with the warning *"This is experimental and very early research code. Don't use this."*.


Installation
------------

You can install ``autoray`` as standard with ``pip``. Alternatively, simply copy the monolithic ``autoray.py`` into your project internally (if dependencies aren't your thing).


Contributing
------------

Pull requests such as extra translations are very welcome!
