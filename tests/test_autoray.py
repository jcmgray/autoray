import importlib

import pytest

import autoray


# find backends to tests
BACKENDS = ['numpy']
for lib in ['cupy', 'dask', 'tensorflow']:
    if importlib.util.find_spec(lib):
        BACKENDS.append(lib)
        if lib == 'tensorflow':
            import tensorflow as tf
            tf.enable_eager_execution()


def gen_rand(shape, backend):
    if backend == 'numpy':
        import numpy as np
        return np.random.uniform(size=shape)

    if backend == 'dask':
        import dask.array as da
        return da.random.uniform(size=shape, chunks=-1)

    if backend == 'tensorflow':
        import tensorflow as tf
        return tf.random.uniform(shape=shape, dtype='float64')

    if backend == 'cupy':  # pragma: no cover
        import cupy as cp
        return cp.random.uniform(size=shape, dtype='float64')


def to_numpy(array, backend=None):

    if backend is None:
        backend = autoray.infer_backend(array)

    if backend == 'dask':
        return array.compute()

    if backend == 'tensorflow':
        return array.numpy()

    if backend == 'cupy':  # pragma: no cover
        return array.get()

    return array


@pytest.mark.parametrize('backend', BACKENDS)
@pytest.mark.parametrize('fn', ['sqrt', 'exp', 'sum'])
def test_basic(backend, fn):
    x = gen_rand((2, 3, 4), backend)
    y = autoray.do(fn, x)
    assert autoray.infer_backend(x) == autoray.infer_backend(y) == backend


@pytest.mark.parametrize('backend', BACKENDS)
@pytest.mark.parametrize('fn,args', [
    (autoray.conj, []),
    (autoray.transpose, []),
    (autoray.real, []),
    (autoray.imag, []),
    (autoray.reshape, [(5, 3)]),
])
def test_attribute_prefs(backend, fn, args):
    x = gen_rand((3, 5), backend)
    y = fn(x, *args)
    assert autoray.infer_backend(x) == autoray.infer_backend(y) == backend


def modified_gram_schmidt(X):

    Q = []
    for j in range(0, X.shape[0]):

        q = X[j, :]
        for i in range(0, j):
            rij = autoray.do('tensordot', autoray.do('conj', Q[i]), q, 1)
            q = q - rij * Q[i]

        rjj = autoray.do('linalg.norm', q, 2)
        Q.append(q / rjj)

    return autoray.do('stack', Q, axis=0, like=X)


@pytest.mark.parametrize('backend', BACKENDS)
def test_mgs(backend):
    x = gen_rand((3, 5), backend)
    Ux = modified_gram_schmidt(x)
    y = autoray.do('sum', Ux @ autoray.dag(Ux))
    assert to_numpy(y, backend) == pytest.approx(3)


def modified_gram_schmidt_np_mimic(X):
    from autoray import numpy as np
    print(np)

    Q = []
    for j in range(0, X.shape[0]):

        q = X[j, :]
        for i in range(0, j):
            rij = np.tensordot(np.conj(Q[i]), q, 1)
            q = q - rij * Q[i]

        rjj = np.linalg.norm(q, 2)
        Q.append(q / rjj)

    return np.stack(Q, axis=0, like=X)


@pytest.mark.parametrize('backend', BACKENDS)
def test_mgs_np_mimic(backend):
    x = gen_rand((3, 5), backend)
    Ux = modified_gram_schmidt_np_mimic(x)
    y = autoray.do('sum', Ux @ autoray.dag(Ux))
    assert to_numpy(y, backend) == pytest.approx(3)


@pytest.mark.parametrize('backend', BACKENDS)
def test_linalg_svd_square(backend):
    x = gen_rand((5, 4), backend)
    U, s, V = autoray.do('linalg.svd', x)
    assert (
        autoray.infer_backend(x) ==
        autoray.infer_backend(U) ==
        autoray.infer_backend(s) ==
        autoray.infer_backend(V) ==
        backend
    )
    y = U @ autoray.do('diag', s, like=x) @ V
    diff = autoray.do('sum', abs(y - x))
    assert to_numpy(diff, backend) < 1e-8


@pytest.mark.parametrize('backend', BACKENDS)
def test_translator_random_uniform(backend):
    from autoray import numpy as apn

    x = apn.random.uniform(low=-10, size=(4, 5), like=backend)
    assert (to_numpy(x, backend) > -10).all()
    assert (to_numpy(x, backend) < 1.0).all()


@pytest.mark.parametrize('backend', BACKENDS)
def test_translator_random_normal(backend):
    from autoray import numpy as anp

    x = anp.random.normal(100.0, 0.1, size=(4, 5), like=backend)
    assert (to_numpy(x, backend) > 90.0).all()
    assert (to_numpy(x, backend) < 110.0).all()

    if backend == 'tensorflow':
        x32 = autoray.do('random.normal', 100.0, 0.1, dtype='float32',
                         size=(4, 5), like=backend)
        assert x32.dtype == 'float32'
        assert (to_numpy(x32, backend) > 90.0).all()
        assert (to_numpy(x32, backend) < 110.0).all()


@pytest.mark.parametrize('backend', BACKENDS)
def test_tril(backend):
    x = autoray.do('random.uniform', size=(4, 4), like=backend)
    xl = autoray.do('tril', x)
    xln = to_numpy(xl)
    assert xln[0, 1] == 0.0
    assert (xln > 0.0).sum() == 10
    xl = autoray.do('tril', x, k=1)
    xln = to_numpy(xl)
    assert xln[0, 1] != 0.0
    assert xln[0, 2] == 0.0
    assert (xln > 0.0).sum() == 13

    if backend == 'tensorflow':
        with pytest.raises(ValueError):
            autoray.do('tril', x, -1)


@pytest.mark.parametrize('backend', BACKENDS)
def test_triu(backend):
    x = autoray.do('random.uniform', size=(4, 4), like=backend)
    xl = autoray.do('triu', x)
    xln = to_numpy(xl)
    assert xln[1, 0] == 0.0
    assert (xln > 0.0).sum() == 10
    xl = autoray.do('triu', x, k=-1)
    xln = to_numpy(xl)
    assert xln[1, 0] != 0.0
    assert xln[2, 0] == 0.0
    assert (xln > 0.0).sum() == 13

    if backend == 'tensorflow':
        with pytest.raises(ValueError):
            autoray.do('triu', x, 1)


def test_pseudo_submodules():
    x = gen_rand((2, 3), 'numpy')
    xT = autoray.do('numpy.transpose', x, like='autoray')
    assert xT.shape == (3, 2)
