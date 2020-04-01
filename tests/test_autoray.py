import importlib

import pytest

import autoray as ar


# find backends to tests
BACKENDS = ['numpy']
for lib in ['cupy', 'dask', 'tensorflow', 'torch', 'mars', 'jax']:
    if importlib.util.find_spec(lib):
        BACKENDS.append(lib)

        if lib == 'jax':
            from jax.config import config
            config.update("jax_enable_x64", True)


JAX_RANDOM_KEY = None


def gen_rand(shape, backend, dtype='float64'):
    if backend == 'jax':
        from jax import random as jrandom

        global JAX_RANDOM_KEY

        if JAX_RANDOM_KEY is None:
            JAX_RANDOM_KEY = jrandom.PRNGKey(42)
        JAX_RANDOM_KEY, subkey = jrandom.split(JAX_RANDOM_KEY)

        return jrandom.uniform(subkey, shape=shape, dtype=dtype)

    x = ar.do('random.uniform', size=shape, like=backend)
    x = ar.astype(x, ar.to_backend_dtype(dtype, backend))
    assert ar.get_dtype_name(x) == dtype
    return x


@pytest.mark.parametrize('backend', BACKENDS)
@pytest.mark.parametrize('fn', ['sqrt', 'exp', 'sum'])
def test_basic(backend, fn):
    x = gen_rand((2, 3, 4), backend)
    y = ar.do(fn, x)
    assert ar.infer_backend(x) == ar.infer_backend(y) == backend


@pytest.mark.parametrize('backend', BACKENDS)
@pytest.mark.parametrize('fn,args', [
    (ar.conj, []),
    (ar.transpose, []),
    (ar.real, []),
    (ar.imag, []),
    (ar.reshape, [(5, 3)]),
])
def test_attribute_prefs(backend, fn, args):
    x = gen_rand((3, 5), backend)
    y = fn(x, *args)
    assert ar.infer_backend(x) == ar.infer_backend(y) == backend


def modified_gram_schmidt(X):

    Q = []
    for j in range(0, X.shape[0]):

        q = X[j, :]
        for i in range(0, j):
            rij = ar.do('tensordot', ar.do('conj', Q[i]), q, 1)
            q = q - rij * Q[i]

        rjj = ar.do('linalg.norm', q, 2)
        Q.append(q / rjj)

    return ar.do('stack', Q, axis=0, like=X)


@pytest.mark.parametrize('backend', BACKENDS)
def test_mgs(backend):
    x = gen_rand((3, 5), backend)
    Ux = modified_gram_schmidt(x)
    y = ar.do('sum', Ux @ ar.dag(Ux))
    assert ar.to_numpy(y) == pytest.approx(3)


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
    y = ar.do('sum', Ux @ ar.dag(Ux))
    assert ar.to_numpy(y) == pytest.approx(3)


@pytest.mark.parametrize('backend', BACKENDS)
def test_linalg_svd_square(backend):
    x = gen_rand((5, 4), backend)
    U, s, V = ar.do('linalg.svd', x)
    assert (
        ar.infer_backend(x) ==
        ar.infer_backend(U) ==
        ar.infer_backend(s) ==
        ar.infer_backend(V) ==
        backend
    )
    y = U @ ar.do('diag', s, like=x) @ V
    diff = ar.do('sum', abs(y - x))
    assert ar.to_numpy(diff) < 1e-8


@pytest.mark.parametrize('backend', BACKENDS)
def test_translator_random_uniform(backend):
    from autoray import numpy as anp

    x = anp.random.uniform(low=-10, size=(4, 5), like=backend)
    assert (ar.to_numpy(x) > -10).all()
    assert (ar.to_numpy(x) < 1.0).all()

    # test default single scalar
    x = anp.random.uniform(low=1000, high=2000, like=backend)
    assert 1000 <= ar.to_numpy(x) < 2000


@pytest.mark.parametrize('backend', BACKENDS)
def test_translator_random_normal(backend):
    from autoray import numpy as anp

    x = anp.random.normal(100.0, 0.1, size=(4, 5), like=backend)
    assert (ar.to_numpy(x) > 90.0).all()
    assert (ar.to_numpy(x) < 110.0).all()

    if backend == 'tensorflow':
        x32 = ar.do('random.normal', 100.0, 0.1, dtype='float32',
                    size=(4, 5), like=backend)
        assert x32.dtype == 'float32'
        assert (ar.to_numpy(x32) > 90.0).all()
        assert (ar.to_numpy(x32) < 110.0).all()

    # test default single scalar
    x = anp.random.normal(loc=1500, scale=10, like=backend)
    assert 1000 <= ar.to_numpy(x) < 2000


@pytest.mark.parametrize('backend', BACKENDS)
def test_tril(backend):
    x = gen_rand((4, 4), backend)
    xl = ar.do('tril', x)
    xln = ar.to_numpy(xl)
    assert xln[0, 1] == 0.0
    assert (xln > 0.0).sum() == 10
    xl = ar.do('tril', x, k=1)
    xln = ar.to_numpy(xl)
    assert xln[0, 1] != 0.0
    assert xln[0, 2] == 0.0
    assert (xln > 0.0).sum() == 13

    if backend == 'tensorflow':
        with pytest.raises(ValueError):
            ar.do('tril', x, -1)


@pytest.mark.parametrize('backend', BACKENDS)
def test_triu(backend):
    x = gen_rand((4, 4), backend)
    xl = ar.do('triu', x)
    xln = ar.to_numpy(xl)
    assert xln[1, 0] == 0.0
    assert (xln > 0.0).sum() == 10
    xl = ar.do('triu', x, k=-1)
    xln = ar.to_numpy(xl)
    assert xln[1, 0] != 0.0
    assert xln[2, 0] == 0.0
    assert (xln > 0.0).sum() == 13

    if backend == 'tensorflow':
        with pytest.raises(ValueError):
            ar.do('triu', x, 1)


@pytest.mark.parametrize('backend', BACKENDS)
@pytest.mark.parametrize('shape', [(4, 3), (4, 4), (3, 4)])
def test_qr_thin_square_fat(backend, shape):
    x = gen_rand(shape, backend)
    Q, R = ar.do('linalg.qr', x)
    xn, Qn, Rn = map(ar.to_numpy, (x, Q, R))
    assert ar.do('allclose', xn, Qn @ Rn)


@pytest.mark.parametrize('backend', BACKENDS)
def test_count_nonzero(backend):
    if backend == 'mars':
        import mars
        if mars._version.version_info < (0, 4, 0, ''):
            pytest.xfail('mars count_nonzero bug fixed in version 0.4.')

    x = ar.do('array', [0, 1, 2, 0, 3], like=backend)
    nz = ar.do('count_nonzero', x)
    assert ar.to_numpy(nz) == 3

    x = ar.do('array', [0., 1., 2., 0., 3.], like=backend)
    nz = ar.do('count_nonzero', x)
    assert ar.to_numpy(nz) == 3

    x = ar.do('array', [False, True, True, False, True], like=backend)
    nz = ar.do('count_nonzero', x)
    assert ar.to_numpy(nz) == 3


def test_pseudo_submodules():
    x = gen_rand((2, 3), 'numpy')
    xT = ar.do('numpy.transpose', x, like='autoray')
    assert xT.shape == (3, 2)


@pytest.mark.parametrize('backend', BACKENDS)
@pytest.mark.parametrize('creation', ['ones', 'zeros'])
@pytest.mark.parametrize('dtype', ['float32', 'float64',
                                   'complex64', 'complex128'])
def test_dtype_specials(backend, creation, dtype):
    import numpy as np
    x = ar.do(creation, shape=(2, 3), like=backend)

    if backend == 'torch' and 'complex' in dtype:
        pytest.xfail("Pytorch doesn't support complex numbers yet...")

    x = ar.astype(x, dtype)
    assert ar.get_dtype_name(x) == dtype
    x = ar.to_numpy(x)
    assert isinstance(x, np.ndarray)
    assert ar.get_dtype_name(x) == dtype


@pytest.mark.parametrize('backend', BACKENDS)
@pytest.mark.parametrize('real_dtype', ['float32', 'float64'])
def test_complex_creation(backend, real_dtype):
    if backend == 'torch':
        pytest.xfail("Pytorch doesn't support complex numbers yet...")

    x = ar.do(
        'complex',
        ar.astype(ar.do('random.normal', size=(3, 4),
                        like=backend), real_dtype),
        ar.astype(ar.do('random.normal', size=(3, 4),
                        like=backend), real_dtype)
    )
    assert ar.get_dtype_name(x) == {'float32': 'complex64',
                                    'float64': 'complex128'}[real_dtype]


@pytest.mark.parametrize('backend', BACKENDS)
def test_register_function(backend):
    x = ar.do('ones', shape=(2, 3), like=backend)

    def direct_fn(x):
        return 1

    # first test we can provide the function directly
    ar.register_function(backend, 'test_register', direct_fn)
    assert ar.do('test_register', x) == 1

    def wrap_fn(fn):

        def new_fn(*args, **kwargs):
            res = fn(*args, **kwargs)
            return res + 1

        return new_fn

    # then check we can wrap the old (previous) function
    ar.register_function(backend, 'test_register', wrap_fn, wrap=True)
    assert ar.do('test_register', x) == 2
