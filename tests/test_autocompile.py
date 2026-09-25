import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
from numpy.testing import assert_allclose

from autoray import autojit, do, infer_backend, shape, to_numpy

from .conftest import gen_params, gen_rand

_COMPILE_BACKENDS = ["jax", "torch", "tensorflow"]
BACKENDS = gen_params(backends=_COMPILE_BACKENDS)


def modified_gram_schmidt(X):
    Q = []
    for j in range(0, shape(X)[0]):
        q = X[j, :]
        for i in range(0, j):
            rij = do("tensordot", do("conj", Q[i]), q, axes=1)
            q = q - rij * Q[i]
        rjj = do("linalg.norm", q, 2)
        Q.append(q / rjj)
    return do("stack", tuple(Q), axis=0)


@pytest.fixture
def mgs_case():
    x = gen_rand((10, 10), "numpy")
    y = modified_gram_schmidt(x)
    return x, y


@pytest.mark.parametrize("share_intermediates", [False, True])
@pytest.mark.parametrize("nested", [False, True])
def test_compile_python(mgs_case, share_intermediates, nested):
    x, y = mgs_case
    compiler_opts = {"python": {"share_intermediates": share_intermediates}}
    mgs = autojit(modified_gram_schmidt, compiler_opts=compiler_opts)
    if nested:
        mgs = autojit(mgs, compiler_opts=compiler_opts)
    y2 = mgs(x)
    assert_allclose(y, y2)


@pytest.mark.parametrize("backend", BACKENDS)
def test_others_numpy(backend, mgs_case):
    x, y = mgs_case
    mgs = autojit(modified_gram_schmidt)
    y2 = mgs(x, backend=backend)
    assert infer_backend(y2) == "numpy"
    assert_allclose(y, y2)


@pytest.mark.parametrize("backend", BACKENDS)
def test_autodispatch(backend, mgs_case):
    x, y = mgs_case
    x = do("array", x, like=backend)
    mgs = autojit(modified_gram_schmidt)
    y2 = mgs(x, backend=backend)
    assert infer_backend(y2) == backend
    assert_allclose(y, to_numpy(y2))


def test_complicated_signature():
    @autojit
    def foo(a, b, c):
        a1, a2 = a
        b1 = b["1"]
        c1, c2 = c["sub"]
        return do("sum", do("stack", (a1, a2, b1, c1, c2)), axis=0)

    x = do("random.uniform", size=(5, 7), like="numpy")
    y = foo((x[0, :], x[1, :]), {"1": x[2, :]}, c={"sub": (x[3, :], x[4, :])})
    assert_allclose(y, x.sum(0))


def test_astype_lazy_dtype():
    @autojit
    def foo(x, y):
        return do("astype", x, y.dtype) + y

    x = gen_rand((3, 4), "numpy", dtype="float32")
    y = gen_rand((3, 4), "numpy", dtype="float64")
    z = foo(x, y)
    assert z.dtype.name == "float64"
    assert_allclose(z, x.astype("float64") + y)


def test_multi_output():
    @autojit
    def foo(a, b, c):
        a = a - do("sum", b)
        b = b - do("sum", a)
        return a + c, b - c

    a = gen_rand((2, 3), "numpy")
    b = gen_rand((4, 5), "numpy")
    x, y = foo(a, b, 1)

    assert_allclose(x, a - b.sum() + 1)
    assert_allclose(y, b - (a - b.sum()).sum() - 1)


def _run_pair(call_a, call_b):
    """Run two functions in separate threads and wait for their results.

    Parameters
    ----------
    call_a, call_b : callable
        Zero-argument functions to run.

    Returns
    -------
    results : list
        The return values of ``call_a`` and ``call_b``, re-raising any error.
    """
    with ThreadPoolExecutor(2) as pool:
        futures = [pool.submit(call_a), pool.submit(call_b)]
        return [f.result(timeout=10) for f in futures]


class _SignalOnContention:
    """Lock that waits on ``barrier`` before blocking, if already held.

    Parameters
    ----------
    barrier : threading.Barrier
        Barrier to wait on when the lock is contended.
    """

    def __init__(self, barrier):
        self._lock = threading.Lock()
        self._barrier = barrier

    def __enter__(self):
        if not self._lock.acquire(blocking=False):
            self._barrier.wait()
            self._lock.acquire()

    def __exit__(self, *exc):
        self._lock.release()


def test_setup_interlaced(mgs_case):
    from autoray import compiler

    x, y = mgs_case
    # A is tracing (holding the setup lock) -> B starts its call
    b_start = threading.Barrier(2, timeout=10)
    # B is queued on the lock (or, if unlocked, also tracing) -> A resumes
    b_queued = threading.Barrier(2, timeout=10)
    ntraces = 0

    def traced_mgs(X):
        nonlocal ntraces
        ntraces += 1
        if ntraces == 1:
            b_start.wait()
        b_queued.wait()
        return modified_gram_schmidt(X)

    cfn = compiler.CompilePython(traced_mgs)
    compiler._SETUP_LOCKS[cfn] = _SignalOnContention(b_queued)

    def call_b():
        b_start.wait()
        return cfn(x)

    results = _run_pair(lambda: cfn(x), call_b)

    assert ntraces == 1
    for y2 in results:
        assert_allclose(y, y2)


def test_compiler_creation_interlaced(mgs_case, monkeypatch):
    from autoray import compiler

    x, y = mgs_case
    # A is creating its compiler -> B makes a full call
    b_start = threading.Barrier(2, timeout=10)
    # B has finished its call -> A resumes
    b_done = threading.Barrier(2, timeout=10)
    ncreated = ntraces = 0

    def make_compiler(fn, **kwargs):
        nonlocal ncreated
        ncreated += 1
        if ncreated == 1:
            b_start.wait()
            b_done.wait()
        return compiler.CompilePython(fn, **kwargs)

    def traced_mgs(X):
        nonlocal ntraces
        ntraces += 1
        return modified_gram_schmidt(X)

    monkeypatch.setitem(compiler._compiler_lookup, "python", make_compiler)
    cfn = autojit(traced_mgs)

    def call_b():
        b_start.wait()
        try:
            return cfn(x)
        finally:
            b_done.wait()

    results = _run_pair(lambda: cfn(x), call_b)

    # both threads missed the cache, but A must reuse B's compiler
    assert ncreated == 2
    assert ntraces == 1
    assert len(cfn._compiled_fns) == 1
    for y2 in results:
        assert_allclose(y, y2)


def test_autojit_pickle(mgs_case):
    import pickle

    from autoray import compiler

    x, y = mgs_case
    cfn = autojit(modified_gram_schmidt)
    cfn(x)
    cfn2 = pickle.loads(pickle.dumps(cfn))
    assert_allclose(y, cfn2(x))
    # the unpickled compiler should get its own setup lock
    (c1,) = cfn._compiled_fns.values()
    (c2,) = cfn2._compiled_fns.values()
    assert compiler._get_setup_lock(c1) is not compiler._get_setup_lock(c2)
