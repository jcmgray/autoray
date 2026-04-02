import pytest

from autoray import do

from .conftest import gen_params, gen_rand

_GRAD_BACKENDS = ["jax", "torch", "tensorflow", "paddle", "autograd"]


@pytest.mark.parametrize("backend", gen_params(backends=_GRAD_BACKENDS))
def test_stop_gradient(backend):
    x = gen_rand((10, 10), backend)
    y = do("stop_gradient", x)
    assert do("allclose", x, y)
