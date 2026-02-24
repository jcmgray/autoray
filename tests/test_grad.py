import pytest
from autoray import do

from .test_autoray import BACKENDS, gen_rand

BACKENDS = ("jax", "torch", "tensorflow", "paddle", "autograd")


@pytest.mark.parametrize("backend", BACKENDS)
def test_stop_gradient(backend):
    x = gen_rand((10, 10), backend)
    y = do("stop_gradient", x)
    assert do("allclose", x, y)
