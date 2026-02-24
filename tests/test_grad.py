import pytest
from autoray import do

from .test_autoray import BACKENDS, gen_rand


_GRAD_BACKENDS = ["jax", "torch", "tensorflow", "paddle", "autograd"]
BACKENDS = [p for p in BACKENDS if p.values[0] in _GRAD_BACKENDS]


@pytest.mark.parametrize("backend", BACKENDS)
def test_stop_gradient(backend):
    x = gen_rand((10, 10), backend)
    y = do("stop_gradient", x)
    assert do("allclose", x, y)
