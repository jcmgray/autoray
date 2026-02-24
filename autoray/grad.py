from .autoray import register_function, do


_GRAD_BACKENDS = {"jax", "torch", "tensorflow", "paddle", "autograd"}
_NO_GRAD_BACKENDS = {"numpy", "cupy", "dask", "sparse", "mars"}


# -------------------- stop_gradient -------------------- #


def jax_stop_gradient(x):
    from jax.lax import stop_gradient

    return stop_gradient(x)


def torch_stop_gradient(x):
    return x.detach()


def tensorflow_stop_gradient(x):
    import tensorflow as tf

    return tf.stop_gradient(x)


def paddle_stop_gradient(x):
    return x.detach()


def autograd_stop_gradient(x):
    import autograd.numpy as anp
    from autograd.tracer import getval

    return anp.array(getval(x))


def do_nothing(x):
    """For backends without grad, keep x unchanged."""
    return x


# register for each backend
register_function("torch", "stop_gradient", torch_stop_gradient)
register_function("jax", "stop_gradient", jax_stop_gradient)
register_function("tensorflow", "stop_gradient", tensorflow_stop_gradient)
register_function("paddle", "stop_gradient", paddle_stop_gradient)
register_function("autograd", "stop_gradient", autograd_stop_gradient)
for _backend in _NO_GRAD_BACKENDS:
    register_function(_backend, "stop_gradient", do_nothing)


def stop_gradient(x):
    """Stop gradient flow through array ``x``.

    In autodiff backends (JAX, PyTorch, TensorFlow, etc.), this detaches
    ``x`` from the computational graph so that no gradients are propagated
    through it. For non-autodiff backends (NumPy, CuPy, etc.), this is a
    no-op and returns ``x`` unchanged.

    Parameters
    ----------
    x : array
        The array to stop gradient flow through.

    Returns
    -------
    array
        An array with the same value as ``x`` but detached from the
        autodiff computational graph.

    Examples
    --------

    With JAX::

        >>> import jax, jax.numpy as jnp, autoray as ar
        >>> x = jnp.array([1.0, 2.0, 3.0])
        >>> ar.stop_gradient(x)  # equivalent to jax.lax.stop_gradient(x)

    With PyTorch::

        >>> import torch, autoray as ar
        >>> x = torch.tensor([1.0, 2.0], requires_grad=True)
        >>> y = ar.stop_gradient(x)  # equivalent to x.detach()
        >>> y.requires_grad
        False

    With NumPy (no-op)::

        >>> import numpy as np, autoray as ar
        >>> x = np.array([1.0, 2.0])
        >>> ar.stop_gradient(x) is x
        True
    """
    return do("stop_gradient", x)
