"""
Functionality for tracing through an autoray.lazy computation and estimating
the cost and scaling.

In the following there are ``cost_*`` functions that estimate the total cost
of a given operation, including sub-leading factors. There are also
`cost_scaling_*` functions that only consider the leading factor of the cost,
so that we can prime number decompose it and extract the scaling.
"""

import math


def _get_batch_matrix_sizes(x):
    """Get the batch size and ordered matrix dimensions of ``x``."""
    (A,) = x.deps
    *batch_shape, dm, dn = A.shape
    return math.prod(batch_shape), max(dm, dn), min(dm, dn)


def cost_tensordot(x):
    x1, x2, axes = x.args
    shape1, shape2 = x1.shape, x2.shape
    cost = math.prod(shape1) * math.prod(shape2)
    for d in axes[0]:
        cost //= shape1[d]
    return cost


cost_scaling_tensordot = cost_tensordot


def cost_qr(x):
    batch_size, m, n = _get_batch_matrix_sizes(x)
    return batch_size * (2 * m * n**2 - (2 / 3) * n**3)


def cost_svd(x):
    batch_size, m, n = _get_batch_matrix_sizes(x)
    return batch_size * (4 * m * n**2 - (4 / 3) * n**3)


def cost_eigh(x):
    (A,) = x.deps
    *batch_shape, m, _ = A.shape
    return math.prod(batch_shape) * (8 / 3) * m**3


def cost_scaling_linalg(x):
    """Here we only care about the leading factor of the cost, which we need to
    preserve so that we can prime number decompose it.
    """
    batch_size, m, n = _get_batch_matrix_sizes(x)
    return batch_size * m * n**2


cost_scaling_qr = cost_scaling_svd = cost_scaling_linalg


def cost_matmul(x):
    A, B = x.deps
    shape_a = A.shape
    shape_b = B.shape

    if len(shape_a) == 1:
        batch_shape_a = ()
        m = 1
    else:
        batch_shape_a = shape_a[:-2]
        m = shape_a[-2]

    if len(shape_b) == 1:
        batch_shape_b = ()
        n = 1
    else:
        batch_shape_b = shape_b[:-2]
        n = shape_b[-1]

    ndim = max(len(batch_shape_a), len(batch_shape_b))
    batch_shape_a = (1,) * (ndim - len(batch_shape_a)) + batch_shape_a
    batch_shape_b = (1,) * (ndim - len(batch_shape_b)) + batch_shape_b
    batch_size = math.prod(
        max(da, db) for da, db in zip(batch_shape_a, batch_shape_b)
    )

    return batch_size * m * shape_a[-1] * n


cost_scaling_matmul = cost_matmul


def cost_einsum(x):
    eq, *operands = x.args
    lhs = eq.split("->")[0]
    terms = lhs.split(",")
    size_dict = {
        ix: d
        for term, x in zip(terms, operands)
        for ix, d in zip(term, x.shape)
    }
    return math.prod(size_dict.values())


cost_scaling_einsum = cost_einsum


def cost_linear(x):
    return math.prod(x.shape)


def cost_nothing(x):
    return 0


_LINEAR_COSTS = {
    "absolute",
    "add",
    "clamp",
    "clip",
    "conj",
    "conjugate",
    "cupy_conjugate",
    "cupy_log10",
    "cupy_sqrt",
    "flip",
    "gt",
    "log10",
    "max",
    "mul",
    "neg",
    "norm",
    "linalg_norm",
    "pow",
    "reshape",
    "sqrt",
    "sum",
    "torch_transpose",
    "trace",
    "transpose",
    "truediv",
    "where",
}

_NOTHING_COSTS = {
    "getitem",
    "None",
}


COSTS = {
    "qr": cost_qr,
    "qr_stabilized": cost_qr,
    "qr_stabilized_numba": cost_qr,
    "qr_stabilized_numpy": cost_qr,
    "svd": cost_svd,
    "svd_truncated": cost_svd,
    "svd_truncated_numba": cost_svd,
    "svd_truncated_numpy": cost_svd,
    "eigh": cost_eigh,
    "linalg_eigh": cost_eigh,
    "tensordot": cost_tensordot,
    "matmul": cost_matmul,
    "einsum": cost_einsum,
    **dict.fromkeys(_LINEAR_COSTS, cost_linear),
    **dict.fromkeys(_NOTHING_COSTS, cost_nothing),
}


def cost_node(x, allow_missed=True):
    f = x.fn_name
    if f in COSTS:
        return COSTS[f](x)
    elif allow_missed:
        return 0
    else:
        raise ValueError(f"Cost for {f} not implemented.")


def compute_cost(z, print_missed=True, allow_missed=True):
    """Estimate the total cost of one or more lazy output nodes.

    Shared dependencies of multiple output nodes are counted once.

    Parameters
    ----------
    z : pytree of LazyArray
        The output node or nodes to trace.
    print_missed : bool, optional
        Whether to warn about operations without a registered cost.
    allow_missed : bool, optional
        Whether to omit operations without a registered cost. If ``False``,
        raise a ``ValueError`` listing them instead.
    """
    from autoray.lazy import descend

    C = 0
    missed = {}
    for node in descend(z):
        f = node.fn_name
        if f in COSTS:
            C += COSTS[f](node)
        else:
            missed[f] = missed.get(f, 0) + 1

    if missed:
        if not allow_missed:
            raise ValueError(f"Costs for {missed} not implemented.")
        if print_missed:
            import warnings

            warnings.warn(f"Missed {missed} in cost computation.")

    return C


COST_SCALINGS = {
    "qr": cost_scaling_qr,
    "qr_stabilized": cost_scaling_qr,
    "qr_stabilized_numba": cost_scaling_qr,
    "qr_stabilized_numpy": cost_scaling_qr,
    "svd": cost_scaling_svd,
    "svd_truncated": cost_scaling_svd,
    "svd_truncated_numba": cost_scaling_svd,
    "svd_truncated_numpy": cost_scaling_svd,
    "eigh": cost_scaling_linalg,
    "linalg_eigh": cost_scaling_linalg,
    "tensordot": cost_scaling_tensordot,
    "matmul": cost_scaling_matmul,
    "einsum": cost_scaling_einsum,
    **dict.fromkeys(_LINEAR_COSTS, cost_linear),
    **dict.fromkeys(_NOTHING_COSTS, cost_nothing),
}


def prime_factors(n) -> list[int]:
    fs = []
    if n <= 1:
        return fs

    while n % 2 == 0:
        fs.append(2)
        n = n // 2
    i = 3
    while i * i <= n:
        while n % i == 0:
            fs.append(i)
            n = n // i
        i += 2
    if n > 2:
        fs.append(n)
    return fs


def is_prime(n: int) -> bool:
    for i in range(int(n**0.5), 1, -2 if int(n**0.5) % 2 == 0 else -1):
        if n % i == 0:
            return False
    return False if n in (0, 1) else True


def closest_prime(nt: int) -> int:
    if is_prime(nt):
        return nt
    lower = None
    higher = None
    for i in range(nt if nt % 2 != 0 else nt - 1, 1, -2):
        if is_prime(i):
            lower = i
            break
    c = nt + 1
    while higher is None:
        if is_prime(c):
            higher = c
        else:
            c += 2 if c % 2 != 0 else 1
    return higher if lower is None or higher - nt < nt - lower else lower


def frequencies(it):
    c = {}
    for i in it:
        c[i] = c.get(i, 0) + 1
    return c


def _check_factor_map(factor_map):
    seen = {}
    for name, factor in factor_map.items():
        if factor in seen:
            raise ValueError(
                "factor_map values must be unique, but "
                f"{seen[factor]!r} and {name!r} both map to {factor}."
            )
        if not isinstance(factor, int) or factor < 2 or not is_prime(factor):
            raise ValueError(
                f"factor_map value for {name!r} must be a prime integer, "
                f"got {factor!r}."
            )
        seen[factor] = name


def compute_cost_scalings(
    z,
    factor_map,
    print_missed=True,
    allow_missed=True,
):
    """Estimate cost scalings for one or more lazy output nodes.

    Parameters
    ----------
    z : pytree of LazyArray
        The output node or nodes to trace. Shared dependencies are counted
        once.
    factor_map : dict[str, int]
        Mapping from dimension labels to distinct prime numbers.
    print_missed : bool, optional
        Whether to warn about unregistered operations and prime factors.
    allow_missed : bool, optional
        Whether to omit operations without a registered scaling. If ``False``,
        raise a ``ValueError`` listing them instead.
    """
    from autoray.lazy import descend

    _check_factor_map(factor_map)

    counts = {}
    missed = {}

    for node in descend(z):
        f = node.fn_name

        if f in COST_SCALINGS:
            CS = COST_SCALINGS[f](node)
        else:
            missed[f] = missed.get(f, 0) + 1
            continue

        # group operations
        key = (CS, f)
        counts[key] = counts.get(key, 0) + 1

    if missed:
        if not allow_missed:
            raise ValueError(f"Cost scalings for {missed} not implemented.")
        if print_missed:
            import warnings

            warnings.warn(f"Missed {missed} in cost scaling computation.")

    scalings = []

    for key, freq in counts.items():
        op = {
            "cost": key[0],
            "name": key[1],
            "freq": freq,
        }
        pf = frequencies(prime_factors(op["cost"]))
        for name, factor in factor_map.items():
            op[name] = pf.pop(factor, 0)

        if pf and print_missed:
            import warnings

            warnings.warn(
                f"Missed prime factor(s) {pf} in cost scaling computation, "
                f" for operation {op}."
            )

        scalings.append(op)

    scalings.sort(key=lambda x: x["cost"], reverse=True)
    return scalings
