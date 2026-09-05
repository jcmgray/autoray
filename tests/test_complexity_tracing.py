import operator

import pytest

from autoray import lazy
from autoray.experimental.complexity_tracing import (
    COST_SCALINGS,
    COSTS,
    compute_cost,
    compute_cost_scalings,
    cost_node,
)


def make_node(name, shape, deps):
    def fn(*args):
        pass

    fn.__name__ = name
    return lazy.LazyArray(
        backend="numpy",
        fn=fn,
        args=deps,
        kwargs=None,
        shape=shape,
        deps=deps,
    )


def test_operation_registrations_are_consistent():
    assert COSTS.keys() == COST_SCALINGS.keys()
    assert {
        "flip",
        "gt",
        "max",
        "qr_stabilized_numpy",
        "where",
    } <= COSTS.keys()


def test_quimb_trace_operation_costs_are_covered():
    x = lazy.Variable((2, 3), backend="numpy")
    y = lazy.flip(x)
    y = lazy.where(y > 0, y, x)
    y = lazy.max(y)

    assert compute_cost(y, allow_missed=False) == 19

    qr = make_node("qr_stabilized_numpy", (3,), (x,))
    assert cost_node(qr, allow_missed=False) == pytest.approx(56 / 3)


@pytest.mark.parametrize(
    ("operation", "shape", "cost", "scaling"),
    [
        ("qr", (2, 3, 5), 144, 90),
        ("svd", (2, 3, 5), 288, 90),
        ("eigh", (2, 5, 5), pytest.approx(2000 / 3), 250),
    ],
)
def test_batched_decomposition_costs(operation, shape, cost, scaling):
    x = lazy.Variable(shape, backend="numpy")
    node = make_node(operation, (2,), (x,))

    assert COSTS[operation](node) == cost
    assert COST_SCALINGS[operation](node) == scaling


@pytest.mark.parametrize(
    ("shape_a", "shape_b", "expected"),
    [
        ((2, 3, 5), (2, 5, 7), 210),
        ((3, 1, 4, 5), (1, 7, 5, 6), 2520),
        ((5,), (5,), 5),
        ((2, 3, 5), (5,), 30),
        ((5,), (2, 5, 7), 70),
    ],
)
def test_batched_matmul_cost(shape_a, shape_b, expected):
    a = lazy.Variable(shape_a, backend="numpy")
    b = lazy.Variable(shape_b, backend="numpy")
    node = lazy.LazyArray(
        backend="numpy",
        fn=operator.matmul,
        args=(a, b),
        kwargs=None,
        shape=(),
        deps=(a, b),
    )

    assert cost_node(node) == expected


def test_compute_cost_multiple_outputs_counts_shared_nodes_once():
    x = lazy.Variable((2, 3), backend="numpy")
    shared = lazy.multiply(x, 2)
    left = lazy.add(shared, 1)
    right = lazy.multiply(shared, 3)
    outputs = (left, {"right": right})

    assert compute_cost(outputs, allow_missed=False) == 18

    scalings = compute_cost_scalings(
        outputs,
        {"a": 2, "b": 3},
        allow_missed=False,
    )
    frequencies = {op["name"]: op["freq"] for op in scalings}
    assert frequencies["add"] == 1
    assert frequencies["mul"] == 2


def test_compute_cost_can_reject_missed_operations():
    x = lazy.Variable((2, 3), backend="numpy")
    unknown = make_node("unknown", x.shape, (x,))

    with pytest.raises(ValueError, match="unknown"):
        compute_cost(unknown, allow_missed=False)
    with pytest.raises(ValueError, match="unknown"):
        compute_cost_scalings(
            unknown,
            {"a": 2, "b": 3},
            allow_missed=False,
        )


@pytest.mark.parametrize(
    "factor_map",
    [
        {"a": 2, "k": 2},
        {"a": 4},
    ],
)
def test_compute_cost_scalings_rejects_ambiguous_factors(factor_map):
    x = lazy.Variable((2, 3), backend="numpy")

    with pytest.raises(ValueError, match="factor_map"):
        compute_cost_scalings(x, factor_map)
