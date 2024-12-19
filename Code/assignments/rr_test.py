import numpy as np
import pytest
from sklearn import datasets

import pipe
from . import hash, rr, base, base_test, spread

X, y = datasets.make_classification(200, 21)


@pytest.mark.parametrize('p', [
    pipe.ExtPipeline(
        name='p1',
        steps=[
            pipe.Step(name='hash', step=hash.Hasher(algo='md5')),
            pipe.Step(name='modulo', step=hash.SingleValuedRouter(algo=hash.SingleValuedRouterType.MODULO, N=5)),
            pipe.Step(name='assignment', step=rr.AssignmentRoundRobinSmart(N=5)),
        ])
])
def test_composed(p: pipe.ExtPipeline):
    base_test._test_assignment_inner(p, N=5)


def test_round_robin_smart():
    points = np.concatenate([np.repeat(0, 10), np.repeat(1, 9), np.repeat(0, 30)])
    bad_idx = np.argwhere(points == 1).flatten()
    # N = number of poisoned points; so 1 point for each model.
    N = 9

    assignment = rr.AssignmentRoundRobinSmart(N=N)
    got = assignment.fit_transform(X=points)

    expected = np.arange(N)

    assert np.all(got[bad_idx] == expected)
    print(got)


@pytest.mark.parametrize('risk_idx, N, weighting_strategy, expected_weights', [
    (
            np.array([1, 0, 1, 1, 0, 0, 0]),
            2,
            base.WeightingStrategy.NONE,
            np.array([0.5, 0.5])
    ),
    (
            np.random.default_rng().choice([1, 0], 100),
            7,
            base.WeightingStrategy.NONE,
            np.repeat(1 / 7, 7)
    ),
    # proportional and extreme result in the same weights
    (
            np.array([1, 0, 1, 1, 0, 0, 0]),
            2,
            base.WeightingStrategy.EXTREME,
            np.array([0, 1])
    ),
    (
            np.random.default_rng().choice([1, 0], 100),
            7,
            base.WeightingStrategy.EXTREME,
            np.concatenate(([0], np.repeat(1 / 6, 6)))
    ),
    (
            np.array([1, 0, 1, 1, 0, 0, 0]),
            2,
            base.WeightingStrategy.PROPORTIONAL,
            np.array([0, 1])
    ),
    (
            np.random.default_rng().choice([1, 0], 100),
            7,
            base.WeightingStrategy.PROPORTIONAL,
            np.concatenate(([0], np.repeat(1 / 6, 6)))
    )
])
def test_sink_basic(risk_idx, N, weighting_strategy, expected_weights):
    assignment = rr.AssignmentSqueezeSink(N=N, weighting_strategy=weighting_strategy)
    got = assignment.fit_transform(X=risk_idx)

    # we retrieve the indices where points are assigned to the first model.

    assert np.all(np.argwhere(risk_idx).flatten()) == np.all(np.argwhere(got == 0).flatten())
    assert assignment.points_to_first == len(np.argwhere(risk_idx).flatten())

    assert np.isclose(np.sum(assignment.get_weights()), 1)
    if expected_weights is not None:
        assert np.all(assignment.get_weights() == expected_weights)


@pytest.mark.parametrize('risk_idx, N, percentage, weighting_strategy, expected_weights', [
    (
            np.array([1, 0, 1, 1, 0, 0, 0]),
            2,
            .5,
            base.WeightingStrategy.NONE,
            np.array([.5, .5])
    ),
    (
            np.random.default_rng().choice([1, 0], 100),
            7,
            .99,
            base.WeightingStrategy.NONE,
            np.repeat(1 / 7, 7)
    ),
    (
            np.array([1, 0, 1, 1, 0, 0, 0]),
            2,
            .5,
            base.WeightingStrategy.EXTREME,
            np.array([0, 1])
    ),
    # here by no means we can know how the weights will look like.
    (
            np.random.default_rng().choice([1, 0], 100),
            7,
            .99,
            base.WeightingStrategy.EXTREME,
            None
    ),
    (
            np.array([1, 0, 1, 1, 0, 0, 0]),
            2,
            .5,
            base.WeightingStrategy.PROPORTIONAL,
            np.array([0, 1])
    ),
    # here by no means we can know how the weights will look like.
    (
            np.random.default_rng().choice([1, 0], 100),
            7,
            .99,
            base.WeightingStrategy.PROPORTIONAL,
            None
    ),
])
def test_sink_with_percentage(risk_idx, N, percentage, weighting_strategy, expected_weights):
    assignment = rr.AssignmentSqueezeSink(N=N, percentage_of_risky=percentage,
                                          weighting_strategy=weighting_strategy)
    got = assignment.fit_transform(X=risk_idx, )

    assert assignment.points_to_first == int(np.round(np.count_nonzero(risk_idx) * percentage))
    # +1 because of rounding
    assert np.count_nonzero(got == 0) == assignment.points_to_first or \
           np.count_nonzero(got == 0) == assignment.points_to_first + 1

    assert np.isclose(np.sum(assignment.get_weights()), 1)
    if expected_weights is not None:
        assert np.all(assignment.get_weights() == expected_weights)
