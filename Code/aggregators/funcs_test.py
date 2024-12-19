import numpy as np
import pytest

import pipe
from aggregators import base_test, funcs


@pytest.mark.parametrize('input_X, expected, p', [
    (
        np.array([[5, 10, 5, 10],
                  [10, 5, 10, 5],
                  [10, 5, 10, 5],
                  [10, 5, 10, 5]]),
        # after the first step we have [7.5, 7.5, 7.5, 7.5]
        # then we have
        # [5, 10, 5, 10, 7.5],
        # [10, 5, 10, 5, 7.5],
        # so when we aggregate, we finally have:
        # [7.5, 7.5, ...]
        np.array([7.5, 7.5, 7.5, 7.5]),
        pipe.ExtPipeline(name='p1', steps=[
            # we use a multiplication just to obtain more data.
            funcs.StepAggregateWeightedSum(name='1', to_aggregate=[-1],
                                           weights=[.25] * 4),
            funcs.StepAggregateFlattener(name='3', to_aggregate=[-1, 0]),
            funcs.StepAggregateWeightedSum(name='4', to_aggregate=[1],
                                           weights=[.2] * 5)
        ])
    ),
    (
            np.array([[5, 10, 5, 10],
                      [10, 5, 10, 5],
                      [10, 5, 10, 5],
                      [10, 5, 10, 5]]),
            # after the first step we have [7.5, 7.5, 7.5, 7.5]
            # then we have
            # [5, 10, 5, 10, 7.5, 5, 10, 5, 10, 7.5,],
            # [10, 5, 10, 5, 7.5, 5, 10, 5, 10, 7.5,],
            # so when we aggregate, we finally have:
            # [7.5, 7.5, ...]
            np.array([7.5, 7.5, 7.5, 7.5]),
            pipe.ExtPipeline(name='p1', steps=[
                # we use a multiplication just to obtain more data.
                funcs.StepAggregateWeightedSum(name='1', to_aggregate=[-1],
                                               weights=[.25] * 4),
                funcs.StepAggregateFlattener(name='3', to_aggregate=[-1, 0],
                                             expansion_mapper={-1: 2, 0: 2}),
                funcs.StepAggregateWeightedSum(name='4', to_aggregate=[1],
                                               weights=[.1] * 10)
            ])
    )
])
def test_funcs(input_X, expected, p: pipe.ExtPipeline):
    base_test.help_test_aggregation(input_X=input_X, expected=expected, p=p)
    
    
@pytest.mark.parametrize('input_X, expected, step', [
    (
        [np.array([5, 10, 5, 10]), np.array([10, 5, 10, 5]), np.array([5, 10, 5, 10])],
        np.array([7.5, 7.5, 7.5]),
        funcs.StepAggregateWeightedSum(name='last', to_aggregate=[0, 1, 2],
                                       weights=[.25] * 4)
    ),
    (
        [np.array([5, 10, 5, 10]), np.array([10, 5, 10, 5]),
        np.array([5, 10, 5, 10]), np.array([10, 5, 10, 5])],
        np.array([7.5, 7.5, 7.5, 7.5]),
        funcs.StepAggregateWeightedAverage(name='last', to_aggregate=[0, 1, 2, 3],
                                           weights=[.25] * 4)
    ),
    (
        [np.array([[5, 10], [10, 5], [1, 1]]),
         np.array([[2, 2, 2], [2, 2, 2], [2, 2, 2]]),
         np.array([[1], [1], [1]])
         ],
        np.array([[5, 10, 2, 2, 2, 1], [10, 5, 2, 2, 2, 1], [1, 1, 2, 2, 2, 1]]),
        funcs.StepAggregateFlattener(name='last', to_aggregate=[0, 1, 2])
    ),
    (
            [np.array([[5, 10], [10, 5], [1, 1]]),
             np.array([[2, 2, 2], [2, 2, 2], [2, 2, 2]]),
             np.array([[1], [1], [1]])
             ],
            np.array([[5, 10, 2, 2, 2, 1, 1], [10, 5, 2, 2, 2, 1, 1], [1, 1, 2, 2, 2, 1, 1]]),
            funcs.StepAggregateFlattener(name='last', to_aggregate=[0, 1, 2],
                                         expansion_mapper={2: 2})
    ),
    (
        # these data are to be considered already flattened.
        [np.array([5, 5, 10]), np.array([1, 2, 10]), np.array([10, 1, 1])],
        np.array([10, 10, 10]),
        funcs.StepAggregateMax(name='last', to_aggregate=[0, 1, 2])
    ),
    (
        [np.array([1, 1, 0]), np.array([1, 0, 1]), np.array([0, 0, 1]), np.array([0, 0, 0]),
         np.array([1, 1, 1])],
        np.array([2, 2, 1, 0, 3]),
        funcs.StepAggregateCount(name='last', to_aggregate=[0, 1, 2, 3, 4])
    ),
    (
        # this is the real concrete use that we made in a pipeline, when we use flattening
        # *before* aggregation
        [np.array([[10, 1], [2, 20], [30, 3], [4, 40]])],
        np.array([1, 2, 3, 4]) * 10,
        funcs.StepAggregateMax(name='last', to_aggregate=[0])
    )
])
def test_aggregation_func(input_X, expected, step: pipe.Step):
    # we have the raw inputs, we need to transform them in the form
    # of StepInput to allow visit.
    inputs = [pipe.StepInput(X=X_, y=None) for X_ in input_X]
    result = step.visit(inputs)
    # result is a tuple of X, y. We are interested in the first part only.
    result = result.actual[0]
    assert len(result) == len(expected)
    for i in range(len(result)):
         assert np.all(result[i] == expected[i])
