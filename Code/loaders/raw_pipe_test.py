import typing

import numpy as np
import pytest
from sklearn import cluster, datasets

import aggregators
import assignments
import iops
import pipe
from . import base, raw_pipe


@pytest.mark.parametrize('initial, expected', [
    (
            raw_pipe.StepRaw(name='s1', step_func_name='_numpy.divide',
                             arg_func_code="lambda step_input: pipe.ArgFuncOutput(args=(step_input[0].X, 1))",
                             output_col_names_pre=['output']),
            pipe.Step(name='s1', step=np.divide, output_col_names_pre=['output'],
                      arg_func=lambda step_input: pipe.ArgFuncOutput(args=(step_input[0].X, 1)))
    ),
    (
            raw_pipe.StepRaw(name='s1', step_func_name='__assignments.Hasher', step_func_kwargs={'algo': 'md5'},
                             output_col_names_pre=['output']),
            pipe.Step(name='s1', step=assignments.Hasher(algo='md5'), output_col_names_pre=['output'])
    ),
    (
            raw_pipe.StepRaw(name='s1', step_func_name='__assignments.Hasher', step_func_kwargs={'algo': 'md5'},
                             output_col_names_pre=['output_pre'], output_col_names_post=['output_post'],
                             post_aggregation_pair=raw_pipe.StepPostAggregationPairRaw(
                                 aggregation_func_name='_numpy.divide',
                                 # aggregation_func_name='_numpy.divide',
                                 aggregation_func_arg_func_code='lambda a: pipe.ArgFuncOutput(args=(a[0], 1))')
                             ),
            pipe.Step(name='s1', step=assignments.Hasher(algo='md5'),
                      post_aggregation=pipe.AggregationFuncPair(
                          aggregation_func=np.divide,
                          aggregation_func_arg_func=lambda a: pipe.ArgFuncOutput(args=(a[0], 1))
                      ),
                      output_col_names_pre=['output_pre'], output_col_names_post=['output_post'])
    ),
    (
            raw_pipe.StepRaw(name='s1', step_func_name='__assignments.Hasher', step_func_kwargs={'algo': 'md5'},
                             output_col_names_pre=['output_pre'], output_col_names_post=['output_post'],
                             post_aggregation_pair=base.FuncPair(
                                 func_name='__aggregators.post_aggregation_take_col_and_where',
                                 # this aggregation function does not make any sense in practice,
                                 # but we do not execute it, so we are fine.
                                 func_kwargs={'col_idx': 0, 'divider': 1, 'left': 0, 'right': 1}
                             )),
            pipe.Step(name='s1', step=assignments.Hasher(algo='md5'),
                      output_col_names_pre=['output_pre'], output_col_names_post=['output_post'],
                      post_aggregation=aggregators.post_aggregation_take_col_and_where(col_idx=0, divider=1, left=0,
                                                                                       right=1)
                      )
    ),
    (
            # one where we have to load the code.
            raw_pipe.FuncPairWithPostAggregationPair(
                func_name='__aggregators.StepAggregateWeightedSum',
                post_aggregation_pair=raw_pipe.StepPostAggregationPairRaw(
                    aggregation_func_code='lambda a: numpy.where(a[..., 0]>=0.75, 1, 0)',
                    aggregation_func_arg_func_code='lambda a: pipe.ArgFuncOutput(args=(a[0],))',
                ),
                func_kwargs={
                    'name': 's1', 'weights': [0.3, 0.3],
                    'to_aggregate': [-1],
                }),
            aggregators.StepAggregateWeightedSum(name='s1',
                                                 weights=[0.3, 0.3],
                                                 to_aggregate=[-1],
                                                 post_aggregation=pipe.AggregationFuncPair(
                                                     aggregation_func=lambda a: np.where(a[..., 0] >= 0.75, 1, 0),
                                                     aggregation_func_arg_func=lambda a: pipe.ArgFuncOutput(
                                                         args=(a[0],))
                                                 ))
    )
])
def test_step_raw(initial: typing.Union[raw_pipe.StepRaw, raw_pipe.StepPostAggregationPairRaw], expected):
    got = initial.parse()
    assert expected == got


@pytest.mark.parametrize('initial_raw, expected', [
    (
            {
                'name': 's1',
                'step_func_name': '_numpy.divide',
                'arg_func_code': 'lambda step_input: pipe.ArgFuncOutput(args=(step_input[0].X, 1))',
                'output_col_names_pre': ['output']
            },
            pipe.Step(name='s1', step=np.divide, output_col_names_pre=['output'],
                      arg_func=lambda step_input: pipe.ArgFuncOutput(args=(step_input[0].X, 1)))
    ),
    (
            {
                'name': 's1',
                'step_func_name': '__assignments.Hasher',
                'step_func_kwargs': {'algo': 'md5'},
                'output_col_names_pre': ['output']
            },
            pipe.Step(name='s1', step=assignments.Hasher(algo='md5'), output_col_names_pre=['output'])
    ),
    (
            {
                'name': 's1',
                'step_func_name': '__assignments.Hasher',
                'step_func_kwargs': {'algo': 'md5'},
                'output_col_names_pre': ['output_pre'],
                'output_col_names_post': ['output_post'],
                'post_aggregation_pair': {
                    'aggregation_func_name': '_numpy.divide',
                    'aggregation_func_arg_func_code': 'lambda a: pipe.ArgFuncOutput(args=(a[0], 1))'
                }
            },
            pipe.Step(name='s1', step=assignments.Hasher(algo='md5'),
                      post_aggregation=pipe.AggregationFuncPair(
                          aggregation_func=np.divide,
                          aggregation_func_arg_func=lambda a: pipe.ArgFuncOutput(args=(a[0], 1))
                      ),
                      output_col_names_pre=['output_pre'], output_col_names_post=['output_post'])
    ),
    (
            {
                'name': 's1',
                'step_func_name': '__assignments.Hasher',
                'step_func_kwargs': {'algo': 'md5'},
                'output_col_names_pre': ['output_pre'],
                'output_col_names_post': ['output_post'],
                'post_aggregation_pair': {
                    'func_name': '__aggregators.post_aggregation_take_col_and_where',
                    # this aggregation function does not make any sense in practice,
                    # but we do not execute it, so we are fine.
                    'func_kwargs': {'col_idx': 0, 'divider': 1, 'left': 0, 'right': 1}
                }
            },
            pipe.Step(name='s1', step=assignments.Hasher(algo='md5'),
                      output_col_names_pre=['output_pre'], output_col_names_post=['output_post'],
                      post_aggregation=aggregators.post_aggregation_take_col_and_where(col_idx=0, divider=1, left=0,
                                                                                       right=1)
                      )
    ),
    (
            # this is the most complex.
            {
                'func_name': '__aggregators.StepAggregateWeightedSum',
                'func_kwargs': {
                    'name': 's1',
                    'to_aggregate': [-1],
                    'weights': [0.3, 0.3],
                },
                'post_aggregation_pair': {
                    'aggregation_func_code': 'lambda a: numpy.where(a[..., 0]>=0.75, 1, 0)',
                    'aggregation_func_arg_func_code': 'lambda a: pipe.ArgFuncOutput(args=(a[0],))',
                }
            },
            aggregators.StepAggregateWeightedSum(name='s1',
                                                 weights=[0.3, 0.3],
                                                 to_aggregate=[-1],
                                                 post_aggregation=pipe.AggregationFuncPair(
                                                     aggregation_func=lambda a: np.where(a[..., 0] >= 0.75, 1, 0),
                                                     aggregation_func_arg_func=lambda a: pipe.ArgFuncOutput(
                                                         args=(a[0],))
                                                 ))
    )
])
def test_step_raw_dict(initial_raw: dict, expected: pipe.Step):
    # this is the same test of above, but we load stuff from raw dict, which is more difficult,
    # but also necessary since this part is quite important.
    if 'name' in initial_raw:
        initial_parsed = raw_pipe.StepRaw.from_dict(initial_raw)
    else:
        # when used within the pipeline, it is automatic.
        initial_parsed = raw_pipe.FuncPairWithPostAggregationPair.from_dict(initial_raw)
    got = initial_parsed.parse()
    assert expected == got


X, y = datasets.make_classification(100, 10)


@pytest.mark.parametrize('initial, step_input, expected', [
    (
            raw_pipe.StepRaw(name='s1', step_func_name='__assignments.Hasher', step_func_kwargs={'algo': 'md5'},
                             output_col_names_pre=['output_pre'], output_col_names_post=['output_post'],
                             post_aggregation_pair=raw_pipe.StepPostAggregationPairRaw(
                                 aggregation_func_name='_numpy.divide',
                                 # aggregation_func_name='_numpy.divide',
                                 aggregation_func_arg_func_code='lambda a: pipe.ArgFuncOutput(args=(a[0], 2))'
                             ),
                             # we need to specify also this value that is normally omitted.
                             steps_to_aggregate=[-1], ),
            [pipe.StepInput(X=X, y=y)],
            pipe.StepOutput.with_actual_only(
                actual=(np.divide(assignments.Hasher(algo='md5').fit(X, y).transform(X, y), 2), None))
    ),
])
def test_step_with_execution(initial: raw_pipe.StepRaw, step_input: typing.List[pipe.StepInput],
                             expected: pipe.StepOutput):
    step = initial.parse()
    result = step.visit(step_input)
    for single_result_got, single_result_expected in zip(result.actual, expected.actual):
        if single_result_got is None:
            assert single_result_expected is None
        else:
            assert np.all(single_result_expected == single_result_got)


@pytest.mark.parametrize('initial, expected', [
    (
            raw_pipe.PipelineRaw(
                name='p1', steps=[
                    raw_pipe.StepRaw(name='s1', step_func_name='__assignments.Hasher', step_func_kwargs={'algo': 'md5'})
                ]
            ),
            pipe.ExtPipeline(name='p1', steps=[pipe.Step(name='s1', step=assignments.Hasher(algo='md5'))])
    ),
    (
            # now we try something where we "instantiate" the Step.
            raw_pipe.PipelineRaw(
                name='p2', steps=[
                    raw_pipe.FuncPairWithPostAggregationPair(
                        func_name='__aggregators.StepAggregateWeightedSum',
                        func_kwargs={'name': 's2', 'to_aggregate': [-1],
                                     'weights': [.3, .3]},
                        post_aggregation_pair=raw_pipe.StepPostAggregationPairRaw(
                            aggregation_func_code='lambda a: numpy.where(a[..., 0] >= 0.75, 1, 0)',
                            aggregation_func_arg_func_code='lambda a: pipe.ArgFuncOutput(args=(a[0],))'
                        ))
                ]
            ),
            pipe.ExtPipeline(name='p2', steps=[
                aggregators.StepAggregateWeightedSum(name='s2', to_aggregate=[-1],
                                                     weights=[.3, .3],
                                                     post_aggregation=pipe.AggregationFuncPair(
                                                         aggregation_func=lambda a: np.where(a[..., 0] >= 0.75, 1, 0),
                                                         aggregation_func_arg_func=lambda a: pipe.ArgFuncOutput(
                                                             args=(a[0],))
                                                     ))
            ])
    ),
])
def test_pipeline_raw(initial: raw_pipe.PipelineRaw, expected):
    got = initial.parse()
    assert expected == got


@pytest.mark.parametrize('initial_raw, expected', [
    (
            {
                'name': 'p1',
                'steps': [
                    {
                        'name': 's1',
                        'step_func_name': '__assignments.Hasher',
                        'step_func_kwargs': {'algo': 'md5'}
                    }
                ]
            },
            pipe.ExtPipeline(name='p1', steps=[pipe.Step(name='s1', step=assignments.Hasher(algo='md5'))])
    ),
    (
            {
                'name': 'p2',
                'steps': [
                    {
                        'func_name': '__aggregators.StepAggregateWeightedSum',
                        'func_kwargs': {'name': 's2', 'to_aggregate': [-1], 'weights': [.3, .3]}
                    }
                ]
            },
            pipe.ExtPipeline(name='p2', steps=[
                aggregators.StepAggregateWeightedSum(name='s2', to_aggregate=[-1],
                                                     weights=[.3, .3])
            ])
    ),
    # finally one quite challenging,
    (
            {
                'name': 'p3',
                "steps_to_figures": [0],
                "steps": [
                    {
                        "name": "distance",
                        "steps_to_aggregate": [0],
                        "step_func_name": "__iops.IoPDistance",
                        "step_func_kwargs": {
                            "how": "_iops.IoPDistanceType.CLUSTERING",
                            "reverse_how": "_iops.ReverseType.SUBTRACT_BY_MAX",
                            "inner_kwargs": {
                                "direction": "_iops.Direction.FROM_BOTH",
                                "clustering_clazz": "_sklearn.cluster.KMeans",
                                "distance_metric_exp": 1
                            }
                        },
                        "post_aggregation_pair": {
                            # where the first column is greater than, returns 0, 1 otherwise.
                            "aggregation_func_code": "lambda a, split: numpy.where(a[..., 0] > split, 0, 1)",
                            "aggregation_func_arg_func_code": "lambda a: pipe.ArgFuncOutput(args=(a[0], 0.1))"
                        },
                        "output_col_names_pre": ["SAME"],
                        "output_col_names_post": ["SAME"],
                    },
                    {
                        "name": "neigh",
                        "steps_to_aggregate": [0],
                        "step_func_name": "__iops.IoPNeighbor",
                        "step_func_kwargs": {
                            "how": "_iops.IoPNeighborType.K_DISTANCE",
                            "reverse_how": "_iops.ReverseType.SUBTRACT_BY_MAX",
                            "inner_kwargs": {
                                "direction": "_iops.Direction.FROM_BOTH",
                                "distance_metric_exp": 1,
                                "neighbor_kwargs": {
                                    "n_neighbors": 5
                                }
                            }
                        },
                        "post_aggregation_pair": {
                            # where the second column is greater than, returns 0, 1 otherwise.
                            "aggregation_func_code": "lambda a, split: numpy.where(a[..., 1] >= split, 0, 1)",
                            "aggregation_func_arg_func_code": "lambda a: pipe.ArgFuncOutput(args=(a[0], 0.195))"
                        }
                    },
                    {
                        "func_name": "__aggregators.StepAggregateFlattener",
                        "func_kwargs": {
                            "name": "aggregate",
                            "to_aggregate": [1, 2]
                        }
                    },
                    {
                        "name": "step_rr",
                        "step_func_name": "__assignments.AssignmentRoundRobinSmart",
                        "step_func_kwargs": {
                            "N": 9
                        }
                    }
                ]
            },
            pipe.ExtPipeline(name='p3',
                             steps_to_figures=[0],
                             steps=[
                                 pipe.Step(name='distance',
                                           steps_to_aggregate=[0],
                                           step=iops.IoPDistance(
                                               how=iops.IoPDistanceType.CLUSTERING,
                                               reverse_how=iops.ReverseType.SUBTRACT_BY_MAX,
                                               inner_kwargs={
                                                   "direction": iops.Direction.FROM_BOTH,
                                                   "clustering_clazz": cluster.KMeans,
                                                   "distance_metric_exp": 1
                                               }
                                           ),
                                           post_aggregation=pipe.AggregationFuncPair(
                                               aggregation_func=lambda a, split: np.where(a[..., 0] > split, 0, 1),
                                               aggregation_func_arg_func=lambda a: pipe.ArgFuncOutput(args=(a[0], 0.1))
                                           ),
                                           output_col_names_pre=["SAME"], output_col_names_post=["SAME"]),
                                 pipe.Step(name='neigh',
                                           steps_to_aggregate=[0],
                                           step=iops.IoPNeighbor(
                                               how=iops.IoPNeighborType.K_DISTANCE,
                                               reverse_how=iops.ReverseType.SUBTRACT_BY_MAX,
                                               inner_kwargs={
                                                   "direction": iops.Direction.FROM_BOTH,
                                                   "distance_metric_exp": 1,
                                                   "neighbor_kwargs": {
                                                       "n_neighbors": 5
                                                   }
                                               },
                                           ),
                                           post_aggregation=pipe.AggregationFuncPair(
                                               aggregation_func=lambda a, split: np.where(a[..., 1] >= split, 0, 1),
                                               aggregation_func_arg_func=lambda a: pipe.ArgFuncOutput(args=(a[
                                                                                                                0],
                                                                                                            0.195))
                                           )
                                           ),
                                 aggregators.StepAggregateFlattener('aggregate',
                                                                    to_aggregate=[1, 2]),
                                 pipe.Step(name='step_rr',
                                           step=assignments.AssignmentRoundRobinSmart(N=9))
                             ])
    ),
    (
            # this one is simpler but is really tricky.
            {
                'name': 'p4',
                'steps': [
                    {
                        'func_name': '__aggregators.StepAggregateWeightedSum',
                        'func_kwargs': {
                            'name': 's1',
                            'to_aggregate': [-1],
                            'weights': [0.3, 0.3],
                        },
                        'post_aggregation_pair': {
                            'aggregation_func_code': 'lambda a: numpy.where(a[..., 0]>=0.75, 1, 0)',
                            'aggregation_func_arg_func_code': 'lambda a: pipe.ArgFuncOutput(args=(a[0],))',
                        }
                    }
                ]
            },
            pipe.ExtPipeline(name='p4',
                             steps=[
                                 aggregators.StepAggregateWeightedSum(
                                     name='s1',
                                     weights=[0.3, 0.3],
                                     to_aggregate=[-1],
                                     post_aggregation=pipe.AggregationFuncPair(
                                         aggregation_func=lambda a: np.where(a[..., 0] >= 0.75, 1, 0),
                                         aggregation_func_arg_func=lambda a: pipe.ArgFuncOutput(args=(a[0],))
                                     ))
                             ])
    )
])
def test_pipeline_raw_dict(initial_raw: dict, expected: pipe.ExtPipeline):
    # also here, we do the very same tests of above but from dict, because
    # this is what really matters (e.g., the Union that are difficult to match).
    initial_parsed = raw_pipe.PipelineRaw.from_dict(initial_raw)
    got = initial_parsed.parse()
    assert expected == got


def test_specific():
    # this is more specific because I want to test a specific aspect (i.e.,
    # the correct handling of numpy.inf).

    conf = raw_pipe.StepRaw(
        name='s1',
        step_func_name='__iops.IoPNeighbor',
        step_func_kwargs={
            'how': '_iops.IoPNeighborType.K_LABEL_COUNT',
            'reverse_how': '_iops.ReverseType.SUBTRACT_BY_MAX',
            'inner_kwargs': {
                'distance_metric_exp': '_numpy.inf'
            }
        }
    )

    expected = pipe.Step(
        name='s1',
        step=iops.IoPNeighbor(how=iops.IoPNeighborType.K_LABEL_COUNT,
                              reverse_how=iops.ReverseType.SUBTRACT_BY_MAX,
                              inner_kwargs={'distance_metric_exp': np.inf})
    )

    got = conf.parse()
    assert isinstance(got.step, expected.step.__class__)
    assert got.step.inner_kwargs['distance_metric_exp'] == np.inf


def test_no_steps():
    # here we test that the pipeline can't be instantiated if there are no steps in it.
    with pytest.raises(ValueError):
        raw_pipe.PipelineRaw(name='p1', steps=[])
