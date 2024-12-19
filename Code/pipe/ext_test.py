import typing

import numpy as np
import pytest

from . import ext, steps as st


@pytest.mark.parametrize('steps, expected_steps_to_save, save_all, important_steps', [
    (
            # if not specified, it takes as input the input passed to the list.
            [st.Step(name='s1', step=ext.ExtendedFunctionTransformer(func=np.divide, positional_args=[10]))],
            set(),
            False, []
    ),
    (
            # if explicitly specified as empty, then we expect an empty set.
            [st.Step(name='s1', step=ext.ExtendedFunctionTransformer(func=np.divide, positional_args=[10]),
                     steps_to_aggregate=[])],
            set(),
            False, []
    ),
    (
            # two steps one after the other (standard pipeline)
            [st.Step(name='s1', step=ext.ExtendedFunctionTransformer(func=np.divide, positional_args=[10])),
             st.Step(name='s2', step=ext.ExtendedFunctionTransformer(func=np.divide, positional_args=[10])),
             ],
            {0},
            False, []
    ),
    (
            # two steps one after the other, no input required
            [st.Step(name='s1', step=ext.ExtendedFunctionTransformer(func=np.divide, positional_args=[10]),
                     steps_to_aggregate=[]),
             st.Step(name='s2', step=ext.ExtendedFunctionTransformer(func=np.divide, positional_args=[10]),
                     steps_to_aggregate=[]),
             ],
            set(),
            False, []
    ),
    (
            # three steps, complicated.
            [st.Step(name='s1', step=ext.ExtendedFunctionTransformer(func=np.divide, positional_args=[10])),
             st.Step(name='s2', step=ext.ExtendedFunctionTransformer(func=np.divide, positional_args=[10])),
             st.Step(name='s2', step=ext.ExtendedFunctionTransformer(func=np.divide, positional_args=[10]),
                     steps_to_aggregate=[0, 1], arg_func=lambda val: val),  # basic arg_func, we are not using it now.
             ],
            {0, 1},
            False, []
    ),
    (
            [st.Step(name='s1', step=ext.ExtendedFunctionTransformer(func=np.divide, positional_args=[10]),
                     steps_to_aggregate=[], post_aggregation=st.AggregationFuncPair(aggregation_func=np.sum),
                     output_col_names_pre=['col1'], output_col_names_post=['col1'])],
            set(),
            False, []
    ),
    # playing with save all
    (
            # if not specified, it takes as input the input passed to the list.
            [st.Step(name='s1', step=ext.ExtendedFunctionTransformer(func=np.divide, positional_args=[10]))],
            set(),
            True, []
    ),
    (
            # if explicitly specified as empty, then we expect an empty set.
            [st.Step(name='s1', step=ext.ExtendedFunctionTransformer(func=np.divide, positional_args=[10]),
                     steps_to_aggregate=[])],
            set(),
            True, []
    ),
    (
            # two steps one after the other (standard pipeline)
            [st.Step(name='s1', step=ext.ExtendedFunctionTransformer(func=np.divide, positional_args=[10])),
             st.Step(name='s2', step=ext.ExtendedFunctionTransformer(func=np.divide, positional_args=[10])),
             ],
            {0},
            True, []
    ),
    (
            # two steps one after the other, no input required
            [st.Step(name='s1', step=ext.ExtendedFunctionTransformer(func=np.divide, positional_args=[10]),
                     steps_to_aggregate=[]),
             st.Step(name='s2', step=ext.ExtendedFunctionTransformer(func=np.divide, positional_args=[10]),
                     steps_to_aggregate=[]),
             ],
            set(),
            True, []
    ),
    (
            # three steps, complicated.
            [st.Step(name='s1', step=ext.ExtendedFunctionTransformer(func=np.divide, positional_args=[10])),
             st.Step(name='s2', step=ext.ExtendedFunctionTransformer(func=np.divide, positional_args=[10])),
             st.Step(name='s2', step=ext.ExtendedFunctionTransformer(func=np.divide, positional_args=[10]),
                     steps_to_aggregate=[0, 1], arg_func=lambda val: val),  # basic arg_func, we are not using it now.
             ],
            {0, 1},
            True, []
    ),
    (
            [st.Step(name='s1', step=ext.ExtendedFunctionTransformer(func=np.divide, positional_args=[10]),
                     steps_to_aggregate=[], post_aggregation=st.AggregationFuncPair(aggregation_func=np.sum),
                     output_col_names_pre=['col1'],
                     output_col_names_post=['col1'])],
            set(),
            True, []
    ),
    # playing with important steps
    (
            # three steps, complicated.
            [st.Step(name='s1', step=ext.ExtendedFunctionTransformer(func=np.divide, positional_args=[10]),
                     output_col_names_pre=['col1']),
             st.Step(name='s2', step=ext.ExtendedFunctionTransformer(func=np.divide, positional_args=[10]),
                     output_col_names_pre=['col1']),
             st.Step(name='s2', step=ext.ExtendedFunctionTransformer(func=np.divide, positional_args=[10]),
                     steps_to_aggregate=[0, 1], arg_func=lambda val: val),  # basic arg_func, we are not using it now.
             ],
            {0, 1},
            True, [0, 1]
    ),
    (
            [st.Step(name='s1', step=ext.ExtendedFunctionTransformer(func=np.divide, positional_args=[10]),
                     steps_to_aggregate=[], post_aggregation=st.AggregationFuncPair(aggregation_func=np.sum),
                     output_col_names_pre=['col1'],
                     output_col_names_post=['col1'])],
            set(),
            True, [0]
    ),
])
def test_pipeline_steps_to_save(steps, expected_steps_to_save, save_all, important_steps):
    p = ext.ExtPipeline(steps=steps)
    assert p.steps_to_save == expected_steps_to_save


@pytest.mark.parametrize('step, important_steps', [
    # this fails because this step is in important steps but the step does not have column names
    (
            st.Step(name='s2', step=np.divide),
            [0]
    ),
    # this fails because it has column names pre but not post since it has an aggregation (and is important)
    (
            st.Step(name='s3', step=np.divide, post_aggregation=st.AggregationFuncPair(aggregation_func=np.sum),
                    output_col_names_pre=['col1']),
            [0]
    ),
])
def test_pipeline_fail(step: st.Step, important_steps: typing.List[int]):
    with pytest.raises(ValueError):
        ext.ExtPipeline(steps=[step], steps_to_export=important_steps)


@pytest.mark.parametrize('p, X_, y_, expected_output', [
    (
            ext.ExtPipeline(name='p1', steps=[
                st.Step(name='s1', step=ext.ExtendedFunctionTransformer(np.divide, positional_args=[10]))],
            ),
            np.multiply(np.arange(10), 10),
            None,
            (np.arange(10).astype(float), None)
    ),
    (
            ext.ExtPipeline(name='p1', steps=[
                st.Step(name='s1', step=ext.ExtendedFunctionTransformer(np.add, positional_args=[10])),
                st.Step(name='s2', step=ext.ExtendedFunctionTransformer(np.add, positional_args=[10])),
                st.Step(name='s3', step=ext.ExtendedFunctionTransformer(np.add, consider_X=False),
                        steps_to_aggregate=[0, 1], arg_func=lambda x: st.ArgFuncOutput(args=(x[0].X, x[1].X))),
            ],
            ),
            np.arange(10),
            None,
            (np.arange(10) + 20 + np.arange(10) + 10, None)
    ),
    (
            ext.ExtPipeline(name='p1', steps=[
                st.Step(name='s1', step=ext.ExtendedFunctionTransformer(np.add, positional_args=[10]),
                        steps_to_aggregate=[0]),
                st.Step(name='s2', step=ext.ExtendedFunctionTransformer(np.add, positional_args=[10]),
                        steps_to_aggregate=[-1]),
                st.Step(name='s3', step=ext.ExtendedFunctionTransformer(np.add, positional_args=[10]),
                        steps_to_aggregate=[-1])
            ]),
            np.arange(10),
            None,
            (np.arange(10) + 10, None)
    ),
    (
            ext.ExtPipeline(name='p1', steps=[
                st.Step(name='s1', step=ext.ExtendedFunctionTransformer(np.add, positional_args=[10]),
                        steps_to_aggregate=[0]),
                st.Step(name='s2', step=ext.ExtendedFunctionTransformer(np.add, positional_args=[10]),
                        steps_to_aggregate=[-1]),
                st.Step(name='s3', step=ext.ExtendedFunctionTransformer(np.add, positional_args=[10]),
                        steps_to_aggregate=[-1], output_col_names_pre=['col1'])
            ], steps_to_export=[2]),
            np.arange(10),
            None,
            (np.arange(10) + 10, None)
    ),
    (
            ext.ExtPipeline(name='p1', steps=[
                st.Step(name='s1', step=ext.ExtendedFunctionTransformer(np.add, positional_args=[10]),
                        steps_to_aggregate=[0], output_col_names_pre=['col1']),
                st.Step(name='s2', step=ext.ExtendedFunctionTransformer(np.add, positional_args=[10]),
                        steps_to_aggregate=[-1], output_col_names_pre=['col1']),
                st.Step(name='s3', step=ext.ExtendedFunctionTransformer(np.add, positional_args=[10]),
                        steps_to_aggregate=[-1], output_col_names_pre=['col1'])
            ], steps_to_export=[2]),
            np.arange(10),
            None,
            (np.arange(10) + 10, None)
    )
])
def test_pipeline_execution(p: ext.ExtPipeline, X_, y_, expected_output):
    output = p.fit_transform(X_, y_)
    assert np.all(expected_output[0] == output[0])
    assert (expected_output[1] is None and output[1] is None) or np.all(expected_output[1] == output[1])

    if len(p.important_steps) > 0:
        assert len(p.output_for_export) == len(p.important_steps)
        out_steps = [out[0] for out in p.output_for_export.values()]
        sub_steps = [step for i, step in enumerate(p.steps) if i in p.important_steps]
        assert out_steps == sub_steps
