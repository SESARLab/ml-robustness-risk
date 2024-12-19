import typing

from sklearn import datasets
import numpy as np
import pytest
import xarray as xr

from . import ext, steps as st


def _step_func(val: typing.Sequence[np.ndarray]):
    assert len(val) == 2

    x1 = val[0]
    x2 = val[1]

    return np.hstack([x1, x2])


@pytest.mark.parametrize('step, step_input, expected_output', [
    (
        st.Step('s1', np.multiply, steps_to_aggregate=[-1],
                arg_func=lambda step_input: st.ArgFuncOutput(args=(step_input[0].X, step_input[0].y))),
        [st.StepInput(X=np.arange(10), y=np.arange(10))],
        (np.multiply(np.arange(10), np.arange(10)), None)
    ),
    (
        st.Step('s2', _step_func, steps_to_aggregate=[-1, 0],
                arg_func=lambda step_input: st.ArgFuncOutput(args=([step_input[0].X, step_input[1].X],))),
        [st.StepInput(X=np.array([list(range(10)) * 10]).reshape(10, -1), y=None),
         st.StepInput(X=np.array([list(range(10)) * 10]).reshape(10, -1), y=None)],
        (np.hstack([np.array([list(range(10)) * 10]).reshape(10, -1),
                           np.array([list(range(10)) * 10]).reshape(10, -1)]), None)
    ),
    # now we do the very same as we did above but with the usage of an aggregation
    # function.
    (
        st.Step('s3', _step_func, steps_to_aggregate=[-1, 0],
                arg_func=lambda step_input: st.ArgFuncOutput(args=([step_input[0].X, step_input[1].X],)),
                post_aggregation=st.AggregationFuncPair(
                    aggregation_func=np.sum,
                    # means the array received as input and axis=1
                    aggregation_func_arg_func=lambda a: st.ArgFuncOutput(args=(a[0], 1))
                )
                ),
        [st.StepInput(X=np.array([list(range(10)) * 10]).reshape(10, -1), y=None),
         st.StepInput(X=np.array([list(range(10)) * 10]).reshape(10, -1), y=None)],
        (np.sum(np.hstack([np.array([list(range(10)) * 10]).reshape(10, -1),
                    np.array([list(range(10)) * 10]).reshape(10, -1)]), axis=1), None)
    ),
    # now we do some other checks with a more detailed expected output.
    (
        st.Step('s4', np.multiply, steps_to_aggregate=[0, 1],
                arg_func=lambda step_input: st.ArgFuncOutput(args=(step_input[0].X, step_input[1].X)),
                output_col_names_pre=['col1'], output_col_names_post=['col1']
                ),
        [st.StepInput(X=np.arange(10), y=np.random.default_rng().choice([1, 0], 10)),
         st.StepInput(X=np.arange(10)*1.5, y=np.random.default_rng().choice([1, 0], 10))
         ],
        st.StepOutput(actual=[np.multiply(np.arange(10), np.arange(10) * 1.5), None],
                      pre_aggregation_output=xr.DataArray(np.multiply(np.arange(10), np.arange(10) * 1.5).reshape(-1, 1),
                                                          dims=('x', 'y'), coords={'y': ['col1']}),
                      # post is empty because there is no aggregation function.
                      post_aggregation_output=xr.DataArray()
                      )
    ),
    # now we do some other checks with a more detailed expected output AND
    # aggregation.
    (
        st.Step('s5', step=lambda data: np.vstack(data).T, steps_to_aggregate=[0, 1],
                arg_func=lambda step_input: st.ArgFuncOutput(args=([step_input[0].X, step_input[1].X],)),
                post_aggregation=st.AggregationFuncPair(
                    aggregation_func=np.sum,
                    # per-column sum.
                    aggregation_func_arg_func=lambda a: st.ArgFuncOutput(args=(a[0], 1)),
                ),
                output_col_names_pre=['col1', 'col2'],
                output_col_names_post=['red_col1']
                ),
        [st.StepInput(X=np.arange(10), y=np.random.default_rng().choice([1, 0], 10)),
         st.StepInput(X=np.arange(10)*1.5, y=np.random.default_rng().choice([1, 0], 10))
         ],
        # reshape on raw output is not needed.
        st.StepOutput(actual=[np.sum(np.vstack([np.arange(10), np.arange(10) * 1.5]).T, axis=1), None],
                      pre_aggregation_output=xr.DataArray(np.vstack([np.arange(10), np.arange(10) * 1.5]).T, dims=('x', 'y'),
                                                          coords={'y': ['col1', 'col2']}),
                      post_aggregation_output=xr.DataArray(
                          np.sum(np.vstack([np.arange(10), np.arange(10)*1.5]).T, axis=1).reshape(-1, 1),
                          dims=('x', 'y'), coords={'y': ['red_col1']})
                      )
    )
])
def test_step(step: st.Step, step_input: typing.Sequence[st.StepInput],
              expected_output: typing.Union[typing.Tuple[np.ndarray, np.ndarray], st.StepOutput]):

    # return also non-raw only if the expected output contains data
    # to be checked against, otherwise just return the plain output.
    return_non_raw = isinstance(expected_output, st.StepOutput)

    out = step.visit(val=step_input, return_non_raw=return_non_raw)

    def _check_raw(expected_X_, expected_y_, got_):
        # check X
        assert np.all(expected_X_ == got_.actual[0])
        # check y
        assert np.all(expected_y_ == got_.actual[1])

    if isinstance(expected_output, tuple):
        # just a collection of X and y.
        _check_raw(expected_X_=expected_output[0], expected_y_=expected_output[1],
                   got_=out)
    else:
        # more thorough checks.
        _check_raw(expected_X_=expected_output.actual[0], expected_y_=expected_output.actual[1],
                   got_=out)

        if not np.all(expected_output.post_aggregation_output == out.post_aggregation_output):
            print(step.name)

        assert np.all(expected_output.pre_aggregation_output == out.pre_aggregation_output)
        assert np.all(expected_output.post_aggregation_output == out.post_aggregation_output) or \
               (len(expected_output.post_aggregation_output.shape) == 0 and len(out.post_aggregation_output.shape) == 0)
                # len on an empty array gives as issues. But the shape of an empty array can be checked with no issues.


def test_step_creation_with_error():
    with pytest.raises(ValueError):
        # it is enough to require > 1 steps without providing an arg_func.
        st.Step(name='s1', step=ext.ExtendedFunctionTransformer(func=np.divide, positional_args=[10]),
                steps_to_aggregate=[0, 1])


def test_step_output_with_error():
    step = st.Step(name='s1', step=ext.ExtendedFunctionTransformer(func=np.divide, positional_args=[10]), steps_to_aggregate=[-1])
    X, y = datasets.make_classification(100, 20)
    with pytest.raises(ValueError):
        # this is going to fail because we did not specify the columns for output.
        step.visit([st.StepInput(X=X, y=y)], return_non_raw=True)
