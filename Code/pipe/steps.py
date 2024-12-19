import dataclasses
import typing

import numpy as np
import xarray as xr

from utils import Transformer
from . import ext

T = typing.TypeVar('T')
V = typing.TypeVar('V')


@dataclasses.dataclass
class StepInput:
    X: np.ndarray
    y: typing.Optional[np.ndarray] = dataclasses.field(default=None)
    # in one case, i.e., the first, it is empty
    step: typing.Optional["Step"] = dataclasses.field(default=None)


@dataclasses.dataclass
class StepOutput:
    """
    Container class for the output produced by a `Step`.

    Attributes
    ---------
    actual : typing.Tuple[np.ndarray, typing.Optional[np.ndarray]] is the "concrete" output, that is,
            the output produced by the `Step` in terms of X and, optional y.

    pre_aggregation_output : typing.Tuple[xr.DataArray, xr.DataArray] is the output produced by the `Step.step` function.
            The array is two-dimensional and contains coordinates for each y, that is, each column.
            The second array may be empty.

    post_aggregation_output : typing.Tuple[xr.DataArray, xr.DataArray] is the output produced by applying the eventual
            aggregation function on `initial_output`. It follows the same rules of `initial_output`.

    """
    # this one is meant for consumption.
    actual: typing.Tuple[np.ndarray, typing.Optional[np.ndarray]]
    # the output retrieved before executing the aggregation function.
    pre_aggregation_output: xr.DataArray
    post_aggregation_output: xr.DataArray

    @staticmethod
    def with_actual_only(actual: typing.Tuple[np.ndarray, typing.Optional[np.ndarray]]):
        return StepOutput(actual=actual, pre_aggregation_output=xr.DataArray(), post_aggregation_output=xr.DataArray())

    def get_pre_and_post_as_xr(self) -> xr.DataArray:
        # if empty, len will fail. Use the length of the shape instead.
        if len(self.post_aggregation_output.shape) == 0:
            return self.pre_aggregation_output
        else:
            return xr.concat([self.pre_aggregation_output, self.post_aggregation_output], dim='y')


@dataclasses.dataclass
class ArgFuncOutput(typing.Generic[T, V]):
    args: typing.Tuple[T, ...]
    kwargs: typing.Optional[typing.Dict[str, V]] = dataclasses.field(default_factory=dict)

    def expand(self) -> typing.Tuple[typing.Tuple[T, ...], typing.Optional[typing.Dict[str, V]]]:
        return self.args, self.kwargs if self.kwargs is not None else {}


@dataclasses.dataclass
class AggregationFuncPair:
    # A function executed after the main step.
    # It takes as input the output of the step function as a tuple (X, y) and returns the modified X only.
    aggregation_func: typing.Optional[typing.Callable[[np.ndarray, ...], np.ndarray]] = dataclasses.field(default=None)
    # a function that determines the input of the aggregation_func.
    # This way we can pass arbitrary values to it.
    aggregation_func_arg_func: typing.Optional[
        typing.Callable[[typing.Tuple[np.ndarray, typing.Optional[np.ndarray]]], ArgFuncOutput]] = dataclasses.field(
        default=None)


@dataclasses.dataclass
class Step:
    name: str
    step: typing.Union[typing.Callable[[...], typing.Union[np.ndarray, typing.Tuple[np.ndarray, np.ndarray]]],
    Transformer, "ext.ExtPipeline", "iops.Iop"]  # = dataclasses.field(default=None)
    arg_func: typing.Optional[typing.Callable[[typing.Iterable[StepInput]], ArgFuncOutput]] = dataclasses.field(
        default=None)
    # the list of steps whose output is required as input in this step.
    steps_to_aggregate: typing.Optional[typing.List[int]] = dataclasses.field(default=None)

    # # for aggregation following the execution of the step when e.g., the output is multidimensional,
    # # and we want to reduce to 1d, i.e., one value for data point.
    # # It takes as input the output of the step function as a tuple (X, y) and returns the modified X only.
    # aggregation_func: typing.Optional[typing.Callable[[np.ndarray, ...], np.ndarray]] = dataclasses.field(default=None)
    # aggregation_func_arg_func: typing.Optional[typing.Callable[[typing.Tuple[np.ndarray, typing.Optional[np.ndarray]]], ArgFuncOutput]] = dataclasses.field(default=None)
    post_aggregation: typing.Optional[AggregationFuncPair] = dataclasses.field(default=None)

    # export_output: typing.Optional[dict[str,str]] = dataclasses.field(default=None)

    output_col_names_pre: typing.Optional[typing.List[str]] = dataclasses.field(default_factory=list)
    output_col_names_post: typing.Optional[typing.List[str]] = dataclasses.field(default_factory=list)

    @staticmethod
    def _default_aggregation_func_arg_func(
            step_input: typing.Tuple[np.ndarray, typing.Optional[np.ndarray]]) -> ArgFuncOutput:
        return ArgFuncOutput(args=(step_input[0],))

    @property
    def used_output_columns(self) -> typing.List[str]:
        """
        Returns the columns that are effectively used in the output of the IoP.
        :return:
        """
        if self.post_aggregation is not None:
            return self.output_col_names_post
        return self.output_col_names_pre

    def __post_init__(self):
        # # check if it is a dictionary. Far from being optimal...
        # if isinstance(self.post_aggregation, dict):
        #     self.post_aggregation = AggregationFuncPair(**self.post_aggregation)

        # if self.aggregate is not None and self.arg_func is None:
        if self.steps_to_aggregate is not None and len(self.steps_to_aggregate) == 1 and self.arg_func is None:
            # set as default
            # the callback that returns the first value of the input list.
            self.arg_func = lambda x: ArgFuncOutput(args=(x[0].X, x[0].y))
        elif self.steps_to_aggregate is not None and len(self.steps_to_aggregate) > 1 and self.arg_func is None:
            raise ValueError('if aggregate is not None, then also self.arg_func cannot be None')

        if self.post_aggregation is not None and self.post_aggregation.aggregation_func is not None and self.post_aggregation.aggregation_func_arg_func is None:
            # set default.
            self.aggregation_func_arg_func = Step._default_aggregation_func_arg_func
        if self.post_aggregation is not None and self.post_aggregation.aggregation_func is None and self.post_aggregation.aggregation_func_arg_func is not None:
            raise ValueError('if aggregation is None, then self.aggregation_func_arg_func must be None')

    def visit(self, val: typing.Sequence[StepInput], return_non_raw: bool = False, **fit_params) -> StepOutput:
        if len(self.steps_to_aggregate) != len(val):
            # this should never happen in practice.
            raise ValueError(f'The step required {len(self.steps_to_aggregate)} steps to aggregate, '
                             f'but received as input {len(val)} steps')
        else:
            # args, kwargs = self.arg_func(val)
            # if kwargs is None:
            #    kwargs = {}
            raw_args = self.arg_func(val)
            args, kwargs = raw_args.expand()

        if isinstance(self.step, ext.ExtendedFunctionTransformer) and args is not None:
            if self.step.positional_args is None:
                self.step.positional_args = args

        if hasattr(self.step, 'ext_transform'):
            res = self.step.fit(*args, **kwargs).ext_transform(*args, **kwargs)
        elif hasattr(self.step, "fit_transform"):
            res = self.step.fit_transform(*args, **kwargs)
        elif hasattr(self.step, 'fit') and hasattr(self.step, 'transform'):
            res = self.step.fit(*args, **kwargs).transform(*args, **kwargs)
        else:
            # it is a callable
            res = self.step(*args, **kwargs)
        out = self.finalize_out(res, return_non_raw=return_non_raw, aggregate=True)
        # print(out.actual[0].shape)
        return out

    @staticmethod
    def _fix_out(output: typing.Union[typing.Tuple[np.ndarray, np.ndarray], np.ndarray]):
        if not isinstance(output, tuple):
            output = (output, None)
        return output

    def finalize_out(self, pre_result: typing.Union[typing.Tuple[np.ndarray, np.ndarray], np.ndarray],
                     aggregate: bool, return_non_raw: bool = False) -> StepOutput:
        output = Step._fix_out(pre_result)
        # output = output
        # now, we apply the aggregator if needed.
        if aggregate and self.post_aggregation is not None and self.post_aggregation.aggregation_func is not None:
            args, kwargs = self.post_aggregation.aggregation_func_arg_func(output).expand()
            output = self.post_aggregation.aggregation_func(*args, **kwargs)
            # and then, we re-create a tuple using this very same function.
            # out = self.finalize_out(result=out, aggregate=False)
            output = Step._fix_out(output)

        result = StepOutput.with_actual_only(actual=output)

        def get_data_for_xr(input_data_) -> np.ndarray:

            if isinstance(input_data_, tuple):
                if input_data_[1] is None:
                    output_ = [input_data_[0]]
                else:
                    output_ = [input_data_[0], input_data_[1]]

                for i in range(len(output_)):
                    if len(output_[i].shape) == 1:
                        # if the shape is 1d, we reshape to 2d to facilitate column concatenation.
                        output_[i] = output_[i].reshape(-1, 1)

                output_ = np.hstack(output_)

            else:
                output_ = input_data_
                if len(output_.shape) == 1:
                    output_ = output_.reshape(-1, 1)

            return output_

        if return_non_raw:
            post_result_xr = xr.DataArray()
            if self.post_aggregation is None or self.post_aggregation.aggregation_func is None:
                pre_result_col_names = [f'{self.name}_{col}' for col in self.output_col_names_pre]
            else:

                pre_result_col_names = [f'{self.name}_PRE_{col}' for col in self.output_col_names_pre]
                post_result_col_names = [f'{self.name}_POST_{col}' for col in self.output_col_names_post]
                #
                # if isinstance(self.step, iops.IoPOutlier) or isinstance(self.step, iops.IoPNeighborhood):
                #     print(f'{get_data_for_xr(input_data_=output).shape}: {post_result_col_names}')

                post_result_xr = xr.DataArray(get_data_for_xr(input_data_=output),
                                              dims=('x', 'y'), coords={'y': post_result_col_names})

            pre_result_xr = xr.DataArray(get_data_for_xr(input_data_=pre_result), dims=('x', 'y'),
                                         coords={'y': pre_result_col_names})

            result.pre_aggregation_output = pre_result_xr
            result.post_aggregation_output = post_result_xr

        return result

    def require_previous_steps(self) -> bool:
        return self.steps_to_aggregate is not None and len(self.steps_to_aggregate) > 0

    def __eq__(self, other: "Step"):
        if not isinstance(other, self.__class__):
            return False
        if self.name != other.name:
            return False
        if self.steps_to_aggregate != other.steps_to_aggregate:
            return False
        if type(self.step).__name__ != type(other.step).__name__:
            return False
        if self.output_col_names_pre != other.output_col_names_pre:
            return False
        if self.output_col_names_post != other.output_col_names_post:
            return False
        if (self.post_aggregation is not None and other.post_aggregation is None) or \
            (self.post_aggregation is None and other.post_aggregation is not None):
            return False
        if (
                self.post_aggregation is not None and self.post_aggregation.aggregation_func is not None and other.post_aggregation is not None and other.post_aggregation.aggregation_func is None) or (
                self.post_aggregation is not None and self.post_aggregation.aggregation_func is None and other.post_aggregation is not None and other.post_aggregation.aggregation_func is not None):
            return False
        if (self.arg_func is not None and other.arg_func is None) or (
                self.arg_func is None and other.arg_func is not None):
            return False
        if (
                self.post_aggregation is not None and self.post_aggregation.aggregation_func_arg_func is not None and other.post_aggregation is not None and other.post_aggregation.aggregation_func_arg_func is None) or (
                self.post_aggregation is not None and self.post_aggregation.aggregation_func_arg_func is None and other.post_aggregation is not None and other.post_aggregation.aggregation_func_arg_func is not None):
            return False
        return True
