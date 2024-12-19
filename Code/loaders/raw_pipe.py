import dataclasses
import typing
import warnings

import mashumaro

import pipe
from . import base

# the functions we eval needs this import
# IT CANNOT BE REMOVED
import numpy


@dataclasses.dataclass
class StepPostAggregationPairRaw(mashumaro.DataClassDictMixin, base.RawToParsed[pipe.AggregationFuncPair]):
    aggregation_func_name: typing.Optional[str] = dataclasses.field(default=None)
    aggregation_func_kwargs: typing.Optional[dict] = dataclasses.field(default_factory=dict)

    aggregation_func_code: typing.Optional[str] = dataclasses.field(default=None)
    aggregation_func_arg_func_code: typing.Optional[str] = dataclasses.field(default=None)

    def __post_init__(self):
        if self.aggregation_func_name is None and self.aggregation_func_code is None:
            raise ValueError('At least one between \'aggregation_func\' and \'aggregation_func_code\' '
                             'must be not None')

    def parse(self) -> pipe.AggregationFuncPair:
        # check which one is provided.
        if self.aggregation_func_name is not None:
            aggr_func = base.load_func(self.aggregation_func_name, self.aggregation_func_kwargs)
        else:
            aggr_func = eval(self.aggregation_func_code)
        return pipe.AggregationFuncPair(
            aggregation_func=aggr_func,
            aggregation_func_arg_func=None if self.aggregation_func_arg_func_code is None else
            eval(self.aggregation_func_arg_func_code)
        )


@dataclasses.dataclass
class FuncPairWithPostAggregationPair(base.FuncPair, mashumaro.DataClassDictMixin, base.RawToParsed[pipe.Step]):
    post_aggregation_pair: typing.Optional[typing.Union[base.FuncPair, StepPostAggregationPairRaw]] = dataclasses.field(
        default=None)

    def parse(self) -> pipe.Step:
        # print('correctly inter')
        post_aggregation = self.post_aggregation_pair.parse() if self.post_aggregation_pair is not None else None
        if post_aggregation is not None:
            self.func_kwargs['post_aggregation'] = post_aggregation
        return super().parse()


@dataclasses.dataclass
class StepRaw(mashumaro.DataClassDictMixin, base.RawToParsed[pipe.Step]):
    """
    Dataclass corresponding to `pipe.Step`.
    """
    name: str

    # step_func_name and step_func_kwargs must be called this way to avoid ambiguities
    # with FuncPair.
    step_func_name: str
    # this is used to instantiate the func (i.e., if it is a class or a function (returning
    # another function) these are the arguments to pass to it).
    step_func_kwargs: typing.Optional[typing.Dict] = dataclasses.field(default=None)

    # this is used to pass arguments from List[StepInput] to the concrete func instantiated
    # according to func_name and func_kwargs
    arg_func_code: typing.Optional[str] = dataclasses.field(default=None)

    # list of steps to aggregate
    steps_to_aggregate: typing.Optional[typing.List[int]] = dataclasses.field(default=None)

    # aggregation_func_name: typing.Optional[str] = dataclasses.field(default=None)
    # like above, aggregation_func_kwargs is USED TO INSTANTIATE the function whose name
    # is contained in aggregation_func_name.
    # aggregation_func_kwargs: typing.Optional[typing.Dict] = dataclasses.field(default=None)

    # this is used to pass arguments from the output of the main step function to the aggregation function.
    # aggregation_func_arg_func_code: typing.Optional[str] = dataclasses.field(default=None)
    post_aggregation_pair: typing.Optional[typing.Union[base.FuncPair, StepPostAggregationPairRaw]] = dataclasses.field(
        default=None)

    output_col_names_pre: typing.Optional[typing.List[str]] = dataclasses.field(default_factory=list)
    output_col_names_post: typing.Optional[typing.List[str]] = dataclasses.field(default_factory=list)

    def __post_init__(self):
        if len(self.output_col_names_pre) == 0 and len(self.output_col_names_post):
            warnings.warn(f'No output column names provided for step {self.name}. '
                          f'This lack may cause issues during execution.')

    def parse(self) -> pipe.Step:
        step_func = base.load_func(self.step_func_name, self.step_func_kwargs)

        return pipe.Step(name=self.name,
                         step=step_func,
                         arg_func=None if self.arg_func_code is None else eval(self.arg_func_code),
                         post_aggregation=self.post_aggregation_pair.parse() if self.post_aggregation_pair is not None else None,
                         steps_to_aggregate=self.steps_to_aggregate,
                         output_col_names_pre=self.output_col_names_pre,
                         output_col_names_post=self.output_col_names_post)


@dataclasses.dataclass
class PipelineRaw(mashumaro.DataClassDictMixin, base.RawToParsed[pipe.ExtPipeline]):
    name: str
    steps: typing.List[typing.Union[StepRaw, FuncPairWithPostAggregationPair]]
    short_name: typing.Optional[str] = dataclasses.field(default=None)
    # steps: typing.List[typing.Union[StepRaw, base.FuncPair[StepRaw]]]
    # where risk is computed (or, at least, the "important" value where assignment is computed.
    risk_idx: typing.Optional[int] = dataclasses.field(default=None)
    # these are exported during IoPs experiments.
    steps_to_export: typing.Optional[typing.List[int]] = dataclasses.field(default=None)
    # these are what we really evaluate during IoPs experiment.
    steps_to_evaluate: typing.Optional[typing.List[int]] = dataclasses.field(default=None)
    # these are what we plot
    steps_to_figures: typing.Optional[typing.List[int]] = dataclasses.field(default=None)

    def __post_init__(self):
        # a bit of sanity checks. No need to go on.
        if len(self.steps) == 0:
            raise ValueError('No steps found')

    def parse(self: "PipelineRaw") -> pipe.ExtPipeline:
        steps = [s.parse() for s in self.steps]
        return pipe.ExtPipeline(name=self.name, steps=steps, short_name=self.short_name,
                                steps_to_export=self.steps_to_export, steps_to_evaluate=self.steps_to_evaluate,
                                pre_assignment_idx=self.risk_idx, steps_to_figures=self.steps_to_figures)
