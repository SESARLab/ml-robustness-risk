import copy
import typing

import numpy as np
from sklearn import preprocessing

import utils
from . import steps as st


class ExtendedFunctionTransformer(preprocessing.FunctionTransformer):
    """
    Extends sklearn.preprocessing.FunctionTransformer by taking as input and returning as output
    (with no modification) also y.

    This class acts as a wrapper on sklearn.preprocessing.FunctionTransformer defining the method ext_transform that
    - takes as input X and y
    - applies the transformation on X as in the base class
    - return transformed X and y.
    """

    def __init__(self, func=None, inverse_func=None, *,
                 X_kwarg: typing.Optional[str] = None,
                 positional_args=None,
                 consider_X=True,
                 validate=False, accept_sparse=False, check_inverse=True,
                 feature_names_out=None, kw_args=None, inv_kw_args=None, ):
        super().__init__(func, inverse_func, validate=validate, accept_sparse=accept_sparse,
                         check_inverse=check_inverse, feature_names_out=feature_names_out, kw_args=kw_args,
                         inv_kw_args=inv_kw_args)
        self.X_kwarg = X_kwarg
        self.positional_args = positional_args
        self.consider_X = consider_X

    def fit(self, X, y=None, **fit_kwargs):
        return super().fit(X, y)

    def ext_transform(self, X, y=None, **transform_kwargs):
        X_ = self._check_input(X, reset=False)
        if self.kw_args is None:
            self.kw_args = {}
        if self.X_kwarg is not None:
            self.kw_args[self.X_kwarg] = X_
        if not self.consider_X:
            X_ = None
            y = None

        if self.X_kwarg is not None and self.positional_args is not None:
            return self.func(*self.positional_args, **self.kw_args), y
        elif self.X_kwarg is None and self.kw_args != {} and self.positional_args is not None:
            return self.func(X_, *self.positional_args, **self.kw_args), y
        elif self.X_kwarg is None and self.kw_args != {} and self.positional_args is None:
            return self.func(X_, **self.kw_args), y
        elif self.X_kwarg is None and self.kw_args == {} and self.positional_args is not None and \
                len(self.positional_args) > 0 and X_ is not None:
            return self.func(X_, *self.positional_args), y
        elif self.X_kwarg is None and self.kw_args == {} and self.positional_args is not None and \
                len(self.positional_args) > 0 and X_ is None:
            return self.func(*self.positional_args), y


KEY_LAST_STEP = -np.inf


class ExtPipeline:

    @property
    def name(self):
        return utils.get_pipeline_name_raw(short_name=self.short_name, full_name=self.full_name)

    def __init__(self, steps: typing.List[st.Step],  # save_all_steps: bool = False,
                 steps_to_export: typing.Optional[typing.List[int]] = None,
                 # which steps to consider during IoPs evaluation.
                 steps_to_evaluate: typing.Optional[typing.List[int]] = None,
                 steps_to_figures: typing.Optional[typing.List[int]] = None,
                 name: typing.Optional[str] = None,
                 short_name: typing.Optional[str] = None,
                 # where risk is retrieved. *always* added to the list of steps to save.
                 pre_assignment_idx: typing.Optional[int] = None,
                 risk_binarization_thresholds: typing.Optional[typing.List[float]] = None
                 ):
        if name is None:
            raise ValueError('Name cannot be None')
        if risk_binarization_thresholds is not None and len(risk_binarization_thresholds) > 1:
            raise ValueError('Binarization threshold length can only be 1')

        # the index of the step where risk values are used. If present is always saved.
        self.pre_assignment_idx = pre_assignment_idx
        # TODO can be made automatic
        self.risk_binarization_thresholds = risk_binarization_thresholds

        self.steps = steps
        self.full_name = name
        self.short_name = short_name

        # this list contains the output of each step whose output is taken as input
        # by another step. Such output is already converted in the form of the data
        # structure taken as input.
        self.output_kept: typing.Dict[typing.Union[int, float], st.StepInput] = {}
        # this holds the output to be consumed by experiment classes.
        # it is a dict of the form: step_idx -> output
        self.output_for_export: typing.Dict[int, typing.Tuple[st.Step, st.StepOutput]] = {}

        # make it in a way that
        # a step may require no input from the previous step (in this case, it will
        # take the initial X, y), also remove pass_output, the output is always passed.
        # then during creation we iterate over the steps fixing the attributes we need such that
        # the loop calls visit and stop.
        # self.named_steps: typing.Dict[str, st.Step] = dict()
        # self.save_all_steps = save_all_steps
        self.steps_to_save = set()

        self.steps_to_export = steps_to_export or []
        self.steps_to_evaluate = steps_to_evaluate or []
        self.steps_to_figures = steps_to_figures or []

        # if set(self.steps_to_plot).difference(set(self.steps_to_evaluate)):

        # whether they are for aggregation or export, we treat them the same way.
        self.important_steps = list(set(self.steps_to_export).union(set(self.steps_to_evaluate)).union(
            set(self.steps_to_figures)))
        if self.important_steps is not None:
            for important_step in self.important_steps:
                if important_step not in range(len(self.steps)):
                    raise ValueError(f'Important step {important_step} does not have a '
                                     f'correspondence among {list(range(len(self.steps)))}')
        self.prepare_from_steps()
        self.named_steps = [s.name for s in self.steps]

        if self.pre_assignment_idx is not None:
            if self.pre_assignment_idx not in range(len(steps)):
                raise ValueError(f'The pre_assignment_idx \'{self.pre_assignment_idx}\' does not have '
                                 f'a correspondence among steps {list(range(len(self.steps)))}')
            self.steps_to_save.add(self.pre_assignment_idx)

    def prepare_from_steps(self):
        for i, step in enumerate(self.steps):
            step: st.Step = step
            # # some sanity checks. If we require the save the steps, we need to ensure the columns
            # # are specified.

            if i in self.important_steps:
                if step.output_col_names_pre is None or len(step.output_col_names_pre) == 0 or \
                        (step.post_aggregation is not None and
                         step.post_aggregation.aggregation_func is not None
                         and (step.output_col_names_post is None or len(step.output_col_names_post) == 0)):
                    raise ValueError(f'Pipeline requires to save this step output but step {step} '
                                     f'does not provide columns.')

            # if the step does not specify which steps it takes as input,
            # it means that it takes as input the previous step
            # (compatibility with standard sklearn pipeline)
            if step.steps_to_aggregate is None:
                # then input from the previous step.
                if i == 0:
                    # this if is not really necessary, but we take it for clarity.
                    # if this is the first step, then we take as input [-1]
                    # indicating the pipeline input.
                    step.steps_to_aggregate = [-1]
                else:
                    step.steps_to_aggregate = [i - 1]
                # then, we also need to set the default arg_func.
                # Note that x is a list of previous input.
                # So with x[0] we grab the first element of the list.
                # Then, we skip x[0][0] containing the Step instance,
                # and we recover the output in terms of X at x[0][1][0]
                # and y at x[0][1][1]
                step.arg_func = lambda x: st.ArgFuncOutput(args=(x[0].X, x[0].y))

            elif len(step.steps_to_aggregate) == 0:
                # if instead it does not require input, then we're fine.
                # (a bit weird tbh)
                continue

            # we now loop over the steps this step takes as input to do some other checks.
            for j in range(len(step.steps_to_aggregate)):
                # if the first step requires to aggregate the first input,
                # then it means it requires -1.
                if step.steps_to_aggregate[j] == 0 and i == 0:
                    step.steps_to_aggregate[j] = -1
                if step.steps_to_aggregate[j] == -1:
                    # no need to add to the list of required steps.
                    continue
                if step.steps_to_aggregate[j] > len(self.steps):
                    raise IndexError(f'{step.steps_to_aggregate[j]} is out of range for {len(self.steps)} steps')
                self.steps_to_save.add(step.steps_to_aggregate[j])

            # self.named_steps[step.name] = step

    def get_required_steps(self, required_steps: typing.List[int]) -> typing.Sequence[st.StepInput]:
        output = []
        for i in required_steps:
            # if i == -1:
            #     output.append(self.output_kept[0])
            # else:
            #     output.append(self.output_kept[i + 1])
            output.append(self.output_kept[i])
        return output

    def fit_transform(self, X: np.ndarray, y: typing.Optional[np.ndarray] = None, **fit_params):
        Xt = X
        yt = y

        # the first input is *always* saved in position [-1].
        self.output_kept[-1] = st.StepInput(step=None, X=Xt, y=yt)

        last_step_output = None

        for i, (step) in enumerate(self.steps):
            step: st.Step = step

            # recover the required outputs of this step.
            # this can also include the previous executed step.
            previous_steps = self.get_required_steps(required_steps=step.steps_to_aggregate)

            try:
                # specify if we need to return all output.
                # return_non_raw_output = self.save_all_steps or i in self.important_steps
                return_non_raw_output = i in self.important_steps
                res: st.StepOutput = step.visit(previous_steps, return_non_raw=return_non_raw_output)
            except Exception as e:
                raise e from e

            Xt = res.actual[0]
            yt = res.actual[1]

            # if self.save_all_steps:
            #     self.output_for_export_all.append((step, res))

            if i in self.important_steps:
                self.output_for_export[i] = (step, res)

            if i in self.steps_to_save:
                self.output_kept[i] = st.StepInput(step=step, X=Xt, y=yt)

            if i == len(self.steps) - 1:
                last_step_output = st.StepInput(step=step, X=Xt, y=yt)

        # we do this track to make sure to retrieve the last step.
        # self.output_kept[-1] = self.output_kept[i]
        # self.output_kept[KEY_LAST_STEP] = self.output_kept[i]
        self.output_kept[KEY_LAST_STEP] = last_step_output
        return Xt, yt

    def __len__(self):
        return len(self.steps)

    def __eq__(self, other):
        if not isinstance(other, ExtPipeline):
            return False
        if self.full_name != other.full_name:
            return False
        if self.short_name != other.short_name:
            return False
        # if self.save_all_steps != other.save_all_steps:
        #     return False
        if self.important_steps != other.important_steps:
            return False
        if self.steps_to_save != other.steps_to_save:
            return False
        if len(self.steps) != len(other.steps):
            return False
        for step_self, step_other in zip(self.steps, other.steps):
            if step_self != step_other:
                return False
        return True

    def __repr__(self):
        return f'{self.name}: {[s.name for s in self.steps]}'

    # def clone_template(self, starting_point: typing.Optional[int] = 0) -> "ExtPipeline":
    #     """
    #     Returns a "raw" copy of this pipeline, removing all the instantiation we did on the steps
    #     :return:
    #     """
    #     if starting_point is None:
    #         starting_point = 0
    #     if starting_point < 0 or starting_point >= len(self.steps):
    #         raise ValueError(f'Invalid starting point. Admissible values: [0, {len(self.steps)-1}]')
    #     # we copy the steps starting from `starting_point` (inclusive), modifying
    #     # them when needed.
    #     steps = []
    #     # now we need to modify the steps a bit
    #     for i, step in enumerate(self.steps[starting_point:]):
    #         new_step = copy.deepcopy(step)
    #         new_step.steps_to_aggregate = None if len(step.steps_to_aggregate) == 0 else []
    #         # here we check that the step does not have any dependencies on steps that we are not going to export.
    #         # we need to be careful. If we are operating on the first step, then having a dependency on the previous
    #         # step is fine. The dependency will be replaced by [-1].
    #         for j in step.steps_to_aggregate:
    #             if i == 0:
    #                 if j < starting_point and j != -1 and j != starting_point-1:
    #                     raise ValueError(f'Step {step.name} has a dependency on step {j}, which is not included.')
    #             else:
    #                 # if it is not the first one, then we do those checks as normal.
    #                 if j < starting_point and j != -1:
    #                     raise ValueError(f'Step {step.name} has a dependency on step {j}, which is not included.')
    #             # if it passes this check, we still need to modify j, because the counter
    #             # is in any case wrong (we need to shift it back. For instance,
    #             # assume a three-step pipeline, each step depending on the previous one.
    #             # We cut at 1 (included). Step 2 (the last) depends on step 1. After the cut,
    #             # step 2 (now step 1) depends on step 1 (now step 0)).
    #             new_step.steps_to_aggregate.append(i-starting_point)
    #
    #         steps.append(new_step)
    #
    #     new_assignment_idx = self.pre_assignment_idx
    #     if new_assignment_idx is not None:
    #         new_assignment_idx -= starting_point
    #     # note that if this value becomes -1, it means that risk is not retrieved
    #     # within the pipeline. So we set it to None.
    #     if new_assignment_idx < 0:
    #         new_assignment_idx = None
    #
    #     return ExtPipeline(steps=steps, pre_assignment_idx=new_assignment_idx,
    #                        steps_to_export=self.steps_to_export, steps_to_evaluate=self.steps_to_evaluate,
    #                        name=self.name)

    # def clone_template_from_risk(self) -> "ExtPipeline":
    #     """
    #     Returns a non-executed copy of the current pipeline
    #         including all steps that follows the one where the actual risk value is retrieved.
    #     :return:
    #     """
    #     if self.pre_assignment_idx is None:
    #         raise ValueError('pre_assignment_idx is None')
    #     return self.clone_template(starting_point=self.pre_assignment_idx+1)
