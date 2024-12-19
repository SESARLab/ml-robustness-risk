import typing

import numpy as np

import pipe


def reshaper(values: typing.List[np.ndarray],
             reshaping: typing.Optional[typing.Dict[int, tuple]] = None):
    if reshaping is not None:
        for reshaping_index, reshaping_shape in reshaping.items():
            values[reshaping_index] = values[reshaping_index].reshape(reshaping_shape)
    return values


def post_aggregation_take_col_and_where(col_idx: int, divider: float, left, right) -> pipe.AggregationFuncPair:
    return pipe.AggregationFuncPair(
        aggregation_func=lambda arr, divider_val: np.where(arr[..., col_idx] < divider_val, left, right),
        aggregation_func_arg_func=lambda arr: pipe.ArgFuncOutput(args=(arr[0], divider))
    )


def flattener(values: typing.Iterable[pipe.StepInput]) -> pipe.ArgFuncOutput:
    # here we need to flat any input.
    # new_X = []
    # for single_step_input in values:
    #     if len(single_step_input.X) == 1:
    #         new_X.append(single_step_input.X)
    #     else:
    #         # we do not force this conversion. If the item cannot be made 1d
    #         # we raise an Exception, since the purpose of this is
    #         # to flat arrays such as [[1], [2], ...]. If we can't, we signal.
    #         if single_step_input.X.shape[0] != 1:
    #             raise ValueError(f'Cannot coerce to a 1d array, shape: {single_step_input.X.shape}' )
    #         new_X.append(single_step_input.X.reshape(-1, 1))
    ## if all_elements_as_one_args:
    ##     return ArgFuncOutput(args=(new_X, ))
    ## else:
    ##     return ArgFuncOutput(args=tuple(new_X))
    # return ArgFuncOutput(args=(new_X,)
    return pipe.ArgFuncOutput(args=([s.X for s in values],))


def StepAggregateWeightedSum(name: str, to_aggregate: typing.List[int],
                             weights: typing.List[float],
                             reshaping: typing.Optional[typing.Dict[int, tuple]] = None,
                             output_col_names_pre: typing.Optional[typing.List[str]] = None,
                             output_col_names_post: typing.Optional[typing.List[str]] = None,
                             post_aggregation: typing.Optional[pipe.AggregationFuncPair] = None) -> pipe.Step:
    """
    TO BE DONE AFTER FLATTENING.
    :param post_aggregation:
    :param output_col_names_post:
    :param output_col_names_pre:
    :param name:
    :param to_aggregate:
    :param weights:
    :param reshaping:
    :return:
    """
    if to_aggregate is None or weights is None:
        raise ValueError('\'to_aggregated\' and \'weights\' cannot be None')

    def step_func(values: typing.List[np.ndarray]) -> np.ndarray:
        values = reshaper(reshaping=reshaping, values=values)
        return np.sum(np.hstack(values) * weights, axis=1)

    return pipe.Step(name=name, step=step_func, steps_to_aggregate=to_aggregate, arg_func=flattener,
                     output_col_names_pre=output_col_names_pre,
                     output_col_names_post=output_col_names_post, post_aggregation=post_aggregation)


def StepAggregateWeightedAverage(name: str, to_aggregate: typing.List[int],
                                 weights: typing.Optional[typing.List[float]],
                                 reshaping: typing.Optional[typing.Dict[int, tuple]] = None,
                                 output_col_names_pre: typing.Optional[typing.List[str]] = None,
                                 output_col_names_post: typing.Optional[typing.List[str]] = None,
                                 post_aggregation: typing.Optional[pipe.AggregationFuncPair] = None):
    """
    TO BE DONE AFTER FLATTENING.
    :param post_aggregation:
    :param output_col_names_post:
    :param output_col_names_pre:
    :param name:
    :param to_aggregate:
    :param weights:
    :param reshaping:
    :return:
    """
    if to_aggregate is None:
        raise ValueError('\'to_aggregated\' cannot be None')

    def step_func(values: typing.List[np.ndarray]):
        values = reshaper(reshaping=reshaping, values=values)
        return np.average(np.vstack(values), weights=weights, axis=1)

    return pipe.Step(name=name, step=step_func, steps_to_aggregate=to_aggregate, arg_func=flattener,
                     output_col_names_post=output_col_names_post,
                     output_col_names_pre=output_col_names_pre, post_aggregation=post_aggregation)


def StepAggregateMax(name: str, to_aggregate: typing.List[int],
                     reshaping: typing.Optional[typing.Dict[int, tuple]] = None,
                     output_col_names_pre: typing.Optional[typing.List[str]] = None,
                     output_col_names_post: typing.Optional[typing.List[str]] = None,
                     post_aggregation: typing.Optional[pipe.AggregationFuncPair] = None):
    """
    TO BE DONE AFTER FLATTENING.
    :param post_aggregation:
    :param output_col_names_post:
    :param output_col_names_pre:
    :param name:
    :param to_aggregate:
    :param reshaping:
    :return:
    """
    if to_aggregate is None:
        raise ValueError('\'to_aggregated\' cannot be None')

    def step_func(values: typing.List[np.ndarray]):
        values = reshaper(reshaping=reshaping, values=values)
        return np.max(np.vstack(values), axis=1)

    return pipe.Step(name=name, step=step_func, steps_to_aggregate=to_aggregate, arg_func=flattener,
                     output_col_names_pre=output_col_names_pre,
                     output_col_names_post=output_col_names_post, post_aggregation=post_aggregation)


def StepAggregateCount(name: str, to_aggregate: typing.List[int],
                       reshaping: typing.Optional[typing.Dict[int, tuple]] = None,
                       output_col_names_pre: typing.Optional[typing.List[str]] = None,
                       output_col_names_post: typing.Optional[typing.List[str]] = None,
                       post_aggregation: typing.Optional[pipe.AggregationFuncPair] = None):
    if to_aggregate is None:
        raise ValueError('\'to_aggregated\' cannot be None')

    def step_func(values: typing.List[np.ndarray]):
        values = reshaper(reshaping=reshaping, values=values)
        # print(f'{[x.shape for x in values]} -> {np.count_nonzero(np.vstack(values), axis=1).shape}')
        return np.count_nonzero(np.vstack(values), axis=1)

    # this basically takes as input an array (even if we specify a list, it always receives as input one array)
    # resulting from the aggregation of some IoPs. For instance, assuming to have two IoPs already aggregated,
    # values[0] will be (len(dataset), 2), where the first column contains the data of the first IoP
    # while the second on ethe second IoP.
    # It will return as output an array of shape (len(dataset),).

    return pipe.Step(name=name, step=step_func, steps_to_aggregate=to_aggregate, arg_func=flattener,
                     output_col_names_pre=output_col_names_pre,
                     output_col_names_post=output_col_names_post, post_aggregation=post_aggregation)


def StepAggregateFlattener(name: str, to_aggregate: typing.List[int],
                           expansion_mapper: typing.Optional[typing.Dict[int, int]] = None,
                           output_col_names_pre: typing.Optional[typing.List[str]] = None,
                           post_aggregation: typing.Optional[pipe.AggregationFuncPair] = None
                           ) -> pipe.Step:
    """
    aggregating IoPs which are by themselves multidimensional. *OR we make IoPs 1d* or we consider
    something, eg., use a post-aggregator for each IoP. Should not be be too difficult if we
    inherit from a common class.
    :param post_aggregation:
    :param output_col_names_pre:
    :param name:
    :param to_aggregate:
    :param expansion_mapper:
    :return:
    """
    # expansion_mapper specifies if and how to expand (i.e., repeat) some data
    if to_aggregate is None:
        raise ValueError('\'to_aggregated\' cannot be None')

    def step_func(values: typing.List[np.ndarray]):
        # note that here we are receiving as input
        # a list of multidimensional arrays.
        # The length of the list corresponds to the number of steps to aggregate
        # while each element is the vstack'd version of all the outputs of such steps.

        # first, we perform repetition any time it is necessary.
        # for i, single_value in enumerate(values):
        #     if expansion_mapper is not None and
        if expansion_mapper is not None:
            for row_index, expand_number in expansion_mapper.items():
                old_value = values[row_index]
                # we expand and then reshape so that the number of rows is the same e.g., from
                # [[5, 10, 5, 10],
                # [10, 5, 10, 5]]
                # becomes
                # array([[ 5,  5, 10, 10,  5,  5, 10, 10],
                #        [10, 10,  5,  5, 10, 10,  5,  5],
                new_value = np.repeat(old_value, expand_number).reshape(old_value.shape[0], -1)
                values[row_index] = new_value

        new_X = []

        # we might work with 1d array. They need to be reshaped
        # to ensure that we are working with arrays where each data point
        # has at least one feature.

        for i, single_values in enumerate(values):
            if len(single_values.shape) == 1:
                new_X.append(single_values.reshape(-1, 1))
            else:
                new_X.append(single_values)

        # now, the last step is straightforward, just hstack.
        result = np.hstack(new_X)
        return result

    # def arg_func_creator(values: typing.List[StepInput]) -> ArgFuncOutput:
    #     result = [v.X for v in values]
    #     return ArgFuncOutput(args=(result,))
    return pipe.Step(name=name, step=step_func, steps_to_aggregate=to_aggregate, arg_func=flattener,
                     output_col_names_pre=output_col_names_pre, post_aggregation=post_aggregation)

