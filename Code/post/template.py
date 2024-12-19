import copy
import dataclasses
import itertools
import typing

# for some reason, the dump file using json5 creates a strange result
# so we need json as well
import json
import json5 as json5
import mashumaro


@dataclasses.dataclass
class ReplaceDataPair(mashumaro.DataClassDictMixin):
    names_for_n: typing.List[str]
    values_for_n: typing.List[int]

    names_for_routing: typing.List[str]
    values_for_routing: typing.List[str]

    names_for_aggregation: typing.Optional[typing.List[str]] = dataclasses.field(default_factory=list)
    values_for_aggregation: typing.Optional[typing.List[str]] = dataclasses.field(default_factory=list)

    def __post_init__(self):
        if len(self.names_for_n) != len(self.values_for_n):
            raise ValueError(f'len(self.names_for_n) != len(self.values_for_n): '
                             f'{len(self.names_for_n)} != {len(self.values_for_n)}')
        if len(self.names_for_routing) != len(self.values_for_routing):
            raise ValueError(f'len(self.names_for_routing) != len(self.values_for_routing): '
                             f'{len(self.names_for_routing)} != {len(self.values_for_routing)}')
        if len(self.names_for_aggregation) != len(self.values_for_aggregation):
            raise ValueError(f'len(self.names_for_aggregation) != len(self.values_for_aggregation): '
                             f'{len(self.names_for_aggregation)} != {len(self.values_for_aggregation)}')


KEY_ROUTING_ALGO_NAME = 'ROUTING_ALGO_NAME'
KEY_ROUTING_ALGO_VALUE = 'ROUTING_ALGO_VALUE'

KEY_N_NAME = 'N_NAME'
KEY_N_VALUE = 'N_VALUE'

KEY_AGGREGATION_NAME = 'AGGREGATION_NAME'
KEY_AGGREGATION_VALUE = 'AGGREGATION_VALUE'


def _rename_name(new_pipeline: dict,
                 key: str,
                 n_name: str, routing_name: str, aggregation_name: typing.Optional[str] = None,
                 ):
    new_pipeline[key] = new_pipeline[key].replace('{{' + KEY_ROUTING_ALGO_NAME + '}}', routing_name)
    new_pipeline[key] = new_pipeline[key].replace('{{' + KEY_N_NAME + '}}', n_name)
    if aggregation_name is not None:
        new_pipeline[key] = new_pipeline[key].replace('{{' + KEY_AGGREGATION_NAME + '}}', aggregation_name)
    return new_pipeline

def replace_single_pipeline(pipelines: typing.List[dict],
                            replacer: ReplaceDataPair):
    new_pipelines = []

    names = set()

    for pipeline in pipelines:

        # some care is needed to build the iterators.
        # if there is no aggregation, the cartesian product ends up empty
        args = [zip(replacer.names_for_n, replacer.values_for_n),
                                 zip(replacer.names_for_routing, replacer.values_for_routing),]
        if len(replacer.names_for_aggregation) > 0:
            args += [zip(replacer.names_for_aggregation, replacer.values_for_aggregation)]

        loop = itertools.product(*args)

        for pack in loop:

                if len(replacer.names_for_aggregation) > 0:
                    (n_name, n_val), (routing_name, routing_value), (aggregation_name, aggregation_value) = pack[0], pack[1], pack[2]
                    kwargs = {'n_name': n_name, 'routing_name': routing_name, 'aggregation_name': aggregation_name}
                else:
                    (n_name, n_val), (routing_name, routing_value) = pack[0], pack[1]
                    kwargs = {'n_name': n_name, 'routing_name': routing_name}

                new_pipeline = copy.deepcopy(pipeline)

                new_pipeline = _rename_name(new_pipeline=new_pipeline, key='name', **kwargs)
                new_pipeline = _rename_name(new_pipeline=new_pipeline, key='short_name', **kwargs)

                if new_pipeline['name'] not in names:
                    # this check prevents the inclusion of duplicates, e.g., TSUSC where we loop over
                    # n only.
                    names.add(new_pipeline['name'])

                    # now, loop over the steps
                    for i, step in enumerate(pipeline['steps']):
                        if 'step_func_kwargs' in step:
                            if 'N' in step['step_func_kwargs']:
                                if '{{' + KEY_N_VALUE + '}}' in step['step_func_kwargs']['N']:
                                    new_pipeline['steps'][i]['step_func_kwargs']['N'] = n_val
                        if 'step_func_name' in step:
                            if '{{' + KEY_ROUTING_ALGO_VALUE + '}}' in step['step_func_name']:
                                new_pipeline['steps'][i]['step_func_name'] = new_pipeline['steps'][i]['step_func_name'].replace(
                                    '{{' + KEY_ROUTING_ALGO_VALUE + '}}', routing_value
                                )

                        if len(replacer.names_for_aggregation) > 0:
                            if 'func_name' in step:
                                # replace the function
                                if '{{' + KEY_AGGREGATION_VALUE + '}}' in step['func_name']:
                                    new_pipeline['steps'][i]['func_name'] = aggregation_value
                                # replace the name in the column.
                                if 'output_col_names_pre' in step.get('func_kwargs', {}):
                                    if '{{' + KEY_AGGREGATION_NAME + '}}' in step['func_kwargs']['output_col_names_pre'][0]:
                                        # print
                                        new_pipeline['steps'][i]['func_kwargs']['output_col_names_pre'][0] = \
                                            new_pipeline['steps'][i]['func_kwargs']['output_col_names_pre'][0].replace(
                                                '{{' + KEY_AGGREGATION_NAME + '}}', aggregation_name)
                    new_pipelines.append(new_pipeline)
    # note: in the end, we may have duplicate names (e.g., when using some pipelines
    # that do iterate over one
    return new_pipelines




def almost_top_make_template(data: dict, replace_data: ReplaceDataPair) -> dict:

    if 'know_all_pipelines' in data:
        new_know_all_pipelines = replace_single_pipeline(pipelines=data['know_all_pipelines'],
                                                         replacer=replace_data)
        data['know_all_pipelines'] = new_know_all_pipelines
    if 'pipelines' in data:
        new_pipelines = replace_single_pipeline(pipelines=data['pipelines'],
                                                replacer=replace_data)
        data['pipelines'] = new_pipelines

    return data

def top_make_template(args):
    with open(args.input_file) as f:
        data = json5.load(f)

    # just because
    data_old = copy.deepcopy(data)

    with open(args.map_file) as f:
        replace_data = json5.load(f)
        replace_data = ReplaceDataPair.from_dict(replace_data)

    result = almost_top_make_template(data=data, replace_data=replace_data)

    # just because.
    for old, new, text, in [
        (len(data_old.get('know_all_pipelines', [])),
         len(result.get('know_all_pipelines', [])),
         'know_all_pipelines',),
        (len(data_old.get('pipelines', [])),
         len(result.get('pipelines', [])),
         'pipelines',),
    ]:
        print(f'From {old} {text} I generated {new} {text}')

    with open(args.output_file, 'w+') as f:
        f.write(json.dumps(result, indent=4))



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    sub_parsers = parser.add_subparsers()

    parser_make_template = sub_parsers.add_parser('make-template')
    parser_make_template.add_argument('--input-file', required=True, type=str)
    parser_make_template.add_argument('--map-file', required=True, type=str)
    parser_make_template.add_argument('--output-file', required=True, type=str)
    parser_make_template.set_defaults(func=top_make_template)

    args_ = parser.parse_args()
    args_.func(args_)